"""Dry-run planning: which keys may re-run, and what that may cost, without running any rule.

The planner walks recorded traces from the targets. A key is *dirty* when a recorded input
(file, glob, config value) no longer matches, when an LLM or tool step it used is gone or its
volatile result expired, when its rule version changed, when it is forced, or when any key it
needs is dirty. Unlike a real build, the plan cannot see early cutoff (it would have to run the
rule to know the new value), so the dirty set is an upper bound for keys that have traces.

Keys that have never been built are *new*: their dependencies are unknown until they run, and so
are any keys that dirty rules will discover at runtime. New keys are priced at the mean recorded
cost of their rule, or reported as unestimated. Callers treat an unestimated plan conservatively
(approval required) and bound the real build with a budget derived from the plan.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import JsonValue

from incremental_agent_builds.agentic.budget import BudgetLedger, BudgetLimits
from incremental_agent_builds.core.canonical import canonical_json, sha256_hex
from incremental_agent_builds.engine.errors import WorkspaceError
from incremental_agent_builds.engine.hashing import MISSING
from incremental_agent_builds.engine.keys import BuildKey
from incremental_agent_builds.engine.session import BuildSession
from incremental_agent_builds.engine.store import Dep, DepKind, Trace

__all__ = ["Plan", "PlanItem", "explain", "make_plan"]

# A plan-bound build may spend this many LLM tokens / micro-USD beyond the scaled estimate, so a
# plan that projects zero cost can still absorb one or two small steps the plan could not see.
_TOKEN_SLACK = 8_000
_USD_MICRO_SLACK = 100_000
_STEP_SLACK = 2


@dataclass(frozen=True, slots=True)
class PlanItem:
    key_id: str
    rule: str
    status: Literal["new", "dirty"]
    reason: str
    llm_steps: int
    tokens: int
    usd_micros: int
    estimated: bool

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "key": self.key_id,
            "rule": self.rule,
            "status": self.status,
            "reason": self.reason,
            "llm_steps": self.llm_steps,
            "tokens": self.tokens,
            "usd_micros": self.usd_micros,
            "estimated": self.estimated,
        }


@dataclass(frozen=True, slots=True)
class Plan:
    targets: tuple[str, ...]
    force: tuple[str, ...]
    items: tuple[PlanItem, ...]
    clean_keys: int
    store_run: int
    # SHA-256 over every non-rule input the planner examined and the hash it had at plan time.
    input_digest: str = ""

    @property
    def projected_llm_steps(self) -> int:
        return sum(item.llm_steps for item in self.items)

    @property
    def projected_tokens(self) -> int:
        return sum(item.tokens for item in self.items)

    @property
    def projected_usd_micros(self) -> int:
        return sum(item.usd_micros for item in self.items)

    @property
    def unestimated_keys(self) -> int:
        return sum(1 for item in self.items if not item.estimated)

    @property
    def dirty_keys(self) -> tuple[str, ...]:
        return tuple(item.key_id for item in self.items)

    @property
    def plan_hash(self) -> str:
        return sha256_hex(
            canonical_json(
                {
                    "v": 1,
                    "targets": list(self.targets),
                    "force": list(self.force),
                    "items": [[i.key_id, i.status, i.reason] for i in self.items],
                    "llm_steps": self.projected_llm_steps,
                    "tokens": self.projected_tokens,
                    "usd_micros": self.projected_usd_micros,
                    "store_run": self.store_run,
                    "inputs": self.input_digest,
                }
            )
        )

    def approval_required(
        self, *, threshold_usd_micros: int, unestimated_needs_approval: bool
    ) -> bool:
        if self.projected_usd_micros > threshold_usd_micros:
            return True
        return unestimated_needs_approval and self.unestimated_keys > 0

    def budget(self, *, factor: float, ceiling: BudgetLimits) -> BudgetLimits:
        """Budget for executing this plan: the scaled estimate plus slack, never above ceiling.

        An unestimated plan gets the ceiling (the configured per-build budget).
        """
        if self.unestimated_keys:
            return ceiling
        return ceiling.tightest(
            BudgetLimits(
                max_steps=math.ceil(self.projected_llm_steps * factor) + _STEP_SLACK,
                max_tokens=math.ceil(self.projected_tokens * factor) + _TOKEN_SLACK,
                max_usd_micros=math.ceil(self.projected_usd_micros * factor) + _USD_MICRO_SLACK,
            )
        )

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "plan_sha256": self.plan_hash,
            "targets": list(self.targets),
            "force": list(self.force),
            "dirty": [item.to_json() for item in self.items],
            "dirty_count": len(self.items),
            "clean_count": self.clean_keys,
            "projected_llm_steps": self.projected_llm_steps,
            "projected_tokens": self.projected_tokens,
            "projected_usd_micros": self.projected_usd_micros,
            "unestimated_keys": self.unestimated_keys,
            "store_run": self.store_run,
            "input_sha256": self.input_digest,
        }


class _Planner:
    def __init__(self, engine: object, force: frozenset[str]) -> None:
        from incremental_agent_builds.engine.engine import BuildEngine  # noqa: PLC0415

        if not isinstance(engine, BuildEngine):  # pragma: no cover - typed callers only
            raise TypeError("engine must be a BuildEngine")
        self.engine = engine
        self.force = force
        # A session is used only for its trace snapshot and input hashing; nothing is executed.
        self.session = BuildSession(
            engine,
            run_id=0,
            force=force,
            budget=BudgetLedger(BudgetLimits(max_steps=0, max_tokens=0, max_usd_micros=0)),
            principal="planner",
        )
        self.traces = engine.store.load_traces()
        self.status: dict[str, tuple[bool, str]] = {}
        # Every non-rule input the planner looked at, with the hash it has *now*. This is what binds
        # a plan (and an approval of it) to one state of the workspace: two different edits that
        # happen to dirty the same keys at the same projected cost still produce different plan
        # hashes, so an approval never carries over to inputs the approver did not see.
        self.observed: dict[str, str] = {}
        self._rule_means: dict[str, tuple[int, int, int] | None] = {}

    def _current_hash(self, dep: Dep) -> str:  # noqa: PLR0911 - one per dep kind
        """The hash ``dep``'s input has now, or ``MISSING``/``unavailable`` when it has none."""
        session = self.session
        try:
            match dep.kind:
                case DepKind.FILE:
                    return session.file_input(dep.ref)[1]
                case DepKind.GLOB:
                    return session.glob_input(dep.ref)[1]
                case DepKind.CONFIG:
                    path, _, dotted = dep.ref.partition("#")
                    if path != session.config.config_path:
                        return "foreign-config"
                    return session.config_input(dotted)[1]
                case DepKind.LLM | DepKind.TOOL:
                    step = session.step(dep.ref)
                    if step is None:
                        return MISSING
                    if dep.kind is DepKind.TOOL and not session.tool_version_current(step):
                        return "stale-tool-version"
                    ttl = step.meta.get("ttl_seconds")
                    if isinstance(ttl, int | float) and session.now - step.recorded_at > ttl:
                        return "expired"
                    return step.result_hash
                case DepKind.ERROR:
                    # A call that raised during the last execution: the key is dirty whatever the
                    # rest of its trace says, because nothing recorded can show it is up to date.
                    return "failed-call"
                case _:  # pragma: no cover - rule deps are handled separately
                    return dep.hash
        except WorkspaceError:
            return "unavailable"

    def _input_changed(self, dep: Dep) -> str | None:
        current = self._current_hash(dep)
        self.observed[f"{dep.kind.value} {dep.ref}"] = current
        if current == dep.hash:
            return None
        if dep.kind is DepKind.ERROR:
            return f"{dep.ref} failed during the last execution"
        return f"{dep.kind.value} {dep.ref} changed"

    def _own_reason(self, key_id: str, trace: Trace | None) -> str | None:
        key = self.engine.rules.key_from_id(key_id)
        if key is None:
            return "no rule defines this key"
        rule = self.engine.rules.rule_for(key)
        if trace is None:
            return None
        if trace.rule != rule.name or trace.version != rule.version:
            return f"rule version {trace.version} -> {rule.version}"
        if key_id in self.force:
            return "forced"
        return None

    def evaluate(self, roots: Sequence[str]) -> None:  # noqa: PLR0912 - one explicit iterative walk
        visiting: set[str] = set()
        stack: list[tuple[str, bool]] = [(root, False) for root in reversed(roots)]
        while stack:
            key_id, expanded = stack.pop()
            if key_id in self.status:
                continue
            trace = self.traces.get(key_id)
            if not expanded:
                visiting.add(key_id)
                stack.append((key_id, True))
                if trace is not None:
                    children = [d.ref for g in trace.deps for d in g if d.kind is DepKind.RULE]
                    for child in reversed(children):
                        if child not in self.status and child not in visiting:
                            stack.append((child, False))
                continue
            visiting.discard(key_id)
            own = self._own_reason(key_id, trace)
            if trace is None:
                self.status[key_id] = (True, own or "new key")
                continue
            reason = own
            for group in trace.deps:
                for dep in group:
                    if reason is not None:
                        break
                    if dep.kind is DepKind.RULE:
                        # A needed key with no status yet was skipped as already-visiting,
                        # which happens only when the recorded traces contain a cycle.
                        needed = self.status.get(dep.ref)
                        if needed is None:
                            reason = f"{dep.ref} is part of a recorded cycle"
                        elif needed[0]:
                            reason = f"{dep.ref} may change"
                    else:
                        reason = self._input_changed(dep)
            self.status[key_id] = (reason is not None, reason or "")

    def rule_mean(self, rule_name: str) -> tuple[int, int, int] | None:
        if rule_name not in self._rule_means:
            samples = [t.usage for t in self.traces.values() if t.rule == rule_name]
            if not samples:
                self._rule_means[rule_name] = None
            else:
                n = len(samples)
                self._rule_means[rule_name] = (
                    math.ceil(sum(u.llm_steps for u in samples) / n),
                    math.ceil(sum(u.tokens for u in samples) / n),
                    math.ceil(sum(u.usd_micros for u in samples) / n),
                )
        return self._rule_means[rule_name]

    def item(self, key_id: str) -> PlanItem | None:
        dirty, reason = self.status[key_id]
        if not dirty:
            return None
        trace = self.traces.get(key_id)
        key = self.engine.rules.key_from_id(key_id)
        rule_name = key.kind if key is not None else key_id.partition(":")[0]
        if trace is not None:
            usage = trace.usage
            return PlanItem(
                key_id,
                rule_name,
                "dirty",
                reason,
                usage.llm_steps,
                usage.tokens,
                usage.usd_micros,
                estimated=True,
            )
        mean = self.rule_mean(rule_name)
        if mean is None:
            return PlanItem(key_id, rule_name, "new", reason, 0, 0, 0, estimated=False)
        return PlanItem(key_id, rule_name, "new", reason, *mean, estimated=True)


def make_plan(
    engine: object, targets: Sequence[BuildKey], *, force: Iterable[BuildKey | str] = ()
) -> Plan:
    force_ids = frozenset(item if isinstance(item, str) else item.key_id for item in force)
    planner = _Planner(engine, force_ids)
    roots = list(dict.fromkeys(target.key_id for target in targets))
    planner.evaluate(roots)
    items = [item for key_id in planner.status if (item := planner.item(key_id)) is not None]
    items.sort(key=lambda item: item.key_id)
    clean = sum(1 for dirty, _ in planner.status.values() if not dirty)
    return Plan(
        targets=tuple(roots),
        force=tuple(sorted(force_ids)),
        items=tuple(items),
        clean_keys=clean,
        store_run=planner.engine.store.last_run_id(),
        input_digest=sha256_hex(canonical_json(dict(sorted(planner.observed.items())))),
    )


def explain(engine: object, key: BuildKey | str) -> list[str]:
    """Why ``key`` would re-run: the chain of first reasons down to a changed input."""
    key_id = key if isinstance(key, str) else key.key_id
    planner = _Planner(engine, frozenset())
    planner.evaluate([key_id])
    chain: list[str] = []
    current = key_id
    seen: set[str] = set()
    while current not in seen:
        seen.add(current)
        dirty, reason = planner.status.get(current, (False, ""))
        if not dirty:
            chain.append(f"{current}: up to date")
            break
        chain.append(f"{current}: {reason}")
        if not reason.endswith(" may change"):
            break
        current = reason.removesuffix(" may change")
    return chain
