"""BuildEngine: the public entry point (build, value lookup, planning)."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from opentelemetry import trace as otel_trace
from opentelemetry.trace import Tracer
from pydantic import JsonValue

from incremental_agent_builds.agentic.audit import AuditLog
from incremental_agent_builds.agentic.budget import BudgetLedger, BudgetLimits
from incremental_agent_builds.agentic.providers.base import LLMProvider
from incremental_agent_builds.core.canonical import canonical_json, sha256_hex
from incremental_agent_builds.engine import hermetic
from incremental_agent_builds.engine.cas import ContentStore
from incremental_agent_builds.engine.errors import (
    BuildAbortedError,
    CycleError,
    RuleFailedError,
)
from incremental_agent_builds.engine.hermetic import GuardMode
from incremental_agent_builds.engine.keys import BuildKey
from incremental_agent_builds.engine.metrics import BuildMetrics, create_build_metrics
from incremental_agent_builds.engine.rules import RuleSet
from incremental_agent_builds.engine.session import BuildSession, BuildStats
from incremental_agent_builds.engine.signing import TrustStore
from incremental_agent_builds.engine.store import Trace, TraceStore
from incremental_agent_builds.engine.tools import BuildToolRegistry
from incremental_agent_builds.engine.workspace import Workspace

__all__ = ["BuildEngine", "BuildResult", "EngineConfig"]

DEFAULT_BUDGET = BudgetLimits.of(steps=1_000, tokens=5_000_000, usd=50.0)


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Engine behaviour. ``early_cutoff`` and ``step_memo`` exist for ablations; keep both on."""

    jobs: int = 4
    llm_concurrency: int = 4
    early_cutoff: bool = True
    step_memo: bool = True
    guard: GuardMode = "warn"
    redact_prompts: bool = True
    config_path: str = "iab.toml"

    def __post_init__(self) -> None:
        if self.jobs < 1 or self.llm_concurrency < 1:
            raise ValueError("jobs and llm_concurrency must be >= 1")


@dataclass
class BuildResult:
    run_id: int
    targets: tuple[str, ...]
    target_hashes: dict[str, str]
    live: dict[str, str]
    executed: list[str]
    failures: dict[str, str]
    cycles: list[tuple[str, ...]]
    aborted: str | None
    stats: BuildStats
    duration_seconds: float
    errors: dict[str, BaseException] = field(default_factory=dict, repr=False)

    @property
    def ok(self) -> bool:
        return not self.failures and not self.cycles and self.aborted is None

    def raise_for_failures(self) -> None:
        """Raise the most informative error of a failed build (cycles first)."""
        if self.aborted is not None:
            raise BuildAbortedError(self.aborted)
        if self.cycles:
            raise CycleError(self.cycles[0])
        for error in self.errors.values():
            raise error
        if self.failures:
            key_id, message = next(iter(self.failures.items()))
            raise RuleFailedError(key_id, RuntimeError(message))

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "run_id": self.run_id,
            "ok": self.ok,
            "targets": list(self.targets),
            "target_hashes": dict(self.target_hashes),
            "executed": list(self.executed),
            "failures": dict(self.failures),
            "cycles": [list(cycle) for cycle in self.cycles],
            "aborted": self.aborted,
            "stats": self.stats.to_json(),
            "duration_seconds": round(self.duration_seconds, 4),
        }


class BuildEngine:
    """Rules + workspace + stores. One build at a time per engine (and per store)."""

    def __init__(
        self,
        *,
        rules: RuleSet,
        workspace: Workspace,
        store: TraceStore,
        cas: ContentStore,
        provider: LLMProvider | None = None,
        tools: BuildToolRegistry | None = None,
        config: EngineConfig | None = None,
        audit: AuditLog | None = None,
        metrics: BuildMetrics | None = None,
        tracer: Tracer | None = None,
        trust: TrustStore | None = None,
        default_budget: BudgetLimits = DEFAULT_BUDGET,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.rules = rules
        self.workspace = workspace
        self.store = store
        self.cas = cas
        self.provider = provider
        self.tools = tools or BuildToolRegistry()
        self.config = config or EngineConfig()
        self.audit_log = audit
        self.metrics = metrics or create_build_metrics()
        self.tracer = tracer or otel_trace.get_tracer("incremental_agent_builds.engine")
        self.trust = trust
        self.default_budget = default_budget
        self.clock = clock
        self._sync_lock = threading.Lock()
        if self.config.guard != "off":
            hermetic.install_guard()

    # --- building -----------------------------------------------------------------------------

    async def build(
        self,
        targets: Sequence[BuildKey],
        *,
        force: Iterable[BuildKey | str] = (),
        budget: BudgetLimits | None = None,
        principal: str = "local",
    ) -> BuildResult:
        return await self._build(targets, force=force, budget=budget, principal=principal)

    def build_sync(
        self,
        targets: Sequence[BuildKey],
        *,
        force: Iterable[BuildKey | str] = (),
        budget: BudgetLimits | None = None,
        principal: str = "local",
    ) -> BuildResult:
        """Run a build on a fresh event loop. Builds on one engine are serialized."""
        with self._sync_lock:
            return asyncio.run(
                self._build(targets, force=force, budget=budget, principal=principal)
            )

    async def _build(
        self,
        targets: Sequence[BuildKey],
        *,
        force: Iterable[BuildKey | str],
        budget: BudgetLimits | None,
        principal: str,
    ) -> BuildResult:
        started = time.perf_counter()
        unique = list({target.key_id: target for target in targets}.values())
        force_ids = frozenset(item if isinstance(item, str) else item.key_id for item in force)
        run_id = self.store.begin_run()
        ledger = BudgetLedger(budget or self.default_budget)
        session = BuildSession(
            self, run_id=run_id, force=force_ids, budget=ledger, principal=principal
        )
        # Built as comprehensions with the target type in scope: ``list`` is invariant, so a
        # ``list[str]`` returned by ``sorted`` is not a ``list[JsonValue]``.
        forced: list[JsonValue] = [key_id for key_id in sorted(force_ids)]  # noqa: C416
        targets_logged: list[JsonValue] = [target.key_id for target in unique]
        session.audit("build_started", {"run": run_id, "targets": targets_logged, "force": forced})
        violations_before = len(hermetic.violations())
        with self.tracer.start_as_current_span(
            "iab.build", attributes={"iab.run": run_id, "iab.targets": len(unique)}
        ) as span:
            try:
                outcomes = await asyncio.gather(
                    *(session.ensure(target, None) for target in unique), return_exceptions=True
                )
            finally:
                await session.drain()
                session.flush()
            session.finish()
            span.set_attribute("iab.executed", session.stats.executed)
            span.set_attribute("iab.reused", session.stats.reused)
        target_hashes: dict[str, str] = {}
        errors: dict[str, BaseException] = {}
        for target, outcome in zip(unique, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                errors[target.key_id] = outcome
            else:
                target_hashes[target.key_id] = outcome.hash
        failures = {key: f"{type(err).__name__}: {err}" for key, err in session.failures.items()}
        for key, err in errors.items():
            failures.setdefault(key, f"{type(err).__name__}: {err}")
        new_violations = hermetic.violations()[violations_before:]
        live = session.live_hashes()
        session.stats.hermetic_violations = sum(1 for v in new_violations if v.key_id in live)
        stats = session.stats
        duration = time.perf_counter() - started
        aborted = session.abort_error.reason if session.abort_error is not None else None
        result = BuildResult(
            run_id=run_id,
            targets=tuple(t.key_id for t in unique),
            target_hashes=target_hashes,
            live=live,
            executed=list(session.executed_keys),
            failures=failures,
            cycles=list(session.cycles),
            aborted=aborted,
            stats=stats,
            duration_seconds=duration,
            errors=errors,
        )
        self._record_metrics(result, stats)
        reused_digest = sha256_hex(
            canonical_json(sorted(k for k in live if k not in set(session.executed_keys)))
        )
        session.audit(
            "build_finished",
            {
                "run": run_id,
                "ok": result.ok,
                "aborted": aborted,
                "stats": stats.to_json(),
                "reused_keys_sha256": reused_digest,
                "targets": dict(target_hashes),
            },
        )
        if self.audit_log is not None:
            self.audit_log.append_many(session.audit_records)
        self.store.finish_run(run_id, result.to_json())
        return result

    def _record_metrics(self, result: BuildResult, stats: BuildStats) -> None:
        metrics = self.metrics
        for rule_name, count in stats.executed_by_rule.items():
            metrics.rules_executed_total.labels(rule_name).inc(count)
        metrics.rules_reused_total.inc(stats.reused)
        metrics.early_cutoffs_total.inc(stats.early_cutoffs)
        metrics.constructive_hits_total.inc(stats.imported)
        metrics.llm_provider_calls_total.inc(stats.provider_calls)
        metrics.llm_tokens_saved_total.inc(stats.tokens_saved)
        metrics.hermeticity_violations_total.inc(stats.hermetic_violations)
        metrics.build_duration_seconds.observe(result.duration_seconds)

    # --- inspection ---------------------------------------------------------------------------

    def trace(self, key: BuildKey | str) -> Trace | None:
        return self.store.get_trace(key if isinstance(key, str) else key.key_id)

    def value(self, key: BuildKey | str) -> JsonValue:
        """The recorded value of a key (SHA-256 verified). Raises KeyError if never built."""
        found = self.trace(key)
        if found is None:
            raise KeyError(key if isinstance(key, str) else key.key_id)
        return self.cas.get_json(found.result_hash)

    def values(self, hashes: dict[str, str]) -> dict[str, Any]:
        return {key: self.cas.get_json(digest) for key, digest in hashes.items()}
