"""One build: a suspending scheduler over verifying traces with early cutoff.

Scheduling (Build Systems a la Carte's *suspending* scheduler): every key is computed by its own
asyncio task, started the first time anything needs it. A rule that needs a key suspends until
that key is up to date, and releases its concurrency slot while it waits, so ``jobs`` bounds the
number of rule bodies actually running without deadlocking on deep dependency chains.

Rebuilding (*verifying traces*): for a key with a trace, the dependencies recorded at its last
execution are checked in the recorded order. A needed key is first brought up to date (which may
execute it) and its current value hash is compared with the recorded one; an input (file, glob,
config value) is re-hashed; an LLM or tool step must still hold the recorded output. The first
mismatch stops the check and the rule re-runs. Checking in order matters because dependencies are
dynamic: a key requested after a changed value may no longer be requested at all, so it must not
be built. When everything matches, the recorded result is reused without running the rule.

*Early cutoff*: when a rule re-runs and produces the same result hash as before, its dependents
see an unchanged hash and are not re-run. ``early_cutoff=False`` (an ablation) makes every
re-executed key count as changed.

Cycles: a waits-for graph over running keys is checked before every suspension, so a key that
(transitively) waits for itself raises :class:`CycleError` instead of deadlocking, including when
the cycle spans tasks started from different targets.
"""

from __future__ import annotations

import asyncio
import contextvars
import datetime as dt
import tomllib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import JsonValue

from incremental_agent_builds.agentic.budget import BudgetLedger
from incremental_agent_builds.agentic.errors import ToolNotAllowedError
from incremental_agent_builds.core.canonical import CanonicalJSONError, canonical_json, sha256_hex
from incremental_agent_builds.engine import hermetic
from incremental_agent_builds.engine.errors import (
    BuildAbortedError,
    CycleError,
    IntegrityError,
    RuleDefinitionError,
    RuleFailedError,
    ToolStepError,
    WorkspaceError,
)
from incremental_agent_builds.engine.hashing import MISSING, value_hash
from incremental_agent_builds.engine.keys import BuildKey
from incremental_agent_builds.engine.rules import Rule
from incremental_agent_builds.engine.store import (
    Dep,
    DepGroups,
    DepKind,
    ImportedStep,
    StepRecord,
    Trace,
)

if TYPE_CHECKING:
    from incremental_agent_builds.engine.engine import BuildEngine

__all__ = ["BuildSession", "BuildStats", "KeyResult"]


@dataclass
class BuildStats:
    executed: int = 0
    reused: int = 0
    imported: int = 0
    early_cutoffs: int = 0
    failed: int = 0
    regenerated: int = 0
    llm_steps: int = 0
    llm_memo_hits: int = 0
    provider_calls: int = 0
    tool_runs: int = 0
    tool_memo_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd_micros: int = 0
    tokens_saved: int = 0
    integrity_events: int = 0
    hermetic_violations: int = 0
    executed_by_rule: dict[str, int] = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_json(self) -> dict[str, JsonValue]:
        data: dict[str, JsonValue] = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "executed_by_rule"
        }
        data["executed_by_rule"] = dict(sorted(self.executed_by_rule.items()))
        return data


@dataclass(frozen=True, slots=True)
class KeyResult:
    key_id: str
    hash: str
    executed: bool
    changed: bool


def _json_compatible(value: Any) -> JsonValue:
    if isinstance(value, dict):
        return {str(k): _json_compatible(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_compatible(v) for v in value]
    if isinstance(value, dt.datetime | dt.date | dt.time):
        return value.isoformat()
    result: JsonValue = value
    return result


def _first_error(results: Sequence[object]) -> BaseException | None:
    errors = [item for item in results if isinstance(item, BaseException)]
    for preferred in (BuildAbortedError, CycleError):
        for error in errors:
            if isinstance(error, preferred):
                return error
    return errors[0] if errors else None


class _Slot:
    """A rule's claim on one of the ``jobs`` slots; released while the rule is suspended."""

    def __init__(self, semaphore: asyncio.Semaphore) -> None:
        self._semaphore = semaphore
        self.held = False

    async def acquire(self) -> None:
        await self._semaphore.acquire()
        self.held = True

    def release(self) -> None:
        if self.held:
            self.held = False
            self._semaphore.release()


class BuildSession:
    def __init__(
        self,
        engine: BuildEngine,
        *,
        run_id: int,
        force: frozenset[str],
        budget: BudgetLedger,
        principal: str,
    ) -> None:
        self.engine = engine
        self.run_id = run_id
        self.force = force
        self.budget = budget
        self.principal = principal
        self.config = engine.config
        self.now = engine.clock()
        self.stats = BuildStats()
        self.audit_records: list[tuple[str, str, dict[str, JsonValue]]] = []
        self.failures: dict[str, BaseException] = {}
        self.cycles: list[tuple[str, ...]] = []
        self.executed_keys: list[str] = []
        self.abort_error: BuildAbortedError | None = None
        # Captured before the loads below, so a row this session finds unreadable is counted and
        # audited by the build that hit it (events already in the list belong to earlier sessions).
        self._integrity_reported = len(engine.store.integrity_events)
        self._old_traces = engine.store.load_traces()
        self._old_steps = engine.store.load_steps()
        self._imported_steps: dict[str, ImportedStep] = engine.store.load_imported_steps()
        self._verified_imports: dict[str, StepRecord | None] = {}
        self._new_traces: dict[str, Trace] = {}
        self._new_steps: dict[str, StepRecord] = {}
        self._futures: dict[str, asyncio.Future[KeyResult]] = {}
        self._waits: dict[str, Counter[str]] = {}
        self._values: dict[str, JsonValue] = {}
        self._regenerating: dict[str, asyncio.Future[JsonValue]] = {}
        self._files: dict[str, tuple[bytes | None, str]] = {}
        self._globs: dict[str, tuple[list[str], str]] = {}
        self._config_doc: dict[str, Any] | None = None
        self._config_error: str | None = None
        self._config_loaded = False
        self._refreshed: set[str] = set()
        self._tasks: set[asyncio.Task[None]] = set()
        self._rule_slots = asyncio.Semaphore(self.config.jobs)
        self.llm_slots = asyncio.Semaphore(self.config.llm_concurrency)

    # --- records ------------------------------------------------------------------------------

    def trace_of(self, key_id: str) -> Trace | None:
        return self._new_traces.get(key_id) or self._old_traces.get(key_id)

    def old_trace(self, key_id: str) -> Trace | None:
        return self._old_traces.get(key_id)

    def step(self, fingerprint: str) -> StepRecord | None:
        """The memoized LLM output or tool result for ``fingerprint``, if one may be used."""
        local = self._new_steps.get(fingerprint) or self._old_steps.get(fingerprint)
        if local is not None:
            return local
        return self._imported_step(fingerprint)

    def _imported_step(self, fingerprint: str) -> StepRecord | None:
        """A step record from a shared-cache bundle, but only while its signer is still trusted.

        An imported LLM output or tool result is foreign data: a peer chose the value this store
        would otherwise have paid a provider for. Like an imported constructive trace it is only a
        candidate, so its Ed25519 signature is re-checked here, against the allowlist as it is
        *now*, every build. That is what makes dropping a key a retroactive revocation of its step
        records and not only of its traces. The answer is cached per session because the allowlist
        cannot change while a build runs.
        """
        if fingerprint in self._verified_imports:
            return self._verified_imports[fingerprint]
        entry = self._imported_steps.get(fingerprint)
        record: StepRecord | None = None
        if entry is not None:
            trust = self.engine.trust
            if trust is not None and trust.verify(entry.signer, entry.body, entry.signature):
                record = entry.as_step_record()
            else:
                self.engine.metrics.cache_import_rejections_total.labels("untrusted_at_use").inc()
                self.audit(
                    "imported_step_rejected",
                    {"fingerprint": fingerprint, "signer": entry.signer},
                )
        self._verified_imports[fingerprint] = record
        return record

    def record_step(self, step: StepRecord) -> None:
        self._new_steps[step.fingerprint] = step

    def audit(self, action: str, data: dict[str, JsonValue]) -> None:
        self.audit_records.append((self.principal, action, data))

    def integrity(self, kind: str, ref: str, reason: str) -> None:
        self.stats.integrity_events += 1
        self.engine.metrics.integrity_failures_total.labels(kind).inc()
        self.audit("integrity_failure", {"kind": kind, "ref": ref, "reason": reason})

    def _collect_store_integrity_events(self) -> None:
        events = self.engine.store.integrity_events[self._integrity_reported :]
        self._integrity_reported += len(events)
        for event in events:
            self.integrity(event.table, event.ref, event.reason)

    def flush(self) -> None:
        self.engine.store.put_traces(self._new_traces.values())
        self.engine.store.put_steps(self._new_steps.values())

    # --- inputs -------------------------------------------------------------------------------

    def file_input(self, path: str) -> tuple[bytes | None, str]:
        cached = self._files.get(path)
        if cached is None:
            data = self.engine.workspace.read_bytes(path)
            cached = (data, MISSING if data is None else sha256_hex(data))
            self._files[path] = cached
        return cached

    def glob_input(self, pattern: str) -> tuple[list[str], str]:
        cached = self._globs.get(pattern)
        if cached is None:
            paths = self.engine.workspace.glob(pattern)
            cached = (paths, value_hash(list(paths)))
            self._globs[pattern] = cached
        return cached

    def config_input(self, dotted: str) -> tuple[JsonValue | None, str]:
        if not self._config_loaded:
            self._config_loaded = True
            data, _ = self.file_input(self.config.config_path)
            if data is not None:
                try:
                    self._config_doc = tomllib.loads(data.decode("utf-8"))
                except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
                    self._config_error = f"{self.config.config_path} is not valid TOML: {exc}"
        if self._config_error is not None:
            raise WorkspaceError(self._config_error)
        node: Any = self._config_doc or {}
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return None, MISSING
            node = node[part]
        value = _json_compatible(node)
        return value, value_hash(value)

    async def _input_matches(self, dep: Dep) -> bool:  # noqa: PLR0911 - one per dep kind
        try:
            match dep.kind:
                case DepKind.FILE:
                    return self.file_input(dep.ref)[1] == dep.hash
                case DepKind.GLOB:
                    return self.glob_input(dep.ref)[1] == dep.hash
                case DepKind.CONFIG:
                    path, _, dotted = dep.ref.partition("#")
                    if path != self.config.config_path:
                        return False
                    return self.config_input(dotted)[1] == dep.hash
                case DepKind.LLM:
                    step = self.step(dep.ref)
                    return step is not None and step.kind == "llm" and step.result_hash == dep.hash
                case DepKind.TOOL:
                    return await self._tool_matches(dep)
                case DepKind.ERROR:
                    # A call the last execution made raised. Whatever it would return now, this
                    # trace cannot show the key is up to date, so it always re-runs.
                    return False
                case _:  # pragma: no cover - rule deps are handled by _verify
                    return False
        except WorkspaceError:
            return False

    def tool_version_current(self, step: StepRecord) -> bool:
        """Whether the tool that produced ``step`` still has the version it was recorded with.

        A tool's version is part of its step fingerprint, but a key that used the tool records the
        *old* fingerprint, which stays valid in the store. Without this check a tool version bump
        (the tool's code changed meaning) would leave every key that used it reusing the old
        result, exactly the stale-cache failure this engine exists to avoid. A tool that is no
        longer registered also counts as changed, so the rule re-runs and fails loudly.
        """
        name, version = step.meta.get("tool"), step.meta.get("version")
        if not isinstance(name, str) or not isinstance(version, int):
            return False
        try:
            return self.engine.tools.get(name).version == version
        except ToolNotAllowedError:
            return False

    async def _tool_matches(self, dep: Dep) -> bool:
        step = self.step(dep.ref)
        if step is None or step.kind != "tool" or step.result_hash != dep.hash:
            return False
        if not self.tool_version_current(step):
            return False
        ttl = step.meta.get("ttl_seconds")
        if not isinstance(ttl, int | float) or self.now - step.recorded_at <= ttl:
            return True
        if dep.ref in self._refreshed:
            fresh = self.step(dep.ref)
            return fresh is not None and fresh.result_hash == dep.hash
        refreshed = await self.refresh_volatile(step)
        return refreshed is not None and refreshed.result_hash == dep.hash

    async def refresh_volatile(self, step: StepRecord) -> StepRecord | None:
        """Re-run an expired volatile tool step; returns the new record (same fingerprint)."""
        self._refreshed.add(step.fingerprint)
        meta = step.meta
        name, args = meta.get("tool"), meta.get("args")
        inputs_meta = meta.get("inputs")
        if not isinstance(name, str) or not isinstance(args, dict):
            return None
        try:
            tool = self.engine.tools.get(name)
            parsed, _ = tool.validate(args)
            inputs: dict[str, bytes | None] = {}
            if isinstance(inputs_meta, dict):
                for path in inputs_meta:
                    inputs[path] = self.file_input(path)[0]
            result = await asyncio.to_thread(tool.handler, parsed, inputs)
            digest = self.engine.cas.put_json(result)
        except (ToolStepError, WorkspaceError, CanonicalJSONError, ValueError) as exc:
            self.audit("volatile_refresh_failed", {"tool": name, "error": type(exc).__name__})
            return None
        self.stats.tool_runs += 1
        fresh = StepRecord(step.fingerprint, "tool", digest, dict(meta), self.engine.clock())
        self.record_step(fresh)
        return fresh

    # --- scheduling -----------------------------------------------------------------------------

    def _reaches(self, start: str, goal: str) -> bool:
        """Whether ``start`` transitively waits for ``goal`` in the current waits-for graph.

        Only membership is computed, not the path: this runs before every suspension, and building
        the path for the error message is worth doing only for the cycle that actually exists.
        """
        stack = [start]
        seen = {start}
        waits = self._waits
        while stack:
            edges = waits.get(stack.pop())
            if edges is None:
                continue
            for nxt in edges:
                if nxt == goal:
                    return True
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return False

    def _path(self, start: str, goal: str) -> list[str]:
        """The waits-for path from ``start`` to ``goal``, for a cycle error message."""
        stack: list[tuple[str, list[str]]] = [(start, [start])]
        seen = {start}
        while stack:
            node, path = stack.pop()
            if node == goal:
                return path
            for nxt in self._waits.get(node, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append((nxt, [*path, nxt]))
        return [start, goal]  # pragma: no cover - only called when a path was just found

    def _cycle(self, cycle: Sequence[str]) -> CycleError:
        error = CycleError(cycle)
        self.cycles.append(error.cycle)
        return error

    def _add_wait(self, waiter: str, target: str) -> None:
        if waiter == target:
            raise self._cycle([waiter, target])
        # A target that is not itself waiting for anything cannot close a cycle, and that is the
        # common case: the key was just scheduled, or it is running rather than suspended. Checking
        # it first turns the graph search from something that happens on every suspension into
        # something that happens only when two suspended keys could actually form a ring.
        if target in self._waits and self._reaches(target, waiter):
            raise self._cycle([waiter, *self._path(target, waiter)])
        self._waits.setdefault(waiter, Counter())[target] += 1

    def _remove_wait(self, waiter: str, target: str) -> None:
        edges = self._waits.get(waiter)
        if edges is None:
            return
        edges[target] -= 1
        if edges[target] <= 0:
            del edges[target]
        if not edges:
            del self._waits[waiter]

    def spawn(self, coro: Any) -> asyncio.Task[None]:
        """Start an engine task outside any rule's hermetic scope."""
        context = contextvars.copy_context()
        context.run(hermetic.clear_rule_marker)
        task: asyncio.Task[None] = asyncio.get_running_loop().create_task(coro, context=context)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def ensure(self, key: BuildKey, waiter: str | None) -> KeyResult:
        """Bring ``key`` up to date (at most once per session) and return its result hash."""
        if self.abort_error is not None:
            raise self.abort_error
        key_id = key.key_id
        future = self._futures.get(key_id)
        if future is None:
            future = asyncio.get_running_loop().create_future()
            future.add_done_callback(_consume_exception)
            self._futures[key_id] = future
            self.spawn(self._run_key(key, future))
        elif future.done():
            return future.result()
        if waiter is not None:
            self._add_wait(waiter, key_id)
        try:
            return await asyncio.shield(future)
        finally:
            if waiter is not None:
                self._remove_wait(waiter, key_id)

    async def ensure_many(self, keys: Sequence[BuildKey], waiter: str | None) -> list[KeyResult]:
        if len(keys) == 1:
            return [await self.ensure(keys[0], waiter)]
        outcomes = await asyncio.gather(
            *(self.ensure(key, waiter) for key in keys), return_exceptions=True
        )
        error = _first_error(outcomes)
        if error is not None:
            raise error
        return [item for item in outcomes if isinstance(item, KeyResult)]

    def abort(self, reason: str) -> BuildAbortedError:
        if self.abort_error is None:
            self.abort_error = BuildAbortedError(reason)
            self.audit("build_aborted", {"reason": reason})
            current = asyncio.current_task()
            for task in list(self._tasks):
                if task is not current:
                    task.cancel()
        return self.abort_error

    async def drain(self) -> None:
        """Wait for (or cancel, after an abort) every task this session started."""
        while self._tasks:
            pending = list(self._tasks)
            if self.abort_error is not None:
                for task in pending:
                    task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    async def _run_key(self, key: BuildKey, future: asyncio.Future[KeyResult]) -> None:
        try:
            result = await self._compute(key)
        except asyncio.CancelledError:
            if not future.done():
                future.set_exception(self.abort_error or BuildAbortedError("cancelled"))
            raise
        except (BuildAbortedError, CycleError, RuleFailedError) as exc:
            self.failures.setdefault(key.key_id, exc)
            future.set_exception(exc)
        except Exception as exc:  # noqa: BLE001 - any rule error becomes a key failure
            self.stats.failed += 1
            error = RuleFailedError(key.key_id, exc)
            self.failures[key.key_id] = error
            self.audit("rule_failed", {"key": key.key_id, "error": type(exc).__name__})
            future.set_exception(error)
        else:
            future.set_result(result)

    async def _verify(self, groups: DepGroups, waiter: str) -> str | None:
        """Return None when every recorded dependency still matches, else the first mismatch."""
        for group in groups:
            rule_deps = [dep for dep in group if dep.kind is DepKind.RULE]
            if rule_deps:
                keys: list[BuildKey] = []
                for dep in rule_deps:
                    needed = self.engine.rules.key_from_id(dep.ref)
                    if needed is None:
                        return f"{dep.ref}: no rule defines this key any more"
                    keys.append(needed)
                results = await self.ensure_many(keys, waiter)
                for dep, result in zip(rule_deps, results, strict=True):
                    if result.hash != dep.hash:
                        return f"{dep.ref} changed"
                    if not self.config.early_cutoff and result.executed:
                        return f"{dep.ref} was rebuilt (early cutoff disabled)"
            for dep in group:
                if dep.kind is DepKind.RULE or await self._input_matches(dep):
                    continue
                if dep.kind is DepKind.ERROR:
                    return f"{dep.ref} failed during the last execution"
                return f"{dep.kind.value} {dep.ref} changed"
        return None

    async def _compute(self, key: BuildKey) -> KeyResult:
        key_id = key.key_id
        rule = self.engine.rules.rule_for(key)
        old = self._old_traces.get(key_id)
        with self.engine.tracer.start_as_current_span(
            "iab.rule", attributes={"iab.key": key_id, "iab.rule": rule.name}
        ) as span:
            reason: str | None
            if old is None:
                reason = "new key"
            elif old.rule != rule.name or old.version != rule.version:
                reason = f"rule version {old.version} -> {rule.version}"
            elif key_id in self.force:
                reason = "forced"
            else:
                reason = await self._verify(old.deps, key_id)
                if reason is None:
                    self.stats.reused += 1
                    self.stats.tokens_saved += old.usage.tokens
                    span.set_attribute("iab.outcome", "reused")
                    span.set_attribute("iab.deps", old.dep_count())
                    return KeyResult(key_id, old.result_hash, executed=False, changed=False)
            if key_id not in self.force and self.engine.trust is not None:
                imported = await self._try_constructive(key, rule, old)
                if imported is not None:
                    span.set_attribute("iab.outcome", "imported")
                    return imported
            result, trace = await self._execute(key, rule, old, reason)
            span.set_attribute("iab.outcome", "executed" if result.changed else "cutoff")
            span.set_attribute("iab.reason", reason)
            for kind in DepKind:
                count = sum(1 for group in trace.deps for dep in group if dep.kind is kind)
                span.set_attribute(f"iab.deps.{kind.value}", count)
            return result

    async def _execute(
        self, key: BuildKey, rule: Rule[Any], old: Trace | None, reason: str
    ) -> tuple[KeyResult, Trace]:
        from incremental_agent_builds.engine.context import BuildContext  # noqa: PLC0415

        key_id = key.key_id
        slot = _Slot(self._rule_slots)
        await slot.acquire()
        context = BuildContext(self, key, slot)
        try:
            with hermetic.rule_scope(key_id, self.config.guard):
                value = await rule.fn(context, key)
        finally:
            slot.release()
            context.close()
        try:
            canonical_json(value)
        except CanonicalJSONError as exc:
            raise RuleDefinitionError(f"rule {rule.name} returned a non-JSON value: {exc}") from exc
        digest = self.engine.cas.put_json(value)
        self._values[key_id] = value
        changed = old is None or old.result_hash != digest
        trace = Trace(
            key_id=key_id,
            rule=rule.name,
            version=rule.version,
            deps=context.dep_groups(),
            result_hash=digest,
            built_run=self.run_id,
            changed_run=self.run_id if changed or old is None else old.changed_run,
            usage=context.usage(),
            origin="executed",
        )
        self._new_traces[key_id] = trace
        self.executed_keys.append(key_id)
        self.stats.executed += 1
        self.stats.executed_by_rule[rule.name] = self.stats.executed_by_rule.get(rule.name, 0) + 1
        if not changed:
            self.stats.early_cutoffs += 1
        self.audit(
            "rule_executed",
            {
                "key": key_id,
                "reason": reason,
                "result": digest,
                "cutoff": not changed,
                "provider_calls": trace.usage.provider_calls,
                "tokens": trace.usage.tokens,
            },
        )
        return KeyResult(key_id, digest, executed=True, changed=changed), trace

    async def _try_constructive(
        self, key: BuildKey, rule: Rule[Any], old: Trace | None
    ) -> KeyResult | None:
        trust = self.engine.trust
        if trust is None:  # pragma: no cover - checked by the caller
            return None
        key_id = key.key_id
        for entry in self.engine.store.constructive_for(key_id):
            if entry.version != rule.version or entry.body.get("rule") != rule.name:
                continue
            if not trust.verify(entry.signer, entry.body, entry.signature):
                self.engine.metrics.cache_import_rejections_total.labels("untrusted_at_use").inc()
                continue
            try:
                deps = entry.deps
            except (ValueError, KeyError):
                continue
            if await self._verify(deps, key_id) is not None:
                continue
            if not self.engine.cas.contains(entry.result_hash):
                continue
            changed = old is None or old.result_hash != entry.result_hash
            trace = Trace(
                key_id=key_id,
                rule=rule.name,
                version=rule.version,
                deps=deps,
                result_hash=entry.result_hash,
                built_run=self.run_id,
                changed_run=self.run_id if changed or old is None else old.changed_run,
                origin=f"imported:{entry.signer}",
            )
            self._new_traces[key_id] = trace
            self.stats.imported += 1
            self.audit(
                "constructive_hit",
                {"key": key_id, "signer": entry.signer, "result": entry.result_hash},
            )
            return KeyResult(key_id, entry.result_hash, executed=False, changed=changed)
        return None

    # --- values -------------------------------------------------------------------------------

    async def value_of(self, key: BuildKey, expected_hash: str) -> JsonValue:
        key_id = key.key_id
        if key_id in self._values:
            return self._values[key_id]
        try:
            with hermetic.io_permit():
                value = self.engine.cas.get_json(expected_hash)
        except IntegrityError as exc:
            self.integrity("cas", key_id, str(exc))
            value = await self._regenerate(key, expected_hash)
        self._values[key_id] = value
        return value

    async def _regenerate(self, key: BuildKey, expected_hash: str) -> JsonValue:
        """Recompute a value whose blob failed verification; the result must hash the same."""
        key_id = key.key_id
        pending = self._regenerating.get(key_id)
        if pending is None:
            # Bound to a separate name so the closure below closes over a non-optional future:
            # a narrowing assertion would be stripped by `python -O`.
            future: asyncio.Future[JsonValue] = asyncio.get_running_loop().create_future()
            future.add_done_callback(_consume_exception)
            pending = future
            self._regenerating[key_id] = future
            rule = self.engine.rules.rule_for(key)

            async def run() -> None:
                from incremental_agent_builds.engine.context import BuildContext  # noqa: PLC0415

                slot = _Slot(self._rule_slots)
                await slot.acquire()
                context = BuildContext(self, key, slot, recording=False)
                try:
                    with hermetic.rule_scope(key_id, self.config.guard):
                        value = await rule.fn(context, key)
                    digest = self.engine.cas.put_json(value)
                    if digest != expected_hash:
                        # A rule that is not a pure function of its inputs: loud, not silent.
                        raise IntegrityError(  # noqa: TRY301
                            f"regenerated value of {key_id} differs from its trace "
                            "(non-deterministic rule?)"
                        )
                    self.stats.regenerated += 1
                    self.audit("value_regenerated", {"key": key_id, "result": digest})
                    future.set_result(value)
                except BaseException as exc:
                    if not future.done():
                        future.set_exception(exc)
                    if isinstance(exc, asyncio.CancelledError):
                        raise
                finally:
                    slot.release()
                    context.close()

            self.spawn(run())
        return await asyncio.shield(pending)

    def live_hashes(self) -> dict[str, str]:
        live: dict[str, str] = {}
        for key_id, future in self._futures.items():
            if future.done() and not future.cancelled() and future.exception() is None:
                live[key_id] = future.result().hash
        return live

    def finish(self) -> None:
        self._collect_store_integrity_events()


def _consume_exception(future: asyncio.Future[Any]) -> None:
    if not future.cancelled():
        future.exception()
