"""Evaluation oracles: clean-equivalence and minimality of an incremental build.

*Clean equivalence*: the keys an incremental build ended up with (its live set) must be exactly
the keys a clean build from an empty store produces, with equal values.

*Minimality*: every rule the incremental build executed must have had at least one direct
dependency whose value differs between the previous and the current clean build (or be new).
The recorded trace from before the build says which dependencies the rule read and what it saw;
the current clean build and the current inputs say what they are now. LLM and tool steps are
derived from earlier dependencies, so they never count as the changed dependency.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from incremental_agent_builds.agentic.budget import BudgetLedger, BudgetLimits
from incremental_agent_builds.engine.engine import BuildEngine
from incremental_agent_builds.engine.errors import IntegrityError, WorkspaceError
from incremental_agent_builds.engine.session import BuildSession
from incremental_agent_builds.engine.store import DepKind, Trace

__all__ = ["compare_live", "unnecessary_reruns"]


def _probe_session(engine: BuildEngine) -> BuildSession:
    return BuildSession(
        engine,
        run_id=0,
        force=frozenset(),
        budget=BudgetLedger(BudgetLimits(max_steps=0, max_tokens=0, max_usd_micros=0)),
        principal="oracle",
    )


def unnecessary_reruns(  # noqa: PLR0912 - one branch per dependency kind, kept in one place
    engine: BuildEngine,
    old_traces: Mapping[str, Trace],
    executed: Iterable[str],
    clean_live: Mapping[str, str],
) -> list[str]:
    """Executed keys none of whose recorded direct dependencies changed value."""
    session = _probe_session(engine)
    unnecessary: list[str] = []
    for key_id in executed:
        trace = old_traces.get(key_id)
        if trace is None:
            continue  # new key: executing it is necessary
        changed = False
        for group in trace.deps:
            for dep in group:
                try:
                    match dep.kind:
                        case DepKind.RULE:
                            now = clean_live.get(dep.ref)
                        case DepKind.FILE:
                            now = session.file_input(dep.ref)[1]
                        case DepKind.GLOB:
                            now = session.glob_input(dep.ref)[1]
                        case DepKind.CONFIG:
                            now = session.config_input(dep.ref.partition("#")[2])[1]
                        case _:
                            continue
                except WorkspaceError:
                    now = None
                if now != dep.hash:
                    changed = True
                    break
            if changed:
                break
        if not changed:
            unnecessary.append(key_id)
    return unnecessary


def compare_live(
    incremental: BuildEngine,
    incremental_live: Mapping[str, str],
    clean: BuildEngine,
    clean_live: Mapping[str, str],
) -> list[str]:
    """Differences between two live sets, checked key by key and value by value."""
    problems: list[str] = []
    for key_id in sorted(set(incremental_live) | set(clean_live)):
        if key_id not in clean_live:
            problems.append(f"{key_id}: only in the incremental build")
            continue
        if key_id not in incremental_live:
            problems.append(f"{key_id}: only in the clean build")
            continue
        try:
            left = incremental.cas.get_json(incremental_live[key_id])
            right = clean.cas.get_json(clean_live[key_id])
        except IntegrityError as exc:
            problems.append(f"{key_id}: {exc}")
            continue
        if left != right or incremental_live[key_id] != clean_live[key_id]:
            problems.append(f"{key_id}: values differ")
    return problems
