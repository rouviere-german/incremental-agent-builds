"""The evaluation oracles, and the budgets that stop a build spending more than it was allowed.

The oracles are what the property tests judge the engine with, so they are tested against known
cases here: a divergence must be *reported*, not silently passed over, or every property test would
be vacuous.
"""

from __future__ import annotations

import pytest

from incremental_agent_builds.agentic.budget import BudgetLedger, BudgetLimits
from incremental_agent_builds.engine.errors import BuildAbortedError
from incremental_agent_builds.engine.oracle import compare_live, unnecessary_reruns
from incremental_agent_builds.workload.synthetic import (
    Edit,
    Node,
    NodeSpec,
    Program,
    initial_workspace,
    run_case,
)
from tests.engine_helpers import Doubled, Echo, counting_rules, make_engine, workspace_with

pytestmark = pytest.mark.unit


# --- compare_live -------------------------------------------------------------------------------


def test_identical_builds_compare_equal() -> None:
    left = make_engine(workspace_with(), counting_rules()[0])
    right = make_engine(workspace_with(), counting_rules()[0])
    first = left.build_sync([Doubled(path="in/a.txt")])
    second = right.build_sync([Doubled(path="in/a.txt")])
    assert compare_live(left, first.live, right, second.live) == []


def test_a_key_present_in_only_one_build_is_reported() -> None:
    left = make_engine(workspace_with(), counting_rules()[0])
    right = make_engine(workspace_with(), counting_rules()[0])
    first = left.build_sync([Doubled(path="in/a.txt"), Echo(path="in/b.txt")])
    second = right.build_sync([Doubled(path="in/a.txt")])
    problems = compare_live(left, first.live, right, second.live)
    assert any("only in the incremental build" in item for item in problems)
    assert compare_live(right, second.live, left, first.live)[0].endswith("only in the clean build")


def test_a_differing_value_is_reported() -> None:
    left = make_engine(workspace_with(), counting_rules()[0])
    right = make_engine(workspace_with({"in/a.txt": "different"}), counting_rules()[0])
    first = left.build_sync([Echo(path="in/a.txt")])
    second = right.build_sync([Echo(path="in/a.txt")])
    problems = compare_live(left, first.live, right, second.live)
    assert problems == [f"{Echo(path='in/a.txt').key_id}: values differ"]


def test_a_corrupt_blob_is_reported_rather_than_ignored() -> None:
    left = make_engine(workspace_with(), counting_rules()[0])
    right = make_engine(workspace_with(), counting_rules()[0])
    first = left.build_sync([Echo(path="in/a.txt")])
    second = right.build_sync([Echo(path="in/a.txt")])
    digest = first.live[Echo(path="in/a.txt").key_id]
    left.cas.blobs[digest] = b"corrupt"  # type: ignore[attr-defined]
    problems = compare_live(left, first.live, right, second.live)
    assert problems
    assert "SHA-256" in problems[0]


# --- unnecessary_reruns -------------------------------------------------------------------------


def test_a_minimal_rebuild_reports_no_unnecessary_reruns() -> None:
    workspace = workspace_with()
    engine = make_engine(workspace, counting_rules()[0])
    engine.build_sync([Doubled(path="in/a.txt"), Doubled(path="in/b.txt")])
    before = engine.store.load_traces()
    workspace.write("in/a.txt", "9")
    result = engine.build_sync([Doubled(path="in/a.txt"), Doubled(path="in/b.txt")])

    clean = make_engine(workspace, counting_rules()[0])
    clean_result = clean.build_sync([Doubled(path="in/a.txt"), Doubled(path="in/b.txt")])
    assert unnecessary_reruns(engine, before, result.executed, clean_result.live) == []


def test_a_rerun_with_no_changed_dependency_is_reported() -> None:
    """Forcing a key re-runs it without any dependency change: the oracle must notice."""
    workspace = workspace_with()
    engine = make_engine(workspace, counting_rules()[0])
    engine.build_sync([Echo(path="in/a.txt")])
    before = engine.store.load_traces()
    result = engine.build_sync([Echo(path="in/a.txt")], force=[Echo(path="in/a.txt")])

    clean = make_engine(workspace, counting_rules()[0])
    clean_result = clean.build_sync([Echo(path="in/a.txt")])
    reported = unnecessary_reruns(engine, before, result.executed, clean_result.live)
    assert reported == [Echo(path="in/a.txt").key_id]


def test_a_new_key_is_never_unnecessary() -> None:
    workspace = workspace_with()
    engine = make_engine(workspace, counting_rules()[0])
    result = engine.build_sync([Echo(path="in/a.txt")])
    assert unnecessary_reruns(engine, {}, result.executed, result.live) == []


# --- budgets ------------------------------------------------------------------------------------


def _llm_program(n_nodes: int) -> Program:
    return Program(
        n_files=1,
        nodes=tuple(
            NodeSpec(reads=(0,), static_deps=(i - 1,) if i else (), uses_llm=True)
            for i in range(n_nodes)
        ),
    )


def test_a_step_budget_aborts_the_build_and_keeps_finished_traces() -> None:
    import os

    from incremental_agent_builds.engine.cas import MemoryContentStore
    from incremental_agent_builds.engine.engine import BuildEngine, EngineConfig
    from incremental_agent_builds.engine.store import TraceStore
    from incremental_agent_builds.workload.synthetic import make_synthetic_provider, program_rules

    program = _llm_program(6)
    engine = BuildEngine(
        rules=program_rules(program),
        workspace=initial_workspace(program, [1]),
        store=TraceStore(":memory:", mac_key=os.urandom(32)),
        cas=MemoryContentStore(),
        provider=make_synthetic_provider(),
        config=EngineConfig(jobs=1, llm_concurrency=1),
        default_budget=BudgetLimits.of(steps=2, tokens=10**9, usd=100.0),
    )
    result = engine.build_sync([Node(i=5)])
    assert not result.ok
    assert result.aborted is not None
    assert "budget" in result.aborted
    assert result.stats.provider_calls <= 2
    with pytest.raises(BuildAbortedError):
        result.raise_for_failures()
    # The keys that finished before the abort keep their traces, so a retry resumes.
    assert engine.trace(Node(i=0)) is not None


def test_a_token_budget_stops_further_steps() -> None:
    import os

    from incremental_agent_builds.engine.cas import MemoryContentStore
    from incremental_agent_builds.engine.engine import BuildEngine, EngineConfig
    from incremental_agent_builds.engine.store import TraceStore
    from incremental_agent_builds.workload.synthetic import make_synthetic_provider, program_rules

    program = _llm_program(6)
    engine = BuildEngine(
        rules=program_rules(program),
        workspace=initial_workspace(program, [1]),
        store=TraceStore(":memory:", mac_key=os.urandom(32)),
        cas=MemoryContentStore(),
        provider=make_synthetic_provider(),
        config=EngineConfig(jobs=1, llm_concurrency=1),
        default_budget=BudgetLimits.of(steps=100, tokens=1, usd=100.0),
    )
    result = engine.build_sync([Node(i=5)])
    assert result.aborted is not None
    assert result.stats.provider_calls == 1, "the first call cannot be known in advance, no more"


def _fan_in_llm_program(n_leaves: int) -> Program:
    """``n_leaves`` independent LLM nodes plus a root that needs them all at once."""
    return Program(
        n_files=1,
        nodes=(
            *(NodeSpec(reads=(0,), uses_llm=True) for _ in range(n_leaves)),
            NodeSpec(reads=(0,), static_deps=tuple(range(n_leaves))),
        ),
    )


@pytest.mark.parametrize("jobs", [1, 2, 4])
def test_the_token_budget_is_overshot_by_at_most_the_calls_in_flight(jobs: int) -> None:
    """The honest bound on the token and USD limits, which are only known after a call returns.

    A step is reserved before the call; the tokens it will cost are not knowable until it comes
    back. Every rule body already past that reservation therefore still reaches the provider, so
    the limit can be crossed by as many calls as there are rule slots -- not by one, as the docs
    used to say. The count comes from the provider itself because the abort cancels the tasks whose
    calls are still in flight: those calls were made and billed, but never reported their usage.
    This test is what keeps ``agentic/budget.py`` and ``docs/THREAT_MODEL.md`` D2 true.
    """
    import os

    from incremental_agent_builds.engine.cas import MemoryContentStore
    from incremental_agent_builds.engine.engine import BuildEngine, EngineConfig
    from incremental_agent_builds.engine.store import TraceStore
    from incremental_agent_builds.workload.synthetic import make_synthetic_provider, program_rules

    leaves = 8
    program = _fan_in_llm_program(leaves)
    provider = make_synthetic_provider()
    engine = BuildEngine(
        rules=program_rules(program),
        workspace=initial_workspace(program, [1]),
        store=TraceStore(":memory:", mac_key=os.urandom(32)),
        cas=MemoryContentStore(),
        provider=provider,
        config=EngineConfig(jobs=jobs, llm_concurrency=4),
        # One token: the very first call that reports its usage crosses the limit.
        default_budget=BudgetLimits.of(steps=100, tokens=1, usd=100.0),
    )
    result = engine.build_sync([Node(i=leaves)])

    assert result.aborted is not None
    assert "tokens" in result.aborted
    assert result.stats.llm_steps <= leaves
    assert 1 <= provider.calls <= jobs, f"{provider.calls} calls reached the provider"


def test_a_ledger_refuses_to_raise_limits_on_resume(tmp_path: object) -> None:
    from pathlib import Path

    from incremental_agent_builds.agentic.errors import BudgetError

    assert isinstance(tmp_path, Path)
    path = tmp_path / "budget.json"
    BudgetLedger(BudgetLimits.of(steps=5, tokens=100, usd=1.0), path)
    BudgetLedger(BudgetLimits.of(steps=3, tokens=50, usd=0.5), path)  # tightening is allowed
    with pytest.raises(BudgetError, match="cannot be raised"):
        BudgetLedger(BudgetLimits.of(steps=50, tokens=10_000, usd=10.0), path)


def test_rewriting_a_file_with_identical_content_costs_nothing() -> None:
    """Inputs are tracked by content hash, not mtime, so a touched-but-unchanged file is free."""
    program = _llm_program(4)
    report = run_case(program, [1], [Edit("set", 0, 1)])  # writes "1" over "1"
    assert report.failures == 0
    assert report.mismatches == []
    assert report.executed == 0, "nothing changed, so no rule may run"
    assert report.reused == len(program.nodes)


def test_editing_the_root_input_re_runs_the_whole_llm_chain() -> None:
    """The other side of the same coin: a real change to a shared input is not free."""
    program = _llm_program(4)
    report = run_case(program, [1], [Edit("set", 0, 5)])
    assert report.failures == 0
    assert report.mismatches == []
    assert report.executed == len(program.nodes)
