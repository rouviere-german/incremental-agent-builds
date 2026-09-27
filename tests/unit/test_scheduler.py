"""Rebuild semantics: verifying traces, reuse, early cutoff, force, dynamic dependency sets.

These tests drive the scheduler through a small rule set whose bodies count their own executions, so
each assertion separates the three outcomes a key can have on a rebuild: reused without running,
re-run with a new value, or re-run with the same value (an early cutoff).
"""

from __future__ import annotations

import pytest

from incremental_agent_builds.engine.engine import EngineConfig
from incremental_agent_builds.engine.errors import RuleFailedError
from incremental_agent_builds.engine.store import DepKind
from tests.engine_helpers import (
    Counted,
    Doubled,
    Echo,
    Sum,
    Upper,
    counting_rules,
    make_engine,
    workspace_with,
)

pytestmark = pytest.mark.unit


def test_a_first_build_executes_every_reachable_key_and_records_a_trace() -> None:
    workspace = workspace_with()
    rules, counts = counting_rules()
    engine = make_engine(workspace, rules)

    result = engine.build_sync([Doubled(path="in/a.txt")])

    assert result.ok
    assert result.stats.executed == 2  # echo + doubled
    assert result.stats.reused == 0
    assert counts.runs["echo"] == 1
    trace = engine.trace(Doubled(path="in/a.txt"))
    assert trace is not None
    kinds = [dep.kind for group in trace.deps for dep in group]
    assert kinds == [DepKind.RULE, DepKind.CONFIG]
    assert engine.value(Doubled(path="in/a.txt")) == "11"


def test_an_unchanged_rebuild_runs_nothing() -> None:
    workspace = workspace_with()
    rules, counts = counting_rules()
    engine = make_engine(workspace, rules)
    engine.build_sync([Doubled(path="in/a.txt")])
    before = dict(counts.runs)

    second = engine.build_sync([Doubled(path="in/a.txt")])

    assert second.stats.executed == 0
    assert second.stats.reused == 2
    assert dict(counts.runs) == before, "no rule body may run on an unchanged rebuild"


def test_changing_a_file_re_runs_only_the_affected_chain() -> None:
    workspace = workspace_with()
    rules, counts = counting_rules()
    engine = make_engine(workspace, rules)
    engine.build_sync([Doubled(path="in/a.txt"), Doubled(path="in/b.txt")])
    counts.runs.clear()

    workspace.write("in/a.txt", "9")
    result = engine.build_sync([Doubled(path="in/a.txt"), Doubled(path="in/b.txt")])

    assert counts.runs["echo"] == 1, "only the edited file's Echo re-runs"
    assert counts.runs["doubled"] == 1
    assert result.stats.reused == 2
    assert engine.value(Doubled(path="in/a.txt")) == "99"
    assert engine.value(Doubled(path="in/b.txt")) == "22"


def test_early_cutoff_stops_propagation_when_the_value_is_unchanged() -> None:
    """``Counted`` maps 1 and 3 to the same value mod 2, so its dependent must not re-run."""
    workspace = workspace_with()
    rules, counts = counting_rules()
    engine = make_engine(workspace, rules)
    engine.build_sync([Counted(modulus=2), Doubled(path="in/a.txt")])
    counts.runs.clear()

    workspace.write("in/a.txt", "3")  # 1 % 2 == 3 % 2, but the file changed
    result = engine.build_sync([Counted(modulus=2), Doubled(path="in/a.txt")])

    assert counts.runs["counted"] == 1, "the rule whose input changed must re-run"
    assert result.stats.early_cutoffs >= 1
    assert counts.runs["echo"] == 1  # Echo's own file changed, so it re-runs
    assert counts.runs["doubled"] == 1  # Echo's value did change ("1" -> "3")
    assert engine.value(Counted(modulus=2)) == 1


def test_early_cutoff_ablation_propagates_the_rebuild() -> None:
    workspace = workspace_with()
    rules, counts = counting_rules()
    engine = make_engine(
        workspace, rules, config=EngineConfig(jobs=1, llm_concurrency=1, early_cutoff=False)
    )
    engine.build_sync([Doubled(path="in/a.txt")])
    counts.runs.clear()
    workspace.write(
        "in/a.txt", "1 "
    )  # a whitespace edit: Echo's value changes, so this is not a cutoff
    engine.build_sync([Doubled(path="in/a.txt")])
    assert counts.runs["doubled"] == 1


def test_a_new_glob_match_is_discovered_and_a_removed_one_disappears() -> None:
    """The dependency set of ``Sum`` is a value, so adding a file changes what it needs."""
    workspace = workspace_with()
    rules, counts = counting_rules()
    engine = make_engine(workspace, rules)
    first = engine.build_sync([Sum(pattern="in/*.txt")])
    assert engine.value(Sum(pattern="in/*.txt")) == ["1", "2"]
    assert first.stats.executed == 3  # sum + two echoes
    counts.runs.clear()

    workspace.write("in/c.txt", "3")
    engine.build_sync([Sum(pattern="in/*.txt")])
    assert engine.value(Sum(pattern="in/*.txt")) == ["1", "2", "3"]
    assert counts.runs["echo"] == 1, "only the new file's Echo runs"
    assert counts.runs["sum"] == 1

    counts.runs.clear()
    workspace.delete("in/b.txt")
    engine.build_sync([Sum(pattern="in/*.txt")])
    assert engine.value(Sum(pattern="in/*.txt")) == ["1", "3"]
    assert counts.runs["echo"] == 0, "the remaining files' Echo values are reused"


def test_a_key_no_longer_needed_is_not_rebuilt() -> None:
    """Dependencies are verified in recorded order, so a key dropped by an edit is not built."""
    workspace = workspace_with()
    rules, counts = counting_rules()
    engine = make_engine(workspace, rules)
    engine.build_sync([Sum(pattern="in/*.txt")])
    counts.runs.clear()

    workspace.delete("in/b.txt")
    workspace.write("in/a.txt", "7")
    engine.build_sync([Sum(pattern="in/*.txt")])

    assert counts.runs["echo"] == 1, "only the surviving, changed file is echoed"
    assert Echo(path="in/b.txt").key_id not in engine.build_sync([Sum(pattern="in/*.txt")]).live


def test_force_re_runs_one_rule_and_nothing_else() -> None:
    workspace = workspace_with()
    rules, counts = counting_rules()
    engine = make_engine(workspace, rules)
    engine.build_sync([Doubled(path="in/a.txt")])
    counts.runs.clear()

    result = engine.build_sync([Doubled(path="in/a.txt")], force=[Echo(path="in/a.txt")])

    assert counts.runs["echo"] == 1
    assert counts.runs["doubled"] == 0, "the forced rule produced the same value, so cutoff applies"
    assert result.stats.early_cutoffs == 1


def test_a_rule_version_bump_invalidates_its_traces() -> None:
    workspace = workspace_with()
    rules, counts = counting_rules()
    engine = make_engine(workspace, rules)
    engine.build_sync([Echo(path="in/a.txt")])
    counts.runs.clear()

    bumped, new_counts = counting_rules()
    from incremental_agent_builds.engine.rules import Rule, RuleSet

    old = bumped.get("echo")
    assert old is not None
    versioned = RuleSet(
        [Rule(key_type=old.key_type, fn=old.fn, version=2)]
        + [r for kind in bumped.kinds if kind != "echo" and (r := bumped.get(kind)) is not None]
    )
    same_store = make_engine(workspace, versioned, store=engine.store, cas=engine.cas)
    result = same_store.build_sync([Echo(path="in/a.txt")])

    assert result.stats.executed == 1
    assert new_counts.runs["echo"] == 1
    trace = same_store.trace(Echo(path="in/a.txt"))
    assert trace is not None
    assert trace.version == 2


def test_a_tool_result_is_memoized_by_its_fingerprint() -> None:
    workspace = workspace_with()
    rules, _ = counting_rules()
    engine = make_engine(workspace, rules)
    first = engine.build_sync([Upper(path="in/a.txt")])
    assert first.stats.tool_runs == 1
    assert engine.value(Upper(path="in/a.txt")) == {"upper": "1"}

    # A forced key re-runs its steps for real: that is what --force is for. The result is
    # unchanged, so the key still cuts off for its dependents.
    forced = engine.build_sync([Upper(path="in/a.txt")], force=[Upper(path="in/a.txt")])
    assert forced.stats.tool_runs == 1
    assert forced.stats.tool_memo_hits == 0
    assert forced.stats.early_cutoffs == 1

    workspace.write("in/a.txt", "x")
    changed = engine.build_sync([Upper(path="in/a.txt")])
    assert changed.stats.tool_runs == 1
    assert engine.value(Upper(path="in/a.txt")) == {"upper": "X"}

    workspace.write("in/a.txt", "1")  # back to a value whose tool step is already recorded
    restored = engine.build_sync([Upper(path="in/a.txt")])
    assert restored.stats.tool_runs == 0, "the recorded step for this input is reused"
    assert restored.stats.tool_memo_hits == 1


def test_a_failing_rule_fails_its_dependents_and_not_the_rest() -> None:
    workspace = workspace_with({"in/bad.txt": "x"})
    rules, _ = counting_rules()
    engine = make_engine(workspace, rules)
    # Echo of a path outside the jail raises a WorkspaceError inside the rule body.
    result = engine.build_sync([Echo(path="../escape.txt"), Echo(path="in/a.txt")])
    assert not result.ok
    assert Echo(path="../escape.txt").key_id in result.failures
    assert Echo(path="in/a.txt").key_id in result.target_hashes
    with pytest.raises(RuleFailedError):
        result.raise_for_failures()


def test_build_results_serialize_to_json_with_stats() -> None:
    workspace = workspace_with()
    rules, _ = counting_rules()
    engine = make_engine(workspace, rules)
    payload = engine.build_sync([Echo(path="in/a.txt")]).to_json()
    assert payload["ok"] is True
    assert payload["targets"] == [Echo(path="in/a.txt").key_id]
    stats = payload["stats"]
    assert isinstance(stats, dict)
    assert stats["executed"] == 1
    assert isinstance(stats["executed_by_rule"], dict)


def test_values_of_never_built_keys_raise_key_error() -> None:
    workspace = workspace_with()
    rules, _ = counting_rules()
    engine = make_engine(workspace, rules)
    with pytest.raises(KeyError):
        engine.value(Echo(path="in/missing.txt"))


def test_concurrent_jobs_produce_the_same_values_as_one_job() -> None:
    files = {f"in/f{i}.txt": str(i) for i in range(12)}
    rules_one, _ = counting_rules()
    rules_many, _ = counting_rules()
    sequential = make_engine(
        workspace_with(files), rules_one, config=EngineConfig(jobs=1, llm_concurrency=1)
    )
    parallel = make_engine(
        workspace_with(files), rules_many, config=EngineConfig(jobs=8, llm_concurrency=8)
    )
    target = [Sum(pattern="in/*.txt")]
    assert sequential.build_sync(target).target_hashes == parallel.build_sync(target).target_hashes
