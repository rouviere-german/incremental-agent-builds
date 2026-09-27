"""The offline demo and the commit replay, with the claims they make asserted.

The README's Results section is produced by ``bench/`` and the demo; this module is what keeps those
numbers honest. It runs a short replay and asserts the properties the numbers rest on: after every
commit the incremental store equals a clean rebuild, no rule re-ran without a changed dependency,
the ablations cost strictly more, and the naive caches go stale where this system does not.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import JsonValue

from incremental_agent_builds.workload.demo import run_demo
from incremental_agent_builds.workload.history import COMMITS
from incremental_agent_builds.workload.replay import ReplayReport, run_replay

pytestmark = pytest.mark.integration

_REPLAY_COMMITS = 4


def number(value: JsonValue) -> float:
    """A numeric summary field, asserted to be numeric rather than silently coerced."""
    assert isinstance(value, int | float), f"expected a number, got {value!r}"
    return float(value)


def integer(value: JsonValue) -> int:
    assert isinstance(value, int), f"expected an integer, got {value!r}"
    return value


@pytest.fixture(scope="module")
def replay(tmp_path_factory: pytest.TempPathFactory) -> ReplayReport:
    """One replay of the first commits, shared by the assertions below (it is the slow part)."""
    workdir = tmp_path_factory.mktemp("replay")
    return run_replay(
        workdir, commits=_REPLAY_COMMITS, clean=True, ablations=True, naive=True, jobs=4
    )


def test_the_cold_build_runs_every_rule_and_pays_for_every_llm_step(replay: ReplayReport) -> None:
    assert replay.initial["executed"] > 200, "the reference workload is the 80-module package"
    assert replay.initial["provider_calls"] > 50
    assert replay.initial["live_keys"] == replay.initial["executed"]


def test_every_commit_leaves_the_store_equal_to_a_clean_rebuild(replay: ReplayReport) -> None:
    summary = replay.summary()
    assert integer(summary["checked_commits"]) == _REPLAY_COMMITS
    assert integer(summary["equivalent_commits"]) == _REPLAY_COMMITS, (
        "a value diverged from a clean rebuild"
    )


def test_no_rule_re_ran_without_a_changed_dependency(replay: ReplayReport) -> None:
    assert integer(replay.summary()["unnecessary_reruns"]) == 0


def test_the_incremental_build_costs_far_less_than_clean_rebuilds(replay: ReplayReport) -> None:
    summary = replay.summary()
    assert number(summary["total_calls_saved"]) > 0.8
    assert number(summary["total_tokens_saved"]) > 0.8
    assert number(summary["incremental_calls"]) < number(summary["clean_calls"])


def test_the_ablations_cost_strictly_more_than_the_full_system(replay: ReplayReport) -> None:
    """Early cutoff and step memoization are each load-bearing, not decoration."""
    summary = replay.summary()
    assert integer(summary["no_cutoff_executed"]) > integer(summary["incremental_executed"])
    assert integer(summary["no_memo_calls"]) > integer(summary["incremental_calls"])
    assert integer(summary["no_memo_calls"]) >= integer(summary["no_cutoff_calls"])


def test_the_system_never_serves_a_stale_value(replay: ReplayReport) -> None:
    assert integer(replay.summary()["system_newly_stale"]) == 0


def test_a_whitespace_commit_executes_one_rule_and_makes_no_llm_call(replay: ReplayReport) -> None:
    whitespace = [row for row in replay.rows if row.kind == "whitespace"]
    assert whitespace, "the first commits include a whitespace-only edit"
    for row in whitespace:
        assert row.executed == 1
        assert row.early_cutoffs == 1
        assert row.provider_calls == 0


def test_a_replay_report_serializes_rows_and_a_summary(replay: ReplayReport) -> None:
    payload = replay.to_json()
    assert len(payload["rows"]) == _REPLAY_COMMITS
    assert payload["rows"][0]["calls_saved"] is not None
    assert set(payload) == {"initial", "summary", "rows"}


def test_a_replay_can_skip_the_expensive_checks(tmp_path: Path) -> None:
    report = run_replay(tmp_path / "quick", commits=1, clean=False, ablations=False, naive=False)
    summary = report.summary()
    assert integer(summary["commits"]) == 1
    assert "median_calls_saved" not in summary, "with no clean rebuild there is nothing to compare"


@pytest.mark.parametrize("kind", ["model_id", "template"])
def test_the_naive_caches_go_stale_where_this_system_does_not(tmp_path: Path, kind: str) -> None:
    """Replay up to and including the first commit of this kind and compare against the truth."""
    index = next(i for i, commit in enumerate(COMMITS) if commit.kind == kind)
    report = run_replay(tmp_path / kind, commits=index + 1, clean=True, ablations=False, naive=True)
    row = report.rows[index]
    assert row.kind == kind
    assert row.equivalent is True
    assert row.system_stale == 0
    assert row.naive_input_stale is not None
    assert row.naive_prompt_stale is not None
    if kind == "model_id":
        assert row.naive_prompt_stale > 0, "a model change is invisible to a prompt-hash cache"
    else:
        assert row.naive_input_stale > 0, "a template edit is invisible to an input-hash cache"


# --- the demo -----------------------------------------------------------------------------------


@pytest.mark.slow
def test_the_demo_runs_end_to_end_offline(tmp_path: Path) -> None:
    """``uv run python tasks.py demo`` must work with no API key and no network."""
    summary = run_demo(tmp_path / "demo", commits=3)

    assert integer(summary["cold_executed"]) > 200
    assert integer(summary["noop_executed"]) == 0, "a rebuild with no change runs nothing"
    assert integer(summary["noop_calls"]) == 0

    assert integer(summary["commits"]) == 3
    assert integer(summary["clean_equivalent_commits"]) == 3
    assert number(summary["calls_saved_fraction"]) > 0.8

    assert summary["cas_tamper_detected"] is True
    assert integer(summary["cas_integrity_events"]) == 1
    assert integer(summary["values_regenerated"]) == 1, "the corrupt value is recomputed"
    assert summary["trace_tamper_detected"] is True

    assert integer(summary["cache_accepted_traces"]) > 0
    assert integer(summary["cache_forged_rejected"]) > 0
    assert integer(summary["cache_untrusted_rejected"]) > 0

    assert summary["publish_first_status"] == "approval_required"
    assert summary["publish_after_approval"] == "applied"
    assert integer(summary["pages_published"]) > 0

    assert summary["audit_ok"] is True
    assert integer(summary["audit_entries"]) > 100
    assert (tmp_path / "demo" / "site" / "manifest.json").is_file()


def test_the_demo_prints_a_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_demo(tmp_path / "demo", commits=1)
    printed = capsys.readouterr().out
    for heading in (
        "Reference workspace",
        "Replaying",
        "Plan-time cost estimate",
        "Integrity",
        "Signed shared cache",
        "Publishing the site",
        "Audit chain",
    ):
        assert heading in printed, heading
