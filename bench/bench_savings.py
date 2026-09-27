"""The savings benchmark: replay the scripted commit history and measure what was avoided.

This is the benchmark the README's headline numbers come from. For every commit it builds the docs
site incrementally, builds it again from an empty store (the clean rebuild), and compares:

* ``median_calls_saved`` / ``median_tokens_saved`` - per-commit savings against the clean rebuild;
* ``clean_equivalent_pct`` - the fraction of commits after which the incremental store still equals
  the clean rebuild value for value. Anything below 100 is a correctness failure, not a slow-down;
* ``unnecessary_reruns`` - executed rules with no changed direct dependency, against the oracle;
* ``cutoff_executed_ratio`` / ``memo_call_ratio`` - the two ablations. Turning off early cutoff
  multiplies the *rules executed*; turning off step memoization multiplies the *LLM calls*;
* ``naive_*_stale`` - artifacts the two naive cache baselines serve stale, where this system
  serves none at all.

Everything runs offline with the deterministic provider, so the LLM calls a clean rebuild makes are
exactly the calls the incremental build avoided.

Usage:
  uv run python bench/bench_savings.py [--commits N] [--check]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import (
    RESULTS_DIR,
    Metrics,
    Results,
    check_thresholds,
    machine,
    owned_thresholds,
    report,
)
from incremental_agent_builds.workload.history import COMMITS
from incremental_agent_builds.workload.replay import ReplayReport, run_replay


def _number(value: object) -> float:
    return float(value) if isinstance(value, int | float) else 0.0


def metrics_from(replay: ReplayReport) -> Results:
    summary = replay.summary()
    rows = replay.rows
    checked = [row for row in rows if row.equivalent is not None]

    savings: Metrics = {
        "commits": float(len(rows)),
        "median_calls_saved_min": round(_number(summary.get("median_calls_saved")), 4),
        "median_tokens_saved_min": round(_number(summary.get("median_tokens_saved")), 4),
        "total_calls_saved_min": round(_number(summary.get("total_calls_saved")), 4),
        "total_tokens_saved_min": round(_number(summary.get("total_tokens_saved")), 4),
        "incremental_calls": _number(summary.get("incremental_calls")),
        "clean_calls": _number(summary.get("clean_calls")),
        "incremental_tokens": _number(summary.get("incremental_tokens")),
        "clean_tokens": _number(summary.get("clean_tokens")),
    }

    correctness: Metrics = {
        "commits_checked": float(len(checked)),
        "clean_equivalent_pct_min": round(
            100.0 * sum(1 for row in checked if row.equivalent) / max(1, len(checked)), 2
        ),
        "unnecessary_reruns_max": _number(summary.get("unnecessary_reruns")),
        "system_stale_artifacts_max": _number(summary.get("system_newly_stale")),
    }

    work: Metrics = {
        "rules_executed": _number(summary.get("incremental_executed")),
        "early_cutoffs_min": _number(summary.get("incremental_early_cutoffs")),
        "no_cutoff_executed": _number(summary.get("no_cutoff_executed")),
        "no_cutoff_calls": _number(summary.get("no_cutoff_calls")),
        "no_memo_executed": _number(summary.get("no_memo_executed")),
        "no_memo_calls": _number(summary.get("no_memo_calls")),
    }
    # The two ablations save different things, so each is measured against what it affects.
    # Early cutoff stops a *rule* re-running when an unchanged value came back; step memoization
    # stops an *LLM call* being paid again. Comparing cutoff against LLM calls would report a ratio
    # of 1.0 by construction, because step memoization still serves the outputs.
    if work["no_cutoff_executed"]:
        work["cutoff_executed_ratio_min"] = round(
            work["no_cutoff_executed"] / max(1.0, work["rules_executed"]), 3
        )
        # The aggregate ratio is diluted: most commits touch little, and on those cutoff has nothing
        # to stop. The per-commit maximum is where the mechanism shows: a layout-only edit executes
        # one rule with cutoff and the whole downstream fan-out without it.
        per_commit = [
            (row.no_cutoff_executed or 0) / max(1, row.executed)
            for row in rows
            if row.no_cutoff_executed is not None
        ]
        if per_commit:
            work["cutoff_executed_ratio_worst_commit_min"] = round(max(per_commit), 2)
    if work["no_memo_calls"]:
        work["memo_call_ratio_min"] = round(
            work["no_memo_calls"] / max(1.0, savings["incremental_calls"]), 3
        )

    stale_by_kind = summary.get("naive_newly_stale_by_kind")
    stale: Metrics = {}
    if isinstance(stale_by_kind, dict):
        for mode, per_kind in stale_by_kind.items():
            total = (
                sum(v for v in per_kind.values() if isinstance(v, int))
                if isinstance(per_kind, dict)
                else 0
            )
            stale[f"{mode.replace('-', '_')}_stale_events_min"] = float(total)
    end_state = summary.get("naive_stale_at_end")
    if isinstance(end_state, dict):
        for mode, count in end_state.items():
            if isinstance(count, int):
                stale[f"{mode.replace('-', '_')}_stale_at_end"] = float(count)

    results: Results = {
        "replay_savings": savings,
        "replay_correctness": correctness,
        "replay_ablations": work,
    }
    if stale:
        results["naive_cache_staleness"] = stale
    return results


def run(commits: int, jobs: int) -> tuple[Results, ReplayReport]:
    with tempfile.TemporaryDirectory() as tmp:
        replay = run_replay(
            Path(tmp) / "replay",
            commits=commits,
            clean=True,
            ablations=True,
            naive=True,
            jobs=jobs,
        )
    return metrics_from(replay), replay


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--commits", type=int, default=len(COMMITS))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--check", action="store_true", help="enforce bench/thresholds.json")
    args = parser.parse_args(argv)

    results, replay = run(args.commits, args.jobs)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "savings.json").write_text(
        json.dumps({"machine": machine(), "results": results, "replay": replay.to_json()}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"Replayed {args.commits} commits of the scripted history (jobs={args.jobs}).")
    print(
        f"Cold build: {replay.initial['executed']} rules, "
        f"{replay.initial['provider_calls']} LLM calls, {replay.initial['tokens']} tokens, "
        f"{replay.initial['duration_seconds']}s."
    )
    print()
    print(
        "| # | commit kind | executed | reused | cutoffs | LLM | clean LLM | saved | clean-equal |"
    )
    print("|---:|---|---:|---:|---:|---:|---:|---:|---|")
    for row in replay.rows:
        saved = f"{row.calls_saved:.0%}" if row.calls_saved is not None else ""
        equal = "yes" if row.equivalent else "NO"
        print(
            f"| {row.index} | {row.kind} | {row.executed} | {row.reused} | {row.early_cutoffs} "
            f"| {row.provider_calls} | {row.clean_calls} | {saved} | {equal} |"
        )
    report("Savings and correctness over the commit replay", results)
    if args.check:
        problems = check_thresholds(results)
        print()
        for problem in problems:
            print(f"REGRESSION: {problem}")
        if problems:
            return 1
        print(f"all {len(owned_thresholds(results))} thresholds for savings met")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
