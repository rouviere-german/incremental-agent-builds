"""Replay the 30-commit history: incremental builds against clean rebuilds and naive caches.

For every commit the harness applies the edit to the workspace and then:

1. builds ``DocsSite`` incrementally with early cutoff and step memoization (the system);
2. optionally builds the two ablations on their own stores: without early cutoff, and without
   early cutoff *and* without LLM step memoization (a make-style rebuild of everything downstream);
3. optionally builds from an empty store (the clean rebuild) and checks clean equivalence value by
   value, plus minimality of the incremental build against the two clean builds;
4. optionally runs both naive-cache baselines and counts artifacts that differ from the correct
   (clean) values.

All builds use the deterministic fake provider, so a clean rebuild's LLM calls are exactly the
calls the incremental build avoided or made.
"""

from __future__ import annotations

import os
import shutil
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from incremental_agent_builds.engine.ast_tool import AstWorkerPool
from incremental_agent_builds.engine.engine import BuildEngine, BuildResult, EngineConfig
from incremental_agent_builds.engine.oracle import compare_live, unnecessary_reruns
from incremental_agent_builds.workload.factory import file_engine, memory_engine
from incremental_agent_builds.workload.fake_llm import make_fake_provider
from incremental_agent_builds.workload.generator import generate_state, write_workspace
from incremental_agent_builds.workload.history import COMMITS, apply_commit
from incremental_agent_builds.workload.naive import NaivePipeline
from incremental_agent_builds.workload.rules import (
    DocsSite,
    ModuleSummary,
    PackageSummary,
    ReleaseNote,
)

__all__ = ["CommitRow", "ReplayReport", "llm_artifacts", "run_replay"]

TARGETS = [DocsSite()]


@dataclass
class CommitRow:
    index: int
    kind: str
    description: str
    executed: int
    reused: int
    early_cutoffs: int
    provider_calls: int
    tokens: int
    clean_calls: int | None = None
    clean_tokens: int | None = None
    equivalent: bool | None = None
    unnecessary_reruns: int | None = None
    no_cutoff_executed: int | None = None
    no_cutoff_calls: int | None = None
    no_memo_executed: int | None = None
    no_memo_calls: int | None = None
    naive_input_stale: int | None = None
    naive_input_calls: int | None = None
    naive_prompt_stale: int | None = None
    naive_prompt_calls: int | None = None
    system_stale: int | None = None

    @property
    def calls_saved(self) -> float | None:
        if not self.clean_calls:
            return None
        return 1.0 - self.provider_calls / self.clean_calls

    @property
    def tokens_saved(self) -> float | None:
        if not self.clean_tokens:
            return None
        return 1.0 - self.tokens / self.clean_tokens


@dataclass
class ReplayReport:
    rows: list[CommitRow] = field(default_factory=list)
    initial: dict[str, Any] = field(default_factory=dict)
    naive_stale_at_end: dict[str, int] = field(default_factory=dict)

    def summary(self) -> dict[str, JsonValue]:
        rows = self.rows
        calls = [r.calls_saved for r in rows if r.calls_saved is not None]
        tokens = [r.tokens_saved for r in rows if r.tokens_saved is not None]
        checked = [r for r in rows if r.equivalent is not None]
        result: dict[str, JsonValue] = {
            "commits": len(rows),
            "incremental_executed": sum(r.executed for r in rows),
            "incremental_early_cutoffs": sum(r.early_cutoffs for r in rows),
            "incremental_calls": sum(r.provider_calls for r in rows),
            "incremental_tokens": sum(r.tokens for r in rows),
        }
        if calls:
            result["median_calls_saved"] = round(statistics.median(calls), 4)
            result["median_tokens_saved"] = round(statistics.median(tokens), 4)
            clean_calls = sum(r.clean_calls or 0 for r in rows)
            clean_tokens = sum(r.clean_tokens or 0 for r in rows)
            result["clean_calls"] = clean_calls
            result["clean_tokens"] = clean_tokens
            result["total_calls_saved"] = round(1 - result["incremental_calls"] / clean_calls, 4)  # type: ignore[operator]
            result["total_tokens_saved"] = round(1 - result["incremental_tokens"] / clean_tokens, 4)  # type: ignore[operator]
        if checked:
            result["equivalent_commits"] = sum(1 for r in checked if r.equivalent)
            result["checked_commits"] = len(checked)
            result["unnecessary_reruns"] = sum(r.unnecessary_reruns or 0 for r in checked)
        for name in ("no_cutoff", "no_memo"):
            values = [getattr(r, f"{name}_calls") for r in rows]
            if all(v is not None for v in values):
                result[f"{name}_calls"] = sum(values)
                result[f"{name}_executed"] = sum(getattr(r, f"{name}_executed") for r in rows)
        stale: dict[str, JsonValue] = {}
        for mode in ("input", "prompt"):
            per_kind: dict[str, int] = {}
            for r in rows:
                value = getattr(r, f"naive_{mode}_stale")
                if value is not None:
                    per_kind[r.kind] = per_kind.get(r.kind, 0) + value
            if per_kind:
                stale[f"{mode}-hash"] = dict(sorted(per_kind.items()))
        if stale:
            result["naive_newly_stale_by_kind"] = stale
            result["naive_stale_at_end"] = dict(sorted(self.naive_stale_at_end.items()))
            result["system_newly_stale"] = sum(r.system_stale or 0 for r in rows)
        return result

    def to_json(self) -> dict[str, Any]:
        rows = []
        for row in self.rows:
            data = asdict(row)
            data["calls_saved"] = row.calls_saved
            data["tokens_saved"] = row.tokens_saved
            rows.append(data)
        return {"initial": self.initial, "summary": self.summary(), "rows": rows}


def llm_artifacts(engine: BuildEngine, live: dict[str, str]) -> dict[str, JsonValue]:
    """The LLM-derived artifacts of a build, keyed like NaivePipeline.run's output."""
    result: dict[str, JsonValue] = {}
    for key_id, digest in live.items():
        key = engine.rules.key_from_id(key_id)
        if isinstance(key, ModuleSummary):
            result[f"module:{key.path}"] = engine.cas.get_json(digest)
        elif isinstance(key, PackageSummary):
            result["package"] = engine.cas.get_json(digest)
        elif isinstance(key, ReleaseNote):
            result["release"] = engine.cas.get_json(digest)
    return result


def _stale(truth: dict[str, JsonValue], candidate: dict[str, JsonValue]) -> set[str]:
    keys = set(truth) | set(candidate)
    return {key for key in keys if truth.get(key) != candidate.get(key)}


def run_replay(
    workdir: Path,
    *,
    commits: int = len(COMMITS),
    clean: bool = True,
    ablations: bool = True,
    naive: bool = True,
    jobs: int = 4,
    progress: Callable[[CommitRow], None] | None = None,
) -> ReplayReport:
    if workdir.exists():
        shutil.rmtree(workdir)
    workspace = workdir / "workspace"
    workspace.mkdir(parents=True)
    state = generate_state()
    write_workspace(workspace, state)
    files = state.files()
    pool = AstWorkerPool(size=2)
    config = EngineConfig(jobs=jobs, llm_concurrency=jobs)
    mac_key = os.urandom(32)
    # Every engine owns a SQLite connection; they are closed in the finally so the workdir can be
    # removed afterwards (on Windows an open connection keeps the file locked).
    engines: list[BuildEngine] = []

    def track(engine: BuildEngine) -> BuildEngine:
        engines.append(engine)
        return engine

    try:
        main = track(
            file_engine(
                workspace,
                workdir / "store",
                provider=make_fake_provider(),
                pool=pool,
                mac_key=mac_key,
                config=config,
            )
        )
        variants: dict[str, BuildEngine] = {}
        if ablations:
            variants["no_cutoff"] = track(
                file_engine(
                    workspace,
                    workdir / "store-no-cutoff",
                    provider=make_fake_provider(),
                    pool=pool,
                    mac_key=mac_key,
                    config=EngineConfig(jobs=jobs, llm_concurrency=jobs, early_cutoff=False),
                )
            )
            variants["no_memo"] = track(
                file_engine(
                    workspace,
                    workdir / "store-no-memo",
                    provider=make_fake_provider(),
                    pool=pool,
                    mac_key=mac_key,
                    config=EngineConfig(
                        jobs=jobs, llm_concurrency=jobs, early_cutoff=False, step_memo=False
                    ),
                )
            )
        naive_pipelines = (
            {
                "input": NaivePipeline(make_fake_provider(), "input-hash"),
                "prompt": NaivePipeline(make_fake_provider(), "prompt-hash"),
            }
            if naive
            else {}
        )
        report = ReplayReport()
        first = main.build_sync(TARGETS)
        first.raise_for_failures()
        for engine in variants.values():
            engine.build_sync(TARGETS).raise_for_failures()
        for pipeline in naive_pipelines.values():
            pipeline.run(files)
        stale_sets: dict[str, set[str]] = {mode: set() for mode in naive_pipelines}
        previous_clean: BuildResult | None = None
        if clean:
            clean_engine = track(
                memory_engine(workspace, provider=make_fake_provider(), pool=pool, config=config)
            )
            previous_clean = clean_engine.build_sync(TARGETS)
        report.initial = {
            "executed": first.stats.executed,
            "provider_calls": first.stats.provider_calls,
            "tokens": first.stats.tokens,
            "duration_seconds": round(first.duration_seconds, 3),
            "live_keys": len(first.live),
        }
        for index in range(commits):
            kind, description = apply_commit(state, index)
            new_files = state.files()
            write_workspace(workspace, state, previous=files)
            files = new_files
            old_traces = main.store.load_traces()
            result = main.build_sync(TARGETS)
            result.raise_for_failures()
            row = CommitRow(
                index=index + 1,
                kind=kind,
                description=description,
                executed=result.stats.executed,
                reused=result.stats.reused,
                early_cutoffs=result.stats.early_cutoffs,
                provider_calls=result.stats.provider_calls,
                tokens=result.stats.tokens,
            )
            for name, engine in variants.items():
                variant = engine.build_sync(TARGETS)
                variant.raise_for_failures()
                setattr(row, f"{name}_executed", variant.stats.executed)
                setattr(row, f"{name}_calls", variant.stats.provider_calls)
            clean_result: BuildResult | None = None
            clean_engine_now: BuildEngine | None = None
            if clean:
                clean_engine_now = track(
                    memory_engine(
                        workspace, provider=make_fake_provider(), pool=pool, config=config
                    )
                )
                clean_result = clean_engine_now.build_sync(TARGETS)
                clean_result.raise_for_failures()
                row.clean_calls = clean_result.stats.provider_calls
                row.clean_tokens = clean_result.stats.tokens
                problems = compare_live(main, result.live, clean_engine_now, clean_result.live)
                row.equivalent = not problems
                row.unnecessary_reruns = len(
                    unnecessary_reruns(main, old_traces, result.executed, clean_result.live)
                )
                previous_clean = clean_result
            system = llm_artifacts(main, result.live)
            truth = system
            if clean_engine_now is not None and clean_result is not None:
                truth = llm_artifacts(clean_engine_now, clean_result.live)
                row.system_stale = len(_stale(truth, system))
            if naive_pipelines:
                for mode, pipeline in naive_pipelines.items():
                    calls_before = pipeline.provider_calls
                    stale_now = _stale(truth, pipeline.run(files))
                    # Count artifacts this commit made stale, not ones already stale before it.
                    setattr(row, f"naive_{mode}_stale", len(stale_now - stale_sets[mode]))
                    setattr(row, f"naive_{mode}_calls", pipeline.provider_calls - calls_before)
                    stale_sets[mode] = stale_now
                    report.naive_stale_at_end[f"{mode}-hash"] = len(stale_now)
            report.rows.append(row)
            if progress is not None:
                progress(row)
        del previous_clean
        return report
    finally:
        pool.close()
        for engine in engines:
            engine.store.close()
