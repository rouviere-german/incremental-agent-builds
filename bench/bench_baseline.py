"""Micro-benchmarks of the hot paths, including the no-op rebuild target.

The load-bearing number here is ``noop_rebuild_1000_rules``: bringing a thousand up-to-date keys up
to date must stay well under 500 ms, because that is the cost a developer pays on every build where
nothing relevant changed. Everything it does is in that path: loading traces, re-hashing inputs,
comparing dependency hashes and scheduling.

Usage:
  uv run python bench/bench_baseline.py           # run and write bench/results/baseline.json
  uv run python bench/bench_baseline.py --check   # also fail on a bench/thresholds.json violation
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pydantic import JsonValue

from _harness import Metrics, Results, main, timed
from incremental_agent_builds.agentic.audit import AuditLog, verify_file
from incremental_agent_builds.agentic.providers.base import (
    ChatMessage,
    LLMRequest,
)
from incremental_agent_builds.core.ratelimit import RateLimiter
from incremental_agent_builds.core.redaction import redact_text
from incremental_agent_builds.engine.cas import FileContentStore, MemoryContentStore
from incremental_agent_builds.engine.engine import BuildEngine, EngineConfig
from incremental_agent_builds.engine.hashing import value_hash
from incremental_agent_builds.engine.llm import LLMStepSpec, schema_json
from incremental_agent_builds.engine.plan import make_plan
from incremental_agent_builds.engine.store import TraceStore
from incremental_agent_builds.workload.rules import ModuleSummaryOut
from incremental_agent_builds.workload.synthetic import (
    Node,
    initial_workspace,
    make_synthetic_provider,
    program_rules,
    random_program,
)

_RULES = 1_000


def _engine(n_rules: int) -> tuple[BuildEngine, list[Node]]:
    """An engine over a random 1k-rule program, already built once (so every key has a trace)."""
    import random

    program = random_program(
        random.Random(11), n_nodes=n_rules, n_files=64, window=24, llm_fraction=0.0
    )
    workspace = initial_workspace(program, list(range(1, 33)))
    engine = BuildEngine(
        rules=program_rules(program),
        workspace=workspace,
        store=TraceStore(":memory:", mac_key=os.urandom(32)),
        cas=MemoryContentStore(),
        provider=make_synthetic_provider(),
        config=EngineConfig(jobs=4, llm_concurrency=4, guard="off"),
    )
    targets = [Node(i=i) for i in range(n_rules)]
    engine.build_sync(targets).raise_for_failures()
    return engine, targets


def bench_noop_rebuild() -> Metrics:
    """The number the spec pins: a rebuild that finds nothing to do, over 1,000 rules."""
    engine, targets = _engine(_RULES)
    result = engine.build_sync(targets)
    if result.stats.executed or result.stats.reused != _RULES:
        raise AssertionError(f"the warm-up rebuild was not a no-op: {result.stats}")
    stats = timed(lambda: engine.build_sync(targets), runs=7, warmup=2)
    stats["rules"] = float(_RULES)
    stats["p95_ms"] = round(stats["p95_us"] / 1_000.0, 2)
    return stats


def bench_cold_build() -> Metrics:
    """For contrast: the same 1,000 rules from an empty store (not a threshold, a reference)."""
    import random

    program = random_program(
        random.Random(11), n_nodes=_RULES, n_files=64, window=24, llm_fraction=0.0
    )
    targets = [Node(i=i) for i in range(_RULES)]

    def build() -> None:
        engine = BuildEngine(
            rules=program_rules(program),
            workspace=initial_workspace(program, list(range(1, 33))),
            store=TraceStore(":memory:", mac_key=os.urandom(32)),
            cas=MemoryContentStore(),
            provider=make_synthetic_provider(),
            config=EngineConfig(jobs=4, llm_concurrency=4, guard="off"),
        )
        engine.build_sync(targets).raise_for_failures()

    stats = timed(build, runs=3, warmup=1)
    stats["p95_ms"] = round(stats["p95_us"] / 1_000.0, 2)
    return stats


def bench_plan_1000_rules() -> Metrics:
    """Planning is the dry run a caller does before approving a build; it must stay interactive."""
    engine, targets = _engine(_RULES)
    stats = timed(lambda: make_plan(engine, targets), runs=7, warmup=2)
    stats["p95_ms"] = round(stats["p95_us"] / 1_000.0, 2)
    return stats


def bench_value_hash() -> Metrics:
    value: JsonValue = {
        "module": "pkg.core.engine",
        "symbols": [{"name": f"f{i}"} for i in range(40)],
    }
    return timed(lambda: value_hash(value), runs=5_000)


def bench_cas_round_trip(workdir: Path) -> Metrics:
    store = FileContentStore(workdir / "cas")
    payload: JsonValue = {"pages": {f"m{i}.md": "x" * 200 for i in range(20)}}
    digest = store.put_json(payload)
    return timed(lambda: store.get_json(digest), runs=500)


def bench_trace_store_load(workdir: Path) -> Metrics:
    """Every build starts by loading and authenticating every trace row."""
    engine, _ = _engine(_RULES)
    rows = list(engine.store.load_traces().values())
    store = TraceStore(workdir / "traces.sqlite", mac_key=os.urandom(32))
    try:
        store.put_traces(rows)
        stats = timed(store.load_traces, runs=7, warmup=2)
    finally:
        store.close()  # Windows keeps the file locked until the connection is closed
    stats["rows"] = float(len(rows))
    stats["p95_ms"] = round(stats["p95_us"] / 1_000.0, 2)
    return stats


def bench_llm_fingerprint() -> Metrics:
    schema = schema_json(ModuleSummaryOut)
    variables: dict[str, JsonValue] = {
        "module": "pkg.core.engine",
        "outline": {"symbols": ["a"] * 30},
        "imported": [],
    }
    spec = LLMStepSpec.create(
        template_id="module_summary",
        template_text="Summarize {{module}} from {{outline}} and {{imported}}.\n",
        model="claude-opus-5",
        max_tokens=800,
        schema=schema,
        redact=True,
        variables=variables,
    )
    return timed(lambda: spec.fingerprint, runs=5_000)


def bench_redaction() -> Metrics:
    line = (
        "user jane.doe@example.com paid with 4111 1111 1111 1111 from +49 30 1234 5678; "
        + "lorem ipsum dolor sit amet " * 35
    )
    return timed(lambda: redact_text(line), runs=2_000)


def bench_rate_limiter() -> Metrics:
    limiter = RateLimiter(rate=1e9, burst=1_000_000_000)
    return timed(lambda: limiter.check("client"), runs=20_000)


def bench_request_hash() -> Metrics:
    messages = tuple(ChatMessage.user(f"message {i} " + "x" * 400) for i in range(10))
    request = LLMRequest(system="bench", messages=messages)
    return timed(request.sha256, runs=2_000)


def bench_audit_append(workdir: Path) -> Metrics:
    log = AuditLog(workdir / "append.jsonl")
    counter = iter(range(10**9))
    return timed(
        lambda: log.append("bench", "step", {"index": next(counter), "note": "x" * 100}), runs=200
    )


def bench_audit_append_batch(workdir: Path) -> Metrics:
    """A build appends one entry per rule, so the batched path is the one that matters."""
    log = AuditLog(workdir / "batch.jsonl")
    batch = [("bench", "rule_executed", {"key": f"node:{i}", "note": "x" * 80}) for i in range(200)]
    stats = timed(lambda: log.append_many(batch), runs=20, warmup=2)
    stats["entries_per_call"] = float(len(batch))
    return stats


def bench_audit_verify(workdir: Path) -> Metrics:
    path = workdir / "verify.jsonl"
    log = AuditLog(path)
    log.append_many([("bench", "step", {"index": i, "note": "x" * 100}) for i in range(1_000)])
    stats = timed(lambda: verify_file(path), runs=10, warmup=1)
    stats["p95_ms"] = round(stats["p95_us"] / 1_000.0, 2)
    return stats


def run() -> Results:
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        return {
            "noop_rebuild_1000_rules": bench_noop_rebuild(),
            "cold_build_1000_rules": bench_cold_build(),
            "plan_1000_rules": bench_plan_1000_rules(),
            "trace_store_load_1000_rows": bench_trace_store_load(workdir),
            "value_hash_40_symbols": bench_value_hash(),
            "cas_get_json_4kb": bench_cas_round_trip(workdir),
            "llm_step_fingerprint": bench_llm_fingerprint(),
            "redact_text_1kb": bench_redaction(),
            "rate_limiter_check": bench_rate_limiter(),
            "llm_request_sha256_10_messages": bench_request_hash(),
            "audit_append_fsync": bench_audit_append(workdir),
            "audit_append_many_200": bench_audit_append_batch(workdir),
            "audit_verify_1000_entries": bench_audit_verify(workdir),
        }


if __name__ == "__main__":
    raise SystemExit(main(name="baseline", title="Engine and baseline micro-benchmarks", run=run))
