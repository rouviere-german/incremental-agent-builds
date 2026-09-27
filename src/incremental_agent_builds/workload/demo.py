"""The offline demo: build a docs site, apply commits, show savings, staleness and the gate.

``uv run python tasks.py demo`` runs this end to end with the deterministic fake provider, so it
needs no API key and no network. It:

1. generates the 80-module reference package and builds ``DocsSite`` from scratch;
2. applies a slice of the scripted commit history and, per commit, prints executed rules, early
   cutoffs, LLM calls made versus a clean rebuild, and whether the incremental store still equals
   that clean rebuild value for value;
3. re-checks the two naive caches on the commits that change a template or a model id, where a
   prompt-hash cache serves the old model's output and this system serves none;
4. tampers with one CAS blob and one trace row to show both are detected;
5. exports a signed cache bundle, imports it into a second store, and shows a forged entry rejected;
6. asks for approval to publish the site, has a second principal approve it, and publishes.

Every number printed comes from the run itself. ``--commits`` keeps the demo to a few seconds.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from pydantic import JsonValue

from incremental_agent_builds.agentic.approval import ApprovalStore
from incremental_agent_builds.agentic.audit import AuditLog, verify_file
from incremental_agent_builds.engine.ast_tool import AstWorkerPool
from incremental_agent_builds.engine.cas import FileContentStore
from incremental_agent_builds.engine.effects import DocsSiteSink, publish_docs
from incremental_agent_builds.engine.engine import EngineConfig
from incremental_agent_builds.engine.errors import IntegrityError
from incremental_agent_builds.engine.metrics import create_build_metrics
from incremental_agent_builds.engine.oracle import compare_live
from incremental_agent_builds.engine.plan import make_plan
from incremental_agent_builds.engine.shared_cache import export_bundle, import_bundle
from incremental_agent_builds.engine.signing import SigningKey, TrustStore
from incremental_agent_builds.engine.store import TraceStore
from incremental_agent_builds.workload.factory import file_engine, memory_engine
from incremental_agent_builds.workload.fake_llm import make_fake_provider
from incremental_agent_builds.workload.generator import generate_state, write_workspace
from incremental_agent_builds.workload.history import COMMITS, apply_commit
from incremental_agent_builds.workload.naive import NaivePipeline
from incremental_agent_builds.workload.replay import llm_artifacts
from incremental_agent_builds.workload.rules import DocsSite, ModuleOutline, ModuleSummary

__all__ = ["DEMO_COMMITS", "run_demo"]

DEMO_COMMITS = 12
_TARGETS = [DocsSite()]


@dataclass
class _Line:
    index: int
    kind: str
    description: str
    executed: int
    reused: int
    early_cutoffs: int
    calls: int
    clean_calls: int
    equivalent: bool
    naive_prompt_stale: int

    def render(self) -> str:
        saved = 1.0 - (self.calls / self.clean_calls) if self.clean_calls else 0.0
        return (
            f"  {self.index:>2}. {self.kind:<13} ran {self.executed:>3} reused {self.reused:>3} "
            f"cutoff {self.early_cutoffs:>3} | LLM {self.calls:>3} vs {self.clean_calls:>3} clean "
            f"({saved:>6.1%} saved) | clean-equal {'yes' if self.equivalent else 'NO'}"
            f" | naive stale +{self.naive_prompt_stale}"
        )


def _print(text: str = "") -> None:
    print(text)


def _tamper_cas(store: TraceStore, cas: FileContentStore, key_id: str) -> bool:
    """Flip a byte in one stored blob; a read of it must raise instead of returning it."""
    trace = store.get_trace(key_id)
    if trace is None:  # pragma: no cover - the caller builds the key first
        return False
    path = cas.path_for(trace.result_hash)
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x20
    path.write_bytes(bytes(data))
    try:
        cas.get_json(trace.result_hash)
    except IntegrityError:
        return True
    return False  # pragma: no cover - a flipped byte always fails verification


def _tamper_trace(store: TraceStore) -> bool:
    """Rewrite a result hash in SQLite; the row must fail its MAC and be treated as absent."""
    key_id = DocsSite().key_id
    store.raw_execute(
        "UPDATE traces SET result_hash = ? WHERE key_id = ?",
        ("0" * 64, key_id),
    )
    before = len(store.integrity_events)
    reread = store.get_trace(key_id)
    return reread is None and len(store.integrity_events) > before


def _forged_entry(bundle: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """A copy of the bundle whose first trace entry claims a different result hash."""
    forged = json.loads(json.dumps(bundle))
    for entry in forged["entries"]:
        if entry["type"] == "trace":
            entry["body"]["result_hash"] = "f" * 64
            break
    return dict(forged)


def run_demo(workdir: Path, *, commits: int = DEMO_COMMITS) -> dict[str, JsonValue]:
    """Run the demo in ``workdir`` and return a JSON summary (also printed as a report)."""
    if workdir.exists():
        shutil.rmtree(workdir)
    workspace = workdir / "workspace"
    workspace.mkdir(parents=True)
    state = generate_state()
    write_workspace(workspace, state)
    files = state.files()
    pool = AstWorkerPool(size=2)
    config = EngineConfig(jobs=4, llm_concurrency=4, guard="warn")
    mac_key = os.urandom(32)
    signer = SigningKey.generate()
    audit = AuditLog(workdir / "audit.jsonl")
    metrics = create_build_metrics()
    summary: dict[str, JsonValue] = {}
    try:
        engine = file_engine(
            workspace,
            workdir / "store",
            provider=make_fake_provider(),
            pool=pool,
            mac_key=mac_key,
            config=config,
            audit=audit,
            metrics=metrics,
            trust=TrustStore([signer.public_b64]),
        )
        cas = engine.cas
        assert isinstance(cas, FileContentStore)  # noqa: S101 - file_engine always builds one
        naive = NaivePipeline(make_fake_provider(), "prompt-hash")

        _print("=" * 100)
        _print("incremental-agent-builds demo - offline, deterministic fake provider, no API key")
        _print("=" * 100)
        _print()
        _print(f"1. Reference workspace: {len(state.model.modules)} modules in {workspace}")
        first = engine.build_sync(_TARGETS)
        first.raise_for_failures()
        _print(
            f"   Cold build: {first.stats.executed} rules executed, "
            f"{first.stats.provider_calls} LLM calls, {first.stats.tokens} tokens, "
            f"{first.duration_seconds:.2f}s"
        )
        no_op = engine.build_sync(_TARGETS)
        _print(
            f"   No-op rebuild: {no_op.stats.executed} executed, {no_op.stats.reused} reused, "
            f"{no_op.stats.provider_calls} LLM calls, {no_op.duration_seconds * 1000:.0f} ms"
        )
        naive.run(files)
        summary["cold_executed"] = first.stats.executed
        summary["cold_calls"] = first.stats.provider_calls
        summary["noop_executed"] = no_op.stats.executed
        summary["noop_calls"] = no_op.stats.provider_calls

        _print()
        _print(f"2. Replaying {commits} commits of the scripted history:")
        lines: list[_Line] = []
        naive_stale: set[str] = set()
        for index in range(min(commits, len(COMMITS))):
            kind, description = apply_commit(state, index)
            new_files = state.files()
            write_workspace(workspace, state, previous=files)
            files = new_files
            result = engine.build_sync(_TARGETS)
            result.raise_for_failures()
            clean = memory_engine(
                workspace, provider=make_fake_provider(), pool=pool, config=config
            )
            clean_result = clean.build_sync(_TARGETS)
            clean_result.raise_for_failures()
            problems = compare_live(engine, result.live, clean, clean_result.live)
            truth = llm_artifacts(clean, clean_result.live)
            stale_now = {k for k, v in naive.run(files).items() if truth.get(k) != v}
            line = _Line(
                index=index + 1,
                kind=kind,
                description=description,
                executed=result.stats.executed,
                reused=result.stats.reused,
                early_cutoffs=result.stats.early_cutoffs,
                calls=result.stats.provider_calls,
                clean_calls=clean_result.stats.provider_calls,
                equivalent=not problems,
                naive_prompt_stale=len(stale_now - naive_stale),
            )
            naive_stale = stale_now
            lines.append(line)
            _print(line.render())
        total_calls = sum(line.calls for line in lines)
        total_clean = sum(line.clean_calls for line in lines)
        saved = 1.0 - total_calls / total_clean if total_clean else 0.0
        _print()
        _print(
            f"   Total: {total_calls} LLM calls instead of {total_clean} for clean rebuilds "
            f"({saved:.1%} saved); clean-equivalent on {sum(1 for x in lines if x.equivalent)}"
            f"/{len(lines)} commits."
        )
        stale_events = sum(line.naive_prompt_stale for line in lines)
        if stale_events:
            _print(
                f"   The prompt-hash cache baseline went stale {stale_events} time(s) over these "
                f"commits and still serves {len(naive_stale)} stale artifact(s) now; this system "
                "served 0 stale values, proven value-for-value against the clean rebuilds above."
            )
        else:
            _print(
                "   The prompt-hash baseline has not gone stale yet: the commits that break it "
                "(a model-id change) come later in the history. Run with "
                f"--commits {len(COMMITS)} to reach them."
            )
        summary["commits"] = len(lines)
        summary["incremental_calls"] = total_calls
        summary["clean_calls"] = total_clean
        summary["calls_saved_fraction"] = round(saved, 4)
        summary["clean_equivalent_commits"] = sum(1 for x in lines if x.equivalent)
        summary["naive_prompt_hash_stale_events"] = stale_events
        summary["naive_prompt_hash_stale_now"] = len(naive_stale)

        _print()
        _print("3. Plan-time cost estimate and the approval gate:")
        apply_commit(state, len(lines))
        write_workspace(workspace, state, previous=files)
        files = state.files()
        plan = make_plan(engine, _TARGETS)
        _print(
            f"   {len(plan.items)} dirty of {len(plan.items) + plan.clean_keys} keys; projected "
            f"{plan.projected_llm_steps} LLM steps, {plan.projected_tokens} tokens, "
            f"${plan.projected_usd_micros / 1_000_000:.4f}; plan {plan.plan_hash[:12]}"
        )
        summary["plan_dirty_keys"] = len(plan.items)
        summary["plan_projected_tokens"] = plan.projected_tokens
        engine.build_sync(_TARGETS).raise_for_failures()

        _print()
        _print("4. Integrity: a flipped CAS byte and an edited trace row")
        victim = state.model.modules[3].path(state.model.root)
        outline_key = ModuleOutline(path=victim)
        cas_detected = _tamper_cas(engine.store, cas, outline_key.key_id)
        _print(f"   flipped the ModuleOutline blob of {victim}: read raises: {cas_detected}")
        # Force the rule that consumes that value, so the corrupt blob is actually read.
        after_cas = engine.build_sync(_TARGETS, force=[ModuleSummary(path=victim)])
        after_cas.raise_for_failures()
        _print(
            f"   next build: {after_cas.stats.integrity_events} integrity events, "
            f"{after_cas.stats.regenerated} value(s) recomputed to the same hash, ok={after_cas.ok}"
        )
        trace_detected = _tamper_trace(engine.store)
        _print(f"   edited a trace row in SQLite: MAC rejects it: {trace_detected}")
        after_trace = engine.build_sync(_TARGETS)
        after_trace.raise_for_failures()
        _print(
            f"   next build: {after_trace.stats.executed} executed (the key lost its trace), "
            f"ok={after_trace.ok}"
        )
        summary["cas_tamper_detected"] = cas_detected
        summary["cas_integrity_events"] = after_cas.stats.integrity_events
        summary["values_regenerated"] = after_cas.stats.regenerated
        summary["trace_tamper_detected"] = trace_detected

        _print()
        _print("5. Signed shared cache: export, import into a fresh store, reject a forgery")
        bundle = export_bundle(engine.store, engine.cas, signer)
        raw = json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode("utf-8")
        peer_store = TraceStore(workdir / "peer.sqlite", mac_key=os.urandom(32))
        peer_cas = FileContentStore(workdir / "peer-cas")
        accepted = import_bundle(
            raw,
            store=peer_store,
            cas=peer_cas,
            trust=TrustStore([signer.public_b64]),
            metrics=metrics,
        )
        forged = json.dumps(_forged_entry(bundle)).encode("utf-8")
        rejected = import_bundle(
            forged,
            store=TraceStore(":memory:", mac_key=os.urandom(32)),
            cas=peer_cas,
            trust=TrustStore([signer.public_b64]),
            metrics=metrics,
        )
        untrusted = import_bundle(
            raw,
            store=TraceStore(":memory:", mac_key=os.urandom(32)),
            cas=peer_cas,
            trust=TrustStore([SigningKey.generate().public_b64]),
            metrics=metrics,
        )
        _print(
            f"   exported {accepted.accepted_traces} signed traces and "
            f"{accepted.accepted_steps} steps into a peer store"
        )
        _print(f"   forged result hash rejected: {dict(rejected.rejected)}")
        _print(f"   untrusted signer rejected: {dict(untrusted.rejected)}")
        peer_store.close()
        summary["cache_accepted_traces"] = accepted.accepted_traces
        summary["cache_forged_rejected"] = rejected.rejected_total
        summary["cache_untrusted_rejected"] = untrusted.rejected_total

        _print()
        _print("6. Publishing the site: an effect sink behind an approval bound to the payload")
        approvals = ApprovalStore(workdir / "approvals.json")
        build = engine.build_sync(_TARGETS)
        build.raise_for_failures()
        payload = engine.value(DocsSite())
        blocked = publish_docs(
            build=build,
            payload=payload,
            sink=DocsSiteSink(workdir / "site"),
            approvals=approvals,
            principal="builder",
            audit=audit,
        )
        _print(f"   first attempt: {blocked.status} (approval {blocked.approval_id})")
        assert blocked.approval_id is not None  # noqa: S101 - the gate always opens a request
        approvals.decide(blocked.approval_id, approver="reviewer", approve=True)
        applied = publish_docs(
            build=build,
            payload=payload,
            sink=DocsSiteSink(workdir / "site"),
            approvals=approvals,
            principal="builder",
            audit=audit,
        )
        _print(f"   after approval: {applied.status}, {len(applied.written)} pages written")
        summary["publish_first_status"] = blocked.status
        summary["publish_after_approval"] = applied.status
        summary["pages_published"] = len(applied.written)

        _print()
        chain = verify_file(audit.path, expect_head=audit.head)
        _print(f"7. Audit chain: {chain.entries} entries, verified ok={chain.ok}")
        summary["audit_entries"] = chain.entries
        summary["audit_ok"] = chain.ok
        _print()
        return summary
    finally:
        pool.close()
