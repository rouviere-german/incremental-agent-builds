"""Detection rates: tampering, cache poisoning, cycles, and minimality over random programs.

Where ``bench_savings.py`` measures how much work is avoided, this script measures the rates the
spec states must be perfect, on a fixed sweep rather than a random one, so a regression shows up as
a number and not as a flaky test:

* ``cas_byte_flips_detected_pct`` - every mutated blob must fail its SHA-256 check on read;
* ``trace_edits_detected_pct`` - every edited SQLite row must fail its MAC;
* ``poisoned_entries_rejected_pct`` - unsigned, forged, re-signed, untrusted and blob-mismatched
  shared-cache entries must all be rejected;
* ``cycles_detected_pct`` - every injected cycle must raise ``CycleError`` rather than deadlock;
* ``clean_equivalent_pct`` and ``unnecessary_reruns`` - the same two properties the hypothesis suite
  checks, over a deterministic sweep of random programs and edit sequences.

Usage:
  uv run python bench/bench_robustness.py [--cases N] [--check]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pydantic import JsonValue

from _harness import (
    RESULTS_DIR,
    Metrics,
    Results,
    check_thresholds,
    machine,
    owned_thresholds,
    report,
)
from incremental_agent_builds.engine.cas import MemoryContentStore
from incremental_agent_builds.engine.engine import BuildEngine, EngineConfig
from incremental_agent_builds.engine.errors import IntegrityError
from incremental_agent_builds.engine.hashing import canonical_bytes, value_hash
from incremental_agent_builds.engine.metrics import create_build_metrics
from incremental_agent_builds.engine.shared_cache import import_bundle
from incremental_agent_builds.engine.signing import SigningKey, TrustStore
from incremental_agent_builds.engine.store import (
    Dep,
    DepKind,
    Trace,
    TraceStore,
    TraceUsage,
)
from incremental_agent_builds.workload.synthetic import (
    Edit,
    Node,
    NodeSpec,
    Program,
    initial_workspace,
    make_synthetic_provider,
    program_rules,
    random_program,
    run_case,
    with_cycle,
)


def bench_cas_tamper(samples: int) -> Metrics:
    """Flip one byte of a stored blob, ``samples`` times, at varying positions."""
    rng = random.Random(1)
    detected = 0
    for index in range(samples):
        store = MemoryContentStore()
        value: JsonValue = {
            "i": index,
            "text": "x" * (10 + index % 40),
            "items": list(range(index % 7)),
        }
        digest = store.put_json(value)
        data = bytearray(store.blobs[digest])
        position = rng.randrange(len(data))
        data[position] ^= 1 << rng.randrange(8)
        if bytes(data) == store.blobs[digest]:  # pragma: no cover - a flip always changes the byte
            continue
        store.blobs[digest] = bytes(data)
        try:
            store.get_json(digest)
        except IntegrityError:
            detected += 1
    return {
        "samples": float(samples),
        "detected": float(detected),
        "cas_byte_flips_detected_pct_min": round(100.0 * detected / samples, 2),
    }


def bench_trace_tamper(samples: int) -> Metrics:
    """Edit one column of one trace row, cycling through the columns a build relies on."""
    columns = ["deps", "result_hash", "version", "usage", "origin", "rule"]
    detected = 0
    for index in range(samples):
        store = TraceStore(":memory:", mac_key=os.urandom(32))
        trace = Trace(
            key_id=f'node:{{"i":{index}}}',
            rule="node",
            version=1,
            deps=((Dep(DepKind.FILE, "in/a.txt", value_hash(index)),),),
            result_hash=value_hash(index),
            built_run=1,
            changed_run=1,
            usage=TraceUsage(llm_steps=1),
        )
        store.put_traces([trace])
        column = columns[index % len(columns)]
        store.raw_execute(
            f"UPDATE traces SET {column} = ? WHERE key_id = ?",  # noqa: S608 - fixed column list
            (f"tampered-{index}", trace.key_id),
        )
        if store.get_trace(trace.key_id) is None:
            detected += 1
        store.close()
    return {
        "samples": float(samples),
        "detected": float(detected),
        "trace_edits_detected_pct_min": round(100.0 * detected / samples, 2),
    }


def _entry(signer: SigningKey, index: int) -> dict[str, object]:
    body: dict[str, object] = {
        "kind": "trace",
        "key_id": f'node:{{"i":{index}}}',
        "rule": "node",
        "version": 1,
        "deps": [[["file", "in/a.txt", value_hash(index)]]],
        "result_hash": value_hash(index),
        "usage": TraceUsage().to_json(),
    }
    return {
        "format": "iab-cache-bundle",
        "version": 1,
        "entries": [
            {"type": "trace", "body": body, "signer": signer.key_id, "signature": signer.sign(body)}  # type: ignore[arg-type]
        ],
        "blobs": {value_hash(index): base64.b64encode(canonical_bytes(index)).decode()},
    }


def bench_cache_poisoning(samples: int) -> Metrics:
    """Five ways to poison a bundle, cycled over ``samples`` entries; none may be accepted."""
    trusted = SigningKey.generate()
    trust = TrustStore([trusted.public_b64])
    metrics = create_build_metrics()
    attacks = ["unsigned", "forged_body", "foreign_signer", "resigned_claim", "blob_mismatch"]
    rejected = 0
    accepted = 0
    per_attack: dict[str, int] = dict.fromkeys(attacks, 0)
    for index in range(samples):
        attack = attacks[index % len(attacks)]
        bundle = _entry(trusted, index)
        entry = bundle["entries"][0]  # type: ignore[index]
        if attack == "unsigned":
            entry["signature"] = ""
        elif attack == "forged_body":
            entry["body"]["result_hash"] = value_hash(index + 10**6)
        elif attack == "foreign_signer":
            stranger = SigningKey.generate()
            bundle = _entry(stranger, index)
        elif attack == "resigned_claim":
            stranger = SigningKey.generate()
            bundle = _entry(stranger, index)
            bundle["entries"][0]["signer"] = trusted.key_id  # type: ignore[index]
        else:
            blobs: dict[str, str] = bundle["blobs"]  # type: ignore[assignment]
            digest = next(iter(blobs))
            blobs[digest] = base64.b64encode(b"not the value that was signed").decode()
        report_ = import_bundle(
            json.dumps(bundle).encode("utf-8"),
            store=TraceStore(":memory:", mac_key=os.urandom(32)),
            cas=MemoryContentStore(),
            trust=trust,
            metrics=metrics,
        )
        accepted += report_.accepted_traces
        if report_.accepted_traces == 0 and report_.rejected_total > 0:
            rejected += 1
            per_attack[attack] += 1
    result: Metrics = {
        "samples": float(samples),
        "accepted_max": float(accepted),
        "poisoned_entries_rejected_pct_min": round(100.0 * rejected / samples, 2),
    }
    for attack, count in per_attack.items():
        result[f"rejected_{attack}"] = float(count)
    return result


def _chain(n_nodes: int) -> Program:
    return Program(
        n_files=1,
        nodes=tuple(
            NodeSpec(reads=(0,), static_deps=(i - 1,) if i else ()) for i in range(n_nodes)
        ),
    )


def bench_cycles(samples: int) -> Metrics:
    """Inject a cycle of every length into a chain of every length; all must be reported."""
    detected = 0
    for index in range(samples):
        n_nodes = 2 + index % 7
        length = 1 + index % n_nodes
        jobs = 1 + index % 4
        program = with_cycle(_chain(n_nodes), list(range(length)))
        engine = BuildEngine(
            rules=program_rules(program),
            workspace=initial_workspace(program, [1]),
            store=TraceStore(":memory:", mac_key=os.urandom(32)),
            cas=MemoryContentStore(),
            provider=make_synthetic_provider(),
            config=EngineConfig(jobs=jobs, llm_concurrency=jobs, guard="off"),
        )
        result = engine.build_sync([Node(i=i) for i in range(n_nodes)])
        if result.cycles and all(c[0] == c[-1] and len(c) >= 2 for c in result.cycles):
            detected += 1
        engine.store.close()
    return {
        "samples": float(samples),
        "detected": float(detected),
        "cycles_detected_pct_min": round(100.0 * detected / samples, 2),
    }


def bench_semantics(cases: int) -> Metrics:
    """A deterministic sweep of random programs and edit sequences through the same oracles."""
    rng = random.Random(7)
    equivalent = 0
    unnecessary = 0
    builds = 0
    for _ in range(cases):
        program = random_program(
            rng,
            n_nodes=rng.randint(2, 14),
            n_files=rng.randint(1, 6),
            window=8,
            llm_fraction=0.25,
        )
        edits = [
            Edit(
                kind=rng.choice(["set", "set", "delete", "config", "template"]),
                target=rng.randrange(6),
                value=rng.randint(-4, 12),
            )
            for _ in range(rng.randint(1, 3))
        ]
        report_ = run_case(program, [rng.randint(-5, 20) for _ in range(4)], edits)
        builds += report_.builds
        if not report_.mismatches:
            equivalent += 1
        unnecessary += len(report_.unnecessary)
    return {
        "cases": float(cases),
        "builds": float(builds),
        "clean_equivalent_pct_min": round(100.0 * equivalent / cases, 2),
        "unnecessary_reruns_max": float(unnecessary),
    }


def run(samples: int, cases: int) -> Results:
    return {
        "cas_tampering": bench_cas_tamper(samples),
        "trace_tampering": bench_trace_tamper(samples),
        "cache_poisoning": bench_cache_poisoning(samples),
        "cycle_detection": bench_cycles(min(samples, 60)),
        "build_semantics_sweep": bench_semantics(cases),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--cases", type=int, default=60)
    parser.add_argument("--check", action="store_true", help="enforce bench/thresholds.json")
    args = parser.parse_args(argv)

    results = run(args.samples, args.cases)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "robustness.json").write_text(
        json.dumps({"machine": machine(), "results": results}, indent=2) + "\n", encoding="utf-8"
    )
    report("Detection rates and build semantics", results)
    if args.check:
        problems = check_thresholds(results)
        print()
        for problem in problems:
            print(f"REGRESSION: {problem}")
        if problems:
            return 1
        print(f"all {len(owned_thresholds(results))} thresholds for robustness met")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
