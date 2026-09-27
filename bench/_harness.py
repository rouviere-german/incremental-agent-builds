"""Shared plumbing for the benchmark scripts: timing, thresholds, reporting.

Every bench script produces a flat ``{name: {metric: value}}`` mapping, writes it to
``bench/results/<name>.json`` with the machine it ran on, prints a table, and with ``--check``
compares it against ``bench/thresholds.json``. Thresholds are directional: a metric whose name ends
in ``_min`` is a floor and must stay *above* its limit, anything else is a ceiling and must stay
*below* it. That lets one file hold both "no slower than" limits and "no less correct than" floors.
Keys beginning with ``_`` are documentation and are never compared.

Thresholds are set to several times the locally measured value, because shared CI runners are noisy:
they catch order-of-magnitude regressions and broken invariants, not small drifts.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "bench" / "results"
THRESHOLDS = ROOT / "bench" / "thresholds.json"

type Metrics = dict[str, float]
type Results = dict[str, Metrics]


def timed(function: Callable[[], object], *, runs: int, warmup: int = 5) -> Metrics:
    """Run ``function`` ``runs`` times and report the median and 95th percentile in microseconds."""
    for _ in range(warmup):
        function()
    samples: list[float] = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - start) / 1_000.0)
    return {
        "runs": float(runs),
        "p50_us": round(statistics.median(samples), 1),
        "p95_us": round(percentile(samples, 95), 1),
    }


def percentile(samples: list[float], percent: float) -> float:
    """Nearest-rank percentile (no interpolation), so a single sample is also well defined."""
    if not samples:
        raise ValueError("percentile of an empty sample")
    ordered = sorted(samples)
    rank = max(1, min(len(ordered), round(percent / 100.0 * len(ordered))))
    return ordered[rank - 1]


def machine() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "processor": platform.processor() or "unknown",
    }


def check_thresholds(results: Results, *, only: Mapping[str, Any] | None = None) -> list[str]:
    """Compare ``results`` against bench/thresholds.json; return one message per violation."""
    limits: dict[str, dict[str, float]] = (
        dict(only) if only is not None else json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    )
    problems: list[str] = []
    for name, metrics in limits.items():
        if name.startswith("_") or name not in results:
            continue  # documentation, or a benchmark this script does not own
        for metric, limit in metrics.items():
            if metric.startswith("_"):
                continue  # a _note explaining where the limit comes from
            if metric not in results[name]:
                problems.append(f"{name}.{metric}: not measured")
                continue
            value = results[name][metric]
            if metric.endswith("_min"):
                if value < limit:
                    problems.append(f"{name}.{metric} = {value} is below the floor {limit}")
            elif value > limit:
                problems.append(f"{name}.{metric} = {value} exceeds the limit {limit}")
    return problems


def owned_thresholds(results: Results) -> dict[str, dict[str, float]]:
    """The threshold groups this run is responsible for (used only to report how many were met)."""
    limits: dict[str, dict[str, float]] = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    return {
        name: metrics
        for name, metrics in limits.items()
        if name in results and not name.startswith("_")
    }


def report(title: str, results: Results) -> None:
    metrics = sorted({metric for values in results.values() for metric in values})
    print()
    print(f"### {title}")
    print()
    print("| benchmark | " + " | ".join(metrics) + " |")
    print("|---" * (len(metrics) + 1) + "|")
    for name, values in results.items():
        cells = [_format(values.get(metric)) for metric in metrics]
        print(f"| {name} | " + " | ".join(cells) + " |")


def _format(value: float | None) -> str:
    if value is None:
        return ""
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.4g}"


def main(
    *, name: str, title: str, run: Callable[[], Results], argv: list[str] | None = None
) -> int:
    """Standard entry point: run, write results, print a table, optionally enforce thresholds."""
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument("--check", action="store_true", help="enforce bench/thresholds.json")
    args = parser.parse_args(argv)

    results = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"machine": machine(), "results": results}
    (RESULTS_DIR / f"{name}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    report(title, results)
    if args.check:
        problems = check_thresholds(results)
        print()
        for problem in problems:
            print(f"REGRESSION: {problem}")
        if problems:
            return 1
        owned = owned_thresholds(results)
        print(f"all {len(owned)} thresholds for {name} met")
    return 0
