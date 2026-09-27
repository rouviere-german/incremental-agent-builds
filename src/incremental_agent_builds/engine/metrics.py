"""Prometheus metrics of the build engine (per-registry, like the HTTP metrics)."""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Histogram

__all__ = ["BuildMetrics", "create_build_metrics"]


@dataclass(frozen=True, slots=True)
class BuildMetrics:
    registry: CollectorRegistry
    rules_executed_total: Counter
    rules_reused_total: Counter
    early_cutoffs_total: Counter
    constructive_hits_total: Counter
    llm_provider_calls_total: Counter
    llm_tokens_saved_total: Counter
    cache_import_rejections_total: Counter
    integrity_failures_total: Counter
    hermeticity_violations_total: Counter
    build_duration_seconds: Histogram


def create_build_metrics(registry: CollectorRegistry | None = None) -> BuildMetrics:
    reg = registry if registry is not None else CollectorRegistry()
    return BuildMetrics(
        registry=reg,
        rules_executed_total=Counter(
            "rules_executed", "Rule bodies executed.", ["rule"], registry=reg
        ),
        rules_reused_total=Counter(
            "rules_reused", "Keys whose verifying trace matched (rule not run).", registry=reg
        ),
        early_cutoffs_total=Counter(
            "early_cutoffs",
            "Executed rules whose result hash equalled the previous one (dependents kept).",
            registry=reg,
        ),
        constructive_hits_total=Counter(
            "constructive_hits", "Keys served from signed shared-cache entries.", registry=reg
        ),
        llm_provider_calls_total=Counter(
            "llm_provider_calls", "LLM steps that reached the provider.", registry=reg
        ),
        llm_tokens_saved_total=Counter(
            "llm_tokens_saved",
            "Recorded LLM tokens not spent again thanks to reuse or step memoization.",
            registry=reg,
        ),
        cache_import_rejections_total=Counter(
            "cache_import_rejections",
            "Shared-cache entries rejected on import or use.",
            ["reason"],
            registry=reg,
        ),
        integrity_failures_total=Counter(
            "integrity_failures",
            "CAS blobs or store rows that failed verification.",
            ["kind"],
            registry=reg,
        ),
        hermeticity_violations_total=Counter(
            "hermeticity_violations",
            "I/O performed by rule bodies outside BuildContext.",
            registry=reg,
        ),
        build_duration_seconds=Histogram(
            "build_duration_seconds",
            "Wall-clock duration of builds.",
            registry=reg,
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
        ),
    )
