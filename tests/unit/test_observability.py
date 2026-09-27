"""Metrics and spans a build emits, which is how a deployment sees reuse actually happening.

A build that reuses 99% of its keys looks identical to one that rebuilds everything unless the
counters say otherwise, so these are the numbers an operator would page on: how many rules ran, how
many were reused, how many cut off, how many tokens were not spent again, and how many cache imports
were rejected. Each rule also gets its own span carrying the number of dependencies of each kind it
read, which is what makes a slow build explainable after the fact.
"""

from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from prometheus_client import CollectorRegistry

from incremental_agent_builds.core.metrics import render_metrics
from incremental_agent_builds.core.tracing import build_tracer_provider
from incremental_agent_builds.engine.metrics import create_build_metrics
from tests.engine_helpers import Doubled, Echo, Sum, counting_rules, make_engine, workspace_with

pytestmark = pytest.mark.unit


def _families(registry: CollectorRegistry) -> str:
    payload, _ = render_metrics(registry)
    return payload.decode("utf-8")


def test_a_build_reports_executions_reuse_and_cutoffs() -> None:
    registry = CollectorRegistry()
    metrics = create_build_metrics(registry)
    workspace = workspace_with()
    engine = make_engine(workspace, counting_rules()[0], metrics=metrics)

    engine.build_sync([Doubled(path="in/a.txt")])
    text = _families(registry)
    assert 'rules_executed_total{rule="echo"} 1.0' in text
    assert "rules_reused_total 0.0" in text
    assert "early_cutoffs_total 0.0" in text

    engine.build_sync([Doubled(path="in/a.txt")])
    text = _families(registry)
    assert "rules_reused_total 2.0" in text, "the no-op rebuild reused both keys"
    assert 'rules_executed_total{rule="echo"} 1.0' in text, "and ran nothing new"


def test_an_early_cutoff_is_counted_separately_from_an_execution() -> None:
    registry = CollectorRegistry()
    metrics = create_build_metrics(registry)
    engine = make_engine(workspace_with(), counting_rules()[0], metrics=metrics)
    engine.build_sync([Echo(path="in/a.txt")])
    engine.build_sync([Echo(path="in/a.txt")], force=[Echo(path="in/a.txt")])
    text = _families(registry)
    assert 'rules_executed_total{rule="echo"} 2.0' in text
    assert "early_cutoffs_total 1.0" in text, "the forced re-run produced the same value"


def test_tokens_not_spent_again_are_counted() -> None:
    """The saving the repo exists for has to be observable, not only measurable in a benchmark."""
    from tests.engine_helpers import Upper

    registry = CollectorRegistry()
    metrics = create_build_metrics(registry)
    engine = make_engine(workspace_with(), counting_rules()[0], metrics=metrics)
    engine.build_sync([Upper(path="in/a.txt")])
    engine.build_sync([Upper(path="in/a.txt")])
    text = _families(registry)
    assert "llm_tokens_saved_total" in text
    assert "build_duration_seconds_count 2.0" in text


def test_a_rejected_cache_import_is_counted_by_reason() -> None:
    import os

    from incremental_agent_builds.engine.cas import MemoryContentStore
    from incremental_agent_builds.engine.shared_cache import import_bundle
    from incremental_agent_builds.engine.signing import SigningKey, TrustStore
    from incremental_agent_builds.engine.store import TraceStore

    registry = CollectorRegistry()
    metrics = create_build_metrics(registry)
    engine = make_engine(workspace_with(), counting_rules()[0], metrics=metrics)
    engine.build_sync([Echo(path="in/a.txt")])

    import json

    from incremental_agent_builds.engine.shared_cache import export_bundle

    bundle = export_bundle(engine.store, engine.cas, SigningKey.generate())
    import_bundle(
        json.dumps(bundle).encode("utf-8"),
        store=TraceStore(":memory:", mac_key=os.urandom(32)),
        cas=MemoryContentStore(),
        trust=TrustStore([]),  # trusts nobody
        metrics=metrics,
    )
    assert 'cache_import_rejections_total{reason="untrusted_signer"}' in _families(registry)


def test_every_rule_gets_a_span_naming_its_outcome_and_dependency_counts() -> None:
    exporter = InMemorySpanExporter()
    provider = build_tracer_provider("test", extra_exporter=exporter)
    tracer = provider.get_tracer("test")
    engine = make_engine(workspace_with(), counting_rules()[0], tracer=tracer)

    engine.build_sync([Sum(pattern="in/*.txt")])
    provider.force_flush()
    spans: list[ReadableSpan] = list(exporter.get_finished_spans())

    build = [span for span in spans if span.name == "iab.build"]
    rules = [span for span in spans if span.name == "iab.rule"]
    assert len(build) == 1
    assert len(rules) == 3, "one span per key brought up to date (sum plus two echoes)"

    attributes = [dict(span.attributes or {}) for span in rules]
    sum_span = next(a for a in attributes if a["iab.rule"] == "sum")
    assert sum_span["iab.outcome"] == "executed"
    assert sum_span["iab.reason"] == "new key"
    assert sum_span["iab.deps.glob"] == 1
    assert sum_span["iab.deps.rule"] == 2
    assert sum_span["iab.deps.llm"] == 0

    build_attributes = dict(build[0].attributes or {})
    assert build_attributes["iab.executed"] == 3
    assert build_attributes["iab.reused"] == 0


def test_a_reused_key_says_so_on_its_span() -> None:
    exporter = InMemorySpanExporter()
    provider = build_tracer_provider("test", extra_exporter=exporter)
    engine = make_engine(workspace_with(), counting_rules()[0], tracer=provider.get_tracer("test"))
    engine.build_sync([Echo(path="in/a.txt")])
    exporter.clear()
    engine.build_sync([Echo(path="in/a.txt")])
    provider.force_flush()

    rules = [s for s in exporter.get_finished_spans() if s.name == "iab.rule"]
    assert len(rules) == 1
    attributes = dict(rules[0].attributes or {})
    assert attributes["iab.outcome"] == "reused"
    assert attributes["iab.deps"] == 1, "the dependency count of the trace that was verified"
