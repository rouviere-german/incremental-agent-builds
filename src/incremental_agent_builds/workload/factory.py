"""Wiring for engines over a workload workspace (used by the service, demo, replay and tests)."""

from __future__ import annotations

import os
from pathlib import Path

from incremental_agent_builds.agentic.audit import AuditLog
from incremental_agent_builds.agentic.budget import BudgetLimits
from incremental_agent_builds.agentic.providers.base import LLMProvider
from incremental_agent_builds.engine.ast_tool import AstWorkerPool, make_ast_outline_tool
from incremental_agent_builds.engine.cas import FileContentStore, MemoryContentStore
from incremental_agent_builds.engine.engine import DEFAULT_BUDGET, BuildEngine, EngineConfig
from incremental_agent_builds.engine.metrics import BuildMetrics
from incremental_agent_builds.engine.signing import TrustStore
from incremental_agent_builds.engine.store import TraceStore
from incremental_agent_builds.engine.tools import BuildToolRegistry
from incremental_agent_builds.engine.workspace import FileWorkspace
from incremental_agent_builds.workload.rules import workload_rules

__all__ = ["build_tools", "file_engine", "memory_engine"]


def build_tools(pool: AstWorkerPool) -> BuildToolRegistry:
    """The read-only tool registry every engine in this repo uses."""
    return BuildToolRegistry([make_ast_outline_tool(pool)])


def file_engine(
    workspace: Path,
    store_dir: Path,
    *,
    provider: LLMProvider,
    pool: AstWorkerPool,
    mac_key: bytes,
    config: EngineConfig | None = None,
    audit: AuditLog | None = None,
    metrics: BuildMetrics | None = None,
    trust: TrustStore | None = None,
    default_budget: BudgetLimits = DEFAULT_BUDGET,
    max_file_bytes: int = 1_048_576,
) -> BuildEngine:
    """An engine with an on-disk SQLite trace store and CAS under ``store_dir``."""
    store_dir.mkdir(parents=True, exist_ok=True)
    return BuildEngine(
        rules=workload_rules(),
        workspace=FileWorkspace(workspace, max_file_bytes=max_file_bytes),
        store=TraceStore(store_dir / "traces.sqlite", mac_key=mac_key),
        cas=FileContentStore(store_dir / "cas"),
        provider=provider,
        tools=build_tools(pool),
        config=config,
        audit=audit,
        metrics=metrics,
        trust=trust,
        default_budget=default_budget,
    )


def memory_engine(
    workspace: Path,
    *,
    provider: LLMProvider,
    pool: AstWorkerPool,
    config: EngineConfig | None = None,
) -> BuildEngine:
    """An engine with in-memory stores (clean rebuilds in the replay and tests)."""
    return BuildEngine(
        rules=workload_rules(),
        workspace=FileWorkspace(workspace),
        store=TraceStore(":memory:", mac_key=os.urandom(32)),
        cas=MemoryContentStore(),
        provider=provider,
        tools=build_tools(pool),
        config=config,
    )
