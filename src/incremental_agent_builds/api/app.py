"""Wire a :class:`BuildService` into the hardened FastAPI app.

``/readyz`` reports the two dependencies the spec names: the trace store accepts a write
transaction, and the signing keys are loaded (at least one trusted verification key, and a signer
that is itself trusted when one is configured). Liveness (``/healthz``) stays dependency-free.

The caller owns the service's lifetime: ``app.state.build_service.close()`` releases the AST worker
processes and the SQLite connection. ``iab serve`` does that in a ``finally``.
"""

from __future__ import annotations

from fastapi import FastAPI
from prometheus_client import CollectorRegistry

from incremental_agent_builds.api.routes import build_router
from incremental_agent_builds.api.service import BuildService
from incremental_agent_builds.core.app import create_app
from incremental_agent_builds.core.config import Settings, get_settings
from incremental_agent_builds.engine.metrics import create_build_metrics

__all__ = ["create_service_app"]


def create_service_app(
    settings: Settings | None = None,
    *,
    service: BuildService | None = None,
    registry: CollectorRegistry | None = None,
) -> FastAPI:
    """The build API. Pass ``service`` to reuse an existing store (tests, embedded use)."""
    cfg = settings if settings is not None else get_settings()
    shared = registry if registry is not None else CollectorRegistry()
    build_service = service or BuildService.from_settings(cfg, metrics=create_build_metrics(shared))
    app = create_app(
        cfg,
        readiness_checks=build_service.readiness_checks(),
        registry=shared,
        routers=[build_router()],
    )
    app.state.build_service = build_service
    return app
