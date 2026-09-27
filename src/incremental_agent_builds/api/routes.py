"""HTTP routes of the build service, each behind one scope.

``POST /v1/plan``            build:run      dry-run dirty set and projected cost
``GET  /v1/explain``         build:run      why one key would re-run
``POST /v1/builds``          build:run      execute (202 + approval id when the plan needs one)
``GET  /v1/store``           build:run      store summary (traces, imported entries, audit head)
``GET  /v1/cache/export``    build:run      signed bundle of locally computed traces
``POST /v1/cache/import``    cache:import   verify and import a bundle
``GET  /v1/approvals``       build:approve  open requests
``POST /v1/approvals/{id}``  build:approve  decide one request
``POST /v1/publish``         build:run      the approval-gated effect sink

The request models forbid unknown fields and bound every list, so a malformed or oversized request
is rejected by validation before it reaches the engine. Engine errors become 4xx with fixed
messages; nothing derived from workspace content is echoed back.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, Body, Path, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from incremental_agent_builds.api.auth import BuildApprover, BuildRunner, CacheImporter
from incremental_agent_builds.api.service import BuildService, TargetSpec
from incremental_agent_builds.core.errors import AppError
from incremental_agent_builds.core.logging import get_logger
from incremental_agent_builds.engine.errors import (
    BuildEngineError,
    CacheImportError,
    EffectError,
)

__all__ = ["build_router"]

_MAX_FORCE = 256
_MAX_BUNDLE_CHARS = 16 * 1024 * 1024
_log = get_logger("incremental_agent_builds.api.routes")


async def _offload[T](work: Callable[[], T]) -> T:
    """Run a blocking service call in a worker thread.

    Every service operation is synchronous and blocking: it hashes files, talks to SQLite and, for a
    build, drives its own event loop. Calling that from an async handler would block the server (and
    ``asyncio.run`` inside a running loop is an error), so it runs in a thread. Builds on one engine
    are serialized by the engine's own lock, so concurrent build requests queue rather than race.
    """
    return await asyncio.to_thread(work)


class BuildRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    targets: list[TargetSpec] = Field(default_factory=list, max_length=64)
    force: list[str] = Field(default_factory=list, max_length=_MAX_FORCE)


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approve: bool
    reason: str | None = Field(default=None, max_length=500)


def _service(request: Request) -> BuildService:
    service = getattr(request.app.state, "build_service", None)
    if not isinstance(service, BuildService):  # pragma: no cover - wired by create_service_app
        raise AppError(503, "unavailable", "The build service is not available.")
    return service


def _engine_error(exc: BuildEngineError) -> AppError:
    _log.info("request_rejected", error_type=type(exc).__name__)
    return AppError(400, "bad_request", "The request could not be applied to this workspace.")


def build_router() -> APIRouter:  # noqa: PLR0915 - one small handler per route, kept together
    """The scoped routes. Scopes come from the annotated aliases in :mod:`.auth`."""
    router = APIRouter(prefix="/v1")

    @router.post("/plan")
    async def plan(
        request: Request,
        body: Annotated[BuildRequest, Body()],
        principal: BuildRunner,
    ) -> dict[str, JsonValue]:
        """Dry run: which keys may re-run and what that may cost. No rule is executed."""
        service = _service(request)
        try:
            computed = await _offload(lambda: service.plan(body.targets, force=body.force))
        except BuildEngineError as exc:
            raise _engine_error(exc) from exc
        payload = computed.to_json()
        payload["approval_required"] = service.approval_required(computed)
        _log.info("plan_served", subject=principal.subject, dirty=len(computed.items))
        return payload

    @router.get("/explain")
    async def explain(
        request: Request,
        principal: BuildRunner,
        kind: Annotated[str, Query(pattern=r"^[a-z][a-z0-9_]{0,63}$")],
        path: Annotated[str | None, Query(max_length=512)] = None,
    ) -> dict[str, JsonValue]:
        """The chain of reasons a key would re-run, down to the input that changed."""
        service = _service(request)
        args: dict[str, JsonValue] = {"path": path} if path is not None else {}
        try:
            chain = await _offload(lambda: service.explain(TargetSpec(kind=kind, args=args)))
        except BuildEngineError as exc:
            raise _engine_error(exc) from exc
        del principal
        return {"kind": kind, "chain": list(chain)}

    @router.post("/builds")
    async def start_build(
        request: Request,
        body: Annotated[BuildRequest, Body()],
        response: Response,
        principal: BuildRunner,
    ) -> dict[str, JsonValue]:
        """Execute a build. A plan above the cost threshold needs an approval first (202)."""
        service = _service(request)
        try:
            decision = await _offload(
                lambda: service.run(body.targets, force=body.force, principal=principal.subject)
            )
        except BuildEngineError as exc:
            raise _engine_error(exc) from exc
        if decision.status == "approval_required":
            response.status_code = 202
        elif decision.status == "failed":
            response.status_code = 409
        _log.info("build_finished", subject=principal.subject, status=decision.status)
        return decision.to_json()

    @router.get("/store")
    async def store_summary(request: Request, principal: BuildRunner) -> dict[str, JsonValue]:
        """Trace, cache and audit counters of this store."""
        del principal
        service = _service(request)
        return await _offload(service.stats_json)

    @router.get("/cache/export")
    async def cache_export(request: Request, principal: BuildRunner) -> dict[str, JsonValue]:
        """A signed bundle of the traces this store computed itself."""
        service = _service(request)
        try:
            bundle = await _offload(service.export_cache)
        except BuildEngineError as exc:
            raise AppError(409, "no_signing_key", "This store has no cache signing key.") from exc
        _log.info("cache_exported", subject=principal.subject)
        return bundle

    @router.post("/cache/import")
    async def cache_import(
        request: Request,
        principal: CacheImporter,
        bundle: Annotated[dict[str, JsonValue], Body()],
    ) -> dict[str, JsonValue]:
        """Import a bundle. Unsigned, untrusted or tampered entries are rejected and counted."""
        service = _service(request)
        raw = json.dumps(bundle, separators=(",", ":")).encode("utf-8")
        if len(raw) > _MAX_BUNDLE_CHARS:
            raise AppError(413, "payload_too_large", "The cache bundle is too large.")
        try:
            report = await _offload(lambda: service.import_cache(raw, principal=principal.subject))
        except CacheImportError as exc:
            raise AppError(400, "bad_bundle", "The bundle is not a valid cache bundle.") from exc
        _log.info(
            "cache_imported",
            subject=principal.subject,
            accepted=report.accepted_traces,
            rejected=report.rejected_total,
        )
        return report.to_json()

    @router.get("/approvals")
    async def approvals(request: Request, principal: BuildApprover) -> dict[str, JsonValue]:
        """Open approval requests, with the exact arguments each one would release."""
        del principal
        service = _service(request)
        records = await _offload(service.pending_approvals)
        pending: list[JsonValue] = [
            {
                "id": record.id,
                "tool": record.tool,
                "call_hash": record.call_hash,
                "requested_by": record.requested_by,
                "args": dict(record.args),
            }
            for record in records
        ]
        return {"pending": pending}

    @router.post("/approvals/{request_id}")
    async def decide(
        request: Request,
        request_id: Annotated[str, Path(max_length=64)],
        body: Annotated[DecisionRequest, Body()],
        principal: BuildApprover,
    ) -> dict[str, JsonValue]:
        """Approve or deny one request. The requester cannot decide its own request."""
        from incremental_agent_builds.agentic.errors import ApprovalError  # noqa: PLC0415

        service = _service(request)
        try:
            record = await _offload(
                lambda: service.decide(
                    request_id, approver=principal.subject, approve=body.approve, reason=body.reason
                )
            )
        except ApprovalError as exc:
            _log.info("approval_rejected", error_type=type(exc).__name__)
            raise AppError(409, "approval_conflict", "The request cannot be decided.") from exc
        return {"id": record.id, "status": record.status.value, "call_hash": record.call_hash}

    @router.post("/publish")
    async def publish(
        request: Request,
        response: Response,
        principal: BuildRunner,
    ) -> dict[str, JsonValue]:
        """Build, then publish the docs site through the approval-gated effect sink."""
        service = _service(request)
        try:
            decision, outcome = await _offload(
                lambda: service.build_and_publish(principal=principal.subject)
            )
        except (BuildEngineError, EffectError, KeyError) as exc:
            raise AppError(409, "publish_failed", "The site could not be published.") from exc
        if outcome is None:
            response.status_code = 409
            return {"status": decision.status, "plan": decision.plan.to_json()}
        if outcome.status == "approval_required":
            response.status_code = 202
        return {"status": outcome.status, **outcome.to_json()}

    return router
