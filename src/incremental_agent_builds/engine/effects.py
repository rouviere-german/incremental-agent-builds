"""Approval gates for plan execution and effect sinks.

Build steps are read-only and memoized. Anything with a side effect is an *effect sink*: it is
never memoized, never runs inside a rule, and runs only after a successful build, through the
agentic kit's approval gate. The approval is bound to the canonical hash of the exact operation:

* ``execute_build_plan`` binds the plan hash (targets, dirty set, projected cost, store run), so
  a plan approved for one state of the inputs does not authorize a different, possibly costlier
  build;
* ``publish_docs`` binds the SHA-256 of the payload, so changing a single byte of any page after
  approval requires a new approval.

Approvals are single-use, expire, and cannot be granted by the requesting principal (separation
of duties, enforced by :class:`ApprovalStore`).
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, field_validator

from incremental_agent_builds.agentic.approval import ApprovalGate, ApprovalStore
from incremental_agent_builds.agentic.audit import AuditLog
from incremental_agent_builds.agentic.errors import ApprovalRequiredError
from incremental_agent_builds.agentic.providers.base import ToolUseBlock
from incremental_agent_builds.agentic.tools import (
    Capability,
    ToolDefinition,
    ToolRegistry,
    ValidatedCall,
)
from incremental_agent_builds.core.canonical import sha256_hex
from incremental_agent_builds.core.fileio import atomic_write_text
from incremental_agent_builds.engine.engine import BuildResult
from incremental_agent_builds.engine.errors import EffectError, WorkspaceError
from incremental_agent_builds.engine.hashing import value_hash
from incremental_agent_builds.engine.plan import Plan
from incremental_agent_builds.engine.workspace import validate_relative_path

__all__ = [
    "DocsPayload",
    "DocsSiteSink",
    "EffectOutcome",
    "GateDecision",
    "PlanApprovalArgs",
    "PublishArgs",
    "authorize_plan",
    "publish_docs",
]

_MAX_PAGES = 2_000
_MAX_PAGE_CHARS = 200_000


class PlanApprovalArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    targets: list[str]
    dirty_keys: int = Field(ge=0)
    projected_llm_steps: int = Field(ge=0)
    projected_tokens: int = Field(ge=0)
    projected_usd_micros: int = Field(ge=0)
    unestimated_keys: int = Field(ge=0)


class PublishArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    sink: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pages: int = Field(ge=0, le=_MAX_PAGES)


class DocsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    pages: dict[str, str] = Field(max_length=_MAX_PAGES)

    @field_validator("pages")
    @classmethod
    def _check_pages(cls, pages: dict[str, str]) -> dict[str, str]:
        for path, text in pages.items():
            try:
                validate_relative_path(path)
            except WorkspaceError as exc:
                raise ValueError(f"unsafe page path {path!r}") from exc
            if not path.endswith((".md", ".json")) or path == "manifest.json":
                raise ValueError(f"page {path!r} must be a .md or .json file")
            if len(text) > _MAX_PAGE_CHARS:
                raise ValueError(f"page {path!r} is too large")
        return pages


@dataclass(frozen=True, slots=True)
class GateDecision:
    allowed: bool
    call_hash: str
    approval_id: str | None


@dataclass(frozen=True, slots=True)
class EffectOutcome:
    status: Literal["applied", "approval_required"]
    payload_sha256: str
    approval_id: str | None = None
    written: tuple[str, ...] = ()

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "status": self.status,
            "payload_sha256": self.payload_sha256,
            "approval_id": self.approval_id,
            "written": list(self.written),
        }


class DocsSiteSink:
    """A fake docs site: a directory of pages plus a manifest naming the payload hash."""

    name = "docs_site"

    def __init__(self, root: Path) -> None:
        self.root = root

    def apply(self, payload: DocsPayload, payload_sha256: str) -> tuple[str, ...]:
        self.root.mkdir(parents=True, exist_ok=True)
        manifest_path = self.root / "manifest.json"
        previous: set[str] = set()
        if manifest_path.is_file():
            with contextlib.suppress(ValueError, OSError):
                old = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(old, dict) and isinstance(old.get("pages"), dict):
                    previous = {str(p) for p in old["pages"]}
        written: list[str] = []
        for path in sorted(payload.pages):
            atomic_write_text(self.root.joinpath(*path.split("/")), payload.pages[path])
            written.append(path)
        for stale in sorted(previous - payload.pages.keys()):
            with contextlib.suppress(WorkspaceError, OSError):
                validate_relative_path(stale)
                self.root.joinpath(*stale.split("/")).unlink(missing_ok=True)
        manifest = {
            "payload_sha256": payload_sha256,
            "pages": {path: sha256_hex(text) for path, text in sorted(payload.pages.items())},
        }
        atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return tuple(written)


def _gate(store: ApprovalStore, call: ValidatedCall, principal: str) -> GateDecision:
    try:
        call_hash = ApprovalGate(store, principal=principal).authorize(call)
    except ApprovalRequiredError as exc:
        return GateDecision(False, exc.call_hash, exc.request_id)
    return GateDecision(True, call_hash, None)


def _plan_registry() -> ToolRegistry:
    registry = ToolRegistry({Capability.EXECUTE_CODE})
    registry.register(
        ToolDefinition(
            name="execute_build_plan",
            description="Execute a build plan whose projected LLM cost needs approval.",
            args_model=PlanApprovalArgs,
            capabilities=frozenset({Capability.EXECUTE_CODE}),
            handler=lambda args: args.plan_sha256,
        )
    )
    return registry


def plan_call(plan: Plan) -> ValidatedCall:
    arguments: dict[str, JsonValue] = {
        "plan_sha256": plan.plan_hash,
        "targets": list(plan.targets),
        "dirty_keys": len(plan.items),
        "projected_llm_steps": plan.projected_llm_steps,
        "projected_tokens": plan.projected_tokens,
        "projected_usd_micros": plan.projected_usd_micros,
        "unestimated_keys": plan.unestimated_keys,
    }
    block = ToolUseBlock(id="plan", name="execute_build_plan", arguments=arguments)
    return _plan_registry().validate_call(block)


def authorize_plan(store: ApprovalStore, plan: Plan, *, principal: str) -> GateDecision:
    """Consume an approval of exactly this plan, or open a request for it."""
    return _gate(store, plan_call(plan), principal)


def publish_docs(
    *,
    build: BuildResult,
    payload: JsonValue,
    sink: DocsSiteSink,
    approvals: ApprovalStore,
    principal: str,
    audit: AuditLog | None = None,
) -> EffectOutcome:
    """Publish ``payload`` through ``sink`` after a successful build and an approval bound to it."""
    if not build.ok:
        raise EffectError("effects run only after a successful build")
    try:
        pages = DocsPayload.model_validate(payload)
    except ValidationError as exc:
        raise EffectError("payload is not a valid docs site") from exc
    digest = value_hash(payload)

    def apply(args: PublishArgs) -> JsonValue:
        if args.payload_sha256 != digest:  # pragma: no cover - bound by construction
            raise EffectError("payload changed after authorization")
        return list(sink.apply(pages, digest))

    registry = ToolRegistry({Capability.WRITE_DATA, Capability.NETWORK_EGRESS})
    registry.register(
        ToolDefinition(
            name="publish_docs",
            description="Publish the generated documentation site.",
            args_model=PublishArgs,
            capabilities=frozenset({Capability.WRITE_DATA, Capability.NETWORK_EGRESS}),
            handler=apply,
        )
    )
    arguments: dict[str, JsonValue] = {
        "sink": sink.name,
        "payload_sha256": digest,
        "pages": len(pages.pages),
    }
    call = registry.validate_call(
        ToolUseBlock(id="publish", name="publish_docs", arguments=arguments)
    )
    decision = _gate(approvals, call, principal)
    if not decision.allowed:
        if audit is not None:
            audit.append(
                principal,
                "approval_requested",
                {
                    "operation": "publish_docs",
                    "approval_id": decision.approval_id,
                    "payload": digest,
                },
            )
        return EffectOutcome("approval_required", digest, decision.approval_id)
    written = call.run(authorized_hash=decision.call_hash)
    paths = tuple(str(p) for p in written) if isinstance(written, list) else ()
    if audit is not None:
        audit.append(
            principal,
            "effect_applied",
            {
                "operation": "publish_docs",
                "payload": digest,
                "pages": len(paths),
                "run": build.run_id,
            },
        )
    return EffectOutcome("applied", digest, None, paths)
