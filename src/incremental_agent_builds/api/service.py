"""BuildService: one workspace, one trace store, and the operations the API and CLI share.

The service owns the long-lived pieces (SQLite trace store, CAS, AST worker pool, approval store,
audit log, signing keys) and exposes the operations both front ends need:

* ``plan`` — dry-run dirty set and projected cost, no rule executed;
* ``run`` — execute a plan. Above the configured cost threshold, or when the plan could not be
  estimated, execution needs an approval bound to the plan hash; the build is then bounded by a
  budget derived from that plan, so an approved estimate cannot be overspent;
* ``publish`` — the effect sink, never memoized, only after a successful build, with an approval
  bound to the payload hash;
* ``export_cache`` / ``import_cache`` — signed constructive traces;
* ``explain`` — why one key would re-run.

Every operation appends to the hash-chained audit log. The store MAC key is read from settings or
generated once and persisted with owner-only permissions under ``data_dir``.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from incremental_agent_builds.agentic.approval import ApprovalRecord, ApprovalStore
from incremental_agent_builds.agentic.audit import AuditLog
from incremental_agent_builds.agentic.budget import BudgetLimits
from incremental_agent_builds.agentic.providers.base import LLMProvider
from incremental_agent_builds.core.config import Settings
from incremental_agent_builds.core.logging import get_logger
from incremental_agent_builds.engine.ast_tool import AstWorkerPool
from incremental_agent_builds.engine.cas import FileContentStore
from incremental_agent_builds.engine.effects import (
    DocsSiteSink,
    EffectOutcome,
    authorize_plan,
    publish_docs,
)
from incremental_agent_builds.engine.engine import BuildEngine, BuildResult, EngineConfig
from incremental_agent_builds.engine.errors import BuildEngineError, EffectError
from incremental_agent_builds.engine.keys import BuildKey
from incremental_agent_builds.engine.metrics import BuildMetrics, create_build_metrics
from incremental_agent_builds.engine.plan import Plan, explain, make_plan
from incremental_agent_builds.engine.shared_cache import ImportReport, export_bundle, import_bundle
from incremental_agent_builds.engine.signing import SigningKey, TrustStore
from incremental_agent_builds.engine.store import TraceStore
from incremental_agent_builds.engine.workspace import FileWorkspace
from incremental_agent_builds.workload.factory import build_tools
from incremental_agent_builds.workload.fake_llm import make_fake_provider
from incremental_agent_builds.workload.rules import DocsSite, workload_rules

__all__ = [
    "MAC_KEY_FILENAME",
    "BuildService",
    "PlanDecision",
    "TargetSpec",
    "bundle_bytes",
    "resolve_mac_key",
]

MAC_KEY_FILENAME = "store_mac.key"
# A plan may cost this multiple of its estimate before the budget stops the build. Recorded usage
# is a good predictor for deterministic prompts but not an exact one, so the factor is > 1.
_PLAN_BUDGET_FACTOR = 1.5
_MAX_TARGETS = 64
_log = get_logger("incremental_agent_builds.api.service")


class TargetSpec(BaseModel):
    """A build target named by its rule kind plus that key type's arguments."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    args: dict[str, JsonValue] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PlanDecision:
    """Outcome of ``run``: either the build ran, or an approval of this plan is required."""

    status: str
    plan: Plan
    result: BuildResult | None = None
    approval_id: str | None = None

    def to_json(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "status": self.status,
            "plan": self.plan.to_json(),
            "approval_id": self.approval_id,
        }
        if self.result is not None:
            payload["build"] = self.result.to_json()
        return payload


def resolve_mac_key(settings: Settings) -> bytes:
    """The trace-store MAC key: from settings, else generated once and persisted under data_dir.

    The key authenticates rows this store wrote. Losing it is harmless (every trace is re-verified
    against the current inputs, so unverifiable rows only cause extra work), which is why
    generating one on first start is safe.
    """
    configured = settings.store_mac_key
    if configured is not None and configured.get_secret_value():
        raw = base64.b64decode(configured.get_secret_value(), validate=True)
        if len(raw) < 32:
            raise ValueError("APP_STORE_MAC_KEY must decode to at least 32 bytes")
        return raw
    path = settings.data_dir / MAC_KEY_FILENAME
    if path.is_file():
        stored = base64.b64decode(path.read_text(encoding="ascii").strip(), validate=True)
        if len(stored) >= 32:
            return stored
    path.parent.mkdir(parents=True, exist_ok=True)
    generated = secrets.token_bytes(32)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        handle.write(base64.b64encode(generated).decode("ascii") + "\n")
    return generated


class BuildService:
    """Everything one workspace needs.

    Deliberately not a context manager: the HTTP app outlives a request, so the caller owns the
    lifetime and calls ``close`` to release the AST workers and the SQLite connection.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        provider: LLMProvider,
        workspace_dir: Path,
        store_dir: Path,
        data_dir: Path,
        mac_key: bytes,
        metrics: BuildMetrics | None = None,
        pool: AstWorkerPool | None = None,
    ) -> None:
        self.settings = settings
        self.metrics = metrics or create_build_metrics()
        self._own_pool = pool is None
        self.pool = pool or AstWorkerPool(size=max(2, min(4, settings.build_jobs)))
        self.signer = (
            SigningKey.from_seed_b64(settings.cache_signing_seed.get_secret_value())
            if settings.cache_signing_seed is not None
            and settings.cache_signing_seed.get_secret_value()
            else None
        )
        trusted = list(settings.cache_trusted_keys)
        if self.signer is not None and self.signer.public_b64 not in trusted:
            trusted.append(self.signer.public_b64)  # a store always trusts what it signed itself
        self.trust = TrustStore(trusted)
        self.store = TraceStore(store_dir / "traces.sqlite", mac_key=mac_key)
        self.cas = FileContentStore(store_dir / "cas")
        self.audit = AuditLog(data_dir / "audit.jsonl")
        self.approvals = ApprovalStore(
            data_dir / "approvals.json", ttl_seconds=settings.approval_ttl_seconds
        )
        self.sink = DocsSiteSink(data_dir / "site")
        self.engine = BuildEngine(
            rules=workload_rules(),
            workspace=FileWorkspace(workspace_dir),
            store=self.store,
            cas=self.cas,
            provider=provider,
            tools=build_tools(self.pool),
            config=EngineConfig(
                jobs=settings.build_jobs,
                llm_concurrency=settings.build_llm_concurrency,
                guard=settings.build_guard_mode,
            ),
            audit=self.audit,
            metrics=self.metrics,
            trust=self.trust if len(self.trust) else None,
            default_budget=BudgetLimits.of(
                steps=settings.budget_max_steps,
                tokens=settings.budget_max_tokens,
                usd=settings.budget_max_usd,
            ),
        )

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        provider: LLMProvider | None = None,
        metrics: BuildMetrics | None = None,
    ) -> Self:
        """Wire a service from settings. Offline (no API key) it uses the deterministic provider."""
        settings.workspace_dir.mkdir(parents=True, exist_ok=True)
        settings.store_dir.mkdir(parents=True, exist_ok=True)
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        chosen = provider if provider is not None else cls._default_provider(settings)
        return cls(
            settings=settings,
            provider=chosen,
            workspace_dir=settings.workspace_dir,
            store_dir=settings.store_dir,
            data_dir=settings.data_dir,
            mac_key=resolve_mac_key(settings),
            metrics=metrics,
        )

    @staticmethod
    def _default_provider(settings: Settings) -> LLMProvider:
        key = settings.anthropic_api_key
        if key is None or not key.get_secret_value():
            _log.info("provider_offline", reason="no_api_key")
            return make_fake_provider()
        from incremental_agent_builds.agentic.providers.anthropic import (  # noqa: PLC0415
            AnthropicProvider,
        )

        return AnthropicProvider.from_settings(settings)

    def close(self) -> None:
        if self._own_pool:
            self.pool.close()
        self.store.close()

    # --- targets ------------------------------------------------------------------------------

    def key_for(self, spec: TargetSpec) -> BuildKey:
        rule = self.engine.rules.get(spec.kind)
        if rule is None:
            raise BuildEngineError(f"no rule defines the key kind {spec.kind!r}")
        try:
            key: BuildKey = rule.key_type.model_validate(spec.args)
        except ValueError as exc:
            raise BuildEngineError(f"invalid arguments for {spec.kind}") from exc
        return key

    def keys_for(self, specs: Sequence[TargetSpec] | None) -> list[BuildKey]:
        if not specs:
            return [DocsSite()]
        if len(specs) > _MAX_TARGETS:
            raise BuildEngineError(f"at most {_MAX_TARGETS} targets per request")
        return [self.key_for(spec) for spec in specs]

    # --- operations ---------------------------------------------------------------------------

    def plan(
        self, targets: Sequence[TargetSpec] | None = None, *, force: Sequence[str] = ()
    ) -> Plan:
        return make_plan(self.engine, self.keys_for(targets), force=force)

    def explain(self, spec: TargetSpec) -> list[str]:
        return explain(self.engine, self.key_for(spec))

    def approval_required(self, plan: Plan) -> bool:
        return plan.approval_required(
            threshold_usd_micros=self.settings.plan_approval_usd_micros,
            unestimated_needs_approval=self.settings.plan_approve_unestimated,
        )

    def run(
        self,
        targets: Sequence[TargetSpec] | None = None,
        *,
        force: Sequence[str] = (),
        principal: str = "local",
    ) -> PlanDecision:
        """Plan, gate on cost, then build within a budget derived from the approved plan."""
        keys = self.keys_for(targets)
        plan = make_plan(self.engine, keys, force=force)
        ceiling = self.engine.default_budget
        budget = plan.budget(factor=_PLAN_BUDGET_FACTOR, ceiling=ceiling)
        if self.approval_required(plan):
            decision = authorize_plan(self.approvals, plan, principal=principal)
            if not decision.allowed:
                self.audit.append(
                    principal,
                    "plan_approval_requested",
                    {
                        "plan": plan.plan_hash,
                        "approval_id": decision.approval_id,
                        "projected_usd_micros": plan.projected_usd_micros,
                        "dirty_keys": len(plan.items),
                    },
                )
                return PlanDecision("approval_required", plan, None, decision.approval_id)
            self.audit.append(
                principal, "plan_approved", {"plan": plan.plan_hash, "hash": decision.call_hash}
            )
        # The build runs under the budget the plan implies, never the full per-build ceiling: an
        # estimate a human approved is also the limit on what may be spent against it.
        result = self.engine.build_sync(keys, force=force, budget=budget, principal=principal)
        return PlanDecision("built" if result.ok else "failed", plan, result)

    def latest_site_payload(self) -> JsonValue:
        """The recorded DocsSite value: the payload a publish would apply."""
        return self.engine.value(DocsSite())

    def publish(self, *, result: BuildResult, principal: str) -> EffectOutcome:
        payload = self.latest_site_payload()
        return publish_docs(
            build=result,
            payload=payload,
            sink=self.sink,
            approvals=self.approvals,
            principal=principal,
            audit=self.audit,
        )

    def build_and_publish(self, *, principal: str) -> tuple[PlanDecision, EffectOutcome | None]:
        """Build ``DocsSite`` and, if that succeeded, attempt the approval-gated publish."""
        decision = self.run(principal=principal)
        if decision.result is None or not decision.result.ok:
            return decision, None
        try:
            return decision, self.publish(result=decision.result, principal=principal)
        except EffectError:
            return decision, None

    def export_cache(self, *, key_ids: Sequence[str] | None = None) -> dict[str, JsonValue]:
        if self.signer is None:
            raise BuildEngineError("no signing key is configured (set APP_CACHE_SIGNING_SEED)")
        bundle = export_bundle(self.store, self.cas, self.signer, key_ids=key_ids)
        self.audit.append(
            "local",
            "cache_exported",
            {"entries": len(bundle["entries"]) if isinstance(bundle["entries"], list) else 0},
        )
        return bundle

    def import_cache(self, raw: bytes, *, principal: str) -> ImportReport:
        report = import_bundle(
            raw,
            store=self.store,
            cas=self.cas,
            trust=self.trust,
            metrics=self.metrics,
            clock_now=self.engine.clock(),
        )
        self.audit.append(principal, "cache_imported", report.to_json())
        return report

    def pending_approvals(self) -> list[ApprovalRecord]:
        return self.approvals.pending()

    def decide(
        self, request_id: str, *, approver: str, approve: bool, reason: str | None = None
    ) -> ApprovalRecord:
        record = self.approvals.decide(
            request_id, approver=approver, approve=approve, reason=reason
        )
        self.audit.append(
            approver,
            "approval_decided",
            {
                "approval_id": record.id,
                "tool": record.tool,
                "status": record.status.value,
                "call_hash": record.call_hash,
            },
        )
        return record

    # --- health -------------------------------------------------------------------------------

    def store_writable(self) -> bool:
        return self.store.check_writable()

    def signing_keys_loaded(self) -> bool:
        """Readiness: at least one trusted verification key, and a usable signer if one is set."""
        if len(self.trust) == 0:
            return False
        return self.signer is None or self.trust.trusts(self.signer.key_id)

    def readiness_checks(self) -> dict[str, Any]:
        return {"store_writable": self.store_writable, "signing_keys": self.signing_keys_loaded}

    def audit_head(self) -> str:
        return self.audit.head

    def stats_json(self) -> dict[str, JsonValue]:
        return {
            "last_run": self.store.last_run_id(),
            "traces": len(self.store.load_traces()),
            "constructive_entries": self.store.constructive_count(),
            "imported_steps": self.store.imported_step_count(),
            "trusted_signers": list(self.trust.key_ids),
            "audit_head": self.audit.head,
        }


def bundle_bytes(bundle: dict[str, JsonValue]) -> bytes:
    """Serialize an exported bundle exactly as ``import_cache`` expects to read it."""
    return json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode("utf-8")
