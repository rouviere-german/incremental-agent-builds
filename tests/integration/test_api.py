"""The build API end to end: scopes, the cost gate, the approval flow and cache import.

Every route is exercised with a token that has the right scope, one that has a different scope, and
none at all, because "least privilege" is only a property if the wrong token is actually refused.
The whole test module runs offline against a generated workspace and the deterministic provider.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from incremental_agent_builds.api.app import create_service_app
from incremental_agent_builds.api.auth import Scope, issue_token
from incremental_agent_builds.api.service import BuildService, bundle_bytes
from incremental_agent_builds.core.config import Settings
from incremental_agent_builds.engine.signing import SigningKey
from incremental_agent_builds.workload.generator import generate_state, write_workspace

pytestmark = pytest.mark.integration

_SECRET = "integration-test-jwt-secret-long-enough"
_SMALL_MODULES = 4


def _workspace(root: Path) -> None:
    """A four-module workspace, so a cold build is a second rather than a minute."""
    state = generate_state()
    state.model.modules = [m for m in state.model.modules if m.sub == "core"][:_SMALL_MODULES]
    state.model.subpackages = ["core"]
    root.mkdir(parents=True, exist_ok=True)
    write_workspace(root, state)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    _workspace(tmp_path / "workspace")
    return Settings(
        jwt_secret=_SECRET,
        workspace_dir=tmp_path / "workspace",
        store_dir=tmp_path / "store",
        data_dir=tmp_path / "data",
        cache_signing_seed=SigningKey.generate().seed_b64(),
        build_jobs=2,
        build_llm_concurrency=2,
        rate_limit_per_second=1000.0,
        rate_limit_burst=1000,
        # Most tests are not about the gate: price it out of the way and allow unpriceable plans.
        plan_approval_usd_micros=10**9,
        plan_approve_unestimated=False,
    )


@pytest.fixture
def service(settings: Settings) -> Iterator[BuildService]:
    built = BuildService.from_settings(settings)
    try:
        yield built
    finally:
        built.close()


@pytest.fixture
def client(settings: Settings, service: BuildService) -> Iterator[TestClient]:
    app = create_service_app(settings, service=service)
    with TestClient(app) as test_client:
        yield test_client


def _headers(*scopes: Scope, subject: str = "ci") -> dict[str, str]:
    token = issue_token(
        _SECRET,
        subject=subject,
        scopes=scopes,
        issuer="incremental-agent-builds",
        audience="incremental-agent-builds",
    )
    return {"Authorization": f"Bearer {token}"}


RUN = (Scope.BUILD_RUN,)
APPROVE = (Scope.BUILD_APPROVE,)
IMPORT = (Scope.CACHE_IMPORT,)


# --- health -------------------------------------------------------------------------------------


def test_healthz_is_dependency_free(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_reports_the_store_and_the_signing_keys(client: TestClient) -> None:
    body = client.get("/readyz").json()
    assert body["status"] == "ready"
    assert body["checks"] == {"store_writable": "ok", "signing_keys": "ok"}


def test_readyz_is_not_ready_without_signing_keys(settings: Settings) -> None:
    unsigned = settings.model_copy(update={"cache_signing_seed": None, "cache_trusted_keys": []})
    service = BuildService.from_settings(unsigned)
    try:
        with TestClient(create_service_app(unsigned, service=service)) as client:
            response = client.get("/readyz")
            assert response.status_code == 503
            assert response.json()["checks"]["signing_keys"] == "fail"
    finally:
        service.close()


# --- scopes -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path", "scopes", "body"),
    [
        ("post", "/v1/plan", RUN, {}),
        ("get", "/v1/explain?kind=docs_site", RUN, None),
        ("post", "/v1/builds", RUN, {}),
        ("get", "/v1/store", RUN, None),
        ("get", "/v1/cache/export", RUN, None),
        (
            "post",
            "/v1/cache/import",
            IMPORT,
            {"format": "iab-cache-bundle", "version": 1, "entries": []},
        ),
        ("get", "/v1/approvals", APPROVE, None),
        ("post", "/v1/publish", RUN, {}),
    ],
)
def test_a_route_needs_its_own_scope(
    client: TestClient, method: str, path: str, scopes: tuple[Scope, ...], body: Any
) -> None:
    call = getattr(client, method)
    kwargs: dict[str, Any] = {"json": body} if body is not None else {}

    anonymous = call(path, **kwargs)
    assert anonymous.status_code == 401, path
    assert anonymous.json()["error"]["code"] == "unauthorized"

    wrong = next(s for s in Scope if s not in scopes)
    denied = call(path, headers=_headers(wrong), **kwargs)
    assert denied.status_code == 403, path
    assert denied.json()["error"]["code"] == "forbidden"

    allowed = call(path, headers=_headers(*scopes), **kwargs)
    assert allowed.status_code < 500, path
    assert allowed.status_code != 403, path


def test_the_routes_fail_closed_without_a_configured_secret(settings: Settings) -> None:
    """A deployment that forgot the secret must refuse builds, not serve them unauthenticated."""
    unconfigured = settings.model_copy(update={"jwt_secret": None})
    service = BuildService.from_settings(unconfigured)
    try:
        with TestClient(create_service_app(unconfigured, service=service)) as client:
            response = client.post("/v1/plan", json={}, headers=_headers(Scope.BUILD_RUN))
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "auth_not_configured"
    finally:
        service.close()


# --- planning and building ----------------------------------------------------------------------


def test_a_plan_lists_the_dirty_set_without_building(client: TestClient) -> None:
    body = client.post("/v1/plan", json={}, headers=_headers(*RUN)).json()
    assert body["dirty_count"] > 0
    assert body["clean_count"] == 0
    assert len(body["plan_sha256"]) == 64
    assert body["unestimated_keys"] > 0, "a first build cannot be priced from history"
    assert body["approval_required"] is False, "this deployment allows unpriceable plans"
    assert client.get("/v1/store", headers=_headers(*RUN)).json()["traces"] == 0


def test_a_build_runs_and_a_second_build_reuses_everything(client: TestClient) -> None:
    first = client.post("/v1/builds", json={}, headers=_headers(*RUN))
    assert first.status_code == 200
    assert first.json()["status"] == "built"
    stats = first.json()["build"]["stats"]
    assert stats["executed"] > 0
    assert stats["provider_calls"] > 0

    second = client.post("/v1/builds", json={}, headers=_headers(*RUN)).json()
    assert second["build"]["stats"]["executed"] == 0
    assert second["build"]["stats"]["provider_calls"] == 0
    assert second["build"]["stats"]["reused"] == stats["executed"]


def test_a_build_can_target_one_key(client: TestClient) -> None:
    body = client.post(
        "/v1/builds", json={"targets": [{"kind": "module_list"}]}, headers=_headers(*RUN)
    ).json()
    assert body["status"] == "built"
    assert body["build"]["targets"] == ["module_list:{}"]


def test_an_unknown_target_is_a_client_error(client: TestClient) -> None:
    response = client.post(
        "/v1/builds", json={"targets": [{"kind": "not_a_rule"}]}, headers=_headers(*RUN)
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "bad_request"


def test_invalid_target_arguments_are_a_client_error(client: TestClient) -> None:
    response = client.post(
        "/v1/builds",
        json={"targets": [{"kind": "module_outline", "args": {"wrong": 1}}]},
        headers=_headers(*RUN),
    )
    assert response.status_code == 400


def test_too_many_targets_are_refused_by_validation(client: TestClient) -> None:
    response = client.post(
        "/v1/builds", json={"targets": [{"kind": "module_list"}] * 65}, headers=_headers(*RUN)
    )
    assert response.status_code == 422


def test_explain_reports_why_a_key_is_up_to_date_or_dirty(client: TestClient) -> None:
    client.post("/v1/builds", json={}, headers=_headers(*RUN))
    body = client.get("/v1/explain?kind=docs_site", headers=_headers(*RUN)).json()
    assert body["chain"] == ["docs_site:{}: up to date"]


def test_explain_refuses_an_unknown_kind(client: TestClient) -> None:
    assert client.get("/v1/explain?kind=nope", headers=_headers(*RUN)).status_code == 400
    assert client.get("/v1/explain?kind=NotValid", headers=_headers(*RUN)).status_code == 422


# --- the cost gate ------------------------------------------------------------------------------


def test_a_plan_that_cannot_be_priced_needs_an_approval(settings: Settings) -> None:
    """A cold build has no recorded usage to estimate from, so the conservative default gates it."""
    gated = settings.model_copy(update={"plan_approve_unestimated": True})
    service = BuildService.from_settings(gated)
    try:
        with TestClient(create_service_app(gated, service=service)) as client:
            plan = client.post("/v1/plan", json={}, headers=_headers(*RUN)).json()
            assert plan["unestimated_keys"] > 0
            assert plan["approval_required"] is True
            assert client.post("/v1/builds", json={}, headers=_headers(*RUN)).status_code == 202
    finally:
        service.close()


def test_a_costly_plan_needs_an_approval_before_it_runs(
    settings: Settings, client: TestClient
) -> None:
    """Once there is recorded usage a plan can be priced, and a low threshold gates it."""
    client.post("/v1/builds", json={}, headers=_headers(*RUN))  # record usage to estimate from
    module = next(
        p for p in (settings.workspace_dir / "pkg" / "core").glob("*.py") if p.name != "__init__.py"
    )
    added = '\n\ndef added(path: str) -> bool:\n    """Added by this test."""\n    return True\n'
    module.write_text(module.read_text(encoding="utf-8") + added, encoding="utf-8")

    gated = settings.model_copy(update={"plan_approval_usd_micros": 0})
    service = BuildService.from_settings(gated)
    try:
        with TestClient(create_service_app(gated, service=service)) as gated_client:
            plan = gated_client.post("/v1/plan", json={}, headers=_headers(*RUN)).json()
            assert plan["projected_usd_micros"] > 0
            assert plan["unestimated_keys"] == 0
            assert plan["approval_required"] is True
            opened = gated_client.post("/v1/builds", json={}, headers=_headers(*RUN))
            assert opened.status_code == 202
            body = opened.json()
            assert body["status"] == "approval_required"
            approval_id = body["approval_id"]
            assert approval_id
            assert "build" not in body

            pending = gated_client.get("/v1/approvals", headers=_headers(*APPROVE)).json()[
                "pending"
            ]
            plan_requests = [item for item in pending if item["tool"] == "execute_build_plan"]
            assert [item["id"] for item in plan_requests] == [approval_id]
            assert plan_requests[0]["args"]["plan_sha256"] == body["plan"]["plan_sha256"]

            decided = gated_client.post(
                f"/v1/approvals/{approval_id}",
                json={"approve": True},
                headers=_headers(*APPROVE, subject="reviewer"),
            )
            assert decided.status_code == 200
            assert decided.json()["status"] == "approved"

            ran = gated_client.post("/v1/builds", json={}, headers=_headers(*RUN))
            assert ran.status_code == 200
            assert ran.json()["status"] == "built"
    finally:
        service.close()


def test_a_requester_cannot_approve_its_own_plan_over_http(settings: Settings) -> None:
    gated = settings.model_copy(update={"plan_approve_unestimated": True})
    service = BuildService.from_settings(gated)
    try:
        with TestClient(create_service_app(gated, service=service)) as client:
            opened = client.post("/v1/builds", json={}, headers=_headers(*RUN, subject="ci")).json()
            response = client.post(
                f"/v1/approvals/{opened['approval_id']}",
                json={"approve": True},
                headers=_headers(*APPROVE, subject="ci"),
            )
            assert response.status_code == 409
            assert response.json()["error"]["code"] == "approval_conflict"
    finally:
        service.close()


def test_a_build_runs_under_the_budget_its_plan_implies(settings: Settings) -> None:
    """An approved estimate is also the ceiling, not just a prediction.

    The build must run under the plan's budget, never the full per-build ceiling.
    """
    from incremental_agent_builds.agentic.budget import BudgetLimits

    service = BuildService.from_settings(settings)
    captured: list[BudgetLimits | None] = []
    original = service.engine.build_sync

    def spy(*args: Any, **kwargs: Any) -> Any:
        captured.append(kwargs.get("budget"))
        return original(*args, **kwargs)

    service.engine.build_sync = spy  # type: ignore[method-assign]
    try:
        service.run()  # cold build: unpriceable, so it gets the ceiling
        assert captured[-1] == service.engine.default_budget
        module = next(
            p
            for p in (settings.workspace_dir / "pkg" / "core").glob("*.py")
            if p.name != "__init__.py"
        )
        extra = '\n\ndef added(path: str) -> bool:\n    """Added."""\n    return True\n'
        module.write_text(module.read_text(encoding="utf-8") + extra, encoding="utf-8")
        service.run()  # now there is recorded usage, so the plan can price it
        bounded = captured[-1]
        assert bounded is not None
        assert bounded.not_looser_than(service.engine.default_budget)
        assert bounded.max_tokens < service.engine.default_budget.max_tokens
    finally:
        service.close()


def test_deciding_an_unknown_approval_is_a_conflict(client: TestClient) -> None:
    response = client.post(
        "/v1/approvals/does-not-exist", json={"approve": True}, headers=_headers(*APPROVE)
    )
    assert response.status_code == 409


# --- the shared cache ---------------------------------------------------------------------------


def test_an_exported_bundle_can_be_imported_and_a_forged_one_cannot(client: TestClient) -> None:
    client.post("/v1/builds", json={}, headers=_headers(*RUN))
    bundle = client.get("/v1/cache/export", headers=_headers(*RUN)).json()
    assert bundle["format"] == "iab-cache-bundle"
    assert bundle["entries"]

    accepted = client.post("/v1/cache/import", json=bundle, headers=_headers(*IMPORT)).json()
    assert accepted["rejected_total"] == 0

    forged = dict(bundle)
    entries = [dict(entry) for entry in bundle["entries"]]
    for entry in entries:
        if entry["type"] == "trace":
            body = dict(entry["body"])
            body["result_hash"] = "f" * 64
            entry["body"] = body
            break
    forged["entries"] = entries
    rejected = client.post("/v1/cache/import", json=forged, headers=_headers(*IMPORT)).json()
    assert rejected["rejected_total"] > 0


def test_a_bundle_that_is_not_a_bundle_is_a_client_error(client: TestClient) -> None:
    response = client.post("/v1/cache/import", json={"nope": True}, headers=_headers(*IMPORT))
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "bad_bundle"


def test_exporting_without_a_signing_key_is_a_conflict(settings: Settings) -> None:
    unsigned = settings.model_copy(
        update={
            "cache_signing_seed": None,
            "cache_trusted_keys": [SigningKey.generate().public_b64],
        }
    )
    service = BuildService.from_settings(unsigned)
    try:
        with TestClient(create_service_app(unsigned, service=service)) as client:
            response = client.get("/v1/cache/export", headers=_headers(*RUN))
            assert response.status_code == 409
            assert response.json()["error"]["code"] == "no_signing_key"
    finally:
        service.close()


def test_the_service_helper_serializes_a_bundle_the_importer_accepts(service: BuildService) -> None:
    service.run()
    raw = bundle_bytes(service.export_cache())
    assert service.import_cache(raw, principal="ci").rejected_total == 0


# --- publishing ---------------------------------------------------------------------------------


def test_publishing_needs_an_approval_and_then_writes_the_site(
    client: TestClient, settings: Settings
) -> None:
    first = client.post("/v1/publish", json={}, headers=_headers(*RUN))
    assert first.status_code == 202
    body = first.json()
    assert body["status"] == "approval_required"
    assert not (settings.data_dir / "site").exists()

    pending = client.get("/v1/approvals", headers=_headers(*APPROVE)).json()["pending"]
    publish_request = next(item for item in pending if item["tool"] == "publish_docs")
    client.post(
        f"/v1/approvals/{publish_request['id']}",
        json={"approve": True},
        headers=_headers(*APPROVE, subject="reviewer"),
    )

    applied = client.post("/v1/publish", json={}, headers=_headers(*RUN))
    assert applied.status_code == 200
    assert applied.json()["status"] == "applied"
    assert (settings.data_dir / "site" / "manifest.json").is_file()
    assert (settings.data_dir / "site" / "index.md").is_file()


# --- audit --------------------------------------------------------------------------------------


def test_every_operation_is_recorded_in_a_verifiable_chain(
    client: TestClient, settings: Settings
) -> None:
    from incremental_agent_builds.agentic.audit import verify_file

    client.post("/v1/builds", json={}, headers=_headers(*RUN))
    client.get("/v1/cache/export", headers=_headers(*RUN))
    head = client.get("/v1/store", headers=_headers(*RUN)).json()["audit_head"]
    result = verify_file(settings.data_dir / "audit.jsonl", expect_head=head)
    assert result.ok
    assert result.entries > 0
    text = (settings.data_dir / "audit.jsonl").read_text(encoding="utf-8")
    assert "build_started" in text
    assert "cache_exported" in text


# --- HTTP hardening -----------------------------------------------------------------------------


def test_the_service_does_not_expose_its_schema_by_default(client: TestClient) -> None:
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_metrics_require_a_key_and_report_the_build_counters(
    client: TestClient, settings: Settings
) -> None:
    from incremental_agent_builds.core.security import hash_api_key

    assert client.get("/metrics").status_code == 401
    keyed = settings.model_copy(update={"api_key_sha256": [hash_api_key("test-metrics-key")]})
    # No service is passed, so the app builds one on its own registry: that is how a deployment
    # gets the build counters and the HTTP counters on one /metrics endpoint.
    app = create_service_app(keyed)
    try:
        with TestClient(app) as authed:
            authed.post("/v1/builds", json={}, headers=_headers(*RUN))
            response = authed.get("/metrics", headers={"X-API-Key": "test-metrics-key"})
            assert response.status_code == 200
            for family in (
                "rules_executed_total",
                "rules_reused_total",
                "early_cutoffs_total",
                "llm_tokens_saved_total",
                "build_duration_seconds",
                "cache_import_rejections_total",
                "app_http_requests_total",
            ):
                assert family in response.text, family
    finally:
        app.state.build_service.close()


def test_an_oversized_body_is_refused(client: TestClient, settings: Settings) -> None:
    del settings
    response = client.post(
        "/v1/plan",
        content=b"x" * 2_000_000,
        headers={**_headers(*RUN), "Content-Type": "application/json"},
    )
    assert response.status_code == 413


def test_a_request_id_is_echoed_on_errors(client: TestClient) -> None:
    response = client.post("/v1/plan", json={})
    assert response.status_code == 401
    assert response.json()["error"]["request_id"]
