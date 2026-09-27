"""The CLI end to end, over a real workspace and a real on-disk store.

These tests drive the commands a person actually types, in the order they type them: generate a
workspace, plan, build, ask why something would re-run, export the cache, publish behind an
approval, verify the audit log. Every command runs offline with the deterministic provider.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from incremental_agent_builds import cli
from incremental_agent_builds.workload.generator import generate_state, write_workspace

pytestmark = pytest.mark.integration


@pytest.fixture
def repo(tmp_path: Path) -> dict[str, str]:
    """A four-module workspace plus the store/data directories, as CLI arguments."""
    state = generate_state()
    state.model.modules = [m for m in state.model.modules if m.sub == "core"][:4]
    state.model.subpackages = ["core"]
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_workspace(workspace, state)
    return {
        "--workspace-dir": str(workspace),
        "--store-dir": str(tmp_path / "store"),
        "--data-dir": str(tmp_path / "data"),
    }


def _flags(repo: dict[str, str]) -> list[str]:
    return [item for pair in repo.items() for item in pair]


def _run(capsys: pytest.CaptureFixture[str], *args: str, expect: int = 0) -> Any:
    """Run the CLI and return the JSON it printed."""
    code = cli.main(list(args))
    out = capsys.readouterr().out
    assert code == expect, f"{args} exited {code}: {out}"
    return json.loads(out) if out.strip() else None


def _env(monkeypatch: pytest.MonkeyPatch, **values: str) -> None:
    for name, value in values.items():
        monkeypatch.setenv(name, value)


# --- workspace and build ------------------------------------------------------------------------


def test_init_generates_the_reference_workspace(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    body = _run(capsys, "init", "--workspace-dir", str(tmp_path / "ws"))
    assert body["modules"] == 80
    assert body["files_written"] > 80
    assert (tmp_path / "ws" / "iab.toml").is_file()
    assert (tmp_path / "ws" / "templates" / "module_summary.txt").is_file()


def test_plan_then_build_then_a_second_build_that_does_nothing(
    repo: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    plan = _run(capsys, "plan", *_flags(repo))
    assert plan["dirty_count"] > 0
    assert plan["clean_count"] == 0

    first = _run(capsys, "build", *_flags(repo))
    assert first["status"] == "built"
    assert first["build"]["stats"]["executed"] > 0
    assert first["build"]["stats"]["provider_calls"] > 0

    second = _run(capsys, "build", *_flags(repo))
    assert second["build"]["stats"]["executed"] == 0
    assert second["build"]["stats"]["reused"] > 0
    assert second["build"]["stats"]["provider_calls"] == 0

    after = _run(capsys, "plan", *_flags(repo))
    assert after["dirty_count"] == 0
    assert after["clean_count"] > 0


def test_a_whitespace_edit_costs_one_rule_and_no_llm_call(
    repo: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    """The headline behaviour: a layout-only edit must not re-pay for any LLM step."""
    _run(capsys, "build", *_flags(repo))
    module = next(
        p
        for p in Path(repo["--workspace-dir"]).joinpath("pkg", "core").glob("*.py")
        if p.name != "__init__.py"
    )
    module.write_text(module.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")

    result = _run(capsys, "build", *_flags(repo))
    assert result["status"] == "built"
    stats = result["build"]["stats"]
    assert stats["executed"] == 1, "only the outline rule re-runs"
    assert stats["early_cutoffs"] == 1, "and its value is unchanged, so nothing propagates"
    assert stats["provider_calls"] == 0


def test_build_can_target_one_key_and_force_another(
    repo: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    _run(capsys, "build", "--target", "module_list", *_flags(repo))
    forced = _run(
        capsys, "build", "--target", "module_list", "--force", "module_list:{}", *_flags(repo)
    )
    assert forced["build"]["stats"]["executed"] == 1
    assert forced["build"]["stats"]["early_cutoffs"] == 1


def test_explain_says_why_a_key_would_re_run(
    repo: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    _run(capsys, "build", *_flags(repo))
    up_to_date = _run(capsys, "explain", "docs_site", *_flags(repo))
    assert up_to_date["chain"] == ["docs_site:{}: up to date"]

    config = Path(repo["--workspace-dir"]) / "iab.toml"
    config.write_text(
        config.read_text(encoding="utf-8").replace("claude-opus-5", "claude-sonnet-5"),
        encoding="utf-8",
    )
    dirty = _run(capsys, "explain", "docs_site", *_flags(repo))
    assert len(dirty["chain"]) > 1
    assert dirty["chain"][-1].endswith("changed")


def test_explain_reports_an_unknown_key_kind(
    repo: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["explain", "not_a_rule", *_flags(repo)]) == 1
    assert "no rule defines the key kind" in capsys.readouterr().err


# --- the shared cache ---------------------------------------------------------------------------


def test_cache_export_and_import_round_trip(
    repo: dict[str, str],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from incremental_agent_builds.engine.signing import SigningKey

    key = SigningKey.generate()
    _env(monkeypatch, APP_CACHE_SIGNING_SEED=key.seed_b64())
    _run(capsys, "build", *_flags(repo))

    bundle = tmp_path / "bundle.json"
    exported = _run(capsys, "cache", "export", str(bundle), *_flags(repo))
    assert exported["entries"] > 0
    assert bundle.is_file()

    imported = _run(capsys, "cache", "import", str(bundle), *_flags(repo))
    assert imported["rejected_total"] == 0


def test_cache_export_without_a_signing_key_fails_cleanly(
    repo: dict[str, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["cache", "export", str(tmp_path / "b.json"), *_flags(repo)]) == 1
    assert "signing key" in capsys.readouterr().err


def test_importing_a_bundle_from_an_untrusted_signer_rejects_every_entry(
    repo: dict[str, str],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from incremental_agent_builds.engine.signing import SigningKey

    stranger = SigningKey.generate()
    _env(monkeypatch, APP_CACHE_SIGNING_SEED=stranger.seed_b64())
    _run(capsys, "build", *_flags(repo))
    bundle = tmp_path / "bundle.json"
    _run(capsys, "cache", "export", str(bundle), *_flags(repo))

    # A second store that trusts a different key entirely.
    other = SigningKey.generate()
    _env(monkeypatch, APP_CACHE_SIGNING_SEED=other.seed_b64())
    report = _run(
        capsys,
        "cache",
        "import",
        str(bundle),
        "--workspace-dir",
        repo["--workspace-dir"],
        "--store-dir",
        str(tmp_path / "store2"),
        "--data-dir",
        str(tmp_path / "data2"),
    )
    assert report["accepted_traces"] == 0
    assert report["rejected"]["untrusted_signer"] > 0


def test_a_malformed_bundle_file_fails_cleanly(
    repo: dict[str, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not a bundle", encoding="utf-8")
    assert cli.main(["cache", "import", str(bad), *_flags(repo)]) == 1
    capsys.readouterr()


# --- approvals and publishing -------------------------------------------------------------------


def test_publishing_needs_an_approval_from_a_second_principal(
    repo: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["publish", "--principal", "builder", *_flags(repo)]) == 1
    first = json.loads(capsys.readouterr().out)
    assert first["publish"]["status"] == "approval_required"
    approval_id = first["publish"]["approval_id"]

    pending = _run(capsys, "approvals", "list", *_flags(repo))
    assert [item["id"] for item in pending] == [approval_id]
    assert pending[0]["tool"] == "publish_docs"

    assert (
        cli.main(["approvals", "decide", approval_id, "--approver", "builder", *_flags(repo)]) == 1
    )
    assert "separation of duties" in capsys.readouterr().err

    decided = _run(
        capsys, "approvals", "decide", approval_id, "--approver", "reviewer", *_flags(repo)
    )
    assert decided["status"] == "approved"

    applied = _run(capsys, "publish", "--principal", "builder", *_flags(repo))
    assert applied["publish"]["status"] == "applied"
    assert applied["publish"]["written"]
    site = Path(repo["--data-dir"]) / "site"
    assert (site / "manifest.json").is_file()
    assert (site / "index.md").is_file()


def test_denying_an_approval_keeps_the_site_unpublished(
    repo: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    cli.main(["publish", "--principal", "builder", *_flags(repo)])
    approval_id = json.loads(capsys.readouterr().out)["publish"]["approval_id"]
    denied = _run(
        capsys,
        "approvals",
        "decide",
        approval_id,
        "--approver",
        "reviewer",
        "--deny",
        "--reason",
        "not yet",
        *_flags(repo),
    )
    assert denied["status"] == "denied"
    assert cli.main(["publish", "--principal", "builder", *_flags(repo)]) == 1
    capsys.readouterr()
    assert not (Path(repo["--data-dir"]) / "site").exists()


# --- keys, tokens and the audit log -------------------------------------------------------------


def test_keygen_prints_a_usable_key_pair(capsys: pytest.CaptureFixture[str]) -> None:
    from incremental_agent_builds.engine.signing import SigningKey, TrustStore

    body = _run(capsys, "keygen")
    restored = SigningKey.from_seed_b64(body["seed_b64"])
    assert restored.key_id == body["key_id"]
    assert TrustStore([body["public_b64"]]).trusts(body["key_id"])


def test_generate_api_key_prints_a_key_and_its_digest(capsys: pytest.CaptureFixture[str]) -> None:
    from incremental_agent_builds.core.security import hash_api_key

    body = _run(capsys, "generate-api-key")
    assert body["api_key_sha256"] == hash_api_key(body["api_key"])


def test_token_needs_a_secret_and_then_mints_a_verifiable_token(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from incremental_agent_builds.api.auth import JwtVerifier, Scope

    assert cli.main(["token", "--subject", "ci", "--scope", "build:run"]) == 1
    assert "APP_JWT_SECRET" in capsys.readouterr().err

    secret = "cli-integration-test-secret-long-enough"
    _env(monkeypatch, APP_JWT_SECRET=secret)
    assert (
        cli.main(["token", "--subject", "ci", "--scope", "build:run", "--scope", "cache:import"])
        == 0
    )
    token = capsys.readouterr().out.strip()
    principal = JwtVerifier(
        secret,
        issuer="incremental-agent-builds",
        audience="incremental-agent-builds",
        max_ttl_seconds=3600,
    ).verify(token)
    assert principal.subject == "ci"
    assert principal.scopes == {Scope.BUILD_RUN, Scope.CACHE_IMPORT}


def test_token_refuses_an_unknown_scope(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _env(monkeypatch, APP_JWT_SECRET="cli-integration-test-secret-long-enough")
    assert cli.main(["token", "--subject", "ci", "--scope", "admin"]) == 1
    assert "unknown scope" in capsys.readouterr().err


def test_the_audit_log_of_a_build_verifies(
    repo: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    _run(capsys, "build", *_flags(repo))
    log = Path(repo["--data-dir"]) / "audit.jsonl"
    assert cli.main(["audit", "verify", str(log)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["entries"] > 0

    assert cli.main(["audit", "verify", str(log), "--expect-head", "0" * 64]) == 1
    capsys.readouterr()


def test_a_tampered_audit_log_fails_verification(
    repo: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    _run(capsys, "build", *_flags(repo))
    log = Path(repo["--data-dir"]) / "audit.jsonl"
    lines = log.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[0] = lines[0].replace("build_started", "build_stopped")
    log.write_text("".join(lines), encoding="utf-8")
    assert cli.main(["audit", "verify", str(log)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False
    assert report["line"] == 1


# --- replay and demo ----------------------------------------------------------------------------


def test_replay_runs_a_short_history_and_reports_savings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    body = _run(
        capsys,
        "replay",
        "--commits",
        "2",
        "--workdir",
        str(tmp_path / "replay"),
        "--no-ablations",
        "--no-naive",
        "--summary-only",
    )
    assert body["commits"] == 2
    assert body["checked_commits"] == 2
    assert body["equivalent_commits"] == 2
    assert body["unnecessary_reruns"] == 0
    assert body["total_calls_saved"] > 0.5


def test_the_cli_reports_unknown_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["not-a-command"])
    assert exit_info.value.code == 2
    capsys.readouterr()


def test_the_module_entry_point_works(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import runpy
    import sys

    monkeypatch.setattr(sys, "argv", ["incremental_agent_builds", "generate-api-key"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("incremental_agent_builds", run_name="__main__")
    assert exit_info.value.code == 0
    assert "api_key_sha256" in capsys.readouterr().out


@pytest.mark.usefixtures("restore_logging")
def test_serve_configures_uvicorn_safely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repo: dict[str, str]
) -> None:
    import uvicorn
    from opentelemetry import trace as otel_trace

    from incremental_agent_builds.core import tracing

    captured: dict[str, Any] = {}

    def fake_run(app: object, **kwargs: Any) -> None:
        captured.update(kwargs, app=app)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(tracing, "_installed", [])
    monkeypatch.setattr(otel_trace, "set_tracer_provider", lambda provider: None)
    assert cli.main(["serve", "--port", "8123", *_flags(repo)]) == 0
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8123
    assert captured["server_header"] is False
    assert captured["access_log"] is False
