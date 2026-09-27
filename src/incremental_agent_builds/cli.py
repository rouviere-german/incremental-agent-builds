"""Command line entry point: ``iab`` / ``incremental-agent-builds``.

```
iab init [--workspace DIR]              generate the reference workspace
iab build [--target KIND[:PATH]] [-f K] plan, gate on cost, build
iab plan  [--target ...]                dry run: dirty set and projected cost
iab explain KIND[:PATH]                  why that key would re-run
iab replay [--commits N]                 replay the scripted history, print savings and checks
iab demo   [--commits N]                 the offline end-to-end demo
iab cache export|import PATH             signed constructive-trace bundles
iab approvals list|decide ID             the approval queue
iab keygen                               a new Ed25519 cache signing key pair
iab token --subject S --scope build:run  mint a bearer token (needs APP_JWT_SECRET)
iab audit verify PATH                    verify the hash-chained audit log
iab serve                                run the HTTP service
iab generate-api-key                     a new API key and its SHA-256 digest
```

Every subcommand except ``serve`` and ``token`` runs fully offline. Paths come from settings
(``APP_WORKSPACE_DIR``, ``APP_STORE_DIR``, ``APP_DATA_DIR``) unless overridden.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = ["main"]

_EXIT_OK = 0
_EXIT_ERROR = 1


def _emit(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _settings(args: argparse.Namespace) -> Any:
    from incremental_agent_builds.core.config import Settings  # noqa: PLC0415

    overrides: dict[str, Any] = {}
    for name in ("workspace_dir", "store_dir", "data_dir"):
        value = getattr(args, name, None)
        if value is not None:
            overrides[name] = value
    return Settings(**overrides)


def _target_specs(values: Sequence[str] | None) -> list[Any]:
    """Parse ``--target kind`` or ``--target kind:relative/path.py`` into target specs."""
    from incremental_agent_builds.api.service import TargetSpec  # noqa: PLC0415

    specs: list[Any] = []
    for raw in values or ():
        kind, sep, path = raw.partition(":")
        specs.append(TargetSpec(kind=kind, args={"path": path} if sep and path else {}))
    return specs


def _service(args: argparse.Namespace) -> Any:
    from incremental_agent_builds.api.service import BuildService  # noqa: PLC0415

    return BuildService.from_settings(_settings(args))


def _init(args: argparse.Namespace) -> int:
    from incremental_agent_builds.workload.generator import (  # noqa: PLC0415
        generate_state,
        write_workspace,
    )

    settings = _settings(args)
    settings.workspace_dir.mkdir(parents=True, exist_ok=True)
    state = generate_state(args.seed)
    written = write_workspace(settings.workspace_dir, state)
    _emit(
        {
            "workspace": str(settings.workspace_dir),
            "modules": len(state.model.modules),
            "files_written": len(written),
        }
    )
    return _EXIT_OK


def _plan(args: argparse.Namespace) -> int:
    service = _service(args)
    try:
        plan = service.plan(_target_specs(args.target), force=args.force)
        payload = plan.to_json()
        payload["approval_required"] = service.approval_required(plan)
        _emit(payload)
    finally:
        service.close()
    return _EXIT_OK


def _build(args: argparse.Namespace) -> int:
    service = _service(args)
    try:
        decision = service.run(
            _target_specs(args.target), force=args.force, principal=args.principal
        )
        _emit(decision.to_json())
        if decision.status == "built":
            return _EXIT_OK
    finally:
        service.close()
    return _EXIT_ERROR


def _explain(args: argparse.Namespace) -> int:
    from incremental_agent_builds.engine.errors import BuildEngineError  # noqa: PLC0415

    service = _service(args)
    try:
        specs = _target_specs([args.key])
        try:
            chain = service.explain(specs[0])
        except BuildEngineError as exc:
            print(str(exc), file=sys.stderr)
            return _EXIT_ERROR
        _emit({"key": args.key, "chain": chain})
    finally:
        service.close()
    return _EXIT_OK


def _publish(args: argparse.Namespace) -> int:
    service = _service(args)
    try:
        decision, outcome = service.build_and_publish(principal=args.principal)
        _emit(
            {
                "build": decision.status,
                "publish": outcome.to_json() if outcome is not None else None,
            }
        )
        return _EXIT_OK if outcome is not None and outcome.status == "applied" else _EXIT_ERROR
    finally:
        service.close()


def _cache(args: argparse.Namespace) -> int:
    from incremental_agent_builds.engine.errors import (  # noqa: PLC0415
        BuildEngineError,
        CacheImportError,
    )

    service = _service(args)
    try:
        if args.cache_command == "export":
            try:
                bundle = service.export_cache()
            except BuildEngineError as exc:
                print(str(exc), file=sys.stderr)
                return _EXIT_ERROR
            args.path.parent.mkdir(parents=True, exist_ok=True)
            args.path.write_text(json.dumps(bundle, sort_keys=True), encoding="utf-8")
            entries = bundle["entries"]
            _emit({"exported": str(args.path), "entries": len(entries) if entries else 0})
            return _EXIT_OK
        try:
            report = service.import_cache(args.path.read_bytes(), principal=args.principal)
        except CacheImportError as exc:
            print(str(exc), file=sys.stderr)
            return _EXIT_ERROR
        _emit(report.to_json())
        return _EXIT_OK
    finally:
        service.close()


def _approvals(args: argparse.Namespace) -> int:
    from incremental_agent_builds.agentic.errors import ApprovalError  # noqa: PLC0415

    service = _service(args)
    try:
        if args.approvals_command == "list":
            _emit(
                [
                    {
                        "id": r.id,
                        "tool": r.tool,
                        "call_hash": r.call_hash,
                        "requested_by": r.requested_by,
                        "args": r.args,
                    }
                    for r in service.pending_approvals()
                ]
            )
            return _EXIT_OK
        try:
            record = service.decide(
                args.id, approver=args.approver, approve=not args.deny, reason=args.reason
            )
        except ApprovalError as exc:
            print(str(exc), file=sys.stderr)
            return _EXIT_ERROR
        _emit({"id": record.id, "status": record.status.value})
        return _EXIT_OK
    finally:
        service.close()


def _replay(args: argparse.Namespace) -> int:
    from incremental_agent_builds.workload.replay import run_replay  # noqa: PLC0415

    report = run_replay(
        args.workdir,
        commits=args.commits,
        clean=not args.no_clean,
        ablations=not args.no_ablations,
        naive=not args.no_naive,
    )
    _emit(report.summary() if args.summary_only else report.to_json())
    return _EXIT_OK


def _demo(args: argparse.Namespace) -> int:
    from incremental_agent_builds.workload.demo import run_demo  # noqa: PLC0415

    summary = run_demo(args.workdir, commits=args.commits)
    _emit(summary)
    return _EXIT_OK


def _keygen(args: argparse.Namespace) -> int:
    from incremental_agent_builds.engine.signing import SigningKey  # noqa: PLC0415

    del args
    key = SigningKey.generate()
    _emit({"key_id": key.key_id, "public_b64": key.public_b64, "seed_b64": key.seed_b64()})
    print(
        "Put seed_b64 in APP_CACHE_SIGNING_SEED (secret) and public_b64 in peers' "
        "APP_CACHE_TRUSTED_KEYS.",
        file=sys.stderr,
    )
    return _EXIT_OK


def _token(args: argparse.Namespace) -> int:
    from incremental_agent_builds.api.auth import Scope, issue_token  # noqa: PLC0415

    settings = _settings(args)
    if settings.jwt_secret is None or not settings.jwt_secret.get_secret_value():
        print("APP_JWT_SECRET is not set; the API refuses tokens without it.", file=sys.stderr)
        return _EXIT_ERROR
    try:
        scopes = [Scope(name) for name in args.scope]
    except ValueError as exc:
        print(f"unknown scope: {exc}", file=sys.stderr)
        return _EXIT_ERROR
    token = issue_token(
        settings.jwt_secret.get_secret_value(),
        subject=args.subject,
        scopes=scopes,
        issuer=settings.jwt_issuer,
        audience=settings.jwt_audience,
        ttl_seconds=min(args.ttl, settings.jwt_max_ttl_seconds),
    )
    print(token)
    return _EXIT_OK


def _serve(args: argparse.Namespace) -> int:
    import uvicorn  # noqa: PLC0415 - only needed when serving

    from incremental_agent_builds.api.app import create_service_app  # noqa: PLC0415
    from incremental_agent_builds.core.logging import configure_logging  # noqa: PLC0415
    from incremental_agent_builds.core.tracing import (  # noqa: PLC0415
        install_global_tracer_provider,
    )

    settings = _settings(args)
    configure_logging(settings.log_level, json_output=settings.log_json)
    app = create_service_app(settings)
    install_global_tracer_provider(app.state.tracer_provider)
    try:
        uvicorn.run(
            app,
            host=args.host or settings.host,
            port=args.port or settings.port,
            log_config=None,
            access_log=False,
            server_header=False,
        )
    finally:
        app.state.build_service.close()
    return _EXIT_OK


def _generate_api_key(args: argparse.Namespace) -> int:
    from incremental_agent_builds.core.security import generate_api_key  # noqa: PLC0415

    del args
    raw, digest = generate_api_key()
    _emit({"api_key": raw, "api_key_sha256": digest})
    print("Store only api_key_sha256 in APP_API_KEY_SHA256.", file=sys.stderr)
    return _EXIT_OK


def _audit(args: argparse.Namespace) -> int:
    from incremental_agent_builds.agentic.audit import main as audit_main  # noqa: PLC0415

    forwarded = ["verify", str(args.path)]
    if args.expect_head:
        forwarded += ["--expect-head", args.expect_head]
    return audit_main(forwarded)


def _add_store_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace-dir", type=Path, default=None)
    parser.add_argument("--store-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="iab", description="Incremental builds for LLM and tool pipelines."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="generate the reference workspace")
    init.add_argument("--seed", type=int, default=7)
    _add_store_options(init)

    for name, help_text in (("build", "plan and build"), ("plan", "dry run only")):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--target", action="append", default=None, metavar="KIND[:PATH]")
        command.add_argument("-f", "--force", action="append", default=[], metavar="KEY_ID")
        command.add_argument("--principal", default="local")
        _add_store_options(command)

    explain = sub.add_parser("explain", help="why a key would re-run")
    explain.add_argument("key", metavar="KIND[:PATH]")
    _add_store_options(explain)

    publish = sub.add_parser("publish", help="build, then publish the docs site (approval-gated)")
    publish.add_argument("--principal", default="local")
    _add_store_options(publish)

    cache = sub.add_parser("cache", help="signed shared-cache bundles")
    cache_sub = cache.add_subparsers(dest="cache_command", required=True)
    for name in ("export", "import"):
        command = cache_sub.add_parser(name)
        command.add_argument("path", type=Path)
        command.add_argument("--principal", default="local")
        _add_store_options(command)

    approvals = sub.add_parser("approvals", help="the approval queue")
    approvals_sub = approvals.add_subparsers(dest="approvals_command", required=True)
    listing = approvals_sub.add_parser("list")
    _add_store_options(listing)
    decide = approvals_sub.add_parser("decide")
    decide.add_argument("id")
    decide.add_argument("--approver", required=True)
    decide.add_argument("--deny", action="store_true")
    decide.add_argument("--reason", default=None)
    _add_store_options(decide)

    replay = sub.add_parser("replay", help="replay the scripted commit history")
    replay.add_argument("--commits", type=int, default=30)
    replay.add_argument("--workdir", type=Path, default=Path("var/replay"))
    replay.add_argument("--no-clean", action="store_true")
    replay.add_argument("--no-ablations", action="store_true")
    replay.add_argument("--no-naive", action="store_true")
    replay.add_argument("--summary-only", action="store_true")

    demo = sub.add_parser("demo", help="the offline end-to-end demo")
    demo.add_argument("--commits", type=int, default=None)
    demo.add_argument("--workdir", type=Path, default=Path("var/demo"))

    sub.add_parser("keygen", help="a new Ed25519 cache signing key pair")

    token = sub.add_parser("token", help="mint a scoped bearer token")
    token.add_argument("--subject", required=True)
    token.add_argument("--scope", action="append", required=True)
    token.add_argument("--ttl", type=int, default=900)

    serve = sub.add_parser("serve", help="run the HTTP service")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    _add_store_options(serve)

    sub.add_parser("generate-api-key", help="print a new API key and its SHA-256 digest")

    audit = sub.add_parser("audit", help="audit log tools")
    audit_sub = audit.add_subparsers(dest="audit_command", required=True)
    verify = audit_sub.add_parser("verify")
    verify.add_argument("path", type=Path)
    verify.add_argument("--expect-head", default=None)

    return parser


_COMMANDS = {
    "init": _init,
    "plan": _plan,
    "build": _build,
    "explain": _explain,
    "publish": _publish,
    "cache": _cache,
    "approvals": _approvals,
    "replay": _replay,
    "demo": _demo,
    "keygen": _keygen,
    "token": _token,
    "serve": _serve,
    "generate-api-key": _generate_api_key,
    "audit": _audit,
}


def main(argv: list[str] | None = None) -> int:
    from incremental_agent_builds.core.logging import configure_logging  # noqa: PLC0415

    args = build_parser().parse_args(argv)
    if args.command != "serve":
        # Every other command prints JSON on stdout, so logs go to stderr and stay quiet.
        configure_logging("WARNING", json_output=True, stream=sys.stderr)
    if args.command == "demo" and args.commits is None:
        from incremental_agent_builds.workload.demo import DEMO_COMMITS  # noqa: PLC0415

        args.commits = DEMO_COMMITS
    return _COMMANDS[str(args.command)](args)


if __name__ == "__main__":
    raise SystemExit(main())
