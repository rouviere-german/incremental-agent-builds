"""Effect sinks: never memoized, only after a successful build, bound to the exact payload.

Everything a build step does is read-only and memoized. The one operation with a side effect -
publishing the generated site - is deliberately *outside* the build: it is not a rule, has no trace,
and runs only when a human approved that exact payload. These tests pin each of those properties,
including the one that matters most: an approval for payload A does not release payload B.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import JsonValue

from incremental_agent_builds.agentic.approval import ApprovalStore
from incremental_agent_builds.agentic.audit import AuditLog
from incremental_agent_builds.engine.effects import (
    DocsPayload,
    DocsSiteSink,
    authorize_plan,
    publish_docs,
)
from incremental_agent_builds.engine.engine import BuildResult
from incremental_agent_builds.engine.errors import EffectError
from incremental_agent_builds.engine.plan import make_plan
from tests.engine_helpers import Doubled, counting_rules, make_engine, workspace_with
from tests.jsonshape import obj

pytestmark = pytest.mark.unit

_PAYLOAD: JsonValue = {"pages": {"index.md": "# Hello\n", "guide/one.md": "# One\n"}}


@pytest.fixture
def approvals(tmp_path: Path) -> ApprovalStore:
    return ApprovalStore(tmp_path / "approvals.json")


def _ok_build() -> BuildResult:
    engine = make_engine(workspace_with(), counting_rules()[0])
    result = engine.build_sync([Doubled(path="in/a.txt")])
    assert result.ok
    return result


def _failed_build() -> BuildResult:
    from tests.engine_helpers import Echo

    engine = make_engine(workspace_with(), counting_rules()[0])
    result = engine.build_sync([Echo(path="../escape.txt")])
    assert not result.ok
    return result


# --- payload validation -------------------------------------------------------------------------


def test_a_valid_payload_validates() -> None:
    assert set(DocsPayload.model_validate(_PAYLOAD).pages) == {"index.md", "guide/one.md"}


@pytest.mark.parametrize(
    "pages",
    [
        {"../escape.md": "x"},
        {"/abs.md": "x"},
        {"a.exe": "x"},
        {"a.md.bak": "x"},
        {"manifest.json": "{}"},
        {"a\\b.md": "x"},
        {"a.md": "x" * 200_001},
    ],
)
def test_unsafe_or_oversized_pages_are_refused(pages: dict[str, str]) -> None:
    """The payload comes from LLM output, so page paths are validated like any untrusted path."""
    with pytest.raises(ValueError, match="validation error"):
        DocsPayload.model_validate({"pages": pages})


def test_extra_payload_fields_are_refused() -> None:
    with pytest.raises(ValueError, match="validation error"):
        DocsPayload.model_validate({"pages": {}, "extra": 1})


# --- the sink -----------------------------------------------------------------------------------


def test_the_sink_writes_pages_and_a_manifest_naming_the_payload_hash(tmp_path: Path) -> None:
    sink = DocsSiteSink(tmp_path / "site")
    written = sink.apply(DocsPayload.model_validate(_PAYLOAD), "a" * 64)
    assert written == ("guide/one.md", "index.md")
    assert (tmp_path / "site" / "index.md").read_text(encoding="utf-8") == "# Hello\n"
    assert (tmp_path / "site" / "guide" / "one.md").read_text(encoding="utf-8") == "# One\n"
    manifest = json.loads((tmp_path / "site" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["payload_sha256"] == "a" * 64
    assert set(manifest["pages"]) == {"index.md", "guide/one.md"}


def test_the_sink_removes_pages_that_left_the_payload(tmp_path: Path) -> None:
    sink = DocsSiteSink(tmp_path / "site")
    sink.apply(DocsPayload.model_validate(_PAYLOAD), "a" * 64)
    sink.apply(DocsPayload.model_validate({"pages": {"index.md": "# New\n"}}), "b" * 64)
    assert (tmp_path / "site" / "index.md").read_text(encoding="utf-8") == "# New\n"
    assert not (tmp_path / "site" / "guide" / "one.md").exists()


def test_the_sink_tolerates_a_corrupt_previous_manifest(tmp_path: Path) -> None:
    root = tmp_path / "site"
    root.mkdir()
    (root / "manifest.json").write_text("not json", encoding="utf-8")
    assert DocsSiteSink(root).apply(DocsPayload.model_validate(_PAYLOAD), "a" * 64)


# --- publishing ---------------------------------------------------------------------------------


def test_publishing_is_refused_until_approved_and_then_applies(
    approvals: ApprovalStore, tmp_path: Path
) -> None:
    sink = DocsSiteSink(tmp_path / "site")
    audit = AuditLog(tmp_path / "audit.jsonl")
    build = _ok_build()

    first = publish_docs(
        build=build,
        payload=_PAYLOAD,
        sink=sink,
        approvals=approvals,
        principal="builder",
        audit=audit,
    )
    assert first.status == "approval_required"
    assert first.approval_id is not None
    assert first.written == ()
    assert not (tmp_path / "site").exists(), "nothing may be written before approval"

    approvals.decide(first.approval_id, approver="reviewer", approve=True)
    second = publish_docs(
        build=build,
        payload=_PAYLOAD,
        sink=sink,
        approvals=approvals,
        principal="builder",
        audit=audit,
    )
    assert second.status == "applied"
    assert second.payload_sha256 == first.payload_sha256
    assert len(second.written) == 2
    assert "effect_applied" in (tmp_path / "audit.jsonl").read_text(encoding="utf-8")


def test_an_approval_for_one_payload_does_not_release_another(
    approvals: ApprovalStore, tmp_path: Path
) -> None:
    """One changed byte of one page must require a new approval."""
    sink = DocsSiteSink(tmp_path / "site")
    build = _ok_build()
    approved = publish_docs(
        build=build,
        payload=_PAYLOAD,
        sink=sink,
        approvals=approvals,
        principal="builder",
    )
    assert approved.approval_id is not None
    approvals.decide(approved.approval_id, approver="reviewer", approve=True)

    pages: dict[str, JsonValue] = dict(obj(obj(_PAYLOAD)["pages"]))
    pages["index.md"] = "# Hello!\n"
    edited: JsonValue = {"pages": pages}
    outcome = publish_docs(
        build=build,
        payload=edited,
        sink=sink,
        approvals=approvals,
        principal="builder",
    )
    assert outcome.status == "approval_required"
    assert outcome.payload_sha256 != approved.payload_sha256
    assert not (tmp_path / "site").exists()


def test_an_approval_is_single_use(approvals: ApprovalStore, tmp_path: Path) -> None:
    sink = DocsSiteSink(tmp_path / "site")
    build = _ok_build()
    first = publish_docs(
        build=build,
        payload=_PAYLOAD,
        sink=sink,
        approvals=approvals,
        principal="builder",
    )
    assert first.approval_id is not None
    approvals.decide(first.approval_id, approver="reviewer", approve=True)
    assert (
        publish_docs(
            build=build,
            payload=_PAYLOAD,
            sink=sink,
            approvals=approvals,
            principal="builder",
        ).status
        == "applied"
    )
    again = publish_docs(
        build=build,
        payload=_PAYLOAD,
        sink=sink,
        approvals=approvals,
        principal="builder",
    )
    assert again.status == "approval_required", "a consumed approval must not publish twice"


def test_an_approval_does_not_transfer_to_another_principal(
    approvals: ApprovalStore, tmp_path: Path
) -> None:
    sink = DocsSiteSink(tmp_path / "site")
    build = _ok_build()
    opened = publish_docs(
        build=build,
        payload=_PAYLOAD,
        sink=sink,
        approvals=approvals,
        principal="builder",
    )
    assert opened.approval_id is not None
    approvals.decide(opened.approval_id, approver="reviewer", approve=True)
    other = publish_docs(
        build=build,
        payload=_PAYLOAD,
        sink=sink,
        approvals=approvals,
        principal="someone_else",
    )
    assert other.status == "approval_required"


def test_publishing_after_a_failed_build_is_refused(
    approvals: ApprovalStore, tmp_path: Path
) -> None:
    with pytest.raises(EffectError, match="successful build"):
        publish_docs(
            build=_failed_build(),
            payload=_PAYLOAD,
            sink=DocsSiteSink(tmp_path / "site"),
            approvals=approvals,
            principal="builder",
        )


def test_an_invalid_payload_is_refused_before_any_approval(
    approvals: ApprovalStore, tmp_path: Path
) -> None:
    with pytest.raises(EffectError, match="not a valid docs site"):
        publish_docs(
            build=_ok_build(),
            payload={"pages": {"../escape.md": "x"}},
            sink=DocsSiteSink(tmp_path / "site"),
            approvals=approvals,
            principal="builder",
        )
    assert approvals.pending() == []


# --- plan authorization -------------------------------------------------------------------------


def test_a_plan_approval_is_bound_to_the_plan_hash(approvals: ApprovalStore) -> None:
    workspace = workspace_with()
    engine = make_engine(workspace, counting_rules()[0])
    engine.build_sync([Doubled(path="in/a.txt")])
    workspace.write("in/a.txt", "9")
    plan = make_plan(engine, [Doubled(path="in/a.txt")])

    first = authorize_plan(approvals, plan, principal="builder")
    assert not first.allowed
    assert first.approval_id is not None
    record = approvals.get(first.approval_id)
    assert record.args["plan_sha256"] == plan.plan_hash
    assert record.args["projected_tokens"] == plan.projected_tokens

    approvals.decide(first.approval_id, approver="reviewer", approve=True)
    assert authorize_plan(approvals, plan, principal="builder").allowed


def test_an_approval_for_one_plan_does_not_authorize_a_different_one(
    approvals: ApprovalStore,
) -> None:
    workspace = workspace_with()
    engine = make_engine(workspace, counting_rules()[0])
    engine.build_sync([Doubled(path="in/a.txt")])
    workspace.write("in/a.txt", "9")
    approved_plan = make_plan(engine, [Doubled(path="in/a.txt")])
    opened = authorize_plan(approvals, approved_plan, principal="builder")
    assert opened.approval_id is not None
    approvals.decide(opened.approval_id, approver="reviewer", approve=True)

    workspace.write("in/b.txt", "99")
    different = make_plan(engine, [Doubled(path="in/a.txt"), Doubled(path="in/b.txt")])
    assert different.plan_hash != approved_plan.plan_hash
    assert not authorize_plan(approvals, different, principal="builder").allowed


def test_a_requester_cannot_approve_its_own_plan(approvals: ApprovalStore) -> None:
    engine = make_engine(workspace_with(), counting_rules()[0])
    plan = make_plan(engine, [Doubled(path="in/a.txt")])
    opened = authorize_plan(approvals, plan, principal="builder")
    assert opened.approval_id is not None
    from incremental_agent_builds.agentic.errors import ApprovalError

    with pytest.raises(ApprovalError, match="separation of duties"):
        approvals.decide(opened.approval_id, approver="builder", approve=True)


def test_a_denied_plan_stays_unauthorized(approvals: ApprovalStore) -> None:
    engine = make_engine(workspace_with(), counting_rules()[0])
    plan = make_plan(engine, [Doubled(path="in/a.txt")])
    opened = authorize_plan(approvals, plan, principal="builder")
    assert opened.approval_id is not None
    approvals.decide(opened.approval_id, approver="reviewer", approve=False, reason="too costly")
    assert not authorize_plan(approvals, plan, principal="builder").allowed


def test_an_effect_outcome_serializes_its_fields() -> None:
    from incremental_agent_builds.engine.effects import EffectOutcome

    payload = EffectOutcome("applied", "a" * 64, None, ("index.md",)).to_json()
    assert payload == {
        "status": "applied",
        "payload_sha256": "a" * 64,
        "approval_id": None,
        "written": ["index.md"],
    }
