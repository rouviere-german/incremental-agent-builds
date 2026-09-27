"""Signed constructive traces: what is exported, what is accepted, and what a hit still checks.

A constructive trace answers "has anyone computed this key for exactly these dependency values?".
Accepting one is only safe if two things hold, and both are tested here:

1. the entry is authentic (signed by an allowlisted key, over a blob that hashes to the claimed
   result), checked at import time;
2. it is still *applicable*, checked again at use time: the build verifies every recorded dependency
   against its own current values before serving the imported result, and re-checks the signature
   against the allowlist, so revoking a key also revokes its entries.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any

import pytest
from pydantic import JsonValue

from incremental_agent_builds.core.canonical import sha256_hex
from incremental_agent_builds.engine.cas import MemoryContentStore
from incremental_agent_builds.engine.engine import BuildEngine
from incremental_agent_builds.engine.errors import CacheImportError
from incremental_agent_builds.engine.metrics import create_build_metrics
from incremental_agent_builds.engine.shared_cache import (
    BUNDLE_FORMAT,
    export_bundle,
    import_bundle,
)
from incremental_agent_builds.engine.signing import SigningKey, TrustStore
from incremental_agent_builds.engine.store import TraceStore
from incremental_agent_builds.engine.tools import BuildToolRegistry
from tests.engine_helpers import (
    Doubled,
    Upper,
    counting_rules,
    make_engine,
    upper_tool,
    workspace_with,
)
from tests.jsonshape import arr, obj, text

pytestmark = pytest.mark.unit

type Bundle = dict[str, JsonValue]


def _producer() -> BuildEngine:
    engine = make_engine(workspace_with(), counting_rules()[0])
    engine.build_sync([Doubled(path="in/a.txt"), Upper(path="in/a.txt")]).raise_for_failures()
    return engine


def _raw(bundle: Bundle) -> bytes:
    return json.dumps(bundle, sort_keys=True).encode("utf-8")


def _entries(bundle: Bundle, kind: str | None = None) -> list[dict[str, JsonValue]]:
    items = [obj(entry) for entry in arr(bundle["entries"])]
    return [entry for entry in items if kind is None or text(entry["type"]) == kind]


def _blobs(bundle: Bundle) -> dict[str, JsonValue]:
    return obj(bundle["blobs"])


def _fresh_store() -> TraceStore:
    return TraceStore(":memory:", mac_key=os.urandom(32))


def test_an_exported_bundle_carries_traces_steps_and_their_blobs() -> None:
    engine = _producer()
    signer = SigningKey.generate()
    bundle = export_bundle(engine.store, engine.cas, signer)

    assert bundle["format"] == BUNDLE_FORMAT
    assert bundle["version"] == 1
    assert {text(entry["type"]) for entry in _entries(bundle)} == {"trace", "step"}, (
        "the steps a trace relies on travel with it"
    )
    for entry in _entries(bundle):
        assert text(entry["signer"]) == signer.key_id
        assert text(entry["signature"])
        assert text(obj(entry["body"])["result_hash"]) in _blobs(bundle)


def test_exporting_can_be_restricted_to_chosen_keys() -> None:
    from tests.engine_helpers import Echo

    engine = _producer()
    signer = SigningKey.generate()
    bundle = export_bundle(engine.store, engine.cas, signer, key_ids=[Echo(path="in/a.txt").key_id])
    keys = [text(obj(entry["body"])["key_id"]) for entry in _entries(bundle, "trace")]
    assert keys == [Echo(path="in/a.txt").key_id]


def test_a_bundle_imports_into_a_fresh_store_and_is_reusable_there() -> None:
    engine = _producer()
    signer = SigningKey.generate()
    trust = TrustStore([signer.public_b64])
    raw = _raw(export_bundle(engine.store, engine.cas, signer))

    peer_store, peer_cas = _fresh_store(), MemoryContentStore()
    report = import_bundle(raw, store=peer_store, cas=peer_cas, trust=trust)
    assert report.accepted_traces > 0
    assert report.accepted_steps > 0
    assert report.rejected_total == 0

    # A build in the peer store now serves those keys from the imported traces without running them.
    rules, counts = counting_rules()
    peer = make_engine(workspace_with(), rules, store=peer_store, cas=peer_cas, trust=trust)
    result = peer.build_sync([Doubled(path="in/a.txt")])
    assert result.ok
    assert result.stats.imported == 2, "both keys came from the shared cache"
    assert result.stats.executed == 0
    assert counts.runs == {}, "no rule body ran"
    assert peer.value(Doubled(path="in/a.txt")) == engine.value(Doubled(path="in/a.txt"))


def test_an_imported_entry_is_not_used_when_a_dependency_differs() -> None:
    """A constructive hit is still verified against the importer's own inputs."""
    engine = _producer()
    signer = SigningKey.generate()
    trust = TrustStore([signer.public_b64])
    peer_store, peer_cas = _fresh_store(), MemoryContentStore()
    import_bundle(
        _raw(export_bundle(engine.store, engine.cas, signer)),
        store=peer_store,
        cas=peer_cas,
        trust=trust,
    )

    rules, counts = counting_rules()
    peer = make_engine(
        workspace_with({"in/a.txt": "different"}),
        rules,
        store=peer_store,
        cas=peer_cas,
        trust=trust,
    )
    result = peer.build_sync([Doubled(path="in/a.txt")])
    assert result.ok
    assert result.stats.imported == 0, "the entry does not apply to these inputs"
    assert counts.runs["echo"] == 1


def test_revoking_a_signer_stops_its_entries_being_used() -> None:
    """The signature is checked again at use time, so the allowlist is a live revocation list."""
    engine = _producer()
    signer = SigningKey.generate()
    peer_store, peer_cas = _fresh_store(), MemoryContentStore()
    import_bundle(
        _raw(export_bundle(engine.store, engine.cas, signer)),
        store=peer_store,
        cas=peer_cas,
        trust=TrustStore([signer.public_b64]),
    )

    rules, counts = counting_rules()
    revoked = make_engine(
        workspace_with(),
        rules,
        store=peer_store,
        cas=peer_cas,
        trust=TrustStore([SigningKey.generate().public_b64]),
    )
    result = revoked.build_sync([Doubled(path="in/a.txt")])
    assert result.stats.imported == 0
    assert counts.runs["echo"] == 1


def test_only_locally_executed_traces_are_re_exported() -> None:
    """A store must not re-sign results it merely imported, or trust would launder itself."""
    engine = _producer()
    first = SigningKey.generate()
    trust = TrustStore([first.public_b64])
    peer_store, peer_cas = _fresh_store(), MemoryContentStore()
    import_bundle(
        _raw(export_bundle(engine.store, engine.cas, first)),
        store=peer_store,
        cas=peer_cas,
        trust=trust,
    )
    peer = make_engine(
        workspace_with(), counting_rules()[0], store=peer_store, cas=peer_cas, trust=trust
    )
    peer.build_sync([Doubled(path="in/a.txt")])

    re_export = export_bundle(peer_store, peer_cas, SigningKey.generate())
    assert _entries(re_export, "trace") == [], "imported traces are not re-exported"


def test_a_volatile_tool_step_is_never_exported() -> None:
    engine = make_engine(
        workspace_with(),
        counting_rules()[0],
        tools=BuildToolRegistry([upper_tool(ttl_seconds=60.0)]),
    )
    engine.build_sync([Upper(path="in/a.txt")])
    bundle = export_bundle(engine.store, engine.cas, SigningKey.generate())
    assert _entries(bundle, "step") == [], (
        "a result that depends on something outside the workspace is not shared"
    )


def test_a_locally_recorded_step_is_never_overwritten_by_an_import() -> None:
    engine = _producer()
    signer = SigningKey.generate()
    raw = _raw(export_bundle(engine.store, engine.cas, signer))
    report = import_bundle(
        raw, store=engine.store, cas=engine.cas, trust=TrustStore([signer.public_b64])
    )
    assert report.accepted_steps == 0, "the local steps stay authoritative"


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"not json",
        b"{}",
        b'{"format":"other","version":1,"entries":[]}',
        b'{"format":"iab-cache-bundle","version":2,"entries":[]}',
    ],
)
def test_a_bundle_that_is_not_the_expected_document_is_refused(raw: bytes) -> None:
    with pytest.raises(CacheImportError):
        import_bundle(raw, store=_fresh_store(), cas=MemoryContentStore(), trust=TrustStore([]))


def test_an_oversized_bundle_is_refused_before_parsing() -> None:
    from incremental_agent_builds.engine.shared_cache import MAX_BUNDLE_BYTES

    with pytest.raises(CacheImportError, match="exceeds"):
        import_bundle(
            b"x" * (MAX_BUNDLE_BYTES + 1),
            store=_fresh_store(),
            cas=MemoryContentStore(),
            trust=TrustStore([]),
        )


def test_a_malformed_entry_body_is_rejected_individually() -> None:
    """One bad entry must not discard a whole bundle."""
    engine = _producer()
    signer = SigningKey.generate()
    bundle = export_bundle(engine.store, engine.cas, signer)
    entries: list[JsonValue] = list(arr(bundle["entries"]))
    entries.append(
        {"type": "trace", "body": {"kind": "trace"}, "signer": signer.key_id, "signature": "x"}
    )
    bundle["entries"] = entries

    report = import_bundle(
        _raw(bundle),
        store=_fresh_store(),
        cas=MemoryContentStore(),
        trust=TrustStore([signer.public_b64]),
        metrics=create_build_metrics(),
    )
    assert report.rejected["malformed"] == 1
    assert report.accepted_traces > 0, "the good entries still import"


def test_an_entry_whose_blob_is_absent_is_rejected() -> None:
    engine = _producer()
    signer = SigningKey.generate()
    bundle = export_bundle(engine.store, engine.cas, signer)
    bundle["blobs"] = {}
    report = import_bundle(
        _raw(bundle),
        store=_fresh_store(),
        cas=MemoryContentStore(),
        trust=TrustStore([signer.public_b64]),
    )
    assert report.accepted_traces == 0
    assert report.rejected["blob_missing"] > 0


def test_a_non_canonical_blob_is_rejected() -> None:
    """A blob must be the canonical encoding, so one value has exactly one representation."""
    engine = _producer()
    signer = SigningKey.generate()
    bundle = export_bundle(engine.store, engine.cas, signer)
    sloppy = b'{"b": 2, "a": 1}'
    digest = sha256_hex(sloppy)

    body: dict[str, Any] = dict(obj(_entries(bundle, "trace")[0]["body"]))
    body["result_hash"] = digest
    bundle["entries"] = [
        {"type": "trace", "body": body, "signer": signer.key_id, "signature": signer.sign(body)}
    ]
    bundle["blobs"] = {digest: base64.b64encode(sloppy).decode()}

    report = import_bundle(
        _raw(bundle),
        store=_fresh_store(),
        cas=MemoryContentStore(),
        trust=TrustStore([signer.public_b64]),
    )
    assert report.rejected["blob_not_canonical"] == 1


def test_an_import_report_serializes_its_counts() -> None:
    engine = _producer()
    signer = SigningKey.generate()
    report = import_bundle(
        _raw(export_bundle(engine.store, engine.cas, signer)),
        store=_fresh_store(),
        cas=MemoryContentStore(),
        trust=TrustStore([]),
    )
    payload = report.to_json()
    assert payload["accepted_traces"] == 0
    assert payload["rejected"] == {"untrusted_signer": report.rejected_total}
    assert len(text(payload["bundle_sha256"])) == 64
