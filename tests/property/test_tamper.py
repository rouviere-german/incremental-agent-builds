"""Any mutation of stored bytes or records must be detected, for arbitrary mutations.

Three stores hold state that a build trusts, and each has its own check:

* the **CAS** verifies SHA-256 on every read, so a flipped, truncated, extended or swapped blob can
  never be returned as a value;
* the **trace store** carries an HMAC over each row's exact stored text, so an edited dependency
  hash, a swapped result hash or a row copied to another key is treated as absent;
* the **shared cache** requires a valid Ed25519 signature from an allowlisted key over the canonical
  entry, so an unsigned, re-signed, forged or replayed-with-edits entry is rejected and counted.

These are properties rather than examples because the interesting claim is "any mutation", not "this
mutation": hypothesis picks the byte, the field and the value.
"""

from __future__ import annotations

import base64
import json
import os

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from pydantic import JsonValue

from incremental_agent_builds.engine.cas import FileContentStore, MemoryContentStore
from incremental_agent_builds.engine.errors import IntegrityError
from incremental_agent_builds.engine.hashing import canonical_bytes, value_hash
from incremental_agent_builds.engine.metrics import create_build_metrics
from incremental_agent_builds.engine.shared_cache import import_bundle
from incremental_agent_builds.engine.signing import SigningKey, TrustStore
from incremental_agent_builds.engine.store import (
    Dep,
    DepKind,
    StepRecord,
    Trace,
    TraceStore,
    TraceUsage,
)

pytestmark = pytest.mark.property

json_values = st.recursive(
    st.none() | st.booleans() | st.integers(min_value=-1000, max_value=1000) | st.text(max_size=20),
    lambda children: (
        st.lists(children, max_size=4)
        | st.dictionaries(st.text(min_size=1, max_size=8), children, max_size=4)
    ),
    max_leaves=8,
)


@settings(max_examples=100, deadline=None)
@given(value=json_values, position=st.integers(min_value=0), flip=st.integers(1, 255))
def test_any_flipped_byte_in_a_blob_is_detected(value: JsonValue, position: int, flip: int) -> None:
    store = MemoryContentStore()
    digest = store.put_json(value)
    data = bytearray(store.blobs[digest])
    index = position % len(data)
    data[index] ^= flip
    assume(bytes(data) != store.blobs[digest])
    store.blobs[digest] = bytes(data)
    with pytest.raises(IntegrityError):
        store.get_json(digest)
    assert store.integrity_failures == 1


@settings(max_examples=60, deadline=None)
@given(value=json_values, cut=st.integers(min_value=0, max_value=200))
def test_truncated_and_extended_blobs_are_detected(value: JsonValue, cut: int) -> None:
    store = MemoryContentStore()
    digest = store.put_json(value)
    original = store.blobs[digest]
    for mutated in (original[: cut % max(1, len(original))], original + b" ", b""):
        assume(mutated != original)
        store.blobs[digest] = mutated
        with pytest.raises(IntegrityError):
            store.get_bytes(digest)
    store.blobs[digest] = original
    assert store.get_json(digest) == value


@settings(max_examples=60, deadline=None)
@given(left=json_values, right=json_values)
def test_a_blob_swapped_for_another_is_detected(left: JsonValue, right: JsonValue) -> None:
    assume(canonical_bytes(left) != canonical_bytes(right))
    store = MemoryContentStore()
    left_digest = store.put_json(left)
    right_digest = store.put_json(right)
    store.blobs[left_digest] = store.blobs[right_digest]
    with pytest.raises(IntegrityError):
        store.get_json(left_digest)


@settings(max_examples=60, deadline=None)
@given(value=json_values, position=st.integers(min_value=0), flip=st.integers(1, 255))
def test_a_file_backed_blob_survives_no_mutation(
    tmp_path_factory: pytest.TempPathFactory, value: JsonValue, position: int, flip: int
) -> None:
    root = tmp_path_factory.mktemp("cas")
    store = FileContentStore(root)
    digest = store.put_json(value)
    path = store.path_for(digest)
    data = bytearray(path.read_bytes())
    data[position % len(data)] ^= flip
    assume(bytes(data) != path.read_bytes())
    path.write_bytes(bytes(data))
    with pytest.raises(IntegrityError):
        store.get_json(digest)
    # Recomputing the value repairs the blob, so the next read succeeds.
    store.put_json(value)
    assert store.get_json(digest) == value


def _trace(key_id: str = "node:{}", result: str | None = None) -> Trace:
    return Trace(
        key_id=key_id,
        rule="node",
        version=1,
        deps=((Dep(DepKind.FILE, "in/a.txt", value_hash("a")),),),
        result_hash=result or value_hash(42),
        built_run=1,
        changed_run=1,
        usage=TraceUsage(llm_steps=1, input_tokens=10, output_tokens=5),
    )


@settings(max_examples=60, deadline=None)
@given(
    column=st.sampled_from(["deps", "result_hash", "version", "usage", "origin", "rule"]),
    replacement=st.text(min_size=1, max_size=24),
)
def test_editing_any_trace_column_fails_the_mac(column: str, replacement: str) -> None:
    store = TraceStore(":memory:", mac_key=os.urandom(32))
    trace = _trace()
    store.put_traces([trace])
    assert store.get_trace(trace.key_id) is not None
    store.raw_execute(
        f"UPDATE traces SET {column} = ? WHERE key_id = ?",  # noqa: S608 - column from a literal set
        (replacement, trace.key_id),
    )
    assert store.get_trace(trace.key_id) is None, "an edited row must not be returned"
    assert store.integrity_events[-1].reason in {"MAC mismatch", "malformed row"}


@settings(max_examples=40, deadline=None)
@given(other_key=st.text(min_size=1, max_size=16))
def test_a_trace_row_copied_to_another_key_fails_the_mac(other_key: str) -> None:
    """The MAC covers the key id, so a valid row moved to a different key does not verify."""
    assume(other_key != "node:{}")
    store = TraceStore(":memory:", mac_key=os.urandom(32))
    trace = _trace()
    store.put_traces([trace])
    rows = store.raw_execute(
        "SELECT deps, result_hash, usage, mac FROM traces WHERE key_id = ?", (trace.key_id,)
    )
    deps, result_hash, usage, mac = rows[0]
    store.raw_execute(
        "INSERT INTO traces (key_id, rule, version, deps, result_hash, built_run, changed_run, "
        "usage, origin, mac) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (other_key, "node", 1, deps, result_hash, 1, 1, usage, "executed", mac),
    )
    assert store.get_trace(other_key) is None


@settings(max_examples=40, deadline=None)
@given(replacement=st.text(min_size=1, max_size=24))
def test_editing_a_step_row_fails_the_mac(replacement: str) -> None:
    store = TraceStore(":memory:", mac_key=os.urandom(32))
    step = StepRecord("a" * 64, "llm", value_hash("out"), {"model": "m"}, 100.0)
    store.put_steps([step])
    assert store.load_steps()[step.fingerprint].result_hash == step.result_hash
    store.raw_execute(
        "UPDATE steps SET result_hash = ? WHERE fingerprint = ?", (replacement, step.fingerprint)
    )
    assert store.load_steps() == {}


def _signed_bundle(signer: SigningKey) -> dict[str, object]:
    body = {
        "kind": "trace",
        "key_id": "node:{}",
        "rule": "node",
        "version": 1,
        "deps": [[["file", "in/a.txt", value_hash("a")]]],
        "result_hash": value_hash(42),
        "usage": TraceUsage().to_json(),
    }
    return {
        "format": "iab-cache-bundle",
        "version": 1,
        "entries": [
            {"type": "trace", "body": body, "signer": signer.key_id, "signature": signer.sign(body)}  # type: ignore[arg-type]
        ],
        "blobs": {value_hash(42): base64.b64encode(canonical_bytes(42)).decode()},
    }


def _import(bundle: dict[str, object], trust: TrustStore) -> dict[str, int]:
    report = import_bundle(
        json.dumps(bundle).encode("utf-8"),
        store=TraceStore(":memory:", mac_key=os.urandom(32)),
        cas=MemoryContentStore(),
        trust=trust,
        metrics=create_build_metrics(),
    )
    return {"accepted": report.accepted_traces, **dict(report.rejected)}


@settings(max_examples=50, deadline=None)
@given(
    field=st.sampled_from(["key_id", "rule", "version", "result_hash", "deps"]),
    replacement=st.integers(min_value=0, max_value=5),
)
def test_editing_a_signed_cache_entry_breaks_its_signature(field: str, replacement: int) -> None:
    signer = SigningKey.generate()
    trust = TrustStore([signer.public_b64])
    bundle = _signed_bundle(signer)
    entry = bundle["entries"][0]  # type: ignore[index]
    original = entry["body"][field]
    edits: dict[str, object] = {
        "key_id": f'node:{{"i":{replacement}}}',
        "rule": f"other{replacement}",
        "version": replacement + 2,
        "result_hash": value_hash(replacement + 1000),
        "deps": [[["file", "in/a.txt", value_hash(replacement + 7)]]],
    }
    assume(edits[field] != original)
    entry["body"][field] = edits[field]
    outcome = _import(bundle, trust)
    assert outcome["accepted"] == 0
    assert set(outcome) - {"accepted"}, f"an edited entry must be rejected: {outcome}"


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(min_value=0, max_value=1000))
def test_unsigned_and_untrusted_entries_are_rejected(seed: int) -> None:
    del seed
    signer = SigningKey.generate()
    stranger = SigningKey.generate()
    trust = TrustStore([signer.public_b64])

    unsigned = _signed_bundle(signer)
    unsigned["entries"][0]["signature"] = ""  # type: ignore[index]
    assert _import(unsigned, trust) == {"accepted": 0, "unsigned": 1}

    foreign = _signed_bundle(stranger)
    assert _import(foreign, trust) == {"accepted": 0, "untrusted_signer": 1}

    # Re-signing someone else's body with a trusted key is still refused by the signer check on
    # the *claimed* signer, and by verification when the claimed signer is the trusted one.
    swapped = _signed_bundle(stranger)
    swapped["entries"][0]["signer"] = signer.key_id  # type: ignore[index]
    assert _import(swapped, trust) == {"accepted": 0, "bad_signature": 1}

    assert _import(_signed_bundle(signer), trust) == {"accepted": 1}


@settings(max_examples=30, deadline=None)
@given(flip=st.integers(1, 255), position=st.integers(min_value=0))
def test_a_bundle_blob_that_does_not_match_its_hash_is_rejected(flip: int, position: int) -> None:
    signer = SigningKey.generate()
    bundle = _signed_bundle(signer)
    blobs: dict[str, str] = bundle["blobs"]  # type: ignore[assignment]
    digest, encoded = next(iter(blobs.items()))
    raw = bytearray(base64.b64decode(encoded))
    raw[position % len(raw)] ^= flip
    assume(base64.b64encode(bytes(raw)).decode() != encoded)
    blobs[digest] = base64.b64encode(bytes(raw)).decode()
    outcome = _import(bundle, TrustStore([signer.public_b64]))
    assert outcome["accepted"] == 0
    assert outcome.get("blob_mismatch") or outcome.get("blob_not_canonical")
