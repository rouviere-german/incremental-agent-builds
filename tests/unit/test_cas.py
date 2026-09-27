"""The content-addressed value store: verified reads, canonical encoding, safe names."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import JsonValue

from incremental_agent_builds.core.canonical import sha256_hex
from incremental_agent_builds.engine.cas import (
    FileContentStore,
    MemoryContentStore,
    decode_json_blob,
)
from incremental_agent_builds.engine.errors import IntegrityError

pytestmark = pytest.mark.unit


@pytest.fixture(params=["memory", "file"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> MemoryContentStore | FileContentStore:
    if request.param == "memory":
        return MemoryContentStore()
    return FileContentStore(tmp_path / "cas")


def test_a_value_round_trips_through_its_hash(store: MemoryContentStore | FileContentStore) -> None:
    value: JsonValue = {"b": [1, 2, {"c": None}], "a": "x"}
    digest = store.put_json(value)
    assert digest == sha256_hex('{"a":"x","b":[1,2,{"c":null}]}')
    assert store.get_json(digest) == value
    assert store.contains(digest)


def test_equal_values_share_one_blob_whatever_their_key_order(
    store: MemoryContentStore | FileContentStore,
) -> None:
    assert store.put_json({"a": 1, "b": 2}) == store.put_json({"b": 2, "a": 1})


def test_reading_a_missing_blob_raises(store: MemoryContentStore | FileContentStore) -> None:
    with pytest.raises(IntegrityError, match="missing"):
        store.get_bytes("0" * 64)


@pytest.mark.parametrize(
    "digest",
    ["", "abc", "A" * 64, "0" * 63, "0" * 65, "../../etc/passwd", "0" * 32 + "/" + "0" * 31],
)
def test_only_lowercase_hex_names_are_accepted(
    store: MemoryContentStore | FileContentStore, digest: str
) -> None:
    """A hash is the only name a caller can supply, so it can never become a path."""
    with pytest.raises(IntegrityError, match="64 lowercase hex"):
        store.get_bytes(digest)


def test_a_flipped_byte_is_detected_and_counted(
    store: MemoryContentStore | FileContentStore,
) -> None:
    digest = store.put_json({"value": 1})
    if isinstance(store, MemoryContentStore):
        store.blobs[digest] = b'{"value":2}'
    else:
        store.path_for(digest).write_bytes(b'{"value":2}')
    with pytest.raises(IntegrityError, match="SHA-256"):
        store.get_json(digest)
    assert store.integrity_failures == 1


def test_non_canonical_json_is_refused_even_when_the_hash_matches(
    store: MemoryContentStore | FileContentStore,
) -> None:
    """A blob must be the canonical encoding, so one value has exactly one representation."""
    sloppy = b'{"b": 2, "a": 1}'
    digest = store.put_bytes(sloppy)
    assert store.get_bytes(digest) == sloppy
    with pytest.raises(IntegrityError, match="canonical"):
        store.get_json(digest)


def test_invalid_utf8_and_broken_json_are_refused(
    store: MemoryContentStore | FileContentStore,
) -> None:
    for payload in (b"\xff\xfe", b"{not json"):
        digest = store.put_bytes(payload)
        with pytest.raises(IntegrityError):
            store.get_json(digest)


def test_decode_json_blob_reports_the_digest_prefix() -> None:
    with pytest.raises(IntegrityError, match="aaaaaaaaaaaa"):
        decode_json_blob("a" * 64, b"{oops")


def test_writing_over_a_corrupt_blob_repairs_it(tmp_path: Path) -> None:
    """Recomputing a value must heal a corrupt blob, so tampering costs work, not correctness."""
    store = FileContentStore(tmp_path / "cas")
    digest = store.put_json({"value": 1})
    store.path_for(digest).write_bytes(b"corrupt")
    store.put_json({"value": 1})
    assert store.integrity_failures == 1
    assert store.get_json(digest) == {"value": 1}


def test_a_memory_store_also_notices_a_corrupt_overwrite() -> None:
    store = MemoryContentStore()
    digest = store.put_json({"value": 1})
    store.blobs[digest] = b"corrupt"
    store.put_json({"value": 1})
    assert store.integrity_failures == 1
    assert store.get_json(digest) == {"value": 1}


def test_blobs_are_sharded_by_the_first_two_hex_characters(tmp_path: Path) -> None:
    store = FileContentStore(tmp_path / "cas")
    digest = store.put_json("x")
    path = store.path_for(digest)
    assert path.parent.name == digest[:2]
    assert path.name == digest
    assert path.is_file()


def test_the_store_root_is_created_and_exposed(tmp_path: Path) -> None:
    root = tmp_path / "nested" / "cas"
    store = FileContentStore(root)
    assert store.root.is_dir()
    assert store.root == root
