"""Content-addressed value store with SHA-256-verified reads.

Blobs are stored under ``<root>/<first two hex chars>/<sha256>``. Every read recomputes the hash
and raises :class:`IntegrityError` on a mismatch, so a flipped byte, a truncated file or a blob
swapped for another one is never returned as a value. Writes are atomic (temporary file plus
``os.replace``); writing a hash that already exists re-verifies the stored copy and repairs it if
it is corrupt, so a corrupted blob cannot survive a rebuild that recomputes its value.

Only lowercase hex SHA-256 names are accepted, so a hash can never be used for path traversal.
"""

from __future__ import annotations

import abc
import json
import threading
from pathlib import Path
from typing import Protocol

from pydantic import JsonValue

from incremental_agent_builds.core.canonical import canonical_json, sha256_hex
from incremental_agent_builds.core.fileio import atomic_write_bytes
from incremental_agent_builds.engine.errors import IntegrityError
from incremental_agent_builds.engine.hashing import canonical_bytes, is_sha256_hex

__all__ = ["ContentStore", "FileContentStore", "MemoryContentStore", "decode_json_blob"]


def decode_json_blob(digest: str, data: bytes) -> JsonValue:
    """Decode a canonical JSON blob, refusing any non-canonical encoding."""
    try:
        text = data.decode("utf-8")
        value: JsonValue = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"blob {digest[:12]} is not valid JSON") from exc
    if canonical_json(value) != text:
        raise IntegrityError(f"blob {digest[:12]} is not canonical JSON")
    return value


class ContentStore(Protocol):
    def put_bytes(self, data: bytes) -> str: ...

    def get_bytes(self, digest: str) -> bytes: ...

    def contains(self, digest: str) -> bool: ...

    def put_json(self, value: JsonValue) -> str: ...

    def get_json(self, digest: str) -> JsonValue: ...


def _check_digest(digest: str) -> None:
    if not is_sha256_hex(digest):
        raise IntegrityError("content hash must be 64 lowercase hex characters")


class _JsonMixin(abc.ABC):
    """Shared JSON encoding/decoding on top of a byte store."""

    @abc.abstractmethod
    def put_bytes(self, data: bytes) -> str: ...

    @abc.abstractmethod
    def get_bytes(self, digest: str) -> bytes: ...

    def put_json(self, value: JsonValue) -> str:
        return self.put_bytes(canonical_bytes(value))

    def get_json(self, digest: str) -> JsonValue:
        return decode_json_blob(digest, self.get_bytes(digest))


class FileContentStore(_JsonMixin):
    """On-disk CAS. Safe for concurrent writers: blobs are immutable and writes are atomic."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)
        self.integrity_failures = 0
        self._lock = threading.Lock()

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, digest: str) -> Path:
        _check_digest(digest)
        return self._root / digest[:2] / digest

    def contains(self, digest: str) -> bool:
        return self.path_for(digest).is_file()

    def put_bytes(self, data: bytes) -> str:
        digest = sha256_hex(data)
        path = self.path_for(digest)
        if path.is_file():
            try:
                if sha256_hex(path.read_bytes()) == digest:
                    return digest
            except OSError:
                pass
            with self._lock:
                self.integrity_failures += 1
        atomic_write_bytes(path, data)
        return digest

    def get_bytes(self, digest: str) -> bytes:
        path = self.path_for(digest)
        try:
            data = path.read_bytes()
        except FileNotFoundError as exc:
            raise IntegrityError(f"blob {digest[:12]} is missing") from exc
        if sha256_hex(data) != digest:
            with self._lock:
                self.integrity_failures += 1
            raise IntegrityError(f"blob {digest[:12]} failed SHA-256 verification")
        return data


class MemoryContentStore(_JsonMixin):
    """In-memory CAS with the same verification, used by property tests and benchmarks."""

    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}
        self.integrity_failures = 0

    def contains(self, digest: str) -> bool:
        _check_digest(digest)
        return digest in self.blobs

    def put_bytes(self, data: bytes) -> str:
        digest = sha256_hex(data)
        existing = self.blobs.get(digest)
        if existing is not None and sha256_hex(existing) != digest:
            self.integrity_failures += 1
        self.blobs[digest] = bytes(data)
        return digest

    def get_bytes(self, digest: str) -> bytes:
        _check_digest(digest)
        data = self.blobs.get(digest)
        if data is None:
            raise IntegrityError(f"blob {digest[:12]} is missing")
        if sha256_hex(data) != digest:
            self.integrity_failures += 1
            raise IntegrityError(f"blob {digest[:12]} failed SHA-256 verification")
        return data
