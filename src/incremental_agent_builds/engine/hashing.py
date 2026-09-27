"""Value hashing for traces.

Rule values are JSON values. Their hash is the SHA-256 of their canonical JSON encoding
(:mod:`incremental_agent_builds.core.canonical`), so two values hash equally exactly when they are
equal as JSON (key order and whitespace never matter). File dependencies hash raw bytes. A missing
input has the fixed hash :data:`MISSING`, and a call that raised is recorded with
:data:`ERROR_HASH`. Neither is a valid SHA-256 hex digest, so neither ever equals the hash of any
content: a recorded dependency carrying one can only ever mismatch.
"""

from __future__ import annotations

import re

from pydantic import JsonValue

from incremental_agent_builds.core.canonical import canonical_json, sha256_hex

__all__ = [
    "ERROR_HASH",
    "MISSING",
    "SHA256_HEX",
    "canonical_bytes",
    "is_sha256_hex",
    "value_hash",
]

MISSING = "missing"
ERROR_HASH = "error"
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


def canonical_bytes(value: JsonValue) -> bytes:
    """Canonical UTF-8 JSON bytes of a value (the bytes the CAS stores)."""
    return canonical_json(value).encode("utf-8")


def value_hash(value: JsonValue) -> str:
    """SHA-256 of the canonical JSON encoding of ``value``."""
    return sha256_hex(canonical_bytes(value))


def is_sha256_hex(text: str) -> bool:
    return SHA256_HEX.fullmatch(text) is not None
