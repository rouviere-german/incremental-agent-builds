"""Ed25519 signing for shared-cache entries.

Keys travel as base64 of their raw 32-byte form. A signer is identified by ``key_id``, the first
16 hex characters of the SHA-256 of its raw public key. Signatures cover a domain-separation
prefix plus the canonical JSON of the entry, so a signature made for another purpose (or another
format version) never verifies as a cache entry.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Iterable, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from pydantic import JsonValue

from incremental_agent_builds.core.canonical import canonical_json, sha256_hex

__all__ = ["SIGNATURE_DOMAIN", "SigningKey", "TrustStore", "key_id_for", "signed_bytes"]

SIGNATURE_DOMAIN = b"iab-constructive-cache-v1\n"


def signed_bytes(body: Mapping[str, JsonValue]) -> bytes:
    return SIGNATURE_DOMAIN + canonical_json(dict(body)).encode("utf-8")


def key_id_for(public_raw: bytes) -> str:
    return sha256_hex(public_raw)[:16]


def _b64decode(text: str, *, length: int, what: str) -> bytes:
    try:
        raw = base64.b64decode(text.strip(), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{what} is not valid base64") from exc
    if len(raw) != length:
        raise ValueError(f"{what} must decode to {length} bytes")
    return raw


class SigningKey:
    def __init__(self, private: Ed25519PrivateKey) -> None:
        self._private = private
        self._public_raw = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)

    @classmethod
    def generate(cls) -> SigningKey:
        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def from_seed_b64(cls, seed_b64: str) -> SigningKey:
        seed = _b64decode(seed_b64, length=32, what="signing key seed")
        return cls(Ed25519PrivateKey.from_private_bytes(seed))

    def seed_b64(self) -> str:
        from cryptography.hazmat.primitives.serialization import (  # noqa: PLC0415
            NoEncryption,
            PrivateFormat,
        )

        raw = self._private.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
        return base64.b64encode(raw).decode("ascii")

    @property
    def public_b64(self) -> str:
        return base64.b64encode(self._public_raw).decode("ascii")

    @property
    def key_id(self) -> str:
        return key_id_for(self._public_raw)

    def sign(self, body: Mapping[str, JsonValue]) -> str:
        return base64.b64encode(self._private.sign(signed_bytes(body))).decode("ascii")


class TrustStore:
    """Allowlisted public keys. Only entries signed by one of them are accepted."""

    def __init__(self, public_keys_b64: Iterable[str] = ()) -> None:
        self._keys: dict[str, Ed25519PublicKey] = {}
        for text in public_keys_b64:
            raw = _b64decode(text, length=32, what="trusted public key")
            self._keys[key_id_for(raw)] = Ed25519PublicKey.from_public_bytes(raw)

    @property
    def key_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._keys))

    def __len__(self) -> int:
        return len(self._keys)

    def trusts(self, key_id: str) -> bool:
        return key_id in self._keys

    def verify(self, key_id: str, body: Mapping[str, JsonValue], signature_b64: str) -> bool:
        public = self._keys.get(key_id)
        if public is None:
            return False
        try:
            signature = _b64decode(signature_b64, length=64, what="signature")
            public.verify(signature, signed_bytes(body))
        except (ValueError, InvalidSignature):
            return False
        return True
