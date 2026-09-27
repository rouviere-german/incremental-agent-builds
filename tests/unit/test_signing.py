"""Ed25519 signing of shared-cache entries: key handling, domain separation, allowlisting."""

from __future__ import annotations

import base64

import pytest
from pydantic import JsonValue

from incremental_agent_builds.engine.signing import (
    SIGNATURE_DOMAIN,
    SigningKey,
    TrustStore,
    key_id_for,
    signed_bytes,
)

pytestmark = pytest.mark.unit

_BODY: dict[str, JsonValue] = {"kind": "trace", "key_id": "echo:{}", "result_hash": "0" * 64}


def test_a_signature_verifies_against_its_own_public_key() -> None:
    key = SigningKey.generate()
    trust = TrustStore([key.public_b64])
    assert trust.verify(key.key_id, _BODY, key.sign(_BODY))


def test_a_signature_does_not_verify_for_another_key() -> None:
    signer, stranger = SigningKey.generate(), SigningKey.generate()
    trust = TrustStore([stranger.public_b64])
    assert not trust.verify(stranger.key_id, _BODY, signer.sign(_BODY))


def test_an_unknown_signer_never_verifies() -> None:
    key = SigningKey.generate()
    assert not TrustStore([]).verify(key.key_id, _BODY, key.sign(_BODY))


def test_editing_the_body_invalidates_the_signature() -> None:
    key = SigningKey.generate()
    trust = TrustStore([key.public_b64])
    signature = key.sign(_BODY)
    edited: dict[str, JsonValue] = {**_BODY, "result_hash": "f" * 64}
    assert not trust.verify(key.key_id, edited, signature)


def test_key_order_does_not_change_a_signature() -> None:
    """The signature covers canonical JSON, so re-serialization cannot invalidate an entry."""
    key = SigningKey.generate()
    trust = TrustStore([key.public_b64])
    reordered: dict[str, JsonValue] = {
        "result_hash": "0" * 64,
        "key_id": "echo:{}",
        "kind": "trace",
    }
    assert trust.verify(key.key_id, reordered, key.sign(_BODY))


def test_the_signed_bytes_carry_a_domain_prefix() -> None:
    """Domain separation stops a signature made elsewhere from counting as a cache entry."""
    assert signed_bytes(_BODY).startswith(SIGNATURE_DOMAIN)
    assert SIGNATURE_DOMAIN == b"iab-constructive-cache-v1\n"


def test_a_seed_round_trips_to_the_same_identity() -> None:
    key = SigningKey.generate()
    restored = SigningKey.from_seed_b64(key.seed_b64())
    assert restored.key_id == key.key_id
    assert restored.public_b64 == key.public_b64
    assert TrustStore([key.public_b64]).verify(key.key_id, _BODY, restored.sign(_BODY))


def test_a_key_id_is_a_prefix_of_the_public_key_hash() -> None:
    key = SigningKey.generate()
    raw = base64.b64decode(key.public_b64)
    assert key.key_id == key_id_for(raw)
    assert len(key.key_id) == 16


@pytest.mark.parametrize("seed", ["", "not base64!", base64.b64encode(b"short").decode()])
def test_invalid_seeds_are_refused(seed: str) -> None:
    with pytest.raises(ValueError, match=r"base64|32 bytes"):
        SigningKey.from_seed_b64(seed)


@pytest.mark.parametrize("public", ["not base64!", base64.b64encode(b"short").decode()])
def test_invalid_public_keys_are_refused(public: str) -> None:
    with pytest.raises(ValueError, match=r"base64|32 bytes"):
        TrustStore([public])


@pytest.mark.parametrize("signature", ["", "not base64!", base64.b64encode(b"short").decode()])
def test_a_malformed_signature_fails_verification_without_raising(signature: str) -> None:
    key = SigningKey.generate()
    assert not TrustStore([key.public_b64]).verify(key.key_id, _BODY, signature)


def test_a_trust_store_lists_and_counts_its_keys() -> None:
    first, second = SigningKey.generate(), SigningKey.generate()
    trust = TrustStore([first.public_b64, second.public_b64])
    assert len(trust) == 2
    assert trust.key_ids == tuple(sorted({first.key_id, second.key_id}))
    assert trust.trusts(first.key_id)
    assert not trust.trusts("0" * 16)


def test_the_same_key_listed_twice_counts_once() -> None:
    key = SigningKey.generate()
    assert len(TrustStore([key.public_b64, key.public_b64])) == 1
