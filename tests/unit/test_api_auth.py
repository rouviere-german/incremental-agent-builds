"""Scoped bearer tokens: what is accepted, and what every rejection looks like.

Verification is strict on purpose. The tests below cover the forgeries that a lax JWT setup lets
through - an ``alg: none`` token, a token signed with the public key of an asymmetric algorithm, a
token minted for another service, one with no expiry, and one with an expiry far in the future - and
assert that every rejection returns the same opaque 401, so a caller learns nothing from the
difference.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import jwt
import pytest

from incremental_agent_builds.api.auth import (
    ALGORITHM,
    BearerPrincipal,
    JwtVerifier,
    Scope,
    issue_token,
)
from incremental_agent_builds.core.errors import AppError

pytestmark = pytest.mark.unit

_SECRET = "unit-test-signing-secret-not-a-real-one"
_ISSUER = "incremental-agent-builds"
_AUDIENCE = "incremental-agent-builds"


def _verifier(*, max_ttl_seconds: int = 3600, secret: str | None = _SECRET) -> JwtVerifier:
    return JwtVerifier(secret, issuer=_ISSUER, audience=_AUDIENCE, max_ttl_seconds=max_ttl_seconds)


def _token(**overrides: Any) -> str:
    kwargs: dict[str, Any] = {
        "subject": "ci",
        "scopes": [Scope.BUILD_RUN],
        "issuer": _ISSUER,
        "audience": _AUDIENCE,
        "ttl_seconds": 900,
    }
    kwargs.update(overrides)
    token: str = issue_token(_SECRET, **kwargs)
    return token


def _raw(claims: dict[str, Any], *, secret: str = _SECRET, algorithm: str = ALGORITHM) -> str:
    return jwt.encode(claims, secret, algorithm=algorithm)


def _now() -> int:
    return int(dt.datetime.now(dt.UTC).timestamp())


def _claims(**overrides: Any) -> dict[str, Any]:
    now = _now()
    base: dict[str, Any] = {
        "iss": _ISSUER,
        "aud": _AUDIENCE,
        "sub": "ci",
        "iat": now,
        "nbf": now,
        "exp": now + 900,
        "scope": "build:run",
    }
    base.update(overrides)
    return base


# --- acceptance ---------------------------------------------------------------------------------


def test_a_valid_token_yields_its_subject_and_scopes() -> None:
    principal = _verifier().verify(_token(scopes=[Scope.BUILD_RUN, Scope.CACHE_IMPORT]))
    assert isinstance(principal, BearerPrincipal)
    assert principal.subject == "ci"
    assert principal.scopes == {Scope.BUILD_RUN, Scope.CACHE_IMPORT}
    assert principal.has(Scope.BUILD_RUN)
    assert not principal.has(Scope.BUILD_APPROVE)
    assert principal.token_id


def test_scopes_are_a_space_separated_claim() -> None:
    principal = _verifier().verify(_raw(_claims(scope="build:run build:approve cache:import")))
    assert principal.scopes == set(Scope)


def test_an_unknown_scope_name_grants_nothing() -> None:
    principal = _verifier().verify(_raw(_claims(scope="build:run admin:everything")))
    assert principal.scopes == {Scope.BUILD_RUN}


def test_a_token_with_no_scopes_authenticates_but_authorizes_nothing() -> None:
    principal = _verifier().verify(_raw(_claims(scope="")))
    assert principal.subject == "ci"
    assert principal.scopes == frozenset()


# --- rejection ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("description", "token"),
    [
        ("no token", None),
        ("empty", ""),
        ("not a jwt", "garbage"),
        ("oversized", "a" * 5000),
    ],
)
def test_a_missing_or_malformed_token_is_refused(description: str, token: str | None) -> None:
    del description
    with pytest.raises(AppError) as info:
        _verifier().verify(token)
    assert info.value.status_code == 401
    assert info.value.code == "unauthorized"
    assert info.value.headers["WWW-Authenticate"] == "Bearer"


def test_a_token_signed_with_another_secret_is_refused() -> None:
    with pytest.raises(AppError):
        _verifier().verify(_raw(_claims(), secret="a-different-secret-entirely-and-long-enough"))


def test_an_unsigned_alg_none_token_is_refused() -> None:
    """Pinning the algorithm is what stops the classic ``alg: none`` bypass."""
    unsigned = jwt.encode(_claims(), key="", algorithm="none")
    with pytest.raises(AppError):
        _verifier().verify(unsigned)


def test_a_token_using_a_different_algorithm_is_refused() -> None:
    other = jwt.encode(_claims(), _SECRET, algorithm="HS512")
    with pytest.raises(AppError):
        _verifier().verify(other)


def test_an_expired_token_is_refused() -> None:
    past = _now() - 10_000
    with pytest.raises(AppError):
        _verifier().verify(_raw(_claims(iat=past, nbf=past, exp=past + 60)))


def test_a_token_that_is_not_valid_yet_is_refused() -> None:
    future = _now() + 5_000
    with pytest.raises(AppError):
        _verifier().verify(_raw(_claims(nbf=future, exp=future + 900)))


@pytest.mark.parametrize("missing", ["exp", "iat", "nbf", "iss", "aud", "sub"])
def test_every_required_claim_is_required(missing: str) -> None:
    claims = _claims()
    del claims[missing]
    with pytest.raises(AppError):
        _verifier().verify(_raw(claims))


def test_a_token_for_another_issuer_or_audience_is_refused() -> None:
    with pytest.raises(AppError):
        _verifier().verify(_raw(_claims(iss="someone-else")))
    with pytest.raises(AppError):
        _verifier().verify(_raw(_claims(aud="another-service")))


def test_a_token_whose_lifetime_exceeds_the_limit_is_refused() -> None:
    """A long-lived token cannot be revoked, so the service caps the lifetime it will accept."""
    long_lived = _token(ttl_seconds=86_000)
    assert _verifier(max_ttl_seconds=86_400).verify(long_lived).subject == "ci"
    with pytest.raises(AppError):
        _verifier(max_ttl_seconds=900).verify(long_lived)


def test_a_non_string_subject_or_scope_is_refused() -> None:
    with pytest.raises(AppError):
        _verifier().verify(_raw(_claims(sub=12345)))
    with pytest.raises(AppError):
        _verifier().verify(_raw(_claims(scope=["build:run"])))
    with pytest.raises(AppError):
        _verifier().verify(_raw(_claims(sub="")))
    with pytest.raises(AppError):
        _verifier().verify(_raw(_claims(sub="x" * 200)))


def test_non_numeric_timestamps_are_refused() -> None:
    with pytest.raises(AppError):
        _verifier().verify(_raw(_claims(iat="soon", exp=_now() + 60)))


# --- configuration ------------------------------------------------------------------------------


def test_a_verifier_with_no_secret_is_not_configured_and_accepts_nothing() -> None:
    verifier = _verifier(secret=None)
    assert not verifier.configured
    with pytest.raises(AppError):
        verifier.verify(_token())


def test_a_configured_verifier_reports_itself_configured() -> None:
    assert _verifier().configured


def test_issue_token_normalizes_and_sorts_its_scopes() -> None:
    token = issue_token(
        _SECRET,
        subject="ci",
        scopes=["cache:import", Scope.BUILD_RUN, "cache:import"],
        issuer=_ISSUER,
        audience=_AUDIENCE,
    )
    claims = jwt.decode(token, _SECRET, algorithms=[ALGORITHM], audience=_AUDIENCE, issuer=_ISSUER)
    assert claims["scope"] == "build:run cache:import"


def test_issue_token_refuses_an_unknown_scope() -> None:
    with pytest.raises(ValueError, match="is not a valid Scope"):
        issue_token(_SECRET, subject="ci", scopes=["admin"], issuer=_ISSUER, audience=_AUDIENCE)


def test_two_tokens_for_the_same_subject_have_different_ids() -> None:
    first = _verifier().verify(_token())
    second = _verifier().verify(_token())
    assert first.token_id != second.token_id


def test_settings_refuse_a_jwt_secret_that_is_too_short_to_sign_with() -> None:
    from pydantic import ValidationError

    from incremental_agent_builds.core.config import Settings

    with pytest.raises(ValidationError, match="at least 32 bytes"):
        Settings(jwt_secret="too-short")
    assert Settings(jwt_secret=_SECRET).jwt_secret is not None
