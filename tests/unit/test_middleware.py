"""Direct ASGI tests for header edge cases an HTTP client library will not send."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest
from prometheus_client import CollectorRegistry
from starlette.types import Message, Receive, Scope, Send

from incremental_agent_builds.api.auth import JwtVerifier, issue_token
from incremental_agent_builds.api.auth import Scope as TokenScope
from incremental_agent_builds.core.metrics import create_http_metrics
from incremental_agent_builds.core.middleware import BodySizeLimitMiddleware, RateLimitMiddleware
from incremental_agent_builds.core.ratelimit import RateLimiter
from incremental_agent_builds.core.security import ApiKeyVerifier, hash_api_key

pytestmark = pytest.mark.unit

_SECRET = "middleware-rate-limit-secret-value"


async def _unreachable_app(scope: Scope, receive: Receive, send: Send) -> None:
    raise AssertionError("the application must not be reached")


@pytest.mark.parametrize("declared", [b"\xb2", b"\xb9\xb2", b"12x", b"-1"])
def test_malformed_content_length_is_rejected_instead_of_crashing(declared: bytes) -> None:
    # Starlette decodes header bytes as latin-1, where b"\xb2" is "²": str.isdigit() accepts
    # it but int() does not.
    sent: list[Message] = []

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        sent.append(message)

    middleware = BodySizeLimitMiddleware(
        _unreachable_app, max_bytes=10, metrics=create_http_metrics(CollectorRegistry())
    )
    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/redact",
        "headers": [(b"content-length", declared)],
    }
    asyncio.run(middleware(scope, receive, send))
    assert sent[0]["status"] == 413


async def _ok_app(scope: Scope, receive: Receive, send: Send) -> None:
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b""})


def _statuses_for_clients(hosts: list[str]) -> list[int]:
    """Send one unauthenticated request per client address through a burst-1 rate limiter."""
    middleware = RateLimitMiddleware(
        _ok_app,
        limiter=RateLimiter(rate=0.001, burst=1),
        verifier=ApiKeyVerifier([]),
        metrics=create_http_metrics(CollectorRegistry()),
    )
    sent: list[Message] = []

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        sent.append(message)

    statuses: list[int] = []
    for host in hosts:
        sent.clear()
        scope: Scope = {
            "type": "http",
            "method": "GET",
            "path": "/v1/items",
            "headers": [],
            "client": (host, 50000),
        }
        asyncio.run(middleware(scope, receive, send))
        statuses.append(int(sent[0]["status"]))
    return statuses


def test_ipv6_clients_are_limited_per_64_prefix() -> None:
    # One IPv6 host usually controls a whole /64; per-address buckets would give it 2**64 bursts.
    hosts = ["2001:db8:1:2::1", "2001:db8:1:2:ffff:ffff:ffff:9", "2001:db8:1:3::1"]
    assert _statuses_for_clients(hosts) == [200, 429, 200]


def test_ipv4_mapped_ipv6_clients_share_the_ipv4_bucket() -> None:
    assert _statuses_for_clients(["192.0.2.7", "::ffff:192.0.2.7", "192.0.2.8"]) == [200, 429, 200]


# --- per-caller buckets -------------------------------------------------------------------------
#
# The routes that cost money authenticate with scoped bearer tokens, and those are checked by
# FastAPI dependencies, which run *after* this middleware. If the limiter could not see the token,
# every caller behind one proxy would share one bucket and every caller on its own address would get
# a whole one, which is not what "rate limits" next to "JWT scopes" claims.


def _verifier() -> JwtVerifier:
    return JwtVerifier(_SECRET, issuer="iab", audience="iab-api", max_ttl_seconds=900)


def _token(subject: str) -> str:
    return issue_token(
        _SECRET,
        subject=subject,
        scopes=[TokenScope.BUILD_RUN],
        issuer="iab",
        audience="iab-api",
    )


def _statuses_for_headers(
    requests: list[list[tuple[bytes, bytes]]], *, api_keys: Sequence[str] = ()
) -> list[int]:
    """One request per header set, all from the same address, through a burst-1 limiter."""
    middleware = RateLimitMiddleware(
        _ok_app,
        limiter=RateLimiter(rate=0.001, burst=1),
        verifier=ApiKeyVerifier([hash_api_key(key) for key in api_keys]),
        metrics=create_http_metrics(CollectorRegistry()),
        bearer_subject=_verifier().subject_of,
    )
    sent: list[Message] = []

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        sent.append(message)

    statuses: list[int] = []
    for headers in requests:
        sent.clear()
        scope: Scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/builds",
            "headers": headers,
            "client": ("198.51.100.4", 50000),
        }
        asyncio.run(middleware(scope, receive, send))
        statuses.append(int(sent[0]["status"]))
    return statuses


def _bearer(token: str) -> list[tuple[bytes, bytes]]:
    return [(b"authorization", b"Bearer " + token.encode("ascii"))]


def test_two_token_subjects_from_one_address_get_independent_buckets() -> None:
    alice, bob = _bearer(_token("alice")), _bearer(_token("bob"))
    assert _statuses_for_headers([alice, alice, bob]) == [200, 429, 200]


def test_two_tokens_for_one_subject_share_a_bucket() -> None:
    """A fresh token is not a fresh bucket: the subject is the identity, not the token."""
    first, second = _bearer(_token("alice")), _bearer(_token("alice"))
    assert first != second, "the tokens differ (each carries its own jti)"
    assert _statuses_for_headers([first, second]) == [200, 429]


def test_an_invalid_token_shares_the_address_bucket() -> None:
    """An unverifiable token must not be able to mint buckets."""
    forged = _bearer("not.a.token")
    other = _bearer(
        issue_token(
            "a-different-signing-secret-value",
            subject="alice",
            scopes=[TokenScope.BUILD_RUN],
            issuer="iab",
            audience="iab-api",
        )
    )
    assert _statuses_for_headers([forged, other]) == [200, 429]


def test_an_api_key_is_used_when_there_is_no_bearer_token() -> None:
    key = "iab_" + "k" * 40
    headers = [(b"x-api-key", key.encode("ascii"))]
    assert _statuses_for_headers([headers, headers], api_keys=[key]) == [200, 429]
