"""Scoped bearer tokens for the build API.

The build service exposes three privileges and they are deliberately separate:

``build:run``
    plan and execute builds, export the local cache, inspect keys.
``build:approve``
    decide approval requests (a costly plan, a publish payload). The approval store already
    refuses self-approval, so a principal holding both scopes still cannot approve its own request.
``cache:import``
    import a signed shared-cache bundle. Import is the only way foreign data reaches the trace
    store, so it is the narrowest scope.

Tokens are HS256 JWTs signed with a shared secret. Verification is strict: the algorithm is pinned
(so an ``alg: none`` or RS256-with-the-public-key token is refused), ``exp``, ``iat``, ``nbf``,
``iss``, ``aud`` and ``sub`` are all required, and a token whose lifetime exceeds
``jwt_max_ttl_seconds`` is refused even if it is otherwise valid. Scopes are a space-separated
``scope`` claim, as in RFC 8693. Every failure returns the same 401 body, so a caller cannot tell
an expired token from a forged one.

Without a configured secret the service fails closed: scoped routes answer 503 rather than serving
requests unauthenticated.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any

import jwt
from fastapi import Depends, Request

from incremental_agent_builds.core.errors import AppError
from incremental_agent_builds.core.logging import get_logger

__all__ = [
    "ALGORITHM",
    "BearerPrincipal",
    "JwtVerifier",
    "Scope",
    "issue_token",
    "require_scope",
]

ALGORITHM = "HS256"
_MAX_TOKEN_CHARS = 4096
_log = get_logger("incremental_agent_builds.api.auth")


class Scope(StrEnum):
    BUILD_RUN = "build:run"
    BUILD_APPROVE = "build:approve"
    CACHE_IMPORT = "cache:import"


@dataclass(frozen=True, slots=True)
class BearerPrincipal:
    """An authenticated API caller: its subject and the scopes its token carries."""

    subject: str
    scopes: frozenset[Scope]
    token_id: str

    def has(self, scope: Scope) -> bool:
        return scope in self.scopes


def _unauthorized() -> AppError:
    return AppError(
        401,
        "unauthorized",
        "A valid bearer token is required.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def issue_token(
    secret: str,
    *,
    subject: str,
    scopes: Iterable[Scope | str],
    issuer: str,
    audience: str,
    ttl_seconds: int = 900,
    now: dt.datetime | None = None,
) -> str:
    """Mint a token. Used by the CLI (``iab token``) and by tests, never by the service itself."""
    issued = now or dt.datetime.now(dt.UTC)
    names = sorted({Scope(item).value for item in scopes})
    payload: dict[str, Any] = {
        "iss": issuer,
        "aud": audience,
        "sub": subject,
        "iat": int(issued.timestamp()),
        "nbf": int(issued.timestamp()),
        "exp": int((issued + dt.timedelta(seconds=ttl_seconds)).timestamp()),
        "jti": uuid.uuid4().hex,
        "scope": " ".join(names),
    }
    return jwt.encode(payload, secret, algorithm=ALGORITHM)


class JwtVerifier:
    """Verifies bearer tokens against one shared secret.

    ``configured`` is False when no secret is set, which makes the scoped routes answer 503.
    """

    def __init__(
        self, secret: str | None, *, issuer: str, audience: str, max_ttl_seconds: int
    ) -> None:
        self._secret = secret
        self._issuer = issuer
        self._audience = audience
        self._max_ttl = max_ttl_seconds

    @property
    def configured(self) -> bool:
        return bool(self._secret)

    def verify(self, token: str | None) -> BearerPrincipal:
        """Return the principal of a valid token, or raise a uniform 401."""
        if self._secret is None or not token or len(token) > _MAX_TOKEN_CHARS:
            raise _unauthorized()
        try:
            claims: dict[str, Any] = jwt.decode(
                token,
                self._secret,
                algorithms=[ALGORITHM],
                issuer=self._issuer,
                audience=self._audience,
                options={
                    "require": ["exp", "iat", "nbf", "iss", "aud", "sub"],
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_nbf": True,
                    "verify_iss": True,
                    "verify_aud": True,
                },
            )
        except jwt.InvalidTokenError as exc:
            _log.info("token_rejected", reason=type(exc).__name__)
            raise _unauthorized() from exc
        issued, expires = claims.get("iat"), claims.get("exp")
        if not isinstance(issued, int | float) or not isinstance(expires, int | float):
            raise _unauthorized()
        if expires - issued > self._max_ttl:
            _log.info("token_rejected", reason="ttl_above_limit")
            raise _unauthorized()
        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject or len(subject) > 128:
            raise _unauthorized()
        raw_scope = claims.get("scope", "")
        if not isinstance(raw_scope, str):
            raise _unauthorized()
        scopes: set[Scope] = set()
        for name in raw_scope.split():
            try:
                scopes.add(Scope(name))
            except ValueError:
                continue  # unknown scope names grant nothing
        token_id = claims.get("jti")
        return BearerPrincipal(
            subject=subject,
            scopes=frozenset(scopes),
            token_id=token_id if isinstance(token_id, str) else "",
        )


def _bearer_token(request: Request) -> str | None:
    header = request.headers.get("authorization")
    if header and header[:7].lower() == "bearer ":
        return header[7:].strip() or None
    return None


def require_scope(*required: Scope) -> Any:
    """FastAPI dependency: authenticate the bearer token and require every listed scope."""
    wanted: Sequence[Scope] = tuple(required)

    def dependency(request: Request) -> BearerPrincipal:
        verifier: JwtVerifier = request.app.state.jwt_verifier
        if not verifier.configured:
            raise AppError(
                503,
                "auth_not_configured",
                "Token authentication is not configured on this deployment.",
            )
        try:
            principal = verifier.verify(_bearer_token(request))
        except AppError:
            request.app.state.metrics.auth_failures_total.inc()
            raise
        missing = [scope.value for scope in wanted if not principal.has(scope)]
        if missing:
            request.app.state.metrics.auth_failures_total.inc()
            _log.info("scope_denied", subject=principal.subject, missing=missing)
            raise AppError(403, "forbidden", f"Missing scope: {', '.join(missing)}.")
        return principal

    return Depends(dependency)


BuildRunner = Annotated[BearerPrincipal, require_scope(Scope.BUILD_RUN)]
BuildApprover = Annotated[BearerPrincipal, require_scope(Scope.BUILD_APPROVE)]
CacheImporter = Annotated[BearerPrincipal, require_scope(Scope.CACHE_IMPORT)]
