"""Typed configuration loaded from environment variables (prefix ``APP_``).

Only ``.env.example`` is committed. Secrets are ``SecretStr`` so they never appear in ``repr``
or logs. API keys are configured as SHA-256 digests, so the process never holds raw keys.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

__all__ = ["Settings", "get_settings"]

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_ORIGIN = re.compile(r"^https?://[A-Za-z0-9.-]+(:\d{1,5})?$")
_MIN_JWT_SECRET_BYTES = 32


def _split_csv(value: object) -> object:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


class Settings(BaseSettings):
    """Service settings. Every field can be set with ``APP_<FIELD_NAME>``."""

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- service ---
    service_name: str = Field(
        default="incremental-agent-builds", pattern=r"^[a-z0-9][a-z0-9-]{0,62}$"
    )
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    enable_docs: bool = False

    # --- HTTP security ---
    api_key_sha256: Annotated[list[str], NoDecode] = Field(default_factory=list)
    cors_allow_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    rate_limit_per_second: float = Field(default=5.0, gt=0)
    rate_limit_burst: int = Field(default=20, ge=1)
    rate_limit_max_clients: int = Field(default=10_000, ge=1)
    max_body_bytes: int = Field(default=1_048_576, ge=1)
    metrics_require_auth: bool = True

    # --- observability ---
    tracing_exporter: Literal["none", "console"] = "none"

    # --- agentic kit ---
    anthropic_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("ANTHROPIC_API_KEY", "anthropic_api_key")
    )
    llm_model: str = "claude-opus-5"
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    llm_max_retries: int = Field(default=2, ge=0, le=10)
    # The SDK default. Non-streaming requests up to 16K output tokens can legitimately take minutes;
    # a shorter total timeout cuts them off, and each retry can be billed again.
    llm_timeout_seconds: float = Field(default=600.0, gt=0)
    llm_refusal_fallback: bool = True
    budget_max_steps: int = Field(default=50, ge=0)
    budget_max_tokens: int = Field(default=200_000, ge=0)
    budget_max_usd: float = Field(default=1.0, ge=0)

    # --- storage ---
    data_dir: Path = Path("var")

    # --- JWT bearer tokens for the build API ---
    # HS256 with a shared secret. Without a secret the token routes fail closed (503), so a
    # misconfigured deployment refuses build requests instead of serving them unauthenticated.
    jwt_secret: SecretStr | None = None
    jwt_issuer: str = "incremental-agent-builds"
    jwt_audience: str = "incremental-agent-builds"
    jwt_max_ttl_seconds: int = Field(default=3600, ge=1, le=86_400)

    # --- build engine ---
    workspace_dir: Path = Path("var/workspace")
    store_dir: Path = Path("var/store")
    build_jobs: int = Field(default=4, ge=1, le=64)
    build_llm_concurrency: int = Field(default=4, ge=1, le=64)
    build_guard_mode: Literal["off", "warn", "strict"] = "warn"
    # The store MAC key (base64, >= 32 bytes). Generated and persisted under data_dir when unset.
    store_mac_key: SecretStr | None = None
    # Ed25519 seed (base64, 32 bytes) used to sign exported shared-cache entries.
    cache_signing_seed: SecretStr | None = None
    # Comma-separated base64 Ed25519 public keys whose cache entries may be imported.
    cache_trusted_keys: Annotated[list[str], NoDecode] = Field(default_factory=list)
    # A plan projected above this many micro-USD needs an approval before it may execute.
    plan_approval_usd_micros: int = Field(default=250_000, ge=0)
    # Whether a plan whose cost could not be estimated (a first build, or new keys of a rule that
    # has never run) also needs an approval. Off by default: a cold build has no recorded usage to
    # price, and the per-build budget (APP_BUDGET_MAX_*) is what bounds it. Turn it on where no
    # unpriceable spend may happen without a human, at the cost of gating every first build.
    plan_approve_unestimated: bool = False
    approval_ttl_seconds: float = Field(default=3600.0, gt=0)

    @field_validator("api_key_sha256", mode="before")
    @classmethod
    def _parse_key_hashes(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator("api_key_sha256")
    @classmethod
    def _check_key_hashes(cls, value: list[str]) -> list[str]:
        for digest in value:
            if not _SHA256_HEX.fullmatch(digest):
                raise ValueError("api_key_sha256 entries must be lowercase hex SHA-256 digests")
        return value

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _parse_origins(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator("cors_allow_origins")
    @classmethod
    def _check_origins(cls, value: list[str]) -> list[str]:
        for origin in value:
            if not _ORIGIN.fullmatch(origin):
                raise ValueError(
                    "cors_allow_origins must be explicit scheme://host[:port] origins "
                    "('*' is refused)"
                )
        return value

    @field_validator("jwt_secret")
    @classmethod
    def _check_jwt_secret(cls, value: SecretStr | None) -> SecretStr | None:
        # RFC 7518 3.2: an HS256 key should be at least as long as the hash output. A short shared
        # secret is brute-forceable offline from a single captured token.
        secret = value.get_secret_value() if value is not None else ""
        if secret and len(secret.encode("utf-8")) < _MIN_JWT_SECRET_BYTES:
            raise ValueError(
                f"jwt_secret must be at least {_MIN_JWT_SECRET_BYTES} bytes (RFC 7518 3.2)"
            )
        return value

    @field_validator("cache_trusted_keys", mode="before")
    @classmethod
    def _parse_trusted_keys(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator("cache_trusted_keys")
    @classmethod
    def _check_trusted_keys(cls, value: list[str]) -> list[str]:
        from incremental_agent_builds.engine.signing import TrustStore  # noqa: PLC0415

        TrustStore(value)  # raises ValueError on a key that is not 32 base64 bytes
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings (cached). Tests construct ``Settings(...)`` directly instead."""
    return Settings()
