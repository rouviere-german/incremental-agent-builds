# syntax=docker/dockerfile:1
#
# Multi-stage, non-root image. Base images are pinned by digest (resolved 2026-09-13); Dependabot's
# docker ecosystem proposes digest bumps. python:3.12-slim is multi-arch, so the index digest works
# on amd64 and arm64.

ARG PYTHON_IMAGE=python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

FROM ghcr.io/astral-sh/uv:0.13.1@sha256:4a1f3666816dafadd20ceb71eac9a7d56d7ef30279f7d64048519ab901ada336 AS uv

FROM ${PYTHON_IMAGE} AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3.12 \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /build
# Dependencies first, so source edits do not invalidate the dependency layer.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM ${PYTHON_IMAGE} AS runtime
LABEL org.opencontainers.image.title="incremental-agent-builds" \
      org.opencontainers.image.licenses="MIT"
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --home-dir /nonexistent \
       --shell /usr/sbin/nologin app
COPY --from=builder /opt/venv /opt/venv
# The service needs somewhere to write: the SQLite trace store, the CAS, the audit log, the approval
# file and the generated site. The root filesystem is read-only in production (see
# docker-compose.yml), so all of it lives under one directory that is mounted as a volume. It is
# created here owned by the runtime user, so the image also starts without a volume for a smoke test.
RUN mkdir -p /var/lib/iab/workspace /var/lib/iab/store /var/lib/iab/data \
    && chown -R 10001:10001 /var/lib/iab
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    APP_WORKSPACE_DIR=/var/lib/iab/workspace \
    APP_STORE_DIR=/var/lib/iab/store \
    APP_DATA_DIR=/var/lib/iab/data
WORKDIR /app
VOLUME ["/var/lib/iab"]
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-m", "incremental_agent_builds.core.healthcheck"]
ENTRYPOINT ["python", "-m", "incremental_agent_builds"]
CMD ["serve"]
