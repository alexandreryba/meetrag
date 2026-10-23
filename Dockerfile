# syntax=docker/dockerfile:1.7

# ---- Build: dependencies, project, and the pre-built index -----------------------------------
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed

WORKDIR /app

# Dependencies first, in their own layer: rebuilt only when the lockfile changes.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-install-project --extra vertex

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY data ./data
COPY eval ./eval
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra vertex

# Bake the embedding model, the vector index and the analytics database into the image:
# a cold start on Cloud Run downloads nothing and serves the first request immediately.
RUN /app/.venv/bin/polyrag ingest

# ---- Runtime: slim, non-root ----------------------------------------------------------------
FROM python:3.12-slim AS runtime

RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=builder --chown=app:app /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    POLYRAG_LOG_JSON=true \
    POLYRAG_AUTO_INGEST=false \
    PORT=8080

USER app
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz')"

CMD ["sh", "-c", "exec uvicorn polyrag.api.app:create_app --factory --host 0.0.0.0 --port ${PORT}"]
