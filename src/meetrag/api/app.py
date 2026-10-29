"""FastAPI service: JSON and Server-Sent Events endpoints, plus a static demo UI."""

from __future__ import annotations

import json
import logging
import secrets
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import anthropic
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from meetrag import __version__
from meetrag.api.schemas import AskRequest, Readiness, SourceInfo
from meetrag.config import get_settings
from meetrag.container import Container
from meetrag.domain.models import AnswerResult, Source
from meetrag.domain.ports import LLMConfigurationError, LLMUnavailableError
from meetrag.observability import configure_logging

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from fastapi import Response

logger = logging.getLogger("meetrag.api")
STATIC_DIR = Path(__file__).parent / "static"


def _unavailable_detail(exc: LLMUnavailableError) -> str:
    if isinstance(exc, LLMConfigurationError):
        return f"LLM provider not configured: {exc}"
    return f"LLM provider unavailable: {exc}"


def create_app(container: Container | None = None) -> FastAPI:
    container = container or Container(get_settings())
    settings = container.settings
    configure_logging(settings.log_level, json_logs=settings.log_json)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if settings.auto_ingest and not container.is_ready:
            logger.info("index not found: running ingestion")
            container.ingest()
        graph = container.graph  # fail fast on configuration errors
        logger.info("graph %s ready", graph.name)
        yield
        container.close()

    app = FastAPI(
        title="MeetRAG",
        version=__version__,
        description="Multi-agent, multi-source RAG assistant with cited, verified answers.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        logger.info(
            "%s %s -> %d",
            request.method,
            request.url.path,
            response.status_code,
            extra={
                "extra": {
                    "request_id": request_id,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                }
            },
        )
        return response

    @app.exception_handler(anthropic.APIError)
    async def upstream_error(_: Request, exc: anthropic.APIError) -> JSONResponse:
        logger.error("model provider error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"detail": "The language model provider returned an error. Try again."},
        )

    @app.exception_handler(LLMUnavailableError)
    async def unavailable(_: Request, exc: LLMUnavailableError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": _unavailable_detail(exc)},
        )

    def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
        expected = settings.api_key
        if expected and not secrets.compare_digest(x_api_key or "", expected.get_secret_value()):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing X-API-Key")

    protected = [Depends(require_api_key)]

    @app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/healthz", tags=["ops"])
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/readyz", tags=["ops"], response_model=Readiness)
    async def readyz() -> JSONResponse:
        chunks = container.store.count()
        ready = container.is_ready
        return JSONResponse(
            status_code=status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "ready" if ready else "not ready", "chunks": chunks},
        )

    @app.get("/v1/sources", tags=["assistant"], dependencies=protected)
    async def sources() -> list[SourceInfo]:
        counts = Counter(chunk.source for chunk in container.bm25.chunks)
        return [
            SourceInfo(
                name=source,
                kind="documents" if source.is_document_source else "sql",
                chunks=counts.get(source) if source.is_document_source else None,
            )
            for source in Source
        ]

    @app.post("/v1/ask", tags=["assistant"], dependencies=protected)
    async def ask(payload: AskRequest) -> AnswerResult:
        return await container.assistant.ask(payload.question)

    @app.post("/v1/ask/stream", tags=["assistant"], dependencies=protected)
    async def ask_stream(payload: AskRequest) -> StreamingResponse:
        async def events() -> AsyncIterator[str]:
            try:
                async for event in container.assistant.stream(payload.question):
                    yield f"event: {event['type']}\ndata: {json.dumps(event, default=str)}\n\n"
            except (anthropic.APIError, LLMUnavailableError) as exc:
                logger.error("model provider error during stream: %s", exc)
                detail = (
                    _unavailable_detail(exc)
                    if isinstance(exc, LLMUnavailableError)
                    else "The language model provider returned an error."
                )
                error = {"type": "error", "detail": detail}
                yield f"event: error\ndata: {json.dumps(error)}\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
