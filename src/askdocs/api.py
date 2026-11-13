"""HTTP API (FastAPI). Build an app with create_app(); there is no module-level app,
so nothing global is created at import time.

CORS: no CORS middleware is installed on purpose. The web page is served by
this same app (same origin), and without CORS headers browsers block other
sites from reading responses. That is the "same-origin only" policy.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from askdocs.config import Settings, load_settings
from askdocs.embed import make_embedder
from askdocs.generate import Generator, answer_question, make_generator
from askdocs.ingest import CorpusNotFoundError, ingest_corpus
from askdocs.models import Answer, Document, IngestReport
from askdocs.retrieve import Mode, Retriever
from askdocs.store import EmbedderMismatchError, Store

log = logging.getLogger("askdocs.api")
WEB_PAGE = Path(__file__).parent / "web" / "index.html"


class AskRequest(BaseModel):
    """Body of POST /ask."""

    question: str
    k: int | None = Field(default=None, ge=1, le=50)
    mode: Mode = "hybrid"


class Health(BaseModel):
    """Body of GET /health."""

    status: str
    documents: int
    chunks: int
    embedder: str
    generator: str


def create_app(settings: Settings | None = None, generator: Generator | None = None) -> FastAPI:
    """Wire settings, store, embedder, retriever and generator into a FastAPI app."""
    settings = settings or load_settings()
    store = Store(settings.db_path)
    embedder = make_embedder(settings)
    retriever = Retriever(settings, store, embedder)
    generator = generator or make_generator(settings)

    app = FastAPI(title="AskDocs", version="0.1.0")
    # Kept on app.state (not a global) so each app has its own lock and tests can reach it.
    app.state.ingest_lock = threading.Lock()

    @app.middleware("http")
    async def log_request(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """One log line per request. Never logs the question or any document text."""
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        extra = getattr(request.state, "log_extra", "")
        log.info("%s %s %d %.1fms%s", request.method, request.url.path,
                 response.status_code, elapsed, extra)
        return response

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        """Serve the single-page web UI."""
        return FileResponse(WEB_PAGE, media_type="text/html")

    @app.get("/health")
    def health() -> Health:
        """Liveness plus index size and the active backends."""
        documents, chunks = store.counts()
        return Health(status="ok", documents=documents, chunks=chunks,
                      embedder=embedder.name, generator=generator.name)

    @app.get("/documents")
    def documents() -> list[Document]:
        """Every ingested document with page and chunk counts."""
        return store.list_documents()

    @app.post("/ask")
    def ask(body: AskRequest, request: Request) -> Answer:
        """Answer a question with citations, or abstain."""
        question = body.question.strip()
        if not question:
            raise HTTPException(422, "The question is empty. Type a question and try again.")
        if len(question) > settings.max_question_chars:
            raise HTTPException(
                422,
                f"The question is {len(question)} characters long; the limit is "
                f"{settings.max_question_chars}. Please shorten it.",
            )
        try:
            answer = answer_question(question, settings, retriever, generator,
                                     mode=body.mode, k=body.k)
        except EmbedderMismatchError as exc:
            raise HTTPException(409, str(exc)) from exc
        request.state.log_extra = (
            f" mode={body.mode} abstained={answer.abstained} latency_ms={answer.latency_ms}"
        )
        return answer

    @app.post("/ingest")
    def ingest() -> IngestReport:
        """Re-scan the corpus folder. Returns 409 if an ingest is already running."""
        lock: threading.Lock = app.state.ingest_lock
        if not lock.acquire(blocking=False):
            raise HTTPException(409, "An ingest is already running. Try again when it finishes.")
        try:
            return ingest_corpus(settings, store, embedder)
        except CorpusNotFoundError as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            lock.release()

    return app
