"""HTTP API (FastAPI). Build an app with create_app(); there is no module-level app,
so nothing global is created at import time.

CORS: no CORS middleware is installed on purpose. The web page is served by
this same app (same origin), and without CORS headers browsers block other
sites from reading responses. That is the "same-origin only" policy.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from askdocs.chunking import chunk_pages
from askdocs.config import Settings, load_settings
from askdocs.embed import make_embedder
from askdocs.generate import Generator, answer_question, make_generator
from askdocs.ingest import (
    CorpusNotFoundError,
    InvalidDocumentNameError,
    file_sha256,
    ingest_corpus,
    read_pages,
    safe_doc_name,
)
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


class UploadResponse(BaseModel):
    """Body returned by PUT /documents/{filename}."""

    doc_name: str
    report: IngestReport


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

    def checked_name(filename: str) -> str:
        """Validate a file name from the URL; bad names are a 422 with the reason."""
        try:
            return safe_doc_name(filename)
        except InvalidDocumentNameError as exc:
            raise HTTPException(422, str(exc)) from exc

    def acquire_ingest_lock() -> threading.Lock:
        """Take the shared ingest lock without waiting, or answer 409."""
        lock: threading.Lock = app.state.ingest_lock
        if not lock.acquire(blocking=False):
            raise HTTPException(409, "An ingest is already running. Try again when it finishes.")
        return lock

    def save_upload(name: str, data: bytes) -> UploadResponse:
        """Check an uploaded file, store it in the corpus folder and index it.

        Everything that can be rejected is checked BEFORE anything is written,
        so a bad upload never leaves a file behind in the corpus folder.
        """
        if not data:
            raise HTTPException(422, "The uploaded file is empty.")
        pages, warnings = read_pages(name, data)
        if not chunk_pages(pages, settings.chunk_words, settings.chunk_overlap):
            detail = "No text could be extracted from this file, so it was not added."
            if warnings:
                detail += " " + "; ".join(warnings)
            raise HTTPException(422, detail)

        sha = file_sha256(data)
        lock = acquire_ingest_lock()
        try:
            for other, other_sha in store.document_hashes().items():
                if other_sha == sha and other != name:
                    raise HTTPException(409, f"This file has the same content as {other}.")
            target = settings.corpus_dir / name
            # Write to a hidden temporary name, then rename, so a half-written
            # file is never picked up by an ingest (hidden files are ignored).
            temp = settings.corpus_dir / f".{name}.part"
            try:
                settings.corpus_dir.mkdir(parents=True, exist_ok=True)
                temp.write_bytes(data)
                os.replace(temp, target)
            except OSError as exc:
                temp.unlink(missing_ok=True)
                raise HTTPException(
                    403,
                    "The corpus folder is not writable (is it mounted read-only?). "
                    "Add the file to the folder by hand instead.",
                ) from exc
            return UploadResponse(doc_name=name, report=ingest_corpus(settings, store, embedder))
        finally:
            lock.release()

    @app.put("/documents/{filename}")
    async def upload_document(filename: str, request: Request) -> UploadResponse:
        """Add or replace one document. The request body is the raw file bytes."""
        name = checked_name(filename)
        limit = settings.max_upload_mb * 1024 * 1024
        too_big = HTTPException(
            413, f"The file is larger than the {settings.max_upload_mb} MB upload limit."
        )
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > limit:
            raise too_big
        # Read the body in pieces so a client that lies about (or omits) its
        # size still cannot make the server hold more than the limit.
        data = bytearray()
        async for piece in request.stream():
            data += piece
            if len(data) > limit:
                raise too_big
        # Parsing, writing and embedding are blocking work: run them off the event loop.
        return await run_in_threadpool(save_upload, name, bytes(data))

    @app.delete("/documents/{filename}")
    def delete_document(filename: str) -> IngestReport:
        """Remove one document from the corpus folder and the index."""
        name = checked_name(filename)
        lock = acquire_ingest_lock()
        try:
            target = settings.corpus_dir / name
            if not target.is_file():
                raise HTTPException(404, f"{name} is not in the corpus folder.")
            try:
                target.unlink()
            except OSError as exc:
                raise HTTPException(
                    403, "The corpus folder is not writable (is it mounted read-only?)."
                ) from exc
            return ingest_corpus(settings, store, embedder)
        finally:
            lock.release()

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
        lock = acquire_ingest_lock()
        try:
            return ingest_corpus(settings, store, embedder)
        except CorpusNotFoundError as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            lock.release()

    return app
