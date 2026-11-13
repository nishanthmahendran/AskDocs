"""Data types shared across modules, plus the project's base exception.

Pydantic models are used everywhere so the same objects can be returned
directly as JSON by the API and printed by the CLI with --json.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskDocsError(Exception):
    """Base class for expected, user-facing errors (bad config, missing folder, ...)."""


class Page(BaseModel):
    """The normalized text of one page of one document (pages are 1-based)."""

    doc_name: str
    page: int
    text: str


class Chunk(BaseModel):
    """A passage of text that is indexed and can be cited."""

    chunk_id: str
    doc_name: str
    page: int
    index: int = Field(description="Position of this chunk within its page, starting at 0.")
    text: str


class Document(BaseModel):
    """One ingested file as recorded in the index."""

    doc_name: str
    sha256: str
    pages: int
    chunks: int
    ingested_at: str


class SearchHit(BaseModel):
    """A retrieved chunk with every intermediate rank and score kept for debugging."""

    chunk: Chunk
    keyword_rank: int | None = None
    vector_rank: int | None = None
    vector_score: float | None = None
    fused_score: float = 0.0


class Citation(BaseModel):
    """A source shown to the user next to an answer."""

    doc_name: str
    page: int
    chunk_id: str
    snippet: str
    score: float | None = Field(
        default=None, description="Cosine similarity between the question and the chunk."
    )


class Answer(BaseModel):
    """The full result of asking one question."""

    answer: str
    abstained: bool
    abstain_reason: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    hits: list[SearchHit] = Field(default_factory=list)
    latency_ms: float = 0.0
    backend: str = ""
    warnings: list[str] = Field(default_factory=list)


class IngestReport(BaseModel):
    """Counts from one ingest run, plus any warnings (scanned pages, bad PDFs, ...)."""

    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    skipped: int = 0
    warnings: list[str] = Field(default_factory=list)
