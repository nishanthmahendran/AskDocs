"""SQLite storage: documents, chunks, an FTS5 keyword index, vectors and metadata.

Only the standard-library sqlite3 module is used. Every query is
parameterized; user text only ever reaches SQL as a bound parameter.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from askdocs.models import AskDocsError, Chunk, Document
from askdocs.text import content_terms

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_name    TEXT PRIMARY KEY,
    sha256      TEXT NOT NULL,
    pages       INTEGER NOT NULL,
    ingested_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_name TEXT NOT NULL,
    page     INTEGER NOT NULL,
    idx      INTEGER NOT NULL,
    text     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_by_doc ON chunks(doc_name);
-- Keyword index. Each FTS row uses the same rowid as its row in `chunks`.
-- remove_diacritics 2 makes "cafe" match "café".
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, tokenize = 'unicode61 remove_diacritics 2'
);
CREATE TABLE IF NOT EXISTS vectors (
    chunk_id TEXT PRIMARY KEY,
    vec      BLOB NOT NULL  -- little-endian float32, L2-normalized
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

VECTOR_DTYPE = np.dtype("<f4")


class StoreError(AskDocsError):
    """The database cannot be used (for example, SQLite was built without FTS5)."""


class EmbedderMismatchError(AskDocsError):
    """The index was built with a different embedder than the one configured now."""


def check_fts5() -> None:
    """Raise StoreError if this Python's SQLite does not include FTS5."""
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE VIRTUAL TABLE fts5_probe USING fts5(x)")
    except sqlite3.OperationalError as exc:
        raise StoreError(
            f"Your Python's SQLite ({sqlite3.sqlite_version}) does not include FTS5, "
            "which AskDocs needs for keyword search. Use the official python.org "
            "or Docker python image, which include it."
        ) from exc
    finally:
        conn.close()


def fts_query(question: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression, or "" if no terms are left.

    Only word characters survive content_terms(), so FTS5 operators and quotes
    (AND, OR, NOT, NEAR, *, ", parentheses, ...) cannot reach the query. Each
    term is still wrapped in double quotes so a word like "and" or "near" is
    treated as a plain string. Terms are joined with OR so a passage does not
    need every word; bm25 ranks passages that match more terms higher.
    """
    return " OR ".join(f'"{term}"' for term in content_terms(question))


def now_iso() -> str:
    """Current UTC time as an ISO-8601 string (seconds precision)."""
    return datetime.now(UTC).isoformat(timespec="seconds")


class Store:
    """All database access. Opens a short-lived connection per operation."""

    def __init__(self, db_path: Path) -> None:
        """Check FTS5, create the folder and the schema if needed."""
        check_fts5()
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection; commit on success, roll back on error, always close.

        WAL mode lets the API read while an ingest is writing. A fresh
        connection per call keeps this safe across FastAPI's worker threads.
        """
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ----- metadata -------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        """Read one metadata value."""
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _set_meta(self, conn: sqlite3.Connection, key: str, value: str) -> None:
        """Write one metadata value inside an existing transaction."""
        conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def _bump_version(self, conn: sqlite3.Connection) -> None:
        """Increase index_version so cached vectors know they are stale."""
        current = conn.execute("SELECT value FROM meta WHERE key = 'index_version'").fetchone()
        self._set_meta(conn, "index_version", str(int(current[0]) + 1 if current else 1))

    def index_version(self) -> int:
        """A counter that changes every time the index content changes."""
        return int(self.get_meta("index_version") or 0)

    def set_index_info(self, info: dict[str, str]) -> None:
        """Record what built this index (embedder, dim, chunk settings, ingest version)."""
        with self.connect() as conn:
            for key, value in info.items():
                self._set_meta(conn, key, value)

    def index_matches(self, info: dict[str, str]) -> bool:
        """True if every recorded value equals the one in `info`."""
        return all(self.get_meta(key) == value for key, value in info.items())

    def check_embedder(self, embedder: str, dim: int) -> None:
        """Raise EmbedderMismatchError if a non-empty index used another embedder."""
        if self.counts()[1] == 0:
            return
        built_with, built_dim = self.get_meta("embedder"), self.get_meta("dim")
        if built_with != embedder or built_dim != str(dim):
            raise EmbedderMismatchError(
                f"The index was built with embedder {built_with!r} (dim {built_dim}) but the "
                f"current embedder is {embedder!r} (dim {dim}). "
                "Run `askdocs ingest` to rebuild the index with the current embedder."
            )

    # ----- writing ---------------------------------------------------------

    def _delete_document(self, conn: sqlite3.Connection, doc_name: str) -> None:
        """Delete a document and all its chunks, keyword rows and vectors."""
        conn.execute(
            "DELETE FROM chunks_fts WHERE rowid IN (SELECT rowid FROM chunks WHERE doc_name = ?)",
            (doc_name,),
        )
        conn.execute(
            "DELETE FROM vectors "
            "WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE doc_name = ?)",
            (doc_name,),
        )
        conn.execute("DELETE FROM chunks WHERE doc_name = ?", (doc_name,))
        conn.execute("DELETE FROM documents WHERE doc_name = ?", (doc_name,))

    def replace_document(
        self, doc_name: str, sha256: str, pages: int, chunks: list[Chunk], vectors: np.ndarray
    ) -> None:
        """Atomically replace everything stored for doc_name."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        with self.connect() as conn:
            self._delete_document(conn, doc_name)
            conn.execute(
                "INSERT INTO documents(doc_name, sha256, pages, ingested_at) VALUES (?, ?, ?, ?)",
                (doc_name, sha256, pages, now_iso()),
            )
            for chunk, vec in zip(chunks, vectors, strict=True):
                cur = conn.execute(
                    "INSERT INTO chunks(chunk_id, doc_name, page, idx, text) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (chunk.chunk_id, chunk.doc_name, chunk.page, chunk.index, chunk.text),
                )
                conn.execute(
                    "INSERT INTO chunks_fts(rowid, text) VALUES (?, ?)",
                    (cur.lastrowid, chunk.text),
                )
                conn.execute(
                    "INSERT INTO vectors(chunk_id, vec) VALUES (?, ?)",
                    (chunk.chunk_id, np.asarray(vec, dtype=VECTOR_DTYPE).tobytes()),
                )
            self._bump_version(conn)

    def delete_document(self, doc_name: str) -> None:
        """Remove one document from the index."""
        with self.connect() as conn:
            self._delete_document(conn, doc_name)
            self._bump_version(conn)

    def clear(self) -> None:
        """Remove all documents (metadata other than the version is kept)."""
        with self.connect() as conn:
            for table in ("chunks_fts", "vectors", "chunks", "documents"):
                conn.execute(f"DELETE FROM {table}")  # table names are constants
            self._bump_version(conn)

    # ----- reading ---------------------------------------------------------

    def counts(self) -> tuple[int, int]:
        """Return (number of documents, number of chunks)."""
        with self.connect() as conn:
            docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return int(docs), int(chunks)

    def document_hashes(self) -> dict[str, str]:
        """Map doc_name -> sha256 for every indexed document."""
        with self.connect() as conn:
            rows = conn.execute("SELECT doc_name, sha256 FROM documents").fetchall()
        return dict(rows)

    def list_documents(self) -> list[Document]:
        """All documents with their page and chunk counts, sorted by name."""
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT d.doc_name, d.sha256, d.pages, d.ingested_at, COUNT(c.chunk_id) "
                "FROM documents d LEFT JOIN chunks c ON c.doc_name = d.doc_name "
                "GROUP BY d.doc_name ORDER BY d.doc_name"
            ).fetchall()
        return [
            Document(doc_name=r[0], sha256=r[1], pages=r[2], ingested_at=r[3], chunks=r[4])
            for r in rows
        ]

    def keyword_search(self, question: str, limit: int) -> list[tuple[str, float]]:
        """Return (chunk_id, bm25) best first; ties broken by chunk_id. Lower bm25 is better."""
        match = fts_query(question)
        if not match:
            return []
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT c.chunk_id, bm25(chunks_fts) AS score "
                "FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid "
                "WHERE chunks_fts MATCH ? ORDER BY score, c.chunk_id LIMIT ?",
                (match, limit),
            ).fetchall()
        return [(r[0], float(r[1])) for r in rows]

    def load_vectors(self) -> tuple[list[str], np.ndarray]:
        """All (chunk_ids, matrix) sorted by chunk_id; matrix is float32 [n, dim]."""
        with self.connect() as conn:
            rows = conn.execute("SELECT chunk_id, vec FROM vectors ORDER BY chunk_id").fetchall()
        if not rows:
            return [], np.zeros((0, 0), dtype=np.float32)
        ids = [r[0] for r in rows]
        matrix = np.vstack([np.frombuffer(r[1], dtype=VECTOR_DTYPE) for r in rows])
        return ids, matrix.astype(np.float32)

    def get_vectors(self, chunk_ids: list[str]) -> dict[str, np.ndarray]:
        """Vectors for specific chunks."""
        if not chunk_ids:
            return {}
        # The placeholders string is built from a count, never from user text.
        marks = ",".join("?" * len(chunk_ids))
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT chunk_id, vec FROM vectors WHERE chunk_id IN ({marks})", chunk_ids
            ).fetchall()
        return {r[0]: np.frombuffer(r[1], dtype=VECTOR_DTYPE).astype(np.float32) for r in rows}

    def get_chunks(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        """Chunks by id."""
        if not chunk_ids:
            return {}
        marks = ",".join("?" * len(chunk_ids))
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT chunk_id, doc_name, page, idx, text FROM chunks "
                f"WHERE chunk_id IN ({marks})",
                chunk_ids,
            ).fetchall()
        return {
            r[0]: Chunk(chunk_id=r[0], doc_name=r[1], page=r[2], index=r[3], text=r[4])
            for r in rows
        }
