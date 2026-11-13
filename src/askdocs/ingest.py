"""Read files from the corpus folder and keep the index in sync with them.

Two halves:
  * extraction: read_pages() turns one file into normalized Page objects;
  * syncing:    ingest_corpus() adds new files, replaces changed ones and
                removes deleted ones, using a SHA-256 of each file's bytes.
"""

from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path

from pypdf import PdfReader

from askdocs.chunking import chunk_pages
from askdocs.config import Settings
from askdocs.embed import Embedder
from askdocs.models import AskDocsError, IngestReport, Page
from askdocs.store import Store
from askdocs.text import normalize_text, strip_markdown

log = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt"}
# A page with less text than this is probably a scanned image.
MIN_PAGE_CHARS = 20
# corpus/README.md describes the sources; it is metadata, not a document.
CORPUS_README = "README.md"
# Bump this whenever text extraction, normalization or chunking changes, so
# existing indexes are rebuilt even though the files themselves did not change.
INGEST_VERSION = "1"


class CorpusNotFoundError(AskDocsError):
    """The configured corpus folder does not exist."""


class InvalidDocumentNameError(AskDocsError):
    """An uploaded file name is not safe to store in the corpus folder."""


# Characters Windows forbids in file names (also covers path separators and
# ":", which NTFS uses for hidden "alternate data streams").
_FORBIDDEN_NAME_CHARS = set('<>:"/\\|?*')
_WINDOWS_DEVICE_NAMES = {"con", "prn", "aux", "nul"} | {
    f"{base}{n}" for base in ("com", "lpt") for n in range(1, 10)
}
MAX_NAME_CHARS = 200


def safe_doc_name(name: str) -> str:
    """Return name unchanged if it is a plain, supported file name; else raise.

    Uploads may only create files directly inside the corpus folder, so any
    path separator, "..", hidden name or unsupported extension is rejected.
    The rules are deliberately strict so the same name is safe on Windows and Linux.
    """
    if not name or len(name) > MAX_NAME_CHARS:
        raise InvalidDocumentNameError(f"The file name must be 1-{MAX_NAME_CHARS} characters.")
    if any(c in _FORBIDDEN_NAME_CHARS or ord(c) < 32 for c in name):
        raise InvalidDocumentNameError(
            "The file name contains a character that is not allowed "
            '(slashes, : * ? " < > | or control characters).'
        )
    if name.startswith(".") or name.endswith((".", " ")) or name.startswith(" "):
        raise InvalidDocumentNameError(
            "The file name cannot start with a dot or a space, or end with a dot or a space."
        )
    if name == CORPUS_README:
        raise InvalidDocumentNameError(f"{CORPUS_README} is reserved; choose another name.")
    path = Path(name)
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise InvalidDocumentNameError(
            "Unsupported file type. Use .pdf, .md, .markdown or .txt."
        )
    if path.stem.lower() in _WINDOWS_DEVICE_NAMES:
        raise InvalidDocumentNameError(f"{path.stem!r} is a reserved name on Windows.")
    return name


def file_sha256(data: bytes) -> str:
    """Hex SHA-256 of a file's bytes."""
    return hashlib.sha256(data).hexdigest()


def list_corpus_files(corpus_dir: Path) -> list[tuple[str, Path]]:
    """Return (doc_name, path) for supported files, sorted by doc_name.

    doc_name is the path relative to corpus_dir with forward slashes, so the
    same corpus gives the same names on Windows and Linux.
    """
    if not corpus_dir.is_dir():
        raise CorpusNotFoundError(f"Corpus folder not found: {corpus_dir}")
    found = []
    for path in corpus_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(corpus_dir)
        name = rel.as_posix()
        if any(part.startswith(".") for part in rel.parts):
            continue  # hidden files and folders (.git, .DS_Store, ...)
        if name == CORPUS_README:
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            log.info("ignoring unsupported file: %s", name)
            continue
        found.append((name, path))
    return sorted(found)


def _pdf_pages(doc_name: str, data: bytes) -> tuple[list[Page], list[str]]:
    """Extract PDF text page by page (1-based, physical page order)."""
    warnings: list[str] = []
    # pypdf raises many different exception types for damaged files, so a
    # broad except is deliberate here: one bad PDF must never stop an ingest.
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            # Many PDFs are "encrypted" only to restrict editing, with an empty
            # user password; those open with "". Others need a real password.
            return [], [f"{doc_name}: skipped, PDF is password-protected"]
        raw_pages = list(reader.pages)
    except Exception as exc:
        return [], [f"{doc_name}: skipped, could not read PDF ({type(exc).__name__}: {exc})"]

    total = len(raw_pages)
    pages = []
    for number, pdf_page in enumerate(raw_pages, start=1):
        try:
            raw = pdf_page.extract_text() or ""
        except Exception as exc:
            warnings.append(f"{doc_name}: page {number} could not be read ({type(exc).__name__})")
            raw = ""
        text = normalize_text(raw)
        if len(text) < MIN_PAGE_CHARS:
            warnings.append(
                f"{doc_name}: page {number} of {total} has no extractable text, possibly scanned"
            )
        pages.append(Page(doc_name=doc_name, page=number, text=text))
    return pages, warnings


def _text_pages(doc_name: str, data: bytes) -> tuple[list[Page], list[str]]:
    """Markdown/text: one page, or one page per form-feed-separated section."""
    warnings = []
    try:
        content = data.decode("utf-8")
    except UnicodeDecodeError:
        content = data.decode("utf-8", errors="replace")
        warnings.append(f"{doc_name}: not valid UTF-8; invalid bytes were replaced")
    if doc_name.lower().endswith((".md", ".markdown")):
        content = strip_markdown(content)
    sections = content.split("\f")
    pages = [
        Page(doc_name=doc_name, page=number, text=normalize_text(section))
        for number, section in enumerate(sections, start=1)
    ]
    return pages, warnings


def read_pages(doc_name: str, data: bytes) -> tuple[list[Page], list[str]]:
    """Turn one file's bytes into normalized pages plus warnings; never raises on bad input."""
    if doc_name.lower().endswith(".pdf"):
        return _pdf_pages(doc_name, data)
    return _text_pages(doc_name, data)


def ingest_corpus(settings: Settings, store: Store, embedder: Embedder) -> IngestReport:
    """Sync the index with settings.corpus_dir and report what changed."""
    files = list_corpus_files(settings.corpus_dir)
    report = IngestReport()

    # `previous` is what was indexed before this run (used for the counts);
    # `indexed` is what is still valid to keep (used to skip unchanged files).
    previous = store.document_hashes()
    indexed = previous
    # Vectors from two different embedders cannot be compared, and old chunks
    # do not match new chunk settings or new extraction code, so any of these
    # changing rebuilds everything.
    info = {
        "embedder": embedder.name,
        "dim": str(embedder.dim),
        "chunk_words": str(settings.chunk_words),
        "chunk_overlap": str(settings.chunk_overlap),
        "ingest_version": INGEST_VERSION,
    }
    if not store.index_matches(info):
        if previous:
            report.warnings.append(
                "embedder, chunk settings or ingest version changed; rebuilding the index"
            )
        store.clear()
        store.set_index_info(info)
        indexed = {}

    kept_hashes: dict[str, str] = {}  # sha256 -> doc_name of files kept in this run
    present = set()

    for doc_name, path in files:
        present.add(doc_name)
        data = path.read_bytes()
        sha = file_sha256(data)

        if sha in kept_hashes:
            # Same bytes as a file already processed (files are processed in
            # sorted order, so the alphabetically first copy wins).
            report.warnings.append(f"{doc_name}: skipped, duplicate of {kept_hashes[sha]}")
            report.skipped += 1
            if doc_name in indexed:
                store.delete_document(doc_name)
            continue

        if indexed.get(doc_name) == sha:
            report.unchanged += 1
            kept_hashes[sha] = doc_name
            continue

        pages, warnings = read_pages(doc_name, data)
        report.warnings.extend(warnings)
        chunks = chunk_pages(pages, settings.chunk_words, settings.chunk_overlap)
        if not chunks:
            # An unreadable PDF already returned its own "skipped, ..." warning
            # with no pages; anything else readable but empty gets this one.
            if pages or not warnings:
                report.warnings.append(f"{doc_name}: skipped, no text found")
            report.skipped += 1
            if doc_name in indexed:
                store.delete_document(doc_name)  # its old text is no longer valid
            continue

        vectors = embedder.embed([c.text for c in chunks])
        store.replace_document(doc_name, sha, len(pages), chunks, vectors)
        kept_hashes[sha] = doc_name
        if doc_name in previous:
            report.updated += 1
        else:
            report.added += 1

    for doc_name in sorted(set(previous) - present):
        if doc_name in indexed:  # after a rebuild it is already gone
            store.delete_document(doc_name)
        report.removed += 1

    for warning in report.warnings:
        log.warning(warning)
    return report
