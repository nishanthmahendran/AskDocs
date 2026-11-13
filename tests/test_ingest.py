"""Ingest tests: extraction from each file type and syncing the index."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter

import askdocs.ingest as ingest_module
from askdocs.config import Settings, with_overrides
from askdocs.embed import HashEmbedder
from askdocs.ingest import CorpusNotFoundError, ingest_corpus, list_corpus_files, read_pages
from askdocs.store import Store
from pdf_helper import build_pdf, write_pdf

LONG_LINE = "The quick brown fox jumps over the lazy dog near the river bank."


def run_ingest(settings: Settings) -> tuple[Store, object]:
    """Ingest settings.corpus_dir and return (store, report)."""
    store = Store(settings.db_path)
    return store, ingest_corpus(settings, store, HashEmbedder())


# ----- extraction -----------------------------------------------------------


def test_pdf_pages_are_one_based_and_in_order() -> None:
    data = build_pdf([["First page text that is long enough."], ["Second page, also long enough."]])
    pages, warnings = read_pages("a.pdf", data)
    assert [p.page for p in pages] == [1, 2]
    assert pages[0].text == "First page text that is long enough."
    assert pages[1].text.startswith("Second page")
    assert warnings == []


def test_pdf_empty_page_warns_and_continues() -> None:
    data = build_pdf([["A page with plenty of real text on it."], [], ["Third page has text too."]])
    pages, warnings = read_pages("scan.pdf", data)
    assert len(pages) == 3
    assert pages[1].text == ""
    assert warnings == ["scan.pdf: page 2 of 3 has no extractable text, possibly scanned"]


def test_pdf_hyphenation_rejoined() -> None:
    data = build_pdf([["Some words then exam-", "ple continues here."]])
    pages, _ = read_pages("h.pdf", data)
    assert "example continues" in pages[0].text


@pytest.mark.parametrize("data", [b"", b"not a pdf at all", build_pdf([["x" * 40]])[:120]])
def test_corrupt_pdf_is_skipped_not_raised(data: bytes) -> None:
    pages, warnings = read_pages("bad.pdf", data)
    assert pages == []
    assert len(warnings) == 1
    assert "skipped, could not read PDF" in warnings[0]


def _encrypt(user_password: str) -> bytes:
    writer = PdfWriter()
    writer.append(PdfReader(io.BytesIO(build_pdf([["Secret text inside the document."]]))))
    writer.encrypt(user_password, owner_password="owner", algorithm="RC4-128")
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def test_password_protected_pdf_is_skipped() -> None:
    pages, warnings = read_pages("locked.pdf", _encrypt("secret"))
    assert pages == []
    assert warnings == ["locked.pdf: skipped, PDF is password-protected"]


def test_pdf_with_empty_user_password_is_read() -> None:
    pages, warnings = read_pages("restricted.pdf", _encrypt(""))
    assert pages[0].text == "Secret text inside the document."
    assert warnings == []


def test_text_form_feed_splits_pages() -> None:
    pages, _ = read_pages("a.txt", b"page one\fpage two\f page three ")
    expected = [(1, "page one"), (2, "page two"), (3, "page three")]
    assert [(p.page, p.text) for p in pages] == expected


def test_unicode_text_preserved() -> None:
    text = "\N{LEFT DOUBLE QUOTATION MARK}Caf\N{LATIN SMALL LETTER E WITH ACUTE}" \
           "\N{RIGHT DOUBLE QUOTATION MARK}\N{NO-BREAK SPACE}costs\N{NO-BREAK SPACE}3 euros."
    pages, warnings = read_pages("u.md", text.encode("utf-8"))
    expected = "\N{LEFT DOUBLE QUOTATION MARK}Caf\N{LATIN SMALL LETTER E WITH ACUTE}" \
               "\N{RIGHT DOUBLE QUOTATION MARK} costs 3 euros."
    assert pages[0].text == expected
    assert warnings == []


def test_invalid_utf8_is_replaced_with_warning() -> None:
    pages, warnings = read_pages("bad.txt", b"hello \xff world")
    assert "hello" in pages[0].text
    assert "not valid UTF-8" in warnings[0]


def test_list_corpus_files_filters_and_sorts(corpus_dir: Path) -> None:
    (corpus_dir / "sub").mkdir()
    (corpus_dir / ".hidden").mkdir()
    for name in ["b.md", "a.TXT", "sub/c.markdown", "notes.docx", "README.md", ".hidden/x.md"]:
        (corpus_dir / name).write_text("text", encoding="utf-8")
    write_pdf(corpus_dir / "sub" / "d.pdf", [["text"]])
    names = [name for name, _ in list_corpus_files(corpus_dir)]
    assert names == ["a.TXT", "b.md", "sub/c.markdown", "sub/d.pdf"]


def test_missing_corpus_raises(tmp_path: Path) -> None:
    with pytest.raises(CorpusNotFoundError):
        list_corpus_files(tmp_path / "nope")


# ----- syncing ---------------------------------------------------------------


def test_ingest_add_unchanged_update_remove(settings: Settings) -> None:
    corpus = settings.corpus_dir
    (corpus / "a.md").write_text("Alpha document about bicycles.", encoding="utf-8")
    (corpus / "b.txt").write_text("Beta document about printers.", encoding="utf-8")
    write_pdf(corpus / "c.pdf", [["Gamma PDF about lasers and cutting."], [LONG_LINE]])

    store, report = run_ingest(settings)
    assert (report.added, report.updated, report.unchanged, report.removed) == (3, 0, 0, 0)
    assert store.counts() == (3, 4)
    version = store.index_version()

    # Re-ingesting unchanged files does nothing (and does not touch the index).
    report = ingest_corpus(settings, store, HashEmbedder())
    assert (report.added, report.updated, report.unchanged, report.removed) == (0, 0, 3, 0)
    assert store.index_version() == version

    # A changed file replaces all of its old chunks.
    (corpus / "a.md").write_text("Alpha rewritten about scooters.", encoding="utf-8")
    report = ingest_corpus(settings, store, HashEmbedder())
    assert (report.added, report.updated, report.unchanged) == (0, 1, 2)
    texts = [c.text for c in store.get_chunks([i for i in store.load_vectors()[0]]).values()]
    assert "Alpha rewritten about scooters." in texts
    assert "Alpha document about bicycles." not in texts

    # A deleted file is removed from the index.
    (corpus / "b.txt").unlink()
    report = ingest_corpus(settings, store, HashEmbedder())
    assert report.removed == 1
    assert sorted(store.document_hashes()) == ["a.md", "c.pdf"]
    assert store.keyword_search("printers", 5) == []


def test_ingest_skips_empty_and_corrupt_files(settings: Settings) -> None:
    corpus = settings.corpus_dir
    (corpus / "empty.txt").write_text("", encoding="utf-8")
    (corpus / "blank.md").write_text("  \n\t ", encoding="utf-8")
    (corpus / "broken.pdf").write_bytes(b"%PDF-1.4 garbage")
    write_pdf(corpus / "allblank.pdf", [[], []])
    (corpus / "ok.md").write_text("Real content here.", encoding="utf-8")

    store, report = run_ingest(settings)
    assert report.added == 1
    assert report.skipped == 4
    assert store.counts()[0] == 1
    joined = "\n".join(report.warnings)
    assert "empty.txt: skipped, no text found" in joined
    assert "broken.pdf: skipped, could not read PDF" in joined
    assert "allblank.pdf: page 1 of 2 has no extractable text" in joined


def test_ingest_skips_duplicate_files(settings: Settings) -> None:
    corpus = settings.corpus_dir
    (corpus / "a.md").write_text("Same bytes in two files.", encoding="utf-8")
    (corpus / "copy.md").write_text("Same bytes in two files.", encoding="utf-8")
    store, report = run_ingest(settings)
    assert (report.added, report.skipped) == (1, 1)
    assert "copy.md: skipped, duplicate of a.md" in report.warnings
    assert list(store.document_hashes()) == ["a.md"]


def test_file_that_becomes_empty_is_removed_from_index(settings: Settings) -> None:
    path = settings.corpus_dir / "a.md"
    path.write_text("Some text.", encoding="utf-8")
    store, _ = run_ingest(settings)
    path.write_text("", encoding="utf-8")
    report = ingest_corpus(settings, store, HashEmbedder())
    assert report.skipped == 1
    assert store.counts() == (0, 0)


def test_very_long_page_is_chunked(settings: Settings) -> None:
    text = " ".join([LONG_LINE] * 300)  # ~3,900 words on one page
    (settings.corpus_dir / "long.txt").write_text(text, encoding="utf-8")
    store, _ = run_ingest(settings)
    _, n_chunks = store.counts()
    assert n_chunks > 15
    chunks = store.get_chunks(store.load_vectors()[0]).values()
    assert all(c.page == 1 for c in chunks)
    assert max(len(c.text.split()) for c in chunks) <= settings.chunk_words * 1.3


def test_changing_ingest_version_rebuilds_unchanged_files(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    (settings.corpus_dir / "a.md").write_text("Some text.", encoding="utf-8")
    store, _ = run_ingest(settings)
    monkeypatch.setattr(ingest_module, "INGEST_VERSION", "test-next")
    report = ingest_corpus(settings, store, HashEmbedder())
    assert (report.updated, report.unchanged) == (1, 0)


def test_changing_chunk_settings_rebuilds_index(settings: Settings) -> None:
    (settings.corpus_dir / "a.md").write_text(" ".join([LONG_LINE] * 40), encoding="utf-8")
    store, _ = run_ingest(settings)
    before = store.counts()[1]
    smaller = with_overrides(settings, chunk_words=50, chunk_overlap=10)
    report = ingest_corpus(smaller, store, HashEmbedder())
    assert report.updated == 1
    assert "rebuilding the index" in report.warnings[0]
    assert store.counts()[1] > before
