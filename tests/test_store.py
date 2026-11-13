from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

import askdocs.store as store_module
from askdocs.chunking import make_chunk_id
from askdocs.embed import HashEmbedder
from askdocs.models import Chunk
from askdocs.store import EmbedderMismatchError, Store, StoreError, fts_query

TEXTS = [
    "The late fee is 2 dollars per hour.",
    "C++ courses cost 300 dollars; the fee is waived for members.",
    "Caf\N{LATIN SMALL LETTER E WITH ACUTE} hours are 8 to 5.",
]


def make_store(tmp_path: Path) -> Store:
    store = Store(tmp_path / "s.sqlite")
    chunks = [
        Chunk(chunk_id=make_chunk_id("d.md", 1, i, t), doc_name="d.md", page=1, index=i, text=t)
        for i, t in enumerate(TEXTS)
    ]
    store.set_index_info({"embedder": "hash", "dim": "384"})
    store.replace_document("d.md", "sha", 1, chunks, HashEmbedder().embed(TEXTS))
    return store


@pytest.mark.parametrize(
    "question",
    [
        "What's the C++ fee?",
        'text containing " AND ( * do not crash',
        "NEAR(a b) OR NOT fee*",
        '"""',
        "?!?",
        "fee",
        "",
        "\N{RIGHT SINGLE QUOTATION MARK}fee\N{RIGHT SINGLE QUOTATION MARK}",
    ],
)
def test_keyword_search_never_crashes(tmp_path: Path, question: str) -> None:
    results = make_store(tmp_path).keyword_search(question, 10)
    assert isinstance(results, list)


def test_fts_query_quotes_terms_and_drops_operators() -> None:
    assert fts_query("What's the C++ fee?") == '"c" OR "fee"'
    assert fts_query('" AND ( *') == ""
    assert fts_query("near not") == '"near" OR "not"'


def test_keyword_search_ranks_and_accents(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    ids = [cid for cid, _ in store.keyword_search("late fee", 10)]
    assert store.get_chunks(ids[:1])[ids[0]].text == TEXTS[0]
    # remove_diacritics: "cafe" finds "Café".
    hits = store.keyword_search("cafe hours", 10)
    assert store.get_chunks([hits[0][0]])[hits[0][0]].text == TEXTS[2]


def test_vectors_round_trip(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    ids, matrix = store.load_vectors()
    assert ids == sorted(ids)
    assert matrix.shape == (3, 384)
    assert matrix.dtype == np.float32
    one = store.get_vectors([ids[0]])[ids[0]]
    assert np.array_equal(one, matrix[0])


def test_documents_listing_and_delete(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    docs = store.list_documents()
    assert [(d.doc_name, d.pages, d.chunks) for d in docs] == [("d.md", 1, 3)]
    store.delete_document("d.md")
    assert store.counts() == (0, 0)
    assert store.keyword_search("fee", 5) == []
    assert store.load_vectors()[0] == []


def test_embedder_mismatch_is_detected(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.check_embedder("hash", 384)  # same: fine
    with pytest.raises(EmbedderMismatchError, match="askdocs ingest"):
        store.check_embedder("fastembed:BAAI/bge-small-en-v1.5", 384)
    with pytest.raises(EmbedderMismatchError):
        store.check_embedder("hash", 128)


def test_empty_index_never_mismatches(tmp_path: Path) -> None:
    Store(tmp_path / "e.sqlite").check_embedder("anything", 1)


def test_wal_mode_enabled(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    with store.connect() as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_missing_fts5_gives_clear_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class NoFtsConnection:
        def execute(self, sql: str) -> None:
            raise sqlite3.OperationalError("no such module: fts5")

        def close(self) -> None:
            pass

    monkeypatch.setattr(store_module.sqlite3, "connect", lambda *_a, **_k: NoFtsConnection())
    with pytest.raises(StoreError, match="FTS5"):
        Store(tmp_path / "x.sqlite")


def test_failed_write_rolls_back(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    bad_chunk = Chunk(chunk_id="x", doc_name="d.md", page=1, index=0, text="t")
    with pytest.raises(ValueError):
        store.replace_document("d.md", "sha2", 1, [bad_chunk], np.zeros((2, 384)))
    assert store.counts() == (1, 3)
