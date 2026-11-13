from __future__ import annotations

import numpy as np
import pytest

from askdocs.config import Settings, with_overrides
from askdocs.embed import HashEmbedder
from askdocs.ingest import ingest_corpus
from askdocs.retrieve import MODES, BruteForceIndex, Retriever, rrf_score
from askdocs.store import EmbedderMismatchError, Store

DOCS = {
    "bikes.md": "Bike share rules. The late fee is two dollars per hour after the first day.",
    "laser.md": "Laser cutter safety. Never leave the laser cutter running unattended.",
    "printer.md": "The 3D printer queue opens at nine. Large prints need staff approval.",
}


@pytest.fixture
def retriever(settings: Settings) -> Retriever:
    for name, text in DOCS.items():
        (settings.corpus_dir / name).write_text(text, encoding="utf-8")
    store = Store(settings.db_path)
    ingest_corpus(settings, store, HashEmbedder())
    return Retriever(settings, store, HashEmbedder())


@pytest.mark.parametrize("mode", MODES)
def test_each_mode_finds_the_right_document(retriever: Retriever, mode: str) -> None:
    hits = retriever.search("late fee for the bike share", mode=mode)  # type: ignore[arg-type]
    assert hits[0].chunk.doc_name == "bikes.md"
    assert hits[0].vector_score is not None  # cosine always filled in


def test_ranks_and_scores_are_kept(retriever: Retriever) -> None:
    hit = retriever.search("laser cutter safety", mode="hybrid")[0]
    assert hit.chunk.doc_name == "laser.md"
    assert hit.keyword_rank == 1 and hit.vector_rank == 1
    assert hit.fused_score == pytest.approx(2 / 61)


def test_keyword_mode_has_no_vector_rank(retriever: Retriever) -> None:
    for hit in retriever.search("printer queue", mode="keyword"):
        assert hit.vector_rank is None


def test_results_are_deterministic(retriever: Retriever) -> None:
    first = retriever.search("rules for the printer and the laser", k=3)
    second = retriever.search("rules for the printer and the laser", k=3)
    assert [h.model_dump() for h in first] == [h.model_dump() for h in second]


@pytest.mark.parametrize("question", ["", "   ", "?!", "a", 'AND ( * "', "fee"])
def test_odd_questions_do_not_crash(retriever: Retriever, question: str) -> None:
    assert isinstance(retriever.search(question), list)


def test_empty_index_returns_empty_list(settings: Settings) -> None:
    store = Store(settings.db_path)
    assert Retriever(settings, store, HashEmbedder()).search("anything") == []


def test_k_limits_results(retriever: Retriever) -> None:
    assert len(retriever.search("the", mode="vector", k=2)) <= 2


def test_invalid_mode_rejected(retriever: Retriever) -> None:
    with pytest.raises(ValueError):
        retriever.search("x", mode="fuzzy")  # type: ignore[arg-type]


def test_dimension_mismatch_refuses_to_query(retriever: Retriever, settings: Settings) -> None:
    other = Retriever(settings, retriever.store, HashEmbedder(dim=128))
    with pytest.raises(EmbedderMismatchError):
        other.search("late fee")


def test_cache_reloads_after_ingest(retriever: Retriever, settings: Settings) -> None:
    retriever.search("anything")
    (settings.corpus_dir / "new.md").write_text("Kayak rental desk hours.", encoding="utf-8")
    ingest_corpus(settings, retriever.store, HashEmbedder())
    assert retriever.search("kayak rental", mode="vector")[0].chunk.doc_name == "new.md"


def test_brute_force_ties_break_by_chunk_id() -> None:
    matrix = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index = BruteForceIndex(["c", "a", "b"], matrix)
    assert [cid for cid, _ in index.search(np.array([1.0, 0.0], dtype=np.float32), 3)] == [
        "a",
        "c",
        "b",
    ]


def test_rrf_score() -> None:
    assert rrf_score([1, None]) == pytest.approx(1 / 61)
    assert rrf_score([None, None]) == 0.0


def test_top_k_respects_settings(retriever: Retriever, settings: Settings) -> None:
    small = Retriever(with_overrides(settings, top_k=1), retriever.store, HashEmbedder())
    assert len(small.search("the laser and the printer")) == 1
