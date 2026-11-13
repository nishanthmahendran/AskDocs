"""Keyword, vector and hybrid retrieval with Reciprocal Rank Fusion (RRF).

Determinism: every ranking breaks score ties by chunk_id, so the same index
and question always give the same results.
"""

from __future__ import annotations

from typing import Literal, Protocol

import numpy as np

from askdocs.config import Settings
from askdocs.embed import Embedder
from askdocs.models import SearchHit
from askdocs.store import Store

Mode = Literal["keyword", "vector", "hybrid"]
MODES: tuple[Mode, ...] = ("keyword", "vector", "hybrid")
# Standard RRF constant from Cormack et al. (2009). Larger k flattens the
# difference between rank 1 and rank 10.
RRF_K = 60


class VectorIndex(Protocol):
    """The seam for vector search. A different index (for example a C++
    library with an approximate nearest-neighbour index) only needs this method.
    """

    def search(self, query: np.ndarray, limit: int) -> list[tuple[str, float]]:
        """Return up to `limit` (chunk_id, cosine) pairs, best first, ties by chunk_id."""
        ...


class BruteForceIndex:
    """Exact cosine search: one matrix-vector product over every stored vector.

    Fine for thousands of chunks; vectors are already L2-normalized, so the
    dot product equals the cosine similarity.
    """

    def __init__(self, chunk_ids: list[str], matrix: np.ndarray) -> None:
        """Keep the ids and the [n, dim] matrix of normalized vectors."""
        self._ids = np.array(chunk_ids, dtype=object)
        self._matrix = matrix

    def search(self, query: np.ndarray, limit: int) -> list[tuple[str, float]]:
        """Return the `limit` most similar chunks."""
        if len(self._ids) == 0:
            return []
        scores = self._matrix @ query
        # np.lexsort sorts by the LAST key first: highest score, then chunk_id.
        order = np.lexsort((self._ids, -scores))[:limit]
        return [(str(self._ids[i]), float(scores[i])) for i in order]


def rrf_score(ranks: list[int | None]) -> float:
    """Reciprocal Rank Fusion: sum of 1 / (RRF_K + rank) over the lists a chunk appears in."""
    return sum(1.0 / (RRF_K + r) for r in ranks if r is not None)


class Retriever:
    """Runs searches against one Store with one Embedder."""

    def __init__(self, settings: Settings, store: Store, embedder: Embedder) -> None:
        """Remember collaborators; vectors are loaded lazily and cached per index version."""
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self._cached_version: int | None = None
        self._index: VectorIndex = BruteForceIndex([], np.zeros((0, 0), dtype=np.float32))

    def _vector_index(self) -> VectorIndex:
        """Reload vectors only when the index changed since the last query."""
        version = self.store.index_version()
        if version != self._cached_version:
            ids, matrix = self.store.load_vectors()
            self._index = BruteForceIndex(ids, matrix)
            self._cached_version = version
        return self._index

    def search(self, question: str, mode: Mode = "hybrid", k: int | None = None) -> list[SearchHit]:
        """Return the top k hits for the question; [] for an empty question or index."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        k = k or self.settings.top_k
        question = question.strip()
        if not question or self.store.counts()[1] == 0:
            return []
        self.store.check_embedder(self.embedder.name, self.embedder.dim)

        # The query vector is always computed: even keyword mode reports each
        # hit's cosine similarity, because the abstain rule needs it.
        query_vec = self.embedder.embed([question])[0]
        limit = self.settings.candidate_k

        keyword = self.store.keyword_search(question, limit) if mode != "vector" else []
        vector = self._vector_index().search(query_vec, limit) if mode != "keyword" else []

        keyword_rank = {cid: rank for rank, (cid, _) in enumerate(keyword, start=1)}
        vector_rank = {cid: rank for rank, (cid, _) in enumerate(vector, start=1)}
        vector_score = {cid: score for cid, score in vector}

        # In single-mode searches the "fused" score is simply that list's RRF term,
        # so the three modes are directly comparable in the evaluation.
        candidates = set(keyword_rank) | set(vector_rank)
        fused = {
            cid: rrf_score([keyword_rank.get(cid), vector_rank.get(cid)]) for cid in candidates
        }
        top_ids = sorted(candidates, key=lambda cid: (-fused[cid], cid))[:k]

        # Fill in cosine scores for hits that came only from keyword search.
        missing = [cid for cid in top_ids if cid not in vector_score]
        for cid, vec in self.store.get_vectors(missing).items():
            vector_score[cid] = float(vec @ query_vec)

        chunks = self.store.get_chunks(top_ids)
        return [
            SearchHit(
                chunk=chunks[cid],
                keyword_rank=keyword_rank.get(cid),
                vector_rank=vector_rank.get(cid),
                vector_score=vector_score.get(cid),
                fused_score=fused[cid],
            )
            for cid in top_ids
        ]
