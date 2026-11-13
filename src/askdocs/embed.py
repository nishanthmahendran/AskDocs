"""Text embedders: a real one (fastembed, ONNX on CPU) and a hash one for tests.

Every embedder returns float32 rows that are L2-normalized, so the dot
product of two rows is their cosine similarity. (An all-zero row, from text
with no words, stays zero and has similarity 0 with everything.)
"""

from __future__ import annotations

import hashlib
import itertools
from pathlib import Path
from typing import Protocol

import numpy as np

from askdocs.config import Settings
from askdocs.models import AskDocsError
from askdocs.text import words


class EmbedderUnavailableError(AskDocsError):
    """The embedding model could not be loaded (for example, no internet for the download)."""


class Embedder(Protocol):
    """Anything that turns texts into normalized vectors."""

    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return a float32 array of shape [len(texts), dim] with L2-normalized rows."""
        ...


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Divide each row by its length; all-zero rows are left as zeros."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class HashEmbedder:
    """Deterministic, dependency-free embedder for TESTS AND CI ONLY.

    It hashes word unigrams and bigrams into 384 buckets ("feature hashing").
    It only captures exact word overlap, has no notion of meaning or synonyms,
    and is NOT suitable for real retrieval quality. Use fastembed for that.
    """

    name = "hash"

    def __init__(self, dim: int = 384) -> None:
        """Create a hash embedder with `dim` buckets."""
        self.dim = dim

    def _bucket(self, feature: str) -> tuple[int, float]:
        """Map a feature to (bucket index, +1 or -1) using a stable hash.

        Python's built-in hash() is randomized per process, so hashlib is used
        to keep results identical across runs and machines. The sign bit makes
        collisions cancel out on average instead of always adding up.
        """
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        sign = 1.0 if (value >> 63) & 1 else -1.0
        return value % self.dim, sign

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts by feature hashing of word unigrams and bigrams."""
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = words(text)
            features = tokens + [f"{a} {b}" for a, b in itertools.pairwise(tokens)]
            for feature in features:
                index, sign = self._bucket(feature)
                out[row, index] += sign
        return l2_normalize(out)


class FastEmbedEmbedder:
    """Real embeddings from fastembed (ONNX Runtime on CPU, no torch).

    The model is downloaded once into cache_dir on first use and reused after.
    """

    def __init__(self, model_name: str, cache_dir: Path, batch_size: int = 64) -> None:
        """Load (and if needed download) the model; raise a clear error on failure."""
        self.name = f"fastembed:{model_name}"
        self._batch_size = batch_size
        try:
            # Imported here so the hash embedder (tests, CI) never loads onnxruntime.
            from fastembed import TextEmbedding

            cache_dir.mkdir(parents=True, exist_ok=True)
            self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))
            probe = next(iter(self._model.embed(["dimension probe"])))
        except Exception as exc:
            raise EmbedderUnavailableError(
                f"Could not load embedding model {model_name!r} ({type(exc).__name__}: {exc}). "
                "The model is downloaded on first use, which needs internet access once. "
                "To run fully offline for testing, set ASKDOCS_EMBEDDER=hash "
                "(low quality, keyword-like matching only)."
            ) from exc
        self.dim = len(probe)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts in batches and return normalized float32 rows."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        rows = list(self._model.embed(texts, batch_size=self._batch_size))
        return l2_normalize(np.vstack(rows).astype(np.float32))


def make_embedder(settings: Settings) -> Embedder:
    """Create the embedder selected by settings.embedder."""
    if settings.embedder == "hash":
        return HashEmbedder()
    return FastEmbedEmbedder(settings.embed_model, settings.model_cache_dir)
