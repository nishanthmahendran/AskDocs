from pathlib import Path

import numpy as np
import pytest

import askdocs.embed as embed_module
from askdocs.config import Settings, with_overrides
from askdocs.embed import EmbedderUnavailableError, HashEmbedder, l2_normalize, make_embedder


def test_hash_embedder_shape_dtype_and_norm() -> None:
    vecs = HashEmbedder().embed(["bike share fees", "laser cutter safety", ""])
    assert vecs.shape == (3, 384)
    assert vecs.dtype == np.float32
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms[:2], 1.0, atol=1e-6)
    assert norms[2] == 0.0  # no words -> zero vector


def test_hash_embedder_is_deterministic_and_similarity_follows_overlap() -> None:
    e = HashEmbedder()
    a, b, c = e.embed(["late fee for bikes", "bikes late fee rules", "laser cutter training"])
    assert np.array_equal(e.embed(["late fee for bikes"])[0], a)
    assert float(a @ b) > float(a @ c)


def test_hash_embedder_handles_unicode() -> None:
    text = "caf\N{LATIN SMALL LETTER E WITH ACUTE} \N{LEFT DOUBLE QUOTATION MARK}x"
    vec = HashEmbedder().embed([text])
    assert np.isclose(np.linalg.norm(vec), 1.0)


def test_l2_normalize_leaves_zero_rows() -> None:
    out = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(out, [[0.6, 0.8], [0.0, 0.0]])


def test_make_embedder_hash(settings: Settings) -> None:
    assert make_embedder(settings).name == "hash"


def test_fastembed_failure_explains_offline_option(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Simulate a failed model download without touching the network."""

    class FailingTextEmbedding:
        def __init__(self, **kwargs: object) -> None:
            raise OSError("network unreachable")

    import fastembed

    monkeypatch.setattr(fastembed, "TextEmbedding", FailingTextEmbedding)
    with pytest.raises(EmbedderUnavailableError, match="ASKDOCS_EMBEDDER=hash"):
        make_embedder(with_overrides(settings, embedder="fastembed"))
    assert embed_module.FastEmbedEmbedder is not None
