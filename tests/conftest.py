"""Shared pytest fixtures. Every test uses the offline hash embedder."""

from __future__ import annotations

from pathlib import Path

import pytest

from askdocs.config import Settings, load_settings

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    """An empty corpus folder inside the test's temporary directory."""
    path = tmp_path / "corpus"
    path.mkdir()
    return path


@pytest.fixture
def settings(tmp_path: Path, corpus_dir: Path) -> Settings:
    """Settings that never touch the network or the real data/ folder."""
    return load_settings(
        env={},
        corpus_dir=corpus_dir,
        db_path=tmp_path / "index.sqlite",
        model_cache_dir=tmp_path / "models",
        embedder="hash",
        generator="extractive",
    )
