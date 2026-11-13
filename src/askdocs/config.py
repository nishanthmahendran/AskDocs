"""Settings for AskDocs, with ASKDOCS_* environment-variable overrides.

Every field of Settings can be overridden by an environment variable named
ASKDOCS_<FIELD NAME IN UPPER CASE>, for example ASKDOCS_TOP_K=8.
"""

# Note: no `from __future__ import annotations` here, because load_settings()
# reads each field's real type (Path, int, float, str) to parse env values.

import os
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

from askdocs.models import AskDocsError

ENV_PREFIX = "ASKDOCS_"
EMBEDDERS = ("fastembed", "hash")
GENERATORS = ("extractive", "ollama", "openai")


class ConfigError(AskDocsError):
    """Raised when a setting has an invalid value."""


def env_name(field_name: str) -> str:
    """Return the environment variable name for a Settings field."""
    return ENV_PREFIX + field_name.upper()


@dataclass(frozen=True)
class Settings:
    """All tunable settings. Frozen so nothing can change them after startup."""

    corpus_dir: Path = Path("corpus")
    db_path: Path = Path("data/askdocs.sqlite")
    # Where fastembed stores its downloaded model. Inside data/ so it is
    # gitignored and lives on the same Docker volume as the index.
    model_cache_dir: Path = Path("data/models")
    chunk_words: int = 200
    chunk_overlap: int = 40
    top_k: int = 5
    candidate_k: int = 20
    # Starting value only. Tune it with eval/run_eval.py, which sweeps 0.0-1.0.
    abstain_threshold: float = 0.35
    embedder: str = "fastembed"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    generator: str = "extractive"
    ollama_url: str = "http://localhost:11434"
    openai_base_url: str = "http://localhost:8080/v1"
    openai_api_key: str = ""
    model_name: str = "llama3.2:3b"
    request_timeout: float = 60.0
    max_question_chars: int = 1000
    # Largest file the web page may upload, in megabytes.
    max_upload_mb: int = 25

    def __post_init__(self) -> None:
        """Check that values make sense together; errors name the env variable."""
        positive_ints = (
            "chunk_words", "top_k", "candidate_k", "max_question_chars", "max_upload_mb",
        )
        for name in positive_ints:
            if getattr(self, name) < 1:
                raise ConfigError(f"{env_name(name)} must be at least 1, got {getattr(self, name)}")
        if not 0 <= self.chunk_overlap < self.chunk_words:
            raise ConfigError(
                f"{env_name('chunk_overlap')} must be >= 0 and smaller than "
                f"{env_name('chunk_words')} ({self.chunk_words}), got {self.chunk_overlap}"
            )
        if self.candidate_k < self.top_k:
            raise ConfigError(
                f"{env_name('candidate_k')} ({self.candidate_k}) must be >= "
                f"{env_name('top_k')} ({self.top_k})"
            )
        if not 0.0 <= self.abstain_threshold <= 1.0:
            raise ConfigError(
                f"{env_name('abstain_threshold')} must be between 0 and 1, "
                f"got {self.abstain_threshold}"
            )
        if self.request_timeout <= 0:
            raise ConfigError(f"{env_name('request_timeout')} must be positive")
        if self.embedder not in EMBEDDERS:
            raise ConfigError(
                f"{env_name('embedder')} must be one of {EMBEDDERS}, got {self.embedder!r}"
            )
        if self.generator not in GENERATORS:
            raise ConfigError(
                f"{env_name('generator')} must be one of {GENERATORS}, got {self.generator!r}"
            )


def _parse(raw: str, target_type: type, var: str) -> Any:
    """Convert one environment string to the field's type, or raise ConfigError."""
    text = raw.strip()
    try:
        if target_type is Path:
            return Path(text)
        if target_type is int:
            return int(text)
        if target_type is float:
            return float(text)
    except ValueError:
        raise ConfigError(
            f"{var}={raw!r} is not a valid {target_type.__name__}"
        ) from None
    return text


def load_settings(env: Mapping[str, str] | None = None, **overrides: Any) -> Settings:
    """Build Settings from defaults, then environment variables, then keyword overrides."""
    source = os.environ if env is None else env
    values: dict[str, Any] = {}
    for f in fields(Settings):
        var = env_name(f.name)
        if var in source:
            values[f.name] = _parse(source[var], f.type, var)  # type: ignore[arg-type]
    values.update(overrides)
    return Settings(**values)


def with_overrides(settings: Settings, **changes: Any) -> Settings:
    """Return a copy of settings with some fields changed (validation runs again)."""
    return replace(settings, **changes)
