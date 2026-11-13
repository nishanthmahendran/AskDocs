from pathlib import Path

import pytest

from askdocs.config import ConfigError, Settings, load_settings, with_overrides


def test_defaults_match_spec() -> None:
    s = load_settings(env={})
    assert s.db_path == Path("data/askdocs.sqlite")
    assert (s.chunk_words, s.chunk_overlap, s.top_k, s.candidate_k) == (200, 40, 5, 20)
    assert s.abstain_threshold == 0.35
    assert s.embedder == "fastembed"
    assert s.embed_model == "BAAI/bge-small-en-v1.5"
    assert s.generator == "extractive"
    assert s.request_timeout == 60
    assert s.max_question_chars == 1000


def test_env_overrides_every_type() -> None:
    env = {
        "ASKDOCS_CORPUS_DIR": "some/folder",
        "ASKDOCS_TOP_K": " 7 ",
        "ASKDOCS_ABSTAIN_THRESHOLD": "0.5",
        "ASKDOCS_EMBEDDER": "hash",
        "ASKDOCS_OPENAI_API_KEY": "abc",
    }
    s = load_settings(env=env)
    assert s.corpus_dir == Path("some/folder")
    assert s.top_k == 7
    assert s.abstain_threshold == 0.5
    assert s.embedder == "hash"
    assert s.openai_api_key == "abc"


def test_keyword_overrides_win_over_env() -> None:
    s = load_settings(env={"ASKDOCS_TOP_K": "7"}, top_k=3)
    assert s.top_k == 3


@pytest.mark.parametrize(
    ("var", "value"),
    [
        ("ASKDOCS_TOP_K", "five"),
        ("ASKDOCS_ABSTAIN_THRESHOLD", "high"),
        ("ASKDOCS_ABSTAIN_THRESHOLD", "1.5"),
        ("ASKDOCS_EMBEDDER", "bert"),
        ("ASKDOCS_GENERATOR", "gpt"),
        ("ASKDOCS_CHUNK_OVERLAP", "200"),
        ("ASKDOCS_CHUNK_WORDS", "0"),
        ("ASKDOCS_CANDIDATE_K", "2"),
    ],
)
def test_invalid_values_name_the_variable(var: str, value: str) -> None:
    with pytest.raises(ConfigError, match=var):
        load_settings(env={var: value})


def test_settings_are_immutable_and_overrides_revalidate() -> None:
    s = Settings()
    with pytest.raises(AttributeError):
        s.top_k = 9  # type: ignore[misc]
    assert with_overrides(s, top_k=9).top_k == 9
    with pytest.raises(ConfigError):
        with_overrides(s, chunk_overlap=-1)
