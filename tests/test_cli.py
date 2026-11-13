"""CLI tests: run main() in-process with ASKDOCS_* settings pointing at a temp folder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import askdocs.cli as cli
from askdocs.cli import main


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, corpus_dir: Path) -> None:
    monkeypatch.setenv("ASKDOCS_DB_PATH", str(tmp_path / "cli.sqlite"))
    monkeypatch.setenv("ASKDOCS_CORPUS_DIR", str(corpus_dir))
    monkeypatch.setenv("ASKDOCS_EMBEDDER", "hash")
    monkeypatch.setenv("ASKDOCS_GENERATOR", "extractive")
    (corpus_dir / "bikes.md").write_text(
        "Bike share rules. The late fee is two dollars per hour.", encoding="utf-8"
    )


def test_ingest_then_ask(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ingest"]) == 0
    assert "added 1, updated 0, unchanged 0, removed 0, skipped 0" in capsys.readouterr().out
    assert main(["ask", "What is the late fee?"]) == 0
    out = capsys.readouterr().out
    assert "two dollars" in out
    assert "1. bikes.md, page 1" in out


def test_ask_json(capsys: pytest.CaptureFixture[str]) -> None:
    main(["ingest"])
    capsys.readouterr()
    assert main(["ask", "late fee", "--json", "--mode", "keyword", "--k", "2"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["abstained"] is False
    assert data["citations"][0]["doc_name"] == "bikes.md"


def test_docs_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["docs"]) == 0
    assert "No documents indexed" in capsys.readouterr().out
    main(["ingest"])
    capsys.readouterr()
    main(["docs"])
    assert "bikes.md" in capsys.readouterr().out


def test_missing_folder_is_user_error_without_traceback(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert main(["ingest", "--corpus", str(tmp_path / "missing")]) == 1
    err = capsys.readouterr().err
    assert "Corpus folder not found" in err
    assert "Traceback" not in err


def test_empty_index_is_user_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ask", "anything"]) == 1
    assert "index is empty" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [["ask", "   "], ["ask", "q", "--k", "0"]])
def test_bad_question_is_user_error(argv: list[str]) -> None:
    main(["ingest"])
    assert main(argv) == 1


@pytest.mark.parametrize("argv", [[], ["ask"], ["ask", "q", "--mode", "fuzzy"], ["frobnicate"]])
def test_bad_arguments_exit_1(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 1


def test_bad_env_is_user_error(monkeypatch: pytest.MonkeyPatch,
                               capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("ASKDOCS_TOP_K", "lots")
    assert main(["docs"]) == 1
    assert "ASKDOCS_TOP_K" in capsys.readouterr().err


def test_unexpected_error_exits_2(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("bug")

    monkeypatch.setattr(cli, "ingest_corpus", boom)
    assert main(["ingest"]) == 2


def test_abstain_output(capsys: pytest.CaptureFixture[str]) -> None:
    main(["ingest"])
    capsys.readouterr()
    assert main(["ask", "Who painted the chapel ceiling?"]) == 0
    assert "No answer found" in capsys.readouterr().out
