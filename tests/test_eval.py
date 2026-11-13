"""Tests for eval/run_eval.py (loaded from its file path; eval/ is not a package)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from askdocs.models import Chunk, SearchHit
from conftest import FIXTURES_DIR

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_eval", ROOT / "eval" / "run_eval.py")
assert _spec and _spec.loader
run_eval = importlib.util.module_from_spec(_spec)
# Register before executing: @dataclass looks its module up in sys.modules.
sys.modules["run_eval"] = run_eval
_spec.loader.exec_module(run_eval)


def write_lines(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


GOOD = ('{"id": "a", "question": "q?", "answerable": true, "gold_doc": "d.md", '
        '"gold_pages": [1], "expected_phrase": "x"}')
UNANSWERABLE = ('{"id": "b", "question": "q2?", "answerable": false, "gold_doc": null, '
                '"gold_pages": null, "expected_phrase": null}')


def test_load_questions_reports_every_bad_line(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "q.jsonl", [
        GOOD,
        "{not json",
        '{"id": "c", "question": "", "answerable": "yes"}',
        '{"id": "d", "question": "q", "answerable": true, "gold_doc": null}',
        '{"id": "e", "question": "q", "answerable": true, "gold_doc": "d", "gold_pages": [0]}',
        "",
        GOOD,  # duplicate id
        "[1, 2]",
    ])
    questions, errors = run_eval.load_questions(path)
    assert [q.id for q in questions] == ["a"]
    joined = "\n".join(errors)
    for expected in ("line 2: invalid JSON", "line 3: 'question'", "line 3: 'answerable'",
                     "line 4: answerable questions need a 'gold_doc'", "line 5: 'gold_pages'",
                     "line 7: duplicate id", "line 8: must be a JSON object"):
        assert expected in joined


def test_empty_file_is_an_error(tmp_path: Path) -> None:
    _, errors = run_eval.load_questions(write_lines(tmp_path / "q.jsonl", [""]))
    assert errors == ["the file contains no questions"]


def test_ratio_never_divides_by_zero() -> None:
    assert run_eval.ratio(0, 0) is None
    assert run_eval.ratio(1, 4) == 0.25


def _hit(doc: str, page: int, text: str = "t", cosine: float = 0.9,
         kw: int | None = 1) -> SearchHit:
    chunk = Chunk(chunk_id=f"{doc}{page}", doc_name=doc, page=page, index=0, text=text)
    return SearchHit(chunk=chunk, vector_score=cosine, keyword_rank=kw)


def test_relevance_and_phrase_matching() -> None:
    q = run_eval.Question("a", "q", True, "d.md", [2], "Late  Fee")
    hits = [_hit("d.md", 1), _hit("x.md", 2), _hit("d.md", 2, text="the late\nfee is")]
    assert run_eval.first_relevant_rank(hits, q) == 3
    assert run_eval.contains_phrase(hits, "Late  Fee")
    any_page = run_eval.Question("b", "q", True, "d.md", None, None)
    assert run_eval.first_relevant_rank(hits, any_page) == 1


def test_metrics_with_only_answerable_or_only_unanswerable(settings) -> None:
    ans = run_eval.Record(run_eval.Question("a", "q", True, "d.md", [1], "t"), [_hit("d.md", 1)])
    una = run_eval.Record(run_eval.Question("b", "q", False, None, None, None), [])
    only_ans = run_eval.retrieval_metrics([ans], settings)
    assert only_ans["hit@1"] == 1.0 and only_ans["abstain_recall"] is None
    only_una = run_eval.retrieval_metrics([una], settings)
    assert only_una["hit@1"] is None and only_una["mrr"] is None
    assert only_una["citation_correctness"] is None
    assert only_una["abstain_accuracy"] == 1.0 and only_una["abstain_precision"] == 1.0


def test_check_gates() -> None:
    results = {"modes": {"hybrid": {"hit@5": 0.9, "mrr": 0.5, "abstain_accuracy": None}}}
    failures = run_eval.check_gates(
        results, {"min_hit_at_5": 0.8, "min_mrr": 0.6, "min_abstain_accuracy": 0.5}
    )
    assert len(failures) == 2
    assert "mrr = 0.500 is below the gate 0.6" in failures[0]
    assert "cannot be measured" in failures[1]


def test_full_run_on_fixtures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASKDOCS_EMBEDDER", "hash")
    monkeypatch.setenv("ASKDOCS_GENERATOR", "extractive")
    questions = FIXTURES_DIR / "fixtures_questions.jsonl"
    code = run_eval.main(["--corpus", str(FIXTURES_DIR), "--questions", str(questions),
                          "--out-dir", str(tmp_path), "--compare-chunks"])
    assert code == 0
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    results = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert "## Abstain threshold sweep" in report
    assert "## Chunk size comparison" in report
    assert len(results["threshold_sweep"]) == 21
    assert {row["chunk_words"] for row in results["chunk_comparison"]} == {100, 200, 400}
    assert results["gates"]["failures"] == []

    strict = tmp_path / "strict.json"
    strict.write_text(json.dumps({"mode": "hybrid", "min_mrr": 1.01}), encoding="utf-8")
    code = run_eval.main(["--corpus", str(FIXTURES_DIR), "--questions", str(questions),
                          "--out-dir", str(tmp_path), "--baseline", str(strict)])
    assert code == 1


def test_malformed_questions_exit_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = write_lines(tmp_path / "bad.jsonl", [GOOD, "{oops", UNANSWERABLE, "{oops"])
    assert run_eval.main(["--questions", str(bad), "--out-dir", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "line 2" in err and "line 4" in err
