"""Evaluate retrieval, citations and abstention on a JSONL question set.

Usage (from the repository root):
    python eval/run_eval.py --corpus tests/fixtures \
        --questions tests/fixtures/fixtures_questions.jsonl
    python eval/run_eval.py                      # your corpus/ and eval/questions.jsonl
    python eval/run_eval.py --compare-chunks     # also compare chunk sizes 100/200/400

Every run ingests the corpus into a fresh temporary database, so results
depend only on the documents, the questions and the settings printed in the
report. Outputs: eval/report.md and eval/results.json. Exit code 1 if a gate
in the baseline file is violated or the questions file is invalid.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from askdocs.abstain import should_abstain
from askdocs.config import Settings, load_settings, with_overrides
from askdocs.embed import Embedder, make_embedder
from askdocs.ingest import ingest_corpus
from askdocs.models import SearchHit
from askdocs.retrieve import MODES, Retriever
from askdocs.store import Store
from askdocs.text import normalize_text

EVAL_DIR = Path(__file__).resolve().parent
THRESHOLDS = [round(i * 0.05, 2) for i in range(21)]  # 0.00, 0.05, ..., 1.00
HIT_KS = (1, 3, 5)
CHUNK_SIZES = (100, 200, 400)


# ----- questions --------------------------------------------------------------


@dataclass(frozen=True)
class Question:
    """One evaluation question."""

    id: str
    question: str
    answerable: bool
    gold_doc: str | None
    gold_pages: list[int] | None
    expected_phrase: str | None


def _check_line(data: Any) -> list[str]:
    """Return a list of problems with one parsed JSON line (empty if valid)."""
    if not isinstance(data, dict):
        return ["must be a JSON object"]
    problems = []
    for key in ("id", "question"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            problems.append(f"'{key}' must be a non-empty string")
    if not isinstance(data.get("answerable"), bool):
        problems.append("'answerable' must be true or false")
    gold_doc = data.get("gold_doc")
    pages = data.get("gold_pages")
    phrase = data.get("expected_phrase")
    if gold_doc is not None and not isinstance(gold_doc, str):
        problems.append("'gold_doc' must be a string or null")
    if data.get("answerable") is True and not gold_doc:
        problems.append("answerable questions need a 'gold_doc'")
    if pages is not None and (
        not isinstance(pages, list)
        or not all(isinstance(p, int) and not isinstance(p, bool) and p >= 1 for p in pages)
    ):
        problems.append("'gold_pages' must be a list of page numbers (1 or more) or null")
    if phrase is not None and (not isinstance(phrase, str) or not phrase.strip()):
        problems.append("'expected_phrase' must be a non-empty string or null")
    return problems


def load_questions(path: Path) -> tuple[list[Question], list[str]]:
    """Parse every line; return (questions, errors), collecting ALL errors, not just the first."""
    questions: list[Question] = []
    errors: list[str] = []
    seen_ids: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {number}: invalid JSON ({exc.msg})")
            continue
        problems = _check_line(data)
        if not problems and data["id"] in seen_ids:
            problems.append(f"duplicate id {data['id']!r}")
        if problems:
            errors.extend(f"line {number}: {p}" for p in problems)
            continue
        seen_ids.add(data["id"])
        questions.append(
            Question(
                id=data["id"],
                question=data["question"],
                answerable=data["answerable"],
                gold_doc=data.get("gold_doc"),
                gold_pages=data.get("gold_pages"),
                expected_phrase=data.get("expected_phrase"),
            )
        )
    if not questions and not errors:
        errors.append("the file contains no questions")
    return questions, errors


# ----- metrics ----------------------------------------------------------------


def ratio(numerator: int, denominator: int) -> float | None:
    """numerator / denominator, or None when the denominator is 0 (never divides by zero)."""
    return numerator / denominator if denominator else None


def is_relevant(hit: SearchHit, q: Question) -> bool:
    """A hit is relevant if it comes from gold_doc and (when given) one of gold_pages."""
    if hit.chunk.doc_name != q.gold_doc:
        return False
    return q.gold_pages is None or hit.chunk.page in q.gold_pages


def first_relevant_rank(hits: list[SearchHit], q: Question) -> int | None:
    """1-based rank of the first relevant hit, or None."""
    for rank, hit in enumerate(hits, start=1):
        if is_relevant(hit, q):
            return rank
    return None


def contains_phrase(hits: list[SearchHit], phrase: str) -> bool:
    """True if any hit's text contains phrase (case-insensitive, whitespace-normalized)."""
    wanted = normalize_text(phrase).lower()
    return any(wanted in normalize_text(h.chunk.text).lower() for h in hits)


@dataclass
class Record:
    """Retrieval results for one question in one mode."""

    question: Question
    hits: list[SearchHit]  # candidate_k hits, used for hit@k and MRR


def abstain_counts(records: list[Record], threshold: float, top_k: int) -> dict[str, Any]:
    """Abstain accuracy / precision / recall at one threshold.

    Abstaining on an unanswerable question is correct (true positive);
    abstaining on an answerable one is a miss (false abstain).
    """
    tp = fp = fn = tn = 0
    for r in records:
        abstained, _ = should_abstain(r.question.question, r.hits[:top_k], threshold)
        if not r.question.answerable:
            tp, fn = tp + abstained, fn + (not abstained)
        else:
            fp, tn = fp + abstained, tn + (not abstained)
    return {
        "accuracy": ratio(tp + tn, len(records)),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "false_abstain_rate": ratio(fp, fp + tn),
    }


def retrieval_metrics(records: list[Record], settings: Settings) -> dict[str, Any]:
    """hit@k, MRR, citation correctness and abstain metrics at the configured threshold."""
    answerable = [r for r in records if r.question.answerable]
    ranks = [first_relevant_rank(r.hits, r.question) for r in answerable]
    metrics: dict[str, Any] = {
        f"hit@{k}": ratio(sum(1 for rank in ranks if rank is not None and rank <= k), len(ranks))
        for k in HIT_KS
    }
    metrics["mrr"] = (
        sum(1 / rank for rank in ranks if rank is not None) / len(ranks) if ranks else None
    )

    # Citation correctness: of answerable questions with an expected_phrase, how
    # many were answered (not abstained) with a cited chunk containing it.
    # The cited chunks are the top_k hits, exactly as the app cites them.
    with_phrase = [r for r in answerable if r.question.expected_phrase]
    correct = 0
    for r in with_phrase:
        cited = r.hits[: settings.top_k]
        abstained, _ = should_abstain(r.question.question, cited, settings.abstain_threshold)
        if not abstained and contains_phrase(cited, r.question.expected_phrase or ""):
            correct += 1
    metrics["citation_correctness"] = ratio(correct, len(with_phrase))
    metrics.update(
        {f"abstain_{k}": v for k, v in
         abstain_counts(records, settings.abstain_threshold, settings.top_k).items()}
    )
    return metrics


def run_mode(questions: list[Question], retriever: Retriever, mode: str,
             settings: Settings) -> list[Record]:
    """Retrieve candidate_k hits for every question in one mode."""
    return [
        Record(q, retriever.search(q.question, mode=mode, k=settings.candidate_k))  # type: ignore[arg-type]
        for q in questions
    ]


def build_index(settings: Settings, embedder: Embedder) -> tuple[Retriever, list[str]]:
    """Ingest settings.corpus_dir into settings.db_path; return a retriever and warnings."""
    store = Store(settings.db_path)
    report = ingest_corpus(settings, store, embedder)
    return Retriever(settings, store, embedder), report.warnings


def evaluate(questions: list[Question], settings: Settings, embedder: Embedder,
             compare_chunks: bool) -> dict[str, Any]:
    """Run every measurement and return one results dictionary."""
    with tempfile.TemporaryDirectory(prefix="askdocs-eval-", ignore_cleanup_errors=True) as tmp:
        main_settings = with_overrides(settings, db_path=Path(tmp) / "eval.sqlite")
        retriever, warnings = build_index(main_settings, embedder)
        records = {mode: run_mode(questions, retriever, mode, settings) for mode in MODES}

        modes = {mode: retrieval_metrics(records[mode], settings) for mode in MODES}
        sweep = [
            {"threshold": t,
             **{mode: abstain_counts(records[mode], t, settings.top_k) for mode in MODES}}
            for t in THRESHOLDS
        ]
        best = {}
        for mode in MODES:
            scored = [(row[mode]["accuracy"], row["threshold"]) for row in sweep
                      if row[mode]["accuracy"] is not None]
            if scored:
                # Highest accuracy; on ties the lowest threshold (abstains least).
                accuracy, threshold = max(scored, key=lambda s: (s[0], -s[1]))
                best[mode] = {"threshold": threshold, "accuracy": accuracy}

        per_question = []
        for r in records["hybrid"]:
            top = r.hits[0] if r.hits else None
            abstained, _ = should_abstain(r.question.question, r.hits[: settings.top_k],
                                          settings.abstain_threshold)
            per_question.append({
                "id": r.question.id,
                "answerable": r.question.answerable,
                "first_gold_rank": first_relevant_rank(r.hits, r.question)
                if r.question.answerable else None,
                "top_doc": top.chunk.doc_name if top else None,
                "top_page": top.chunk.page if top else None,
                "best_cosine": max((h.vector_score or 0.0 for h in r.hits[: settings.top_k]),
                                   default=None),
                "abstained": abstained,
            })

        chunk_rows = []
        if compare_chunks:
            for size in CHUNK_SIZES:
                # Overlap is kept at 20% of the chunk size (40 for the default 200).
                s = with_overrides(settings, chunk_words=size, chunk_overlap=size // 5,
                                   db_path=Path(tmp) / f"chunks_{size}.sqlite")
                chunk_retriever, _ = build_index(s, embedder)
                for mode in MODES:
                    m = retrieval_metrics(run_mode(questions, chunk_retriever, mode, s), s)
                    chunk_rows.append({"chunk_words": size, "chunk_overlap": size // 5,
                                       "mode": mode, **m})

    return {
        "modes": modes,
        "threshold_sweep": sweep,
        "best_threshold": best,
        "per_question_hybrid": per_question,
        "chunk_comparison": chunk_rows,
        "ingest_warnings": warnings,
    }


# ----- gates and report -------------------------------------------------------


def check_gates(results: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    """Return a description of every violated gate (empty list = all passed)."""
    mode = baseline.get("mode", "hybrid")
    metrics = results["modes"][mode]
    gates = {"min_hit_at_5": "hit@5", "min_mrr": "mrr", "min_abstain_accuracy": "abstain_accuracy"}
    failures = []
    for gate, metric in gates.items():
        if gate not in baseline:
            continue
        value = metrics.get(metric)
        if value is None:
            failures.append(f"{mode} {metric}: cannot be measured (no matching questions)")
        elif value < baseline[gate]:
            failures.append(f"{mode} {metric} = {value:.3f} is below the gate {baseline[gate]}")
    return failures


def setting_text(settings: Settings, name: str) -> str:
    """One setting as report text: paths with forward slashes, the API key hidden."""
    value = getattr(settings, name)
    if name == "db_path":
        return "temporary database, rebuilt for every evaluation run"
    if name == "openai_api_key":
        return "***" if value else "(not set)"
    return value.as_posix() if isinstance(value, Path) else str(value)


def fmt(value: float | None) -> str:
    """Format a metric for the report."""
    return "n/a" if value is None else f"{value:.3f}"


def corpus_table(corpus_dir: Path) -> list[str]:
    """The Markdown table rows from <corpus>/README.md (document versions), if any."""
    readme = corpus_dir / "README.md"
    if not readme.is_file():
        return []
    return [line for line in readme.read_text(encoding="utf-8").splitlines()
            if line.strip().startswith("|")]


def render_report(results: dict[str, Any], settings: Settings, questions: list[Question],
                  questions_path: Path, gate_failures: list[str] | None) -> str:
    """Build the Markdown report."""
    n_ans = sum(q.answerable for q in questions)
    lines = [
        "# AskDocs evaluation report",
        "",
        f"- Date (UTC): {results['date']}",
        f"- Questions: `{questions_path.as_posix()}` ({len(questions)} total: "
        f"{n_ans} answerable, {len(questions) - n_ans} unanswerable)",
        f"- Corpus: `{settings.corpus_dir.as_posix()}`",
        "",
        "## Settings",
        "",
        "| setting | value |",
        "|---|---|",
    ]
    for key, value in results["settings"].items():
        lines.append(f"| {key} | `{value}` |")

    lines += ["", "## Document versions (from corpus README.md)", ""]
    table = corpus_table(settings.corpus_dir)
    lines += table if table else ["No README.md table found in the corpus folder."]

    lines += [
        "",
        f"## Retrieval and abstention (threshold {settings.abstain_threshold}, "
        f"top_k {settings.top_k})",
        "",
        "| mode | hit@1 | hit@3 | hit@5 | MRR | citation correct | abstain acc "
        "| abstain precision | abstain recall | false abstain |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for mode, m in results["modes"].items():
        lines.append(
            f"| {mode} | {fmt(m['hit@1'])} | {fmt(m['hit@3'])} | {fmt(m['hit@5'])} "
            f"| {fmt(m['mrr'])} | {fmt(m['citation_correctness'])} "
            f"| {fmt(m['abstain_accuracy'])} | {fmt(m['abstain_precision'])} "
            f"| {fmt(m['abstain_recall'])} | {fmt(m['abstain_false_abstain_rate'])} |"
        )

    lines += [
        "",
        "## Abstain threshold sweep",
        "",
        "Accuracy = share of questions where the abstain decision was right. "
        "Precision/recall/false-abstain columns are for hybrid mode.",
        "",
        "| threshold | keyword acc | vector acc | hybrid acc | hybrid precision "
        "| hybrid recall | hybrid false abstain |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in results["threshold_sweep"]:
        h = row["hybrid"]
        lines.append(
            f"| {row['threshold']:.2f} | {fmt(row['keyword']['accuracy'])} "
            f"| {fmt(row['vector']['accuracy'])} | {fmt(h['accuracy'])} | {fmt(h['precision'])} "
            f"| {fmt(h['recall'])} | {fmt(h['false_abstain_rate'])} |"
        )
    lines += ["", "Best threshold per mode (highest accuracy; ties go to the lower threshold):", ""]
    for mode, b in results["best_threshold"].items():
        lines.append(f"- {mode}: {b['threshold']:.2f} (accuracy {fmt(b['accuracy'])})")

    if results["chunk_comparison"]:
        lines += [
            "",
            "## Chunk size comparison",
            "",
            "| chunk_words | overlap | mode | hit@1 | hit@3 | hit@5 | MRR | abstain acc |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for row in results["chunk_comparison"]:
            lines.append(
                f"| {row['chunk_words']} | {row['chunk_overlap']} | {row['mode']} "
                f"| {fmt(row['hit@1'])} | {fmt(row['hit@3'])} | {fmt(row['hit@5'])} "
                f"| {fmt(row['mrr'])} | {fmt(row['abstain_accuracy'])} |"
            )

    lines += [
        "",
        "## Per-question results (hybrid)",
        "",
        "| id | answerable | first gold rank | top hit | best cosine | abstained |",
        "|---|---|---|---|---|---|",
    ]
    for q in results["per_question_hybrid"]:
        top = f"{q['top_doc']} p.{q['top_page']}" if q["top_doc"] else "-"
        rank = q["first_gold_rank"] if q["first_gold_rank"] is not None else "-"
        lines.append(
            f"| {q['id']} | {q['answerable']} | {rank} | {top} "
            f"| {fmt(q['best_cosine'])} | {q['abstained']} |"
        )

    if results["ingest_warnings"]:
        lines += ["", "## Ingest warnings", ""]
        lines += [f"- {w}" for w in results["ingest_warnings"]]

    lines += ["", "## Gates", ""]
    if gate_failures is None:
        lines.append("No baseline file used.")
    elif gate_failures:
        lines += [f"- FAILED: {f}" for f in gate_failures]
    else:
        lines.append("All gates passed.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation and write report.md and results.json."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=Path, default=EVAL_DIR / "questions.jsonl")
    parser.add_argument("--corpus", type=Path, help="default: ASKDOCS_CORPUS_DIR")
    parser.add_argument("--out-dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--baseline", type=Path, default=EVAL_DIR / "baseline.json")
    parser.add_argument("--no-gate", action="store_true", help="do not check the baseline")
    parser.add_argument("--compare-chunks", action="store_true",
                        help="also evaluate chunk_words 100, 200 and 400")
    args = parser.parse_args(argv)

    overrides = {"corpus_dir": args.corpus} if args.corpus else {}
    settings = load_settings(**overrides)
    questions, errors = load_questions(args.questions)
    if errors:
        print(f"{args.questions}: {len(errors)} problem(s):", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1

    embedder = make_embedder(settings)
    results = evaluate(questions, settings, embedder, args.compare_chunks)
    results["date"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
    results["questions_file"] = args.questions.as_posix()
    results["settings"] = {f.name: setting_text(settings, f.name) for f in fields(settings)}
    results["settings"]["embedder_name"] = embedder.name
    results["questions"] = [asdict(q) for q in questions]

    gate_failures = None
    if not args.no_gate:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        gate_failures = check_gates(results, baseline)
        results["gates"] = {"baseline": baseline, "failures": gate_failures}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = render_report(results, settings, questions, args.questions, gate_failures)
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")
    (args.out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    for mode, m in results["modes"].items():
        print(f"{mode:8} hit@1={fmt(m['hit@1'])} hit@5={fmt(m['hit@5'])} mrr={fmt(m['mrr'])} "
              f"abstain_acc={fmt(m['abstain_accuracy'])} "
              f"citations={fmt(m['citation_correctness'])}")
    print(f"wrote {args.out_dir / 'report.md'} and {args.out_dir / 'results.json'}")
    if gate_failures:
        for failure in gate_failures:
            print(f"GATE FAILED: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
