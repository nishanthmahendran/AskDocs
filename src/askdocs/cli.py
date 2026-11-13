"""Command-line interface: askdocs ingest | ask | serve | docs.

Exit codes: 0 success, 1 user error (bad input, missing folder, empty index),
2 internal error (a bug or unexpected failure).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import NoReturn

from askdocs.config import load_settings
from askdocs.embed import make_embedder
from askdocs.generate import answer_question, make_generator
from askdocs.ingest import ingest_corpus
from askdocs.models import Answer, AskDocsError
from askdocs.retrieve import MODES, Retriever
from askdocs.store import Store

EXIT_OK, EXIT_USER_ERROR, EXIT_INTERNAL_ERROR = 0, 1, 2


class UserError(AskDocsError):
    """A problem with how the command was used."""


class Parser(argparse.ArgumentParser):
    """ArgumentParser that exits with 1 (user error) instead of argparse's default 2."""

    def error(self, message: str) -> NoReturn:
        """Print usage and the problem, then exit with the user-error code."""
        self.print_usage(sys.stderr)
        self.exit(EXIT_USER_ERROR, f"{self.prog}: error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    """Define the commands and their options."""
    parser = Parser(prog="askdocs", description="Ask questions about your local documents.")
    commands = parser.add_subparsers(dest="command", required=True, parser_class=Parser)

    ingest = commands.add_parser("ingest", help="index (or re-sync) the corpus folder")
    ingest.add_argument(
        "--corpus", type=Path, help="folder of documents (default: ASKDOCS_CORPUS_DIR)"
    )

    ask = commands.add_parser("ask", help="ask one question")
    ask.add_argument("question")
    ask.add_argument("--mode", choices=MODES, default="hybrid")
    ask.add_argument("--k", type=int, help="number of passages to use (default: ASKDOCS_TOP_K)")
    ask.add_argument("--json", action="store_true", help="print the full result as JSON")

    serve = commands.add_parser("serve", help="run the web UI and HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    commands.add_parser("docs", help="list ingested documents")
    return parser


def print_answer(answer: Answer) -> None:
    """Human-readable answer with numbered sources."""
    if answer.abstained:
        print(f"No answer found. {answer.answer}")
        if answer.abstain_reason:
            print(f"Reason: {answer.abstain_reason}")
    else:
        print(answer.answer)
    for warning in answer.warnings:
        print(f"Warning: {warning}")
    if answer.citations:
        print("\nSources:")
        for i, c in enumerate(answer.citations, start=1):
            score = f" (similarity {c.score:.2f})" if c.score is not None else ""
            print(f"  {i}. {c.doc_name}, page {c.page}{score}")
            print(f"     {c.snippet}")
    print(f"\n[{answer.backend}, {answer.latency_ms:.0f} ms]")


def run(args: argparse.Namespace) -> int:
    """Execute one parsed command and return its exit code."""
    overrides = {"corpus_dir": args.corpus} if getattr(args, "corpus", None) else {}
    settings = load_settings(**overrides)

    if args.command == "serve":
        import uvicorn  # imported here so other commands start faster

        from askdocs.api import create_app

        app = create_app(settings)
        print(f"AskDocs running at http://{args.host}:{args.port}  (Ctrl+C to stop)")
        uvicorn.run(app, host=args.host, port=args.port, access_log=False, log_level="info")
        return EXIT_OK

    store = Store(settings.db_path)

    if args.command == "docs":
        documents = store.list_documents()
        if not documents:
            print("No documents indexed. Run: askdocs ingest --corpus <folder>")
            return EXIT_OK
        print(f"{'pages':>5} {'chunks':>6}  document")
        for d in documents:
            print(f"{d.pages:>5} {d.chunks:>6}  {d.doc_name}")
        return EXIT_OK

    embedder = make_embedder(settings)

    if args.command == "ingest":
        report = ingest_corpus(settings, store, embedder)
        for warning in report.warnings:
            print(f"warning: {warning}")
        print(
            f"added {report.added}, updated {report.updated}, unchanged {report.unchanged}, "
            f"removed {report.removed}, skipped {report.skipped}"
        )
        docs, chunks = store.counts()
        print(f"index now holds {docs} documents, {chunks} chunks ({settings.db_path})")
        return EXIT_OK

    # args.command == "ask"
    question = args.question.strip()
    if not question:
        raise UserError("the question is empty")
    if len(question) > settings.max_question_chars:
        raise UserError(f"the question is longer than {settings.max_question_chars} characters")
    if args.k is not None and args.k < 1:
        raise UserError("--k must be at least 1")
    if store.counts()[1] == 0:
        raise UserError("the index is empty. Run: askdocs ingest --corpus <folder>")
    retriever = Retriever(settings, store, embedder)
    answer = answer_question(question, settings, retriever, make_generator(settings),
                             mode=args.mode, k=args.k)
    if args.json:
        # ensure_ascii keeps the output safe for any console or file encoding.
        print(json.dumps(answer.model_dump(), indent=2, ensure_ascii=True))
    else:
        print_answer(answer)
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `askdocs` command."""
    # On Windows, output redirected to a file may use a legacy code page;
    # replace unprintable characters instead of crashing.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.command == "serve" else logging.ERROR,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return run(args)
    except AskDocsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USER_ERROR
    except KeyboardInterrupt:
        return EXIT_USER_ERROR
    except Exception:
        print("internal error (this is a bug; details below):", file=sys.stderr)
        traceback.print_exc()
        return EXIT_INTERNAL_ERROR


if __name__ == "__main__":
    sys.exit(main())
