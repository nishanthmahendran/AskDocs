"""End-to-end: ingest tests/fixtures through the API and ask every fixture question.

Uses the default settings (abstain_threshold 0.35, top_k 5) with the offline
hash embedder and the extractive generator, exactly like CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from askdocs.api import create_app
from askdocs.config import load_settings
from conftest import FIXTURES_DIR

QUESTIONS = [
    json.loads(line)
    for line in (FIXTURES_DIR / "fixtures_questions.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    tmp: Path = tmp_path_factory.mktemp("e2e")
    settings = load_settings(env={}, corpus_dir=FIXTURES_DIR, db_path=tmp / "e2e.sqlite",
                             embedder="hash", generator="extractive")
    with TestClient(create_app(settings)) as c:
        report = c.post("/ingest").json()
        assert report["added"] == 3 and report["skipped"] == 0
        yield c


def test_fixture_set_shape() -> None:
    assert len(QUESTIONS) == 12
    assert sum(not q["answerable"] for q in QUESTIONS) == 3


def test_three_documents_with_exact_pages(client: TestClient) -> None:
    docs = {d["doc_name"]: d["pages"] for d in client.get("/documents").json()}
    assert docs == {"bikeshare_rules.md": 1, "makerspace_handbook.txt": 3, "release_notes.md": 1}


@pytest.mark.parametrize("q", [q for q in QUESTIONS if q["answerable"]], ids=lambda q: q["id"])
def test_answerable_questions_cite_the_right_page(client: TestClient, q: dict) -> None:
    body = client.post("/ask", json={"question": q["question"]}).json()
    assert body["abstained"] is False, body["abstain_reason"]
    cited = {(c["doc_name"], c["page"]) for c in body["citations"]}
    assert any((q["gold_doc"], page) in cited for page in q["gold_pages"])
    # The top-ranked citation is from the right document.
    assert body["citations"][0]["doc_name"] == q["gold_doc"]


@pytest.mark.parametrize("q", [q for q in QUESTIONS if not q["answerable"]], ids=lambda q: q["id"])
def test_unanswerable_questions_abstain(client: TestClient, q: dict) -> None:
    body = client.post("/ask", json={"question": q["question"]}).json()
    assert body["abstained"] is True
    assert body["citations"] == []
