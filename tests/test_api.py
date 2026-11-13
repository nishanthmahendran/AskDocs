from __future__ import annotations

import logging
import shutil

import pytest
from fastapi.testclient import TestClient

from askdocs.api import create_app
from askdocs.config import Settings, with_overrides
from askdocs.embed import HashEmbedder
from askdocs.ingest import ingest_corpus
from askdocs.store import Store

BIKES = "Bike share rules. The late fee is two dollars per hour after the first day."


@pytest.fixture
def client(settings: Settings) -> TestClient:
    (settings.corpus_dir / "bikes.md").write_text(BIKES, encoding="utf-8")
    app = create_app(settings)
    with TestClient(app) as c:
        assert c.post("/ingest").status_code == 200
        yield c


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body == {"status": "ok", "documents": 1, "chunks": 1,
                    "embedder": "hash", "generator": "extractive"}


def test_documents(client: TestClient) -> None:
    docs = client.get("/documents").json()
    assert [(d["doc_name"], d["pages"], d["chunks"]) for d in docs] == [("bikes.md", 1, 1)]


def test_ask_answers_with_citation(client: TestClient) -> None:
    body = client.post("/ask", json={"question": "What is the late fee?"}).json()
    assert body["abstained"] is False
    assert "two dollars" in body["answer"]
    assert body["citations"][0]["doc_name"] == "bikes.md"
    assert body["citations"][0]["page"] == 1
    assert body["backend"] == "extractive"
    assert body["latency_ms"] >= 0


@pytest.mark.parametrize("mode", ["keyword", "vector", "hybrid"])
def test_ask_modes(client: TestClient, mode: str) -> None:
    r = client.post("/ask", json={"question": "late fee", "mode": mode, "k": 2})
    assert r.status_code == 200


def test_ask_abstains(client: TestClient) -> None:
    body = client.post("/ask", json={"question": "Who wrote the national anthem?"}).json()
    assert body["abstained"] is True
    assert body["citations"] == []


@pytest.mark.parametrize("question", ["", "   ", "x" * 1001])
def test_ask_rejects_empty_or_long(client: TestClient, question: str) -> None:
    r = client.post("/ask", json={"question": question})
    assert r.status_code == 422
    assert "question" in r.json()["detail"].lower()


BAD_BODIES = [{}, {"question": "q", "mode": "fuzzy"}, {"question": "q", "k": 0}]


@pytest.mark.parametrize("body", BAD_BODIES)
def test_ask_rejects_invalid_body(client: TestClient, body: dict) -> None:
    assert client.post("/ask", json=body).status_code == 422


@pytest.mark.parametrize("question", ["a", "?!", 'AND ( * "', "What's the C++ fee?"])
def test_ask_odd_questions_return_200(client: TestClient, question: str) -> None:
    assert client.post("/ask", json={"question": question}).status_code == 200


def test_concurrent_ingest_returns_409(client: TestClient) -> None:
    lock = client.app.state.ingest_lock  # type: ignore[attr-defined]
    assert lock.acquire(blocking=False)  # simulate an ingest in progress
    try:
        r = client.post("/ingest")
        assert r.status_code == 409
    finally:
        lock.release()
    assert client.post("/ingest").status_code == 200


def test_ingest_reports_counts(client: TestClient, settings: Settings) -> None:
    (settings.corpus_dir / "new.md").write_text("Kayak rentals.", encoding="utf-8")
    body = client.post("/ingest").json()
    assert (body["added"], body["unchanged"]) == (1, 1)


def test_ingest_missing_corpus_is_400(client: TestClient, settings: Settings) -> None:
    shutil.rmtree(settings.corpus_dir)
    r = client.post("/ingest")
    assert r.status_code == 400
    assert "not found" in r.json()["detail"]


def test_index_page_served(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "<title>AskDocs</title>" in r.text


def test_no_cors_headers(client: TestClient) -> None:
    r = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers


def test_embedder_mismatch_is_409(settings: Settings) -> None:
    (settings.corpus_dir / "bikes.md").write_text(BIKES, encoding="utf-8")
    ingest_corpus(settings, Store(settings.db_path), HashEmbedder(dim=128))
    with TestClient(create_app(settings)) as c:
        r = c.post("/ask", json={"question": "late fee"})
    assert r.status_code == 409
    assert "askdocs ingest" in r.json()["detail"]


def test_request_log_has_no_question_text(client: TestClient,
                                          caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="askdocs.api"):
        client.post("/ask", json={"question": "secret-question-words late fee"})
    lines = [r.getMessage() for r in caplog.records if r.name == "askdocs.api"]
    assert any("POST /ask 200" in line and "mode=hybrid" in line and "abstained=" in line
               for line in lines)
    assert not any("secret-question-words" in line or "two dollars" in line for line in lines)


def test_generator_outage_never_500(settings: Settings) -> None:
    (settings.corpus_dir / "bikes.md").write_text(BIKES, encoding="utf-8")
    broken = with_overrides(settings, generator="ollama", ollama_url="http://127.0.0.1:9",
                            request_timeout=2)
    with TestClient(create_app(broken)) as c:
        c.post("/ingest")
        body = c.post("/ask", json={"question": "What is the late fee?"}).json()
    assert body["backend"] == "extractive (fallback)"
    assert body["warnings"]
