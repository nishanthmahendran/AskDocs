from __future__ import annotations

import logging
import shutil
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from askdocs.api import create_app
from askdocs.config import Settings, with_overrides
from askdocs.embed import HashEmbedder
from askdocs.ingest import ingest_corpus
from askdocs.store import Store
from pdf_helper import build_pdf

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


# ----- upload and delete ------------------------------------------------------

KAYAK = b"Kayak rentals cost twelve dollars per hour. Life jackets are provided."


def put(client: TestClient, name: str, data: bytes) -> object:
    return client.put(f"/documents/{quote(name)}", content=data)


def test_upload_makes_a_document_searchable(client: TestClient, settings: Settings) -> None:
    r = put(client, "kayaks.md", KAYAK)
    assert r.status_code == 200
    body = r.json()
    assert body["doc_name"] == "kayaks.md"
    assert (body["report"]["added"], body["report"]["unchanged"]) == (1, 1)
    assert (settings.corpus_dir / "kayaks.md").read_bytes() == KAYAK
    names = [d["doc_name"] for d in client.get("/documents").json()]
    assert names == ["bikes.md", "kayaks.md"]
    answer = client.post("/ask", json={"question": "How much do kayak rentals cost?"}).json()
    assert answer["abstained"] is False
    assert answer["citations"][0]["doc_name"] == "kayaks.md"


def test_upload_pdf(client: TestClient) -> None:
    pdf = build_pdf([["Page one is about turtles and their long lives."], ["Page two: owls."]])
    r = put(client, "animals.pdf", pdf)
    assert r.status_code == 200
    docs = {d["doc_name"]: d["pages"] for d in client.get("/documents").json()}
    assert docs["animals.pdf"] == 2


def test_upload_unicode_and_spaces_in_name(client: TestClient) -> None:
    name = "café menu 2026.txt"
    assert put(client, name, b"Espresso costs three dollars at the cafe.").status_code == 200
    assert name in [d["doc_name"] for d in client.get("/documents").json()]


def test_upload_same_name_replaces_content(client: TestClient, settings: Settings) -> None:
    put(client, "kayaks.md", KAYAK)
    r = put(client, "kayaks.md", b"Kayak rentals now cost fifteen dollars per hour.")
    assert r.status_code == 200
    assert (r.json()["report"]["updated"], r.json()["report"]["added"]) == (1, 0)
    assert b"fifteen" in (settings.corpus_dir / "kayaks.md").read_bytes()


UNSAFE_NAMES = [
    ".hidden.md", "a.exe", "README.md", "CON.txt", "a%5Cb.md", "a%3Ab.md", "x" * 201 + ".md",
]


@pytest.mark.parametrize("name", UNSAFE_NAMES)
def test_upload_rejects_unsafe_names(client: TestClient, settings: Settings, name: str) -> None:
    before = sorted(p.name for p in settings.corpus_dir.iterdir())
    r = client.put(f"/documents/{name}", content=KAYAK)
    assert r.status_code in (404, 422)
    assert sorted(p.name for p in settings.corpus_dir.iterdir()) == before


def test_upload_cannot_escape_the_corpus_folder(client: TestClient, settings: Settings) -> None:
    for path in ("/documents/..%2Fescaped.md", "/documents/%2E%2E/escaped.md"):
        assert client.put(path, content=KAYAK).status_code in (404, 422)
    assert not (settings.corpus_dir.parent / "escaped.md").exists()


def test_upload_rejects_empty_and_textless_files(client: TestClient, settings: Settings) -> None:
    assert put(client, "empty.md", b"").status_code == 422
    r = put(client, "blank.txt", b"   \n  ")
    assert r.status_code == 422
    assert "No text could be extracted" in r.json()["detail"]
    r = put(client, "broken.pdf", b"%PDF-1.4 garbage")
    assert r.status_code == 422
    assert "could not read PDF" in r.json()["detail"]
    # Nothing was left behind in the corpus folder, hidden temp files included.
    assert sorted(p.name for p in settings.corpus_dir.iterdir()) == ["bikes.md"]


def test_upload_too_large(settings: Settings) -> None:
    small = with_overrides(settings, max_upload_mb=1)
    (small.corpus_dir / "bikes.md").write_text(BIKES, encoding="utf-8")
    with TestClient(create_app(small)) as c:
        limit = 1024 * 1024
        # Declared size too large (Content-Length header).
        assert put(c, "big.txt", b"a " * limit).status_code == 413

        # Size not declared (chunked upload): still stopped by the streaming limit.
        def chunks() -> Iterator[bytes]:
            for _ in range(limit // 1024 + 2):
                yield b"a" * 1024

        assert c.put("/documents/chunked.txt", content=chunks()).status_code == 413
        # Exactly at the limit is fine.
        ok = (b"word " * limit)[:limit]
        assert put(c, "ok.txt", ok).status_code == 200
    assert not (small.corpus_dir / "big.txt").exists()
    assert not (small.corpus_dir / "chunked.txt").exists()


def test_upload_duplicate_content_is_409(client: TestClient, settings: Settings) -> None:
    r = put(client, "copy.md", BIKES.encode())
    assert r.status_code == 409
    assert "bikes.md" in r.json()["detail"]
    assert not (settings.corpus_dir / "copy.md").exists()


def test_upload_during_ingest_is_409(client: TestClient, settings: Settings) -> None:
    lock = client.app.state.ingest_lock  # type: ignore[attr-defined]
    assert lock.acquire(blocking=False)
    try:
        assert put(client, "kayaks.md", KAYAK).status_code == 409
        assert client.delete("/documents/bikes.md").status_code == 409
    finally:
        lock.release()
    assert not (settings.corpus_dir / "kayaks.md").exists()
    assert (settings.corpus_dir / "bikes.md").exists()
    assert put(client, "kayaks.md", KAYAK).status_code == 200  # lock released again


def test_upload_to_read_only_folder_is_403(
    client: TestClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("read-only file system")

    monkeypatch.setattr(Path, "write_bytes", refuse)
    r = put(client, "kayaks.md", KAYAK)
    assert r.status_code == 403
    assert "not writable" in r.json()["detail"]
    monkeypatch.undo()
    assert client.get("/documents").status_code == 200


def test_delete_removes_file_and_index_entry(client: TestClient, settings: Settings) -> None:
    put(client, "kayaks.md", KAYAK)
    r = client.delete("/documents/kayaks.md")
    assert r.status_code == 200
    assert r.json()["removed"] == 1
    assert not (settings.corpus_dir / "kayaks.md").exists()
    assert [d["doc_name"] for d in client.get("/documents").json()] == ["bikes.md"]
    ask = client.post("/ask", json={"question": "kayak rentals cost"}).json()
    assert all(c["doc_name"] != "kayaks.md" for c in ask["citations"])


def test_delete_unknown_or_unsafe_name(client: TestClient) -> None:
    assert client.delete("/documents/nope.md").status_code == 404
    assert client.delete("/documents/.hidden.md").status_code == 422
    assert client.delete("/documents/a.exe").status_code == 422


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
