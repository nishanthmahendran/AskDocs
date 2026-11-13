# AskDocs

Ask questions about a folder of your own PDF, Markdown and text files, and get
an answer with citations (document, page, passage). If the documents do not
contain good evidence, AskDocs says it could not find the answer instead of
guessing. Everything runs locally on a CPU, and nothing leaves your machine by
default.

It is meant for one person who wants trustworthy lookups in a fixed set of
documents (manuals, policies, course handbooks, release notes) from a laptop
or a phone on the same network.

## 60-second quickstart

You need Python 3.11 or newer. The first `ingest` with the default embedder
downloads a ~70 MB model once (into `data/models/`); after that everything
works offline.

### Windows 11 (PowerShell)

```powershell
py -V:3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -e ".[dev]"
askdocs ingest --corpus tests/fixtures
askdocs ask "How old do you have to be to rent a bike on your own?"
askdocs serve                         # then open http://127.0.0.1:8000
```

### Linux / Kubuntu

```bash
sudo apt install python3-venv          # once, if venv is missing
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
askdocs ingest --corpus tests/fixtures
askdocs ask "How old do you have to be to rent a bike on your own?"
askdocs serve                          # then open http://127.0.0.1:8000
```

To try it without downloading anything, set `ASKDOCS_EMBEDDER=hash` first
(`$env:ASKDOCS_EMBEDDER="hash"` in PowerShell, `export ASKDOCS_EMBEDDER=hash`
in bash). The hash embedder is for tests only and gives poor results.

### With Docker

**You must test the container on your own machine; it was not run while
building this project.** Put your documents in `corpus/` first.

Windows 11 (Docker Desktop with the WSL2 backend), in PowerShell:

```powershell
docker compose build
docker compose run --rm askdocs askdocs ingest     # first run downloads the model into the data volume
docker compose up -d
curl.exe http://127.0.0.1:8000/health
docker compose logs -f askdocs                      # Ctrl+C to stop following
docker compose down
```

Kubuntu (Docker Engine with the compose plugin):

```bash
sudo apt install docker.io docker-compose-v2      # once; or Docker's own packages
sudo usermod -aG docker "$USER"                   # once, then log out and back in
docker compose build
docker compose run --rm askdocs askdocs ingest
docker compose up -d
curl http://127.0.0.1:8000/health
docker compose down
```

Notes:

- The model downloads on first use and is cached in the `askdocs-data` volume.
  Running `ingest` once (as above) is how you pre-download it before starting
  the server. Re-run `ingest` whenever you change `corpus/`, or use
  `POST /ingest`.
- `corpus/` is mounted writable so the web page can upload and delete
  documents (add `:ro` in `docker-compose.yml` to forbid that). On Linux, if
  uploads fail with "not writable", the host folder is not writable by the
  container's user (uid 10001); `chmod a+w corpus` fixes it. The container
  runs as a non-root user.
- The port is bound to `127.0.0.1`. To reach it from your phone, change it to
  `"8000:8000"` in `docker-compose.yml` and open `http://<your-PC-IP>:8000`.
  There is no login, so do this only on a network you trust (and allow the
  port in Windows Firewall if needed). Without Docker, use
  `askdocs serve --host 0.0.0.0` for the same effect.

## Commands

| command | what it does |
|---|---|
| `askdocs ingest [--corpus DIR]` | add new files, re-index changed ones, drop deleted ones |
| `askdocs ask "question" [--mode hybrid\|keyword\|vector] [--k 5] [--json]` | answer one question |
| `askdocs serve [--host 127.0.0.1 --port 8000]` | web page at `/` and the HTTP API |
| `askdocs docs` | list indexed documents |

Exit codes: 0 success, 1 user error (missing folder, empty index, bad input),
2 internal error.

HTTP API: `POST /ask {"question": "...", "k": 5, "mode": "hybrid"}`,
`GET /health`, `GET /documents`, `POST /ingest` (409 if one is already running),
`PUT /documents/{filename}` (body = the raw file; adds or replaces one
document), `DELETE /documents/{filename}`.

## Configuration

Every setting can be overridden with an `ASKDOCS_*` environment variable
(see `src/askdocs/config.py`). Invalid values stop the program with an error
naming the variable.

| variable | default | meaning |
|---|---|---|
| `ASKDOCS_CORPUS_DIR` | `corpus` | folder of documents |
| `ASKDOCS_DB_PATH` | `data/askdocs.sqlite` | the index |
| `ASKDOCS_MODEL_CACHE_DIR` | `data/models` | where the embedding model is cached |
| `ASKDOCS_CHUNK_WORDS` / `ASKDOCS_CHUNK_OVERLAP` | `200` / `40` | chunk size and overlap, in words |
| `ASKDOCS_TOP_K` / `ASKDOCS_CANDIDATE_K` | `5` / `20` | passages returned / candidates per search |
| `ASKDOCS_ABSTAIN_THRESHOLD` | `0.35` | starting value; tune it with the evaluation |
| `ASKDOCS_EMBEDDER` | `fastembed` | `fastembed` or `hash` (tests only) |
| `ASKDOCS_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | fastembed model |
| `ASKDOCS_GENERATOR` | `extractive` | `extractive`, `ollama` or `openai` |
| `ASKDOCS_OLLAMA_URL` | `http://localhost:11434` | Ollama server |
| `ASKDOCS_OPENAI_BASE_URL` | `http://localhost:8080/v1` | any OpenAI-compatible server |
| `ASKDOCS_OPENAI_API_KEY` | (empty) | sent only if set |
| `ASKDOCS_MODEL_NAME` | `llama3.2:3b` | model name for Ollama / OpenAI |
| `ASKDOCS_REQUEST_TIMEOUT` | `60` | seconds |
| `ASKDOCS_MAX_QUESTION_CHARS` | `1000` | longer questions are rejected |
| `ASKDOCS_MAX_UPLOAD_MB` | `25` | largest file the web page may upload |

## Architecture

```
                 corpus/ (PDF, .md, .txt)
                          |
                    ingest.py  -- SHA-256 per file: add / update / remove
           (pypdf page by page, whitespace + hyphen fix)
                          |
                   chunking.py  -- ~200-word chunks, 40 overlap, never across pages
                          |
              +-----------+-----------+
              |                       |
          embed.py                 store.py (SQLite)
   fastembed (ONNX, CPU)        documents | chunks | chunks_fts (FTS5)
   or hash (tests only)         vectors (float32) | meta (embedder, dim, ...)
              |                       |
              +-----------+-----------+
                          |
  question --> retrieve.py: FTS5 bm25 top 20 + cosine top 20 --> RRF (k=60) --> top 5
                          |
                    abstain.py  -- weak evidence? --> "I could not find that in the documents."
                          |
                   generate.py  -- extractive (default) | Ollama | OpenAI-compatible
                          |        (any backend failure falls back to extractive)
                          |
          Answer + citations (document, page, snippet, similarity)
                          |
              api.py (FastAPI + web/index.html)   cli.py
```

## How retrieval works

1. **Keyword search.** The question is reduced to its content words (letters
   and digits only, common words like "the" removed), each word is quoted, and
   SQLite FTS5 ranks chunks with bm25. Operators and punctuation in the
   question can never reach the query.
2. **Vector search.** The question is embedded with the same model as the
   chunks, and every chunk vector is compared by cosine similarity (exact,
   brute force; `retrieve.VectorIndex` is the seam for a faster index).
3. **Fusion.** Reciprocal Rank Fusion adds `1 / (60 + rank)` from each list.
   Ties are broken by chunk id, so results are deterministic.

Use `--mode keyword|vector|hybrid` to compare them; the evaluation does this
automatically.

## How abstaining works

The rule lives in one function, `abstain.should_abstain`. AskDocs answers
only if at least one of these is true for the top 5 passages:

- the best cosine similarity is at or above `ASKDOCS_ABSTAIN_THRESHOLD`, or
- a passage in the top 3 keyword results contains at least half of the
  question's content words.

Otherwise it replies "I could not find that in the documents." and gives the
reason. The second condition exists because an OR keyword search matches
almost any question on a single common word. If an LLM backend replies that
the context lacks the answer, that is also reported as an abstention.

The default threshold (0.35) is only a starting point. In a quick check with
bge-small, an unrelated question/passage pair scored 0.39, so you will
probably need a higher value. Use the threshold sweep in the evaluation to
choose it.

## Adding documents

Either use the web page: open the **Documents** section, choose one or more
files, and they are saved into the corpus folder and indexed immediately. Each
document has a Delete button. Or do it by hand:

1. Copy files into `corpus/` (subfolders are fine). Supported: `.pdf`, `.md`,
   `.markdown`, `.txt`. Other files are ignored.
2. Record each one in `corpus/README.md` (source URL, version, license).
3. Run `askdocs ingest`. Unchanged files are skipped, changed files are
   re-indexed, deleted files are removed. Scanned pages, corrupt or
   password-protected PDFs and duplicate files produce warnings, not crashes.

Uploads are checked before anything is saved: the file name must be a plain
name (no folders, no hidden or reserved names) with a supported extension, the
size must be within `ASKDOCS_MAX_UPLOAD_MB`, and text must be extractable.
Uploaded files go directly into the corpus folder (no subfolders). There is
no login: anyone who can open the page can add or delete documents, so keep
the server on `127.0.0.1` or a network you trust.

In text files, a form-feed character starts a new page.

## Evaluation

See [eval/README.md](eval/README.md) for how to write questions and what each
metric means. In short:

```
python eval/run_eval.py --compare-chunks
```

This writes `eval/report.md` and `eval/results.json`. The copies currently in
the repository were produced on the synthetic test fixtures with the hash
embedder; they demonstrate the format, not real-world quality.

## Results

Fill in after running eval on the real corpus (`python eval/run_eval.py --compare-chunks`).

Corpus: fill in after running eval (documents, versions, number of questions).

| mode | hit@1 | hit@3 | hit@5 | MRR | citation correctness | abstain accuracy |
|---|---|---|---|---|---|---|
| keyword | fill in after running eval | | | | | |
| vector | fill in after running eval | | | | | |
| hybrid | fill in after running eval | | | | | |

| chunk_words | hybrid hit@5 | hybrid MRR |
|---|---|---|
| 100 | fill in after running eval | |
| 200 | fill in after running eval | |
| 400 | fill in after running eval | |

Chosen abstain threshold and the trade-off at that value: fill in after running eval.

## Known limitations

- No OCR: scanned PDF pages have no extractable text and are skipped with a warning.
- Tables, multi-column layouts and footnotes extract poorly from PDFs.
- Single user: no login, no per-user libraries (everyone shares one set of
  documents, and anyone who can reach the page can upload or delete); one
  ingest or upload at a time.
- English only: the stopword list and the default embedding model are English.
- No stemming: "fees" and "fee" are different words to keyword search.
- Not legal or financial advice. Answers are only as current as the document
  versions you loaded; always confirm with the official source.
- The extractive answer is a selection of sentences, not a written answer.

## Development

```
pip install -e ".[dev]"
ruff check .
pytest            # offline; tests use ASKDOCS_EMBEDDER=hash and the extractive generator
```

Built with AI assistance. Design decisions, the evaluation set, tuning, and
measurements are mine.
