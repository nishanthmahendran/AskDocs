# Decisions

A short log of every judgment call, grouped by area. "Approved" means the
choice was agreed before implementation.

## Environment and dependencies

- Developed and tested with Python 3.12.10 on Windows 11. Minimum versions in
  `pyproject.toml` are the versions actually installed and tested, except
  `numpy>=2.0`: numpy 2.5 no longer ships Python 3.11 wheels, so a 3.11
  install resolves to numpy 2.4 (checked with a pip dry run for 3.11).
- Build backend: setuptools (a build-time requirement only, not a runtime dependency).
- `pyyaml` is on the allowed list but nothing needs it, so it is not a dependency.
- No new runtime dependencies were added. fastembed brings onnxruntime,
  tokenizers and huggingface-hub as its own dependencies.
- Starlette's TestClient warns that it will move to "httpx2". We keep httpx
  (an allowed dependency) and silence only that warning in the pytest config.
- Packages were installed with `pip --no-cache-dir` (approved) so nothing is
  cached outside the project folder.

## Layout and configuration

- Added `src/askdocs/text.py` (not in the requested layout): one place that
  defines words, stopwords, whitespace normalization and sentence splitting,
  used by ingest, store, abstain and generate. Without it these rules would
  be duplicated or create circular imports.
- Added the setting `model_cache_dir` / `ASKDOCS_MODEL_CACHE_DIR`, default
  `data/models` (approved), so fastembed does not write to the system temp
  folder and the model lives on the Docker data volume.
- `Settings` is a frozen dataclass. `load_settings(env, **overrides)` reads
  `ASKDOCS_*` variables; every validation error names the variable.
  `candidate_k >= top_k` and `0 <= chunk_overlap < chunk_words` are enforced.
- Defaults not given in the spec: generator `extractive` (needs no model and
  sends nothing anywhere), `ollama_url` `http://localhost:11434`,
  `openai_base_url` `http://localhost:8080/v1`, `model_name` `llama3.2:3b`
  (an example; change it to the model you run).

## Ingest

- `doc_name` is the path relative to the corpus folder with forward slashes,
  so names are identical on Windows and Linux.
- Ignored: hidden files and folders, and `README.md` at the corpus root
  (it documents the sources). Unsupported files are logged at INFO level and
  not counted.
- Encrypted PDFs: try the empty password first (many PDFs are "encrypted"
  only to restrict editing); if that fails, skip the file with a warning.
- Duplicate files (identical bytes): the first in sorted order is indexed,
  the others are skipped with a warning.
- A file that becomes empty or unreadable is removed from the index and
  counted as skipped.
- The index records the embedder, vector dimension, chunk settings and an
  `INGEST_VERSION` constant. If any of them differs, ingest rebuilds the whole
  index (the documents are counted as updated). `INGEST_VERSION` was added
  after a re-ingest kept stale chunks: the extraction code had changed but
  the file bytes had not.
- Unicode: no NFC/NFKC normalization (it can change characters); `\s`
  collapsing already covers non-breaking and thin spaces; a leading BOM is
  removed; invalid UTF-8 is replaced with a warning instead of failing.
- Markdown: heading markers (`#`) and bold markers (`**`, `__`) are removed;
  no words change. Links, lists and tables are left as they are. Added after
  the UI showed `## ... **...**` inside answers.
- Hyphens at line breaks are rejoined only between letters, as specified.
  Real compounds split at a line end (`well-` / `known`) become `wellknown`;
  this is a known trade-off.

## Chunking

- `Chunk.index` is the position within its page (0-based).
- The splitter looks back over the last 30% of each window for a sentence end.
  A tail adding few new words is merged into the previous chunk if the result
  is at most 1.3 × `chunk_words`, which is why that is the stated maximum.
- A sentence end is a word ending in `.`, `!` or `?` (plus closing quotes).
  Abbreviations such as "e.g." can end a chunk early. That is harmless,
  because chunk boundaries do not need to be exact sentence boundaries.

## Retrieval and abstention

- FTS5 tokenizer `unicode61 remove_diacritics 2` ("cafe" matches "café"),
  without stemming. The query is the question's content words, each quoted,
  joined with OR. A question with no content words skips keyword search.
- The query vector is always computed, even in keyword mode, so every hit has
  a cosine score for the abstain rule and the UI.
- In keyword-only or vector-only mode, `fused_score` is that list's RRF term,
  so all three modes are comparable.
- Vectors are cached in the retriever and reloaded only when `index_version`
  in the metadata changes.
- Abstain rule: option A (approved). Keyword evidence counts only if a top-3
  keyword hit contains at least 50% of the question's content words.
  "Best hit's cosine" is the maximum cosine over the top_k hits, not only the
  first-ranked hit.
- The default threshold stays at 0.35 as specified. In a quick check with the
  real bge-small model, a related pair scored 0.76 and an unrelated pair 0.39,
  so 0.35 will rarely abstain on real data. Tune it with the sweep. On the
  12 fixture questions with bge-small, retrieval was perfect in all modes
  (hit@1 = 1.0), but at 0.35 nothing abstained: the unanswerable questions
  scored 0.56-0.78 cosine, overlapping the answerable ones (0.56-0.82). The
  sweep's best hybrid value there was 0.80. That is only 12 synthetic
  questions, so it is a warning, not a tuning result.
- Known weakness, deliberately left unchanged (found by hand in the web UI):
  with a threshold of 0.70, "Can I bring my dog?" is still answered. Its
  content words are only `bring` and `dog`; the top keyword hit ("Members may
  bring one guest per visit") contains `bring`, which is exactly 50%, so the
  keyword condition passes regardless of the threshold. Longer questions
  ("...my cat to the store?", 3 content words) abstain correctly. Options
  considered: a coverage constant above 0.5 (also hurts legitimate short
  questions such as "What is the late fee?"), or requiring at least 2 matched
  words. Neither was applied: they would be tuned to a handful of fixture
  questions. Revisit with the real 30-50 question set and compare before/after
  with the evaluation.
- If an LLM answers "I could not find that in the documents", the answer is
  marked as abstained and citations are dropped.

## Generation and API

- Citations are all top_k hits that were given to the generator (document,
  page, chunk id, first 240 characters, cosine score).
- Extractive answers keep only sentences whose word overlap is at least half
  of the best sentence's overlap (at most 3), each followed by
  `[doc, p.N]`. Added after the UI showed sentences matching on one common
  word ("open").
- Prompt-injection hardening: document text appears only inside
  `<context>…</context>` in the user message, with `<` and `>` replaced so a
  document cannot close the block. Both LLM backends use temperature 0.
- No CORS middleware at all: the page is served from the same origin, and
  without CORS headers browsers block other origins. Nothing more is needed
  for "same-origin only".
- Logging: one line per request (method, path, status, latency, plus mode and
  abstained for `/ask`). The question and document text are never logged.
- Status codes: 422 for an empty or too-long question, 409 for a concurrent
  ingest and for an embedder/index mismatch, 400 when the corpus folder is
  missing. Generator failures never produce 500 (extractive fallback).
- CLI: argparse (stdlib). Usage errors exit 1 instead of argparse's default 2.
  Internal errors print a traceback and exit 2. `--json` output is ASCII-only
  so any Windows console or redirect can print it.
- Web page: Enter submits and Shift+Enter adds a new line; no mode selector
  (it was not requested; the API and CLI have `mode`).

## Evaluation

- Each run builds a fresh temporary index, so results never depend on an old
  `data/` folder.
- Citation correctness counts a question as correct only if it was answered
  (not abstained) and a cited chunk contains `expected_phrase`.
- Best threshold = highest abstain accuracy; ties go to the lower threshold.
- `--compare-chunks` uses an overlap of 20% of the chunk size (100/20,
  200/40, 400/80), matching the default ratio.
- Gates apply to hybrid mode. `eval/baseline.json` holds CI gates measured on
  the fixtures (hybrid hit@5 1.000, MRR 0.944, abstain accuracy 1.000), set
  to allow one extra miss. A malformed questions file exits 1 after listing
  every problem. `--no-gate` skips the check.
- `eval/questions.jsonl` ships 6 format examples that point at the fixtures.
- Fixtures are 3 short synthetic documents with 12 questions. The questions
  were written together with the documents and not edited after the first
  measurement. On the fixtures, one unanswerable question (f11) has cosine
  0.340 against a threshold of 0.35, so the CI abstain result has a small
  margin. None of the fixture numbers say anything about real-world quality.

## Docker and CI

- Compose binds the port to `127.0.0.1`; opening it to the LAN for a phone is
  documented as a deliberate edit, because there is no authentication.
- The healthcheck uses Python's urllib (the slim image has no curl), with a
  180-second start period for the first model download.
- CI writes evaluation output to `eval-output/` (gitignored) so it never
  modifies tracked files.
