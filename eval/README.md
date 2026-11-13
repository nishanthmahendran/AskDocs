# Evaluation

`run_eval.py` measures how well AskDocs finds the right passage and how well
it decides when to say "I could not find that". Every run ingests the corpus
into a fresh temporary database, so the numbers depend only on the documents,
the questions and the settings, all of which are printed in the report.

## Writing questions

`questions.jsonl` has one JSON object per line. The 6 lines shipped here are
format examples that point at the test fixtures; replace them with 30-50
questions about your own corpus.

```json
{"id": "q01", "question": "What is the late fee?", "answerable": true, "gold_doc": "rules.pdf", "gold_pages": [4], "expected_phrase": "two dollars per hour"}
{"id": "q31", "question": "Who is the CEO?", "answerable": false, "gold_doc": null, "gold_pages": null, "expected_phrase": null}
```

| field | meaning |
|---|---|
| `id` | unique label, shown in the per-question table |
| `question` | asked exactly as written |
| `answerable` | `true` if the corpus contains the answer |
| `gold_doc` | document name as AskDocs shows it: the path relative to the corpus folder with `/`, e.g. `manuals/rules.pdf` (required when answerable) |
| `gold_pages` | 1-based pages that contain the answer, or `null` to accept any page of `gold_doc` |
| `expected_phrase` | a short exact phrase from the answer passage (case and whitespace are ignored), or `null` |

Tips for an honest set:

- Write questions the way a real user would phrase them, not by copying
  sentences from the documents. Copied wording makes keyword search look
  better than it is.
- Write the questions before you look at results, and do not edit them to
  make a number go up.
- Include roughly 20-30% unanswerable questions, and make some of them
  "near misses" that use the corpus's vocabulary (e.g. asking for a fee
  the documents do not list). Those are the ones that test abstention.
- Use `askdocs ask "..." --json` to see which page a passage is on.

Malformed lines are all reported with their line numbers; nothing runs until
they are fixed.

## Running

From the repository root (with the virtual environment active):

```
python eval/run_eval.py                                   # corpus/ + eval/questions.jsonl
python eval/run_eval.py --compare-chunks                  # also try chunk_words 100/200/400
python eval/run_eval.py --corpus tests/fixtures --questions tests/fixtures/fixtures_questions.jsonl
```

Options: `--questions`, `--corpus`, `--out-dir` (default `eval/`),
`--baseline` (default `eval/baseline.json`), `--no-gate`, `--compare-chunks`.
Settings such as `ASKDOCS_EMBEDDER` and `ASKDOCS_ABSTAIN_THRESHOLD` come from
the environment as usual.

## What is measured

For each retrieval mode (`keyword`, `vector`, `hybrid`):

- **hit@1 / hit@3 / hit@5**: share of answerable questions where a chunk from
  `gold_doc` (and one of `gold_pages`) is in the top 1 / 3 / 5.
- **MRR**: mean of 1 / (rank of the first correct chunk) over the top
  `candidate_k`, counting 0 when none is found.
- **citation correctness**: share of answerable questions with an
  `expected_phrase` that were answered (not abstained) with at least one cited
  chunk (the top `top_k`) containing the phrase.
- **abstain accuracy / precision / recall / false abstain**: abstaining on an
  unanswerable question is correct; abstaining on an answerable one is a
  false abstain. Precision = correct abstentions / all abstentions;
  recall = correct abstentions / unanswerable questions.

Any metric whose denominator is zero (for example, no unanswerable questions)
is reported as `n/a` instead of dividing by zero.

The threshold sweep re-applies the abstain rule at 0.00, 0.05, ..., 1.00 to
the same retrieval results and reports the best threshold per mode (ties go
to the lower threshold). Set the winner with `ASKDOCS_ABSTAIN_THRESHOLD`, but
remember that a threshold tuned on the same questions you report on will look
better than it really is. Hold out some questions if you can.

## Outputs and gates

- `eval/report.md`: tables, the exact settings, document versions copied from
  `corpus/README.md`, and the date.
- `eval/results.json`: the same data, machine-readable.
- Exit code 1 if any gate in the baseline file is violated (minimum hybrid
  hit@5, MRR and abstain accuracy). `eval/baseline.json` holds the CI gates
  for the fixtures; keep a separate file for your corpus, e.g.
  `--baseline eval/baseline_real.json`.
