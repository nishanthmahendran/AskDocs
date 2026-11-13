# AskDocs evaluation report

- Date (UTC): 2026-11-13 08:36
- Questions: `tests/fixtures/fixtures_questions.jsonl` (12 total: 9 answerable, 3 unanswerable)
- Corpus: `tests/fixtures`

## Settings

| setting | value |
|---|---|
| corpus_dir | `tests/fixtures` |
| db_path | `temporary database, rebuilt for every evaluation run` |
| model_cache_dir | `data/models` |
| chunk_words | `200` |
| chunk_overlap | `40` |
| top_k | `5` |
| candidate_k | `20` |
| abstain_threshold | `0.35` |
| embedder | `hash` |
| embed_model | `BAAI/bge-small-en-v1.5` |
| generator | `extractive` |
| ollama_url | `http://localhost:11434` |
| openai_base_url | `http://localhost:8080/v1` |
| openai_api_key | `(not set)` |
| model_name | `llama3.2:3b` |
| request_timeout | `60.0` |
| max_question_chars | `1000` |
| embedder_name | `hash` |

## Document versions (from corpus README.md)

No README.md table found in the corpus folder.

## Retrieval and abstention (threshold 0.35, top_k 5)

| mode | hit@1 | hit@3 | hit@5 | MRR | citation correct | abstain acc | abstain precision | abstain recall | false abstain |
|---|---|---|---|---|---|---|---|---|---|
| keyword | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| vector | 0.889 | 1.000 | 1.000 | 0.944 | 0.111 | 0.333 | 0.273 | 1.000 | 0.889 |
| hybrid | 0.889 | 1.000 | 1.000 | 0.944 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

## Abstain threshold sweep

Accuracy = share of questions where the abstain decision was right. Precision/recall/false-abstain columns are for hybrid mode.

| threshold | keyword acc | vector acc | hybrid acc | hybrid precision | hybrid recall | hybrid false abstain |
|---|---|---|---|---|---|---|
| 0.00 | 0.750 | 0.750 | 0.750 | n/a | 0.000 | 0.000 |
| 0.05 | 0.750 | 0.750 | 0.750 | n/a | 0.000 | 0.000 |
| 0.10 | 0.833 | 0.667 | 0.750 | n/a | 0.000 | 0.000 |
| 0.15 | 0.833 | 0.667 | 0.750 | n/a | 0.000 | 0.000 |
| 0.20 | 0.917 | 0.750 | 0.917 | 1.000 | 0.667 | 0.000 |
| 0.25 | 0.917 | 0.500 | 0.917 | 1.000 | 0.667 | 0.000 |
| 0.30 | 0.917 | 0.417 | 0.917 | 1.000 | 0.667 | 0.000 |
| 0.35 | 1.000 | 0.333 | 1.000 | 1.000 | 1.000 | 0.000 |
| 0.40 | 1.000 | 0.250 | 1.000 | 1.000 | 1.000 | 0.000 |
| 0.45 | 1.000 | 0.250 | 1.000 | 1.000 | 1.000 | 0.000 |
| 0.50 | 1.000 | 0.250 | 1.000 | 1.000 | 1.000 | 0.000 |
| 0.55 | 1.000 | 0.250 | 1.000 | 1.000 | 1.000 | 0.000 |
| 0.60 | 1.000 | 0.250 | 1.000 | 1.000 | 1.000 | 0.000 |
| 0.65 | 1.000 | 0.250 | 1.000 | 1.000 | 1.000 | 0.000 |
| 0.70 | 1.000 | 0.250 | 1.000 | 1.000 | 1.000 | 0.000 |
| 0.75 | 1.000 | 0.250 | 1.000 | 1.000 | 1.000 | 0.000 |
| 0.80 | 1.000 | 0.250 | 1.000 | 1.000 | 1.000 | 0.000 |
| 0.85 | 1.000 | 0.250 | 1.000 | 1.000 | 1.000 | 0.000 |
| 0.90 | 1.000 | 0.250 | 1.000 | 1.000 | 1.000 | 0.000 |
| 0.95 | 1.000 | 0.250 | 1.000 | 1.000 | 1.000 | 0.000 |
| 1.00 | 1.000 | 0.250 | 1.000 | 1.000 | 1.000 | 0.000 |

Best threshold per mode (highest accuracy; ties go to the lower threshold):

- keyword: 0.35 (accuracy 1.000)
- vector: 0.00 (accuracy 0.750)
- hybrid: 0.35 (accuracy 1.000)

## Per-question results (hybrid)

| id | answerable | first gold rank | top hit | best cosine | abstained |
|---|---|---|---|---|---|
| f01 | True | 1 | bikeshare_rules.md p.1 | 0.358 | False |
| f02 | True | 1 | bikeshare_rules.md p.1 | 0.275 | False |
| f03 | True | 1 | bikeshare_rules.md p.1 | 0.220 | False |
| f04 | True | 2 | makerspace_handbook.txt p.3 | 0.249 | False |
| f05 | True | 1 | makerspace_handbook.txt p.2 | 0.333 | False |
| f06 | True | 1 | makerspace_handbook.txt p.3 | 0.072 | False |
| f07 | True | 1 | release_notes.md p.1 | 0.311 | False |
| f08 | True | 1 | release_notes.md p.1 | 0.231 | False |
| f09 | True | 1 | release_notes.md p.1 | 0.191 | False |
| f10 | False | - | makerspace_handbook.txt p.1 | 0.177 | True |
| f11 | False | - | bikeshare_rules.md p.1 | 0.340 | True |
| f12 | False | - | release_notes.md p.1 | 0.182 | True |

## Gates

All gates passed.
