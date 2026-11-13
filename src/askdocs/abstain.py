"""Decide whether the evidence is too weak to answer.

The whole rule lives in should_abstain() so it is easy to read and tune.
eval/run_eval.py sweeps the threshold and reports the trade-off.
"""

from __future__ import annotations

from askdocs.models import SearchHit
from askdocs.text import content_terms

# A keyword hit only counts as evidence if it ranks this high...
KEYWORD_RANK_LIMIT = 3
# ...and contains at least this fraction of the question's content words.
# Without this, an OR keyword query matches almost any question on one common
# word (e.g. "fee"), every question gets keyword rank 1, and we never abstain.
MIN_TERM_COVERAGE = 0.5


def term_coverage(question: str, passage: str) -> float:
    """Fraction of the question's content words that appear in the passage (0.0-1.0)."""
    question_terms = set(content_terms(question))
    if not question_terms:
        return 0.0
    return len(question_terms & set(content_terms(passage))) / len(question_terms)


def should_abstain(question: str, hits: list[SearchHit], threshold: float) -> tuple[bool, str]:
    """Return (abstain?, reason). Abstain when no hit has strong vector OR keyword evidence.

    Rule:
      1. No hits at all -> abstain.
      2. Vector evidence is strong if the best cosine similarity >= threshold.
      3. Keyword evidence is strong if some hit has keyword rank <= 3 AND
         covers >= 50% of the question's content words.
      4. Abstain only if neither 2 nor 3 holds.
    """
    if not hits:
        return True, "no passages matched the question"

    best_cosine = max((h.vector_score for h in hits if h.vector_score is not None), default=0.0)
    if best_cosine >= threshold:
        return False, ""

    for hit in hits:
        ranked_high = hit.keyword_rank is not None and hit.keyword_rank <= KEYWORD_RANK_LIMIT
        if ranked_high and term_coverage(question, hit.chunk.text) >= MIN_TERM_COVERAGE:
            return False, ""

    return True, (
        f"best similarity {best_cosine:.2f} is below the threshold {threshold:.2f} "
        f"and no top-{KEYWORD_RANK_LIMIT} keyword match contains at least "
        f"{MIN_TERM_COVERAGE:.0%} of the question's key words"
    )
