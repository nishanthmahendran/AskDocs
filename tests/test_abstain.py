from askdocs.abstain import should_abstain, term_coverage
from askdocs.models import Chunk, SearchHit


def hit(text: str, keyword_rank: int | None, cosine: float | None) -> SearchHit:
    chunk = Chunk(chunk_id="c1", doc_name="d.md", page=1, index=0, text=text)
    return SearchHit(chunk=chunk, keyword_rank=keyword_rank, vector_score=cosine)


def test_no_hits_abstains() -> None:
    abstain, reason = should_abstain("anything", [], 0.35)
    assert abstain and "no passages" in reason


def test_high_cosine_answers() -> None:
    assert should_abstain("q", [hit("x", None, 0.8)], 0.35) == (False, "")


def test_threshold_is_inclusive() -> None:
    assert should_abstain("q", [hit("x", None, 0.35)], 0.35)[0] is False


def test_low_cosine_and_no_keyword_abstains() -> None:
    abstain, reason = should_abstain("late fee", [hit("unrelated", None, 0.1)], 0.35)
    assert abstain
    assert "0.10" in reason and "0.35" in reason


def test_strong_keyword_match_answers_despite_low_cosine() -> None:
    h = hit("The late fee is two dollars.", keyword_rank=2, cosine=0.1)
    assert should_abstain("What is the late fee?", [h], 0.35)[0] is False


def test_keyword_rank_worse_than_3_does_not_count() -> None:
    h = hit("The late fee is two dollars.", keyword_rank=4, cosine=0.1)
    assert should_abstain("What is the late fee?", [h], 0.35)[0] is True


def test_keyword_match_on_one_common_word_does_not_count() -> None:
    # Rank 1 on "fee" alone: 1 of 4 content words is below 50% coverage.
    h = hit("The fee schedule is posted.", keyword_rank=1, cosine=0.1)
    assert should_abstain("What is the parking fee at the airport lot?", [h], 0.35)[0] is True


def test_term_coverage() -> None:
    assert term_coverage("late fee bikes", "the late fee") == 2 / 3
    assert term_coverage("???", "anything") == 0.0
    cafe = "Caf\N{LATIN SMALL LETTER E WITH ACUTE}"
    assert term_coverage(cafe, f"{cafe.lower()} open") == 1.0
