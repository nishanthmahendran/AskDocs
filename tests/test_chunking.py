"""Chunking tests, including seeded property tests over random pages."""

import itertools
import random

import pytest

from askdocs.chunking import MAX_GROWTH, chunk_page, make_chunk_id, plan_spans
from askdocs.models import Page

VOCAB = ["alpha", "beta", "gamma", "delta", "fee", "rule", "bike", "dock", "time", "pass"]


def random_page(rng: random.Random, n_words: int) -> Page:
    """Build a page of n_words words where roughly 1 in 12 words ends a sentence."""
    out = []
    for _ in range(n_words):
        word = rng.choice(VOCAB)
        if rng.random() < 1 / 12:
            word += rng.choice([".", "!", "?", '."'])
        out.append(word)
    return Page(doc_name="doc.txt", page=3, text=" ".join(out))


CASES = [
    (rng_seed, n_words, chunk_words, overlap)
    for rng_seed in range(5)
    for n_words in (1, 7, 99, 100, 101, 250, 777, 2000)
    for chunk_words, overlap in ((100, 20), (200, 40), (50, 0), (30, 29))
]


@pytest.mark.parametrize(("seed", "n_words", "chunk_words", "overlap"), CASES)
def test_chunk_properties(seed: int, n_words: int, chunk_words: int, overlap: int) -> None:
    page = random_page(random.Random(seed), n_words)
    chunks = chunk_page(page, chunk_words, overlap)
    page_words = page.text.split()

    # Never empty, never too long, page and index preserved.
    assert chunks
    for i, c in enumerate(chunks):
        assert c.text.strip()
        assert len(c.text.split()) <= chunk_words * MAX_GROWTH
        assert (c.page, c.index, c.doc_name) == (3, i, "doc.txt")

    # Overlap is respected: chunk i+1 starts with the last `overlap` words of chunk i.
    for a, b in itertools.pairwise(chunks):
        a_words, b_words = a.text.split(), b.text.split()
        if overlap:
            assert a_words[-overlap:] == b_words[:overlap]

    # Concatenating the non-overlapping parts rebuilds the page exactly.
    rebuilt = chunks[0].text.split()
    for c in chunks[1:]:
        rebuilt += c.text.split()[overlap:]
    assert rebuilt == page_words


def test_short_page_is_one_chunk() -> None:
    page = Page(doc_name="a.md", page=1, text="Just a few words here.")
    chunks = chunk_page(page, 200, 40)
    assert [c.text for c in chunks] == ["Just a few words here."]


def test_empty_page_gives_no_chunks() -> None:
    assert chunk_page(Page(doc_name="a.md", page=1, text="   "), 200, 40) == []


def test_prefers_sentence_end() -> None:
    # 10-word chunks; a sentence ends at word 8, which is inside the 70% window.
    text = "w1 w2 w3 w4 w5 w6 w7 end. w9 w10 w11 w12 w13 w14 w15 w16 w17 w18 w19 w20 w21 w22"
    chunks = chunk_page(Page(doc_name="a", page=1, text=text), 10, 2)
    assert chunks[0].text.endswith("end.")


def test_chunk_id_is_deterministic_and_content_sensitive() -> None:
    a = make_chunk_id("doc", 1, 0, "text")
    assert a == make_chunk_id("doc", 1, 0, "text")
    assert len(a) == 16
    assert a != make_chunk_id("doc", 2, 0, "text")
    assert a != make_chunk_id("doc", 1, 0, "text!")


def test_plan_spans_tail_merge() -> None:
    # 105 words with chunk_words=100 would leave a 5-new-word tail: merged.
    spans = plan_spans([False] * 105, 100, 20)
    assert spans == [(0, 105)]
