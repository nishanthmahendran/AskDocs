"""Split page text into overlapping word-window chunks.

Chunks never span pages, so every citation's page number is exact.
"""

from __future__ import annotations

import hashlib
import re

from askdocs.models import Chunk, Page

# A word "ends a sentence" if it ends with . ! or ?, optionally followed by
# closing quotes or brackets, e.g.  end.  end."  (end.)
_SENTENCE_END_RE = re.compile(
    r"[.!?][\"')\]\N{RIGHT DOUBLE QUOTATION MARK}\N{RIGHT SINGLE QUOTATION MARK}]*$"
)

# When looking for a sentence end to break at, search back at most this far
# (as a fraction of chunk_words) from the hard limit. 0.7 means a chunk is at
# least ~70% of chunk_words long unless the page itself is shorter.
MIN_FILL = 0.7
# A last chunk adding this few new words is merged into the previous chunk,
# as long as the merged chunk stays within MAX_GROWTH * chunk_words.
MAX_GROWTH = 1.3


def make_chunk_id(doc_name: str, page: int, index: int, text: str) -> str:
    """First 16 hex characters of sha256("doc_name|page|index|text")."""
    key = f"{doc_name}|{page}|{index}|{text}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def plan_spans(
    sentence_ends: list[bool], chunk_words: int, chunk_overlap: int
) -> list[tuple[int, int]]:
    """Return (start, end) word-index spans; consecutive spans share chunk_overlap words."""
    n = len(sentence_ends)
    if n == 0:
        return []
    if n <= chunk_words:
        return [(0, n)]

    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        limit = start + chunk_words
        if limit >= n:
            spans.append((start, n))
            break
        # Prefer to end right after a sentence end, searching backwards from the
        # limit. The lower bound is > start + chunk_overlap so the next chunk
        # always starts later than this one (guarantees progress).
        lowest = start + max(chunk_overlap + 1, int(chunk_words * MIN_FILL))
        end = limit
        for candidate in range(limit, lowest - 1, -1):
            if sentence_ends[candidate - 1]:
                end = candidate
                break
        spans.append((start, end))
        start = end - chunk_overlap

    # Merge a short tail into the previous chunk instead of emitting a tiny chunk.
    if len(spans) >= 2:
        prev_start, _ = spans[-2]
        _, last_end = spans[-1]
        if last_end - prev_start <= int(chunk_words * MAX_GROWTH):
            spans[-2:] = [(prev_start, last_end)]
    return spans


def chunk_page(page: Page, chunk_words: int, chunk_overlap: int) -> list[Chunk]:
    """Split one page into chunks. Empty pages produce no chunks."""
    page_words = page.text.split()
    ends = [bool(_SENTENCE_END_RE.search(w)) for w in page_words]
    chunks = []
    for index, (start, end) in enumerate(plan_spans(ends, chunk_words, chunk_overlap)):
        text = " ".join(page_words[start:end])
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(page.doc_name, page.page, index, text),
                doc_name=page.doc_name,
                page=page.page,
                index=index,
                text=text,
            )
        )
    return chunks


def chunk_pages(pages: list[Page], chunk_words: int, chunk_overlap: int) -> list[Chunk]:
    """Chunk every page of a document, in page order."""
    chunks: list[Chunk] = []
    for page in pages:
        chunks.extend(chunk_page(page, chunk_words, chunk_overlap))
    return chunks
