"""Small text helpers shared by ingest, store, abstain and generate.

Kept in one place so that "what counts as a word" is defined exactly once.
"""

from __future__ import annotations

import re

# A short English stopword list. These words are dropped from keyword queries
# and from the word-overlap checks, because they match almost every passage.
STOPWORDS = frozenset(
    [
        "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
        "could", "did", "do", "does", "for", "from", "had", "has", "have", "how",
        "i", "if", "in", "into", "is", "it", "its", "me", "my", "of", "on", "or",
        "our", "s", "should", "so", "t", "than", "that", "the", "their", "them",
        "then", "there", "these", "they", "this", "to", "was", "we", "were",
        "what", "when", "where", "which", "who", "whom", "why", "will", "with",
        "would", "you", "your",
    ]
)  # fmt: skip

_WORD_RE = re.compile(r"\w+")
# Rejoin "exam-\nple" -> "example" only when both sides are letters.
# [^\W\d_] means "a word character that is not a digit or underscore", i.e. a letter.
_HYPHEN_BREAK_RE = re.compile(r"(?<=[^\W\d_])-[ \t]*\r?\n[ \t]*(?=[^\W\d_])")
# In Python 3 str patterns, \s also matches Unicode spaces such as the
# non-breaking space (U+00A0) and thin space (U+2009).
_WHITESPACE_RE = re.compile(r"\s+")
# Split after . ! or ? optionally followed by one closing quote or bracket.
# Two fixed-width lookbehinds because Python lookbehinds cannot vary in width.
_CLOSERS = "\"')]\N{RIGHT DOUBLE QUOTATION MARK}\N{RIGHT SINGLE QUOTATION MARK}"
_SENTENCE_SPLIT_RE = re.compile(
    r"(?:(?<=[.!?])|(?<=[.!?][" + re.escape(_CLOSERS) + r"]))\s+"
)
_BOM = "\N{ZERO WIDTH NO-BREAK SPACE}"  # U+FEFF, the byte-order mark
# Markdown markup that is not part of the words: heading markers at the start
# of a line ("## Title") and bold markers (**text**, __text__).
_MD_HEADING_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+", re.MULTILINE)
_MD_BOLD_RE = re.compile(r"\*\*|__")


def strip_markdown(text: str) -> str:
    """Remove Markdown heading and bold markers; every word is kept."""
    return _MD_BOLD_RE.sub("", _MD_HEADING_RE.sub("", text))


def normalize_text(text: str) -> str:
    """Rejoin hyphenated line breaks and collapse whitespace without changing words."""
    text = text.replace(_BOM, "")
    text = _HYPHEN_BREAK_RE.sub("", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def words(text: str) -> list[str]:
    """Lower-case word tokens (letters, digits, underscore; any script)."""
    return _WORD_RE.findall(text.lower())


def content_terms(text: str) -> list[str]:
    """Word tokens minus stopwords, de-duplicated, in first-seen order."""
    seen: dict[str, None] = {}
    for word in words(text):
        if word not in STOPWORDS:
            seen.setdefault(word, None)
    return list(seen)


def split_sentences(text: str) -> list[str]:
    """Split normalized text into sentences at ., ! or ? followed by whitespace."""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
