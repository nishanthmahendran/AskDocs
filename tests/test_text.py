from askdocs.text import content_terms, normalize_text, split_sentences, strip_markdown, words

# Named escapes keep invisible / look-alike characters readable in the source.
NBSP = "\N{NO-BREAK SPACE}"
THIN = "\N{THIN SPACE}"
LQ, RQ = "\N{LEFT DOUBLE QUOTATION MARK}", "\N{RIGHT DOUBLE QUOTATION MARK}"
APOS = "\N{RIGHT SINGLE QUOTATION MARK}"
E_ACUTE = "\N{LATIN SMALL LETTER E WITH ACUTE}"


def test_normalize_collapses_whitespace_including_unicode_spaces() -> None:
    raw = f"Hello{NBSP} world\n\n\tnext{THIN}line  "
    assert normalize_text(raw) == "Hello world next line"


def test_normalize_rejoins_hyphen_only_between_letters() -> None:
    assert normalize_text("exam-\nple text") == "example text"
    accented = f"caf{E_ACUTE}-\n{E_ACUTE}t{E_ACUTE}"
    assert normalize_text(accented) == f"caf{E_ACUTE}{E_ACUTE}t{E_ACUTE}"
    # Digits on either side: keep the hyphen (probably a range or a code).
    assert normalize_text("pages 10-\n20") == "pages 10- 20"
    # A hyphen without a line break is never touched.
    assert normalize_text("well-known") == "well-known"


def test_normalize_keeps_curly_quotes_and_accents() -> None:
    raw = f"{LQ}Caf{E_ACUTE}{RQ} isn{APOS}t naive"
    assert normalize_text(raw) == raw


def test_normalize_strips_bom() -> None:
    assert normalize_text("\N{ZERO WIDTH NO-BREAK SPACE}Hello") == "Hello"


def test_words_and_content_terms() -> None:
    assert words("What's the C++ fee?") == ["what", "s", "the", "c", "fee"]
    assert content_terms("What's the C++ fee? The fee!") == ["c", "fee"]
    assert content_terms(f"caf{E_ACUTE} AND ( * \"") == [f"caf{E_ACUTE}"]
    assert content_terms("?!...") == []


def test_strip_markdown_keeps_words() -> None:
    raw = "# Title\n  ## Sub heading\n- **Bold.** text C# and C++ #tag __x__ 2**3"
    assert strip_markdown(raw) == "Title\nSub heading\n- Bold. text C# and C++ #tag x 23"


def test_split_sentences_keeps_closing_quotes() -> None:
    text = f'He said "stop." Then left! Why? {LQ}Fine.{RQ} Done'
    expected = ['He said "stop."', "Then left!", "Why?", f"{LQ}Fine.{RQ}", "Done"]
    assert split_sentences(text) == expected
