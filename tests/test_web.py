"""Static checks on the single-file web page (a real browser test is done by hand)."""

import re
from pathlib import Path

PAGE = (Path(__file__).resolve().parents[1] / "src" / "askdocs" / "web" / "index.html").read_text(
    encoding="utf-8"
)


def test_no_external_resources() -> None:
    assert not re.search(r"""(src|href)\s*=\s*["']?(https?:)?//""", PAGE)
    assert "@import" not in PAGE


def test_never_uses_innerhtml_or_similar() -> None:
    # Look for real use (property access or call), so the comment explaining
    # the rule may still mention the word.
    for sink in (".innerHTML", ".outerHTML", "insertAdjacentHTML(", "document.write(", "eval("):
        assert sink not in PAGE


def test_has_document_library_controls() -> None:
    assert 'type="file"' in PAGE
    assert 'method: "PUT"' in PAGE and 'method: "DELETE"' in PAGE
    # File names go into the URL encoded, never raw.
    assert 'encodeURIComponent(file.name)' in PAGE


def test_has_disclaimer_and_states() -> None:
    disclaimer = (
        "Answers come only from the loaded documents and may be out of date. "
        "Not legal or financial advice. Confirm with the official source."
    )
    assert disclaimer in " ".join(PAGE.split())
    assert "No answer found" in PAGE
    assert 'name="viewport"' in PAGE
    assert 'event.key === "Enter"' in PAGE
