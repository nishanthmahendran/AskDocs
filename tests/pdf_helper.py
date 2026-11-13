"""Write tiny but valid PDF files in plain Python, so no binary fixtures are committed.

Each page's text is drawn with the built-in Helvetica font, one line per
entry in the page's list of lines. Only ASCII text is supported, which is all
the tests need (Unicode handling is tested with .txt / .md files).
"""

from __future__ import annotations

from pathlib import Path


def _escape(line: str) -> str:
    """Escape the characters that are special inside a PDF string literal."""
    return line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _content_stream(lines: list[str]) -> bytes:
    """PDF drawing commands that print each line 14 points below the previous one."""
    if not lines:
        return b""
    ops = ["BT", "/F1 10 Tf", "14 TL", "40 800 Td"]
    for line in lines:
        ops.append(f"({_escape(line)}) Tj T*")
    ops.append("ET")
    return "\n".join(ops).encode("ascii")


def build_pdf(pages: list[list[str]]) -> bytes:
    """Return the bytes of a PDF with one page per entry in `pages` (a list of lines)."""
    n = len(pages)
    # Object numbers: 1 catalog, 2 page tree, 3 font, then (page, content) pairs.
    page_ids = [4 + 2 * i for i in range(n)]
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: (
            "<< /Type /Pages /Kids ["
            + " ".join(f"{pid} 0 R" for pid in page_ids)
            + f"] /Count {n} >>"
        ).encode("ascii"),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for pid, lines in zip(page_ids, pages, strict=True):
        stream = _content_stream(lines)
        objects[pid] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {pid + 1} 0 R >>"
        ).encode("ascii")
        objects[pid + 1] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"\nendstream"
        )

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for num in sorted(objects):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode("ascii") + objects[num] + b"\nendobj\n"
    xref_pos = len(out)
    size = max(objects) + 1
    out += f"xref\n0 {size}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for num in range(1, size):
        out += f"{offsets[num]:010d} 00000 n \n".encode("ascii")
    out += f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode(
        "ascii"
    )
    return bytes(out)


def write_pdf(path: Path, pages: list[list[str]]) -> Path:
    """Write build_pdf(pages) to path and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(build_pdf(pages))
    return path
