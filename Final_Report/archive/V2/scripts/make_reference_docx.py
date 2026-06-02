"""Produce a BGU-styled reference.docx for Pandoc.

BGU rules:
  - Times New Roman 12 pt
  - 1.5 line spacing
  - 2.5 cm margins on every side
  - Justify body alignment

Strategy: ask Pandoc to convert a placeholder document into a DOCX (which
gives us a real reference.docx with every Pandoc-relevant style present),
then override the relevant style properties using python-docx.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pypandoc

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Final_Report" / "source" / "bgu_reference.docx"


PLACEHOLDER = """# Heading 1

Sample body paragraph with normal text and *emphasis* and **bold** and `code`.

## Heading 2

A second paragraph.

### Heading 3

A third paragraph.
"""


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("[make_reference_docx] converting placeholder via Pandoc to seed reference.docx")
    pypandoc.convert_text(
        PLACEHOLDER,
        to="docx",
        format="markdown",
        outputfile=str(OUT),
        extra_args=[],
    )
    print(f"  seed -> {OUT}")

    from docx import Document
    from docx.shared import Cm, Pt
    from docx.enum.text import WD_LINE_SPACING, WD_ALIGN_PARAGRAPH

    d = Document(str(OUT))

    for s in d.sections:
        s.top_margin = Cm(2.5)
        s.bottom_margin = Cm(2.5)
        s.left_margin = Cm(2.5)
        s.right_margin = Cm(2.5)

    target_styles = [
        "Normal", "Body Text", "Compact", "First Paragraph",
        "Heading 1", "Heading 2", "Heading 3", "Heading 4",
        "Title", "Subtitle",
        "Caption", "Table Caption", "Image Caption",
        "Author", "Date",
        "Verbatim Char", "Source Code", "Footnote Text",
        "TOC Heading", "toc 1", "toc 2", "toc 3",
        "List Paragraph",
        "Block Text",
    ]
    n_changed = 0
    for name in target_styles:
        try:
            st = d.styles[name]
        except KeyError:
            continue
        try:
            if st.font is not None:
                st.font.name = "Times New Roman"
                st.font.size = Pt(12)
        except AttributeError:
            pass
        try:
            pf = st.paragraph_format
            pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
            pf.line_spacing = 1.5
            if name in ("Normal", "Body Text", "Compact", "First Paragraph", "List Paragraph", "Block Text"):
                pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        except AttributeError:
            pass
        n_changed += 1

    d.save(str(OUT))
    print(f"[make_reference_docx] overrode {n_changed} styles -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
