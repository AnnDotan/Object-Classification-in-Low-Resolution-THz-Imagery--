"""Render Final_Exp.md to artifacts/Final_Exp.pdf.

Reproduces the manual VSCode "Markdown PDF" export the operator was relying
on, so the PDF artifact does not drift again from the markdown source.

Engine: `markdown-pdf` (pure-Python, no system dependencies). Install with
``pip install markdown-pdf`` if missing.

Usage:
    python scripts/render_final_exp_pdf.py
    python scripts/render_final_exp_pdf.py --src Final_Exp.md --out artifacts/Final_Exp.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SRC = _REPO_ROOT / "docs" / "Final_Exp_Report.md"
_DEFAULT_OUT = _REPO_ROOT / "artifacts" / "Final_Exp.pdf"

_CSS = """
body { font-family: "Segoe UI", Arial, sans-serif; color: #1f2530;
       font-size: 11pt; line-height: 1.4;
       word-wrap: break-word; overflow-wrap: break-word; }
h1 { color: #0d4f8b; border-bottom: 2px solid #0d4f8b; padding-bottom: 4px;
     font-size: 20pt; margin-top: 14pt; margin-bottom: 10pt;
     word-wrap: break-word; overflow-wrap: break-word; }
h2 { color: #0d4f8b; border-bottom: 1px solid #cfd6e0; padding-bottom: 2px;
     margin-top: 22pt; font-size: 15pt;
     word-wrap: break-word; overflow-wrap: break-word; }
h3 { color: #2a3957; font-size: 12.5pt; margin-top: 16pt; }
h4 { color: #2a3957; font-size: 11.5pt; margin-top: 12pt; }
p, li { font-size: 10.5pt; }
code { background: #f1f3f6; padding: 1px 4px; border-radius: 3px;
       font-family: Consolas, "Courier New", monospace; font-size: 9.5pt;
       word-break: break-all; }
pre { background: #f1f3f6; padding: 8px; border-radius: 4px;
      font-size: 9pt; white-space: pre-wrap; word-wrap: break-word; }
table { border-collapse: collapse; width: 100%;
        font-size: 9.5pt; margin: 10px 0;
        table-layout: auto; }
th, td { border: 1px solid #cfd6e0; padding: 5px 7px;
         text-align: left; vertical-align: top;
         /* keep identifiers like `transnext_tiny` on a single line: */
         word-break: keep-all; }
th { background: #eef2f7; font-weight: 600; white-space: nowrap; }
/* Numeric-leaning columns (right-aligned in result tables for legibility) */
td.num { text-align: right; font-variant-numeric: tabular-nums; }
/* `code` inside table cells: don't force-break short tags */
td code, th code { word-break: keep-all; white-space: nowrap; }
"""


def render(src: Path, out: Path) -> int:
    try:
        from markdown_pdf import MarkdownPdf, Section  # type: ignore
    except ImportError:
        print(
            "ERROR: markdown-pdf is not installed. Run:\n"
            "  pip install markdown-pdf",
            file=sys.stderr,
        )
        return 2

    if not src.exists():
        print(f"ERROR: source markdown not found: {src}", file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    md = src.read_text(encoding="utf-8")

    pdf = MarkdownPdf(toc_level=2, optimize=True)
    # A3 landscape (1191 x 842 pt = 16.54 x 11.7 inch) gives ~420 mm of
    # usable width — wide enough that the cross-model L5 axis table
    # (7 columns) and the full Phase C result table (8 columns) fit
    # comfortably at 9.5 pt without column truncation.
    # Borders convention is (left, top, right, bottom) where right/bottom
    # are NEGATIVE offsets from the opposite page edge — passing
    # `(36, 36, -36, -36)` yields a 36 pt (~12.7 mm) margin on all four
    # sides. A previous attempt with positive right/bottom collapsed the
    # printable area and clipped table columns on the right.
    pdf.add_section(
        Section(md, paper_size="A3-L", borders=(36, 36, -36, -36)),
        user_css=_CSS,
    )
    pdf.meta["title"] = "Object Classification in Low-Resolution THz-like Imagery — Final Research Report"
    pdf.meta["author"] = "Final-campaign report"
    pdf.save(str(out))

    size = out.stat().st_size
    print(f"wrote {out} ({size:,} bytes)")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Render Final_Exp.md to PDF.")
    p.add_argument("--src", default=str(_DEFAULT_SRC),
                   help=f"Source markdown (default: {_DEFAULT_SRC}).")
    p.add_argument("--out", default=str(_DEFAULT_OUT),
                   help=f"Output PDF path (default: {_DEFAULT_OUT}).")
    args = p.parse_args(argv)
    return render(Path(args.src), Path(args.out))


if __name__ == "__main__":
    sys.exit(main())
