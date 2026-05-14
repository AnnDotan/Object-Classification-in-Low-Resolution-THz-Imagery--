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
_DEFAULT_SRC = _REPO_ROOT / "Final_Exp.md"
_DEFAULT_OUT = _REPO_ROOT / "artifacts" / "Final_Exp.pdf"

_CSS = """
body { font-family: "Segoe UI", Arial, sans-serif; color: #1f2530; }
h1 { color: #0d4f8b; border-bottom: 2px solid #0d4f8b; padding-bottom: 4px; }
h2 { color: #0d4f8b; border-bottom: 1px solid #cfd6e0; padding-bottom: 2px; margin-top: 18px; }
h3 { color: #2a3957; }
code { background: #f1f3f6; padding: 1px 4px; border-radius: 3px;
       font-family: Consolas, "Courier New", monospace; font-size: 0.92em; }
pre { background: #f1f3f6; padding: 8px; border-radius: 4px; overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 0.78em; margin: 8px 0; }
th, td { border: 1px solid #cfd6e0; padding: 4px 6px; text-align: left; vertical-align: top; }
th { background: #eef2f7; }
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
    # A4 landscape so the wide row tables (Phase B / Phase C with 10-11
    # columns) keep their right-side Duration/Started cells legible.
    pdf.add_section(
        Section(md, paper_size="A4-L", borders=(28, 28, 28, 28)),
        user_css=_CSS,
    )
    pdf.meta["title"] = "Final Experiment Matrix — 186 Runs"
    pdf.meta["author"] = "Final-campaign tracker"
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
