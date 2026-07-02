"""V6 build script. Produces DOCX ONLY -- never a PDF.

Per the V6 directive the deliverable is the Microsoft Word document. A
XeLaTeX PDF may be compiled locally as a layout/Hebrew proof but is not a
build target here.

Filter chain (order matters):
  dash_sanitizer.lua  em/en-dash normalization (body text only)
  hebrew_rtl.lua      lang="he" Divs/Spans get dir="rtl" (Word w:bidi)
  docx_lists.lua      caption numbering ("Figure N:", "Table N:") + native
                      Word TOC / LOF / LOT fields under the three front-
                      matter headings
  ieee_caption.lua    caption-position convention check (no-op)

Run from the repository root:
  python Final_Report/V6/scripts/build_v6_docx.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V6 = ROOT / "Final_Report" / "V6"
SRC = V6 / "source"
FIGURES = ROOT / "Final_Report" / "figures"
OUT_DOCX = V6 / "fin-2026-061-v6.docx"

TEX_BASENAME = "fin-2026-061"


def main() -> int:
    print("[v6] Building DOCX only (no PDF emitted by design).")
    try:
        import pypandoc
    except ImportError:
        print("[v6] ERROR: pypandoc is not installed.", file=sys.stderr)
        return 1

    extra = [
        f"--reference-doc={(SRC / 'bgu_reference.docx').as_posix()}",
        # NOTE: citeproc is intentionally OFF. The references are a hand-
        # formatted IEEE thebibliography (sections/08_references.tex); pandoc
        # resolves the \cite commands against its \bibitem labels and renders
        # numeric [n] citations. citeproc would emit author-date instead.
        # NOTE: --toc is intentionally OFF. Pandoc would place its TOC before
        # the cover page; the front-matter lists are injected as native Word
        # fields by docx_lists.lua at the correct location instead.
        "--lua-filter", (SRC / "filters" / "dash_sanitizer.lua").as_posix(),
        "--lua-filter", (SRC / "filters" / "hebrew_rtl.lua").as_posix(),
        "--lua-filter", (SRC / "filters" / "docx_lists.lua").as_posix(),
        "--lua-filter", (SRC / "filters" / "ieee_caption.lua").as_posix(),
        "--number-sections",
        f"--resource-path={SRC.as_posix()};{V6.as_posix()};{FIGURES.as_posix()};{(V6.parent).as_posix()}",
    ]

    cwd_before = Path.cwd()
    try:
        os.chdir(SRC)
        pypandoc.convert_file(
            f"{TEX_BASENAME}.tex",
            to="docx",
            format="latex",
            outputfile=str(OUT_DOCX),
            extra_args=extra,
        )
    finally:
        os.chdir(cwd_before)

    size_kb = OUT_DOCX.stat().st_size / 1024.0
    print(f"[v6] DOCX -> {OUT_DOCX} ({size_kb:.0f} KiB)")
    print("[v6] Reminder: open in Word and press Ctrl+A, F9 to populate the")
    print("[v6] Table of Contents / List of Figures / List of Tables fields.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
