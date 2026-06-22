"""V5 build script. Produces DOCX ONLY -- never a PDF.

Per the V5 refocused-revision directive, the deliverable is the Microsoft Word
document. A XeLaTeX PDF may be compiled locally as a layout/Hebrew proof but is
not a build target here.

Run from the repository root:
  python Final_Report/V5/scripts/build_v5_docx.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V5 = ROOT / "Final_Report" / "V5"
SRC = V5 / "source"
FIGURES = ROOT / "Final_Report" / "figures"
OUT_DOCX = V5 / "fin-2026-061-v5.docx"

TEX_BASENAME = "fin-2026-061"


def main() -> int:
    print("[v5] Building DOCX only (no PDF emitted by design).")
    try:
        import pypandoc
    except ImportError:
        print("[v5] ERROR: pypandoc is not installed.", file=sys.stderr)
        return 1

    extra = [
        f"--reference-doc={(SRC / 'bgu_reference.docx').as_posix()}",
        # NOTE: citeproc is intentionally OFF. The references are a hand-formatted
        # IEEE thebibliography (sections/08_references.tex); pandoc resolves the
        # \cite commands against its \bibitem labels and renders numeric [n]
        # citations. Enabling citeproc would instead emit author-date (Chicago)
        # citations from refs.bib, which is not the required IEEE style.
        "--lua-filter", (SRC / "filters" / "dash_sanitizer.lua").as_posix(),
        "--lua-filter", (SRC / "filters" / "ieee_caption.lua").as_posix(),
        "--number-sections",
        # No --toc: the table of contents is intentionally removed (as in V4).
        f"--resource-path={SRC.as_posix()};{(V5).as_posix()};{FIGURES.as_posix()};{(V5.parent).as_posix()}",
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

    print(f"[v5] DOCX -> {OUT_DOCX}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
