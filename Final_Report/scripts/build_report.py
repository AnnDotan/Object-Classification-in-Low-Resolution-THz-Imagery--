"""Master build script. Produces:
  Final_Report/fin-2026-061.pdf   (reference PDF via pdflatex + bibtex)
  Final_Report/fin-2026-061.docx  (deliverable via Pandoc)

Run from the repository root:
  python Final_Report/scripts/build_report.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "Final_Report" / "source"
OUT_DIR = ROOT / "Final_Report"
BUILD = ROOT / "Final_Report" / "build"
BUILD.mkdir(parents=True, exist_ok=True)

TEX_BASENAME = "fin-2026-061"


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> int:
    print(f"  $ {' '.join(cmd)}")
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if check and p.returncode != 0:
        raise SystemExit(f"command failed with code {p.returncode}: {' '.join(cmd)}")
    return p.returncode


def find_xelatex() -> str:
    """Canonical PDF engine per PRD V2 §6.1 (XeLaTeX + polyglossia for
    Hebrew). Falls back to pdflatex if XeLaTeX is missing, with a warning
    that Hebrew will not render."""
    candidates = [
        shutil.which("xelatex"),
        r"C:\Users\ib94\AppData\Local\Programs\MiKTeX\miktex\bin\x64\xelatex.exe",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    print("[build_report] WARNING: xelatex not found; falling back to pdflatex")
    print("[build_report] WARNING: Hebrew abstract will NOT render in the PDF")
    return find_pdflatex()


def find_pdflatex() -> str:
    candidates = [
        shutil.which("pdflatex"),
        r"C:\Users\ib94\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    raise SystemExit("pdflatex not found on PATH")


def find_bibtex() -> str:
    candidates = [
        shutil.which("bibtex"),
        r"C:\Users\ib94\AppData\Local\Programs\MiKTeX\miktex\bin\x64\bibtex.exe",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    raise SystemExit("bibtex not found on PATH")


def build_pdf() -> Path:
    print("[build_report] BUILDING PDF (reference) ...")
    tex_engine = find_xelatex()    # PRD V2 §6.1 canonical engine
    bibtex = find_bibtex()
    # Run TeX engine from the source dir so \input{sections/...} and
    # \includegraphics{../figures/...} resolve. Aux files land in build/tex.
    work = BUILD / "tex"
    work.mkdir(parents=True, exist_ok=True)
    args_common = [
        tex_engine,
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={work.as_posix()}",
        f"{TEX_BASENAME}.tex",
    ]
    # Pass 1
    run(args_common, cwd=SRC)
    # bibtex needs the aux file from work/
    run([bibtex, TEX_BASENAME], cwd=work, check=False)
    # Pass 2
    run(args_common, cwd=SRC)
    # Pass 3 (TOC / refs settle)
    run(args_common, cwd=SRC)
    out_pdf = OUT_DIR / f"{TEX_BASENAME}.pdf"
    shutil.copy2(work / f"{TEX_BASENAME}.pdf", out_pdf)
    print(f"[build_report] PDF -> {out_pdf}")
    return out_pdf


def build_docx() -> Path:
    print("[build_report] BUILDING DOCX (deliverable) ...")
    import pypandoc
    out_docx = OUT_DIR / f"{TEX_BASENAME}.docx"
    # Resolve \input{sections/...} by running Pandoc from the source dir.
    # We chdir there, then call Pandoc on the bare basename.
    cwd_before = Path.cwd()
    extra = [
        f"--reference-doc={(SRC / 'bgu_reference.docx').as_posix()}",
        f"--bibliography={(SRC / 'refs.bib').as_posix()}",
        "--citeproc",
        "--lua-filter", (SRC / "filters" / "dash_sanitizer.lua").as_posix(),
        "--lua-filter", (SRC / "filters" / "ieee_caption.lua").as_posix(),
        "--number-sections",
        "--toc",
        "--toc-depth=3",
        f"--resource-path={SRC.as_posix()};{OUT_DIR.as_posix()};{(OUT_DIR / 'figures').as_posix()}",
    ]
    try:
        os.chdir(SRC)
        pypandoc.convert_file(
            f"{TEX_BASENAME}.tex",
            to="docx",
            format="latex",
            outputfile=str(out_docx),
            extra_args=extra,
        )
    finally:
        os.chdir(cwd_before)

    # Hebrew is now inline in 01_abstract.tex via the PRD V2 Q5 / FX-16
    # implementation (XeLaTeX + polyglossia for PDF; Pandoc reads the
    # UTF-8 source directly into the DOCX). No post-step append is
    # needed; the previous _append_hebrew_to_docx step is retired.

    print(f"[build_report] DOCX -> {out_docx}")
    return out_docx


def _append_hebrew_to_docx(docx_path: Path, hebrew_txt: Path) -> None:
    """Append the Hebrew abstract block to the end of the DOCX."""
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    d = Document(str(docx_path))
    text = hebrew_txt.read_text(encoding="utf-8")

    # Section break for the Hebrew block (keeps default RTL paragraph defaults
    # local).
    d.add_page_break()
    h = d.add_paragraph()
    h.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    h_run = h.add_run("Hebrew Abstract")
    h_run.bold = True
    h_run.font.size = None  # inherit Heading 1 size

    for block in text.strip().split("\n\n"):
        p = d.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        # Mark RTL
        pPr = p._p.get_or_add_pPr()
        bidi = OxmlElement("w:bidi")
        bidi.set(qn("w:val"), "1")
        pPr.append(bidi)
        run = p.add_run(block)
        # Make sure RTL applies to the run too
        rPr = run._r.get_or_add_rPr()
        rtl = OxmlElement("w:rtl")
        rtl.set(qn("w:val"), "1")
        rPr.append(rtl)

    d.save(str(docx_path))
    print(f"  appended Hebrew abstract -> {docx_path}")


def main() -> int:
    parts = sys.argv[1:] or ["pdf", "docx"]
    try:
        if "pdf" in parts:
            build_pdf()
        if "docx" in parts:
            build_docx()
    except SystemExit as e:
        print(f"[build_report] ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
