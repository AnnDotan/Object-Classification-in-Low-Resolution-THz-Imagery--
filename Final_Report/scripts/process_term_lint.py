"""Banned-term lint for the Final Summary Report (PRD V2 §2 R3, §6.2 step 4).

R3 forbids the following tokens anywhere in the report body or its captions
(they leak the internal process into the reviewer-facing artifact):

  PRD, PRD v4, PRD v5, US-001..US-099, Ralph loop, Ralph, MASTER, Claude,
  Codex, sub-agent, agent spec, sprint, commit, git, branch, PHASE_B2,
  operator-locked, BLOCKED-OPERATOR.

This tool scans either:
  * a directory of LaTeX sources (.tex), passing `--source-dir <path>`, OR
  * a rendered .docx, passing `--docx <path>` (post-build validation), OR
  * a single .txt file via `--text <path>` (cheap unit-test mode).

Exit code 0 if clean; non-zero with a line-by-line violation report
otherwise. The build pipeline gates on exit code (PRD V2 §10 gate #13,
§6.2 step 4).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable


# Banned tokens (case-insensitive for the named entities, but the US-NNN
# pattern is regex). Order matters for reporting: longer / more specific
# terms first so the violation message is informative.
BANNED_LITERALS = [
    "PRD v5",
    "PRD v4",
    "PRD",
    "Ralph loop",
    "Ralph",
    "MASTER",
    "Claude",
    "Codex",
    "sub-agent",
    "agent spec",
    "sprint",
    "PHASE_B2",
    "operator-locked",
    "BLOCKED-OPERATOR",
]
# Process-token words allowed in narrative prose are excluded by the
# regex anchors below (e.g., "branch" appears in "branching factor" — but
# the PRD-banned form is the version-control noun). We require a
# punctuation / whitespace boundary on both sides so "commitment" does
# not match "commit", and "branched" does not match "branch".
BANNED_BOUNDED = [
    r"\bcommit\b",
    r"\bgit\b",
    r"\bbranch\b",
]
US_PATTERN = re.compile(r"\bUS-0?\d{2,3}\b")


def _scan_text(text: str) -> list[tuple[int, str, str]]:
    """Return [(line_no, token, line_snippet)] for every violation."""
    out: list[tuple[int, str, str]] = []
    for ln, line in enumerate(text.splitlines(), start=1):
        # Strip LaTeX comments (everything after %, unless escaped \%).
        clean = _strip_latex_comment(line)
        # Literal scan (case-insensitive).
        low = clean.lower()
        for tok in BANNED_LITERALS:
            if tok.lower() in low:
                out.append((ln, tok, line.strip()))
        for pattern in BANNED_BOUNDED:
            if re.search(pattern, clean, flags=re.IGNORECASE):
                out.append((ln, pattern.strip("\\b"), line.strip()))
        for m in US_PATTERN.finditer(clean):
            out.append((ln, m.group(0), line.strip()))
    return out


def _strip_latex_comment(line: str) -> str:
    out_chars = []
    i = 0
    while i < len(line):
        if line[i] == "\\" and i + 1 < len(line) and line[i + 1] == "%":
            out_chars.append("\\%")
            i += 2
            continue
        if line[i] == "%":
            break
        out_chars.append(line[i])
        i += 1
    return "".join(out_chars)


def _scan_tex_dir(d: Path) -> dict[Path, list[tuple[int, str, str]]]:
    out: dict[Path, list[tuple[int, str, str]]] = {}
    for path in sorted(d.glob("**/*.tex")):
        if "archive" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        hits = _scan_text(text)
        if hits:
            out[path] = hits
    return out


def _scan_docx(path: Path) -> list[tuple[int, str, str]]:
    try:
        from docx import Document  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "[process_term_lint] --docx requires python-docx; install via "
            "`pip install python-docx`. (Original error: " + repr(e) + ")"
        )
    doc = Document(str(path))
    lines: list[str] = []
    for p in doc.paragraphs:
        lines.append(p.text)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                lines.append(cell.text)
    text = "\n".join(lines)
    return _scan_text(text)


def _report(hits: Iterable[tuple[Path | None, int, str, str]]) -> int:
    hits = list(hits)
    if not hits:
        print("[process_term_lint] clean — no banned terms detected.")
        return 0
    print(f"[process_term_lint] {len(hits)} violation(s):")
    for path, ln, tok, snippet in hits:
        loc = f"{path}:{ln}" if path else f"L{ln}"
        # Trim long snippets for readability.
        snip = snippet if len(snippet) <= 140 else snippet[:137] + "..."
        print(f"  {loc}  [{tok}]  {snip}")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--source-dir", type=Path,
                   help="Directory of .tex sources to scan recursively.")
    g.add_argument("--docx", type=Path,
                   help="Rendered .docx to scan after the build.")
    g.add_argument("--text", type=Path,
                   help="Plain-text file to scan (unit-test mode).")
    args = ap.parse_args()

    flat: list[tuple[Path | None, int, str, str]] = []
    if args.source_dir:
        for path, hits in _scan_tex_dir(args.source_dir).items():
            for ln, tok, snip in hits:
                flat.append((path, ln, tok, snip))
    elif args.docx:
        for ln, tok, snip in _scan_docx(args.docx):
            flat.append((args.docx, ln, tok, snip))
    elif args.text:
        text = args.text.read_text(encoding="utf-8", errors="replace")
        for ln, tok, snip in _scan_text(text):
            flat.append((args.text, ln, tok, snip))
    return _report(flat)


if __name__ == "__main__":
    raise SystemExit(main())
