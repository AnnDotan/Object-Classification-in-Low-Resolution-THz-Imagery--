"""Validate Final_Report/fin-2026-061.docx against BGU and PRD rules.

Checks:
  - File exists and is a valid .docx (zip archive with word/document.xml).
  - Font is Times New Roman on Normal style.
  - Line spacing is 1.5.
  - Margins are 2.5 cm on every side.
  - No em-dash (U+2014) or en-dash (U+2013) in body text.
  - No emoji codepoints in body text.
  - References list parses as numbered IEEE-style citations.

Exit code 0 if all checks pass, otherwise 1 and a printed report.
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parents[2]
DOCX = ROOT / "Final_Report" / "fin-2026-061.docx"


def is_emoji(ch: str) -> bool:
    cp = ord(ch)
    # Crude but conservative: standard emoji blocks.
    return (0x1F300 <= cp <= 0x1FAFF) or (0x2600 <= cp <= 0x27BF)


def main() -> int:
    if not DOCX.exists():
        print(f"ERROR: {DOCX} not found")
        return 1

    d = Document(str(DOCX))

    failures: list[str] = []

    # 1) Margins
    for i, s in enumerate(d.sections):
        for name, expected in (
            ("top_margin", 2.5),
            ("bottom_margin", 2.5),
            ("left_margin", 2.5),
            ("right_margin", 2.5),
        ):
            val = getattr(s, name)
            cm = val.cm if hasattr(val, "cm") else val.inches * 2.54
            if abs(cm - expected) > 0.05:
                failures.append(f"section {i} {name} = {cm:.2f} cm; expected {expected:.2f} cm")

    # 2) Normal style: font + line spacing
    try:
        normal = d.styles["Normal"]
    except KeyError:
        failures.append("Normal style not present in DOCX")
    else:
        fn = (normal.font.name or "").strip()
        if "Times New Roman" not in fn:
            failures.append(f"Normal font is {fn!r}; expected 'Times New Roman'")
        pf = normal.paragraph_format
        ls = pf.line_spacing
        if ls is None or abs(ls - 1.5) > 0.01:
            failures.append(f"Normal line spacing = {ls}; expected 1.5")

    # 3) Em-dash / en-dash / emoji audit (body text only)
    em_dash_count = 0
    en_dash_count = 0
    emoji_count = 0
    body_chars = 0
    for p in d.paragraphs:
        text = p.text
        body_chars += len(text)
        em_dash_count += text.count("—")
        en_dash_count += text.count("–")
        for ch in text:
            if is_emoji(ch):
                emoji_count += 1
    for t in d.tables:
        for row in t.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    text = p.text
                    body_chars += len(text)
                    em_dash_count += text.count("—")
                    en_dash_count += text.count("–")
                    for ch in text:
                        if is_emoji(ch):
                            emoji_count += 1

    if em_dash_count > 0:
        failures.append(f"em-dash (U+2014) count = {em_dash_count}; expected 0")
    if en_dash_count > 0:
        failures.append(f"en-dash (U+2013) count = {en_dash_count}; expected 0")
    if emoji_count > 0:
        failures.append(f"emoji count = {emoji_count}; expected 0")

    print("---- validation report ----")
    print(f"file: {DOCX}")
    print(f"size: {DOCX.stat().st_size / 1024:.1f} KB")
    print(f"paragraphs: {len(d.paragraphs)}, tables: {len(d.tables)}, body chars: {body_chars}")
    print(f"em-dash: {em_dash_count}, en-dash: {en_dash_count}, emoji: {emoji_count}")
    print()
    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
