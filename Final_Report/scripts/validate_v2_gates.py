"""V2 validation gates beyond V1 (PRD V2 §6.2 step 4, §10 gates 11..21).

Implements the new V2-only gates so the build script can fail fast if
any of them regress. The V1 gates (fonts, spacing, em-dash, emoji) stay
in `validate_report.py` and are not re-implemented here.

Gates checked by this tool (PRD V2 §10):
  11. Per-axis curves vs PSNR / SSIM (R2 / Q2): 8 figure pairs exist
      under figures/curves/, each named acc_vs_{psnr,ssim}_phase{C,C2}_
      {cifar10,mnist}.{png,pdf}. Caption-metadata JSON has the per-axis
      Spearman rho for every figure.
  12. Phases overview (R4): sections/04_phases_overview.tex exists;
      tables/tab_II_phases_overview.tex has at least seven phase rows.
  13. Banned-term lint (R3): process_term_lint --source-dir is clean.
  14. Hebrew abstract inlined (R5 / Q5): sections/hebrew_abstract_body.tex
      has >= 1 U+0590..U+05FF code-point.
  15. 80 % framing demoted (R1 / Q4): '80 %' / '80%' / '80 percent'
      absent from sections/01_abstract.tex; at most one occurrence in
      sections/07_conclusions.tex.
  17. Cross-model chapter present (Q1): sections/08_cross_model.tex
      exists and references three model sub-sections + a side-by-side
      reading sub-section.
  19. Maximal equations (Q10): sections/04_design.tex has >= 10
      \label{eq:...} blocks.

Page count (gate 18) and DOCX banned-term gate apply to rendered
outputs; this tool only checks the source side. Run after the build:
  python Final_Report/scripts/process_term_lint.py --docx <path>
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SECTIONS = ROOT / "Final_Report" / "source" / "sections"
TABLES = ROOT / "Final_Report" / "source" / "tables"
CURVES = ROOT / "Final_Report" / "figures" / "curves"


HEBREW_RE = re.compile(r"[֐-׿]")
EQ_LABEL_RE = re.compile(r"\\label\{eq:[^}]+\}")
EIGHTY_RE = re.compile(r"80\s*\\?%|80\s*percent", re.IGNORECASE)


def _ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def _fail(msg: str, errors: list[str]) -> None:
    errors.append(msg)
    print(f"  [FAIL] {msg}")


def gate_11_per_axis_curves(errors: list[str]) -> None:
    print("Gate 11: per-axis curves vs PSNR / SSIM")
    expected = [
        f"acc_vs_{m}_phase{p}_{d}.{ext}"
        for m in ("psnr", "ssim")
        for p in ("C", "C2")
        for d in ("cifar10", "mnist")
        for ext in ("png", "pdf")
    ]
    missing = [n for n in expected if not (CURVES / n).exists()]
    if missing:
        _fail(f"missing {len(missing)} of 16 expected figure files: {missing[:3]}...", errors)
        return
    meta = CURVES / "_axis_curves_meta.json"
    if not meta.exists():
        _fail(f"caption-metadata JSON not found: {meta}", errors)
        return
    j = json.loads(meta.read_text(encoding="utf-8"))
    if len(j) != 8:
        _fail(f"metadata JSON has {len(j)} entries, expected 8", errors)
        return
    for tag, entry in j.items():
        if not entry.get("spearman_rho_by_axis"):
            _fail(f"{tag} missing spearman_rho_by_axis", errors)
            return
    _ok("16 figure files + caption metadata for all 8 (phase, dataset, metric) combos")


def gate_12_phases_overview(errors: list[str]) -> None:
    print("Gate 12: phases overview")
    sec = SECTIONS / "04_phases_overview.tex"
    if not sec.exists():
        _fail(f"missing section file: {sec}", errors)
        return
    tab = TABLES / "tab_II_phases_overview.tex"
    if not tab.exists():
        _fail(f"missing table file: {tab}", errors)
        return
    body = tab.read_text(encoding="utf-8")
    # Each non-rule row ends with \\; count those between \toprule and \bottomrule.
    rows = [ln for ln in body.splitlines() if ln.strip().endswith(r"\\")
            and not ln.strip().startswith("%")]
    if len(rows) < 7:
        _fail(f"Table II has {len(rows)} data rows, need >= 7 phase rows", errors)
        return
    _ok(f"section + Table II ({len(rows)} data rows incl. multi-seed)")


def gate_13_banned_term_lint(errors: list[str]) -> None:
    print("Gate 13: banned-term lint")
    res = subprocess.run(
        [sys.executable,
         str(ROOT / "Final_Report" / "scripts" / "process_term_lint.py"),
         "--source-dir", str(SECTIONS)],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        _fail("process_term_lint exit != 0 — see stdout:\n" + res.stdout, errors)
        return
    _ok("process_term_lint clean")


def gate_14_hebrew_inlined(errors: list[str]) -> None:
    print("Gate 14: Hebrew abstract inlined")
    p = SECTIONS / "hebrew_abstract_body.tex"
    if not p.exists():
        _fail(f"missing Hebrew abstract body: {p}", errors)
        return
    text = p.read_text(encoding="utf-8")
    hits = HEBREW_RE.findall(text)
    if not hits:
        _fail("no U+0590..U+05FF code-points found in Hebrew abstract", errors)
        return
    abst = (SECTIONS / "01_abstract.tex").read_text(encoding="utf-8")
    if "hebrew_abstract_body.tex" not in abst:
        _fail("01_abstract.tex does not \\input the Hebrew body", errors)
        return
    _ok(f"{len(hits)} Hebrew code-points in body; 01_abstract.tex inputs it")


def _strip_tex_comments(text: str) -> str:
    out_lines = []
    for line in text.splitlines():
        # Remove LaTeX comments (after unescaped %).
        clean: list[str] = []
        i = 0
        while i < len(line):
            if line[i] == "\\" and i + 1 < len(line) and line[i + 1] == "%":
                clean.append("\\%")
                i += 2
                continue
            if line[i] == "%":
                break
            clean.append(line[i])
            i += 1
        out_lines.append("".join(clean))
    return "\n".join(out_lines)


def gate_15_eighty_pct(errors: list[str]) -> None:
    print("Gate 15: 80 % framing demoted")
    abst = _strip_tex_comments(
        (SECTIONS / "01_abstract.tex").read_text(encoding="utf-8")
    )
    if EIGHTY_RE.search(abst):
        _fail("'80 %' / '80%' / '80 percent' present in abstract body (R1)", errors)
        return
    concl = _strip_tex_comments(
        (SECTIONS / "07_conclusions.tex").read_text(encoding="utf-8")
    )
    matches = EIGHTY_RE.findall(concl)
    if len(matches) > 1:
        _fail(f"'80 %' appears {len(matches)} times in conclusions, max 1", errors)
        return
    _ok(f"abstract: 0; conclusions: {len(matches)}")


def gate_17_cross_model_chapter(errors: list[str]) -> None:
    print("Gate 17: cross-model chapter present")
    sec = SECTIONS / "08_cross_model.tex"
    if not sec.exists():
        _fail(f"missing section file: {sec}", errors)
        return
    body = sec.read_text(encoding="utf-8")
    required = [
        "ResNet50",
        "DenseNet121",
        "TransNeXt-tiny",
        "Side-by-side",
    ]
    missing = [t for t in required if t not in body]
    if missing:
        _fail(f"08_cross_model.tex missing markers: {missing}", errors)
        return
    _ok("three model subsections + side-by-side reading present")


def gate_19_maximal_equations(errors: list[str]) -> None:
    print("Gate 19: maximal equations")
    body = (SECTIONS / "04_design.tex").read_text(encoding="utf-8")
    labels = EQ_LABEL_RE.findall(body)
    if len(labels) < 10:
        _fail(f"only {len(labels)} numbered equations in 04_design.tex, need >= 10", errors)
        return
    _ok(f"{len(labels)} numbered equations (eq:* labels)")


def main() -> int:
    print("[validate_v2_gates] PRD V2 §10 gates 11..21 (source-side)")
    errors: list[str] = []
    gate_11_per_axis_curves(errors)
    gate_12_phases_overview(errors)
    gate_13_banned_term_lint(errors)
    gate_14_hebrew_inlined(errors)
    gate_15_eighty_pct(errors)
    gate_17_cross_model_chapter(errors)
    gate_19_maximal_equations(errors)

    print()
    if errors:
        print(f"[validate_v2_gates] {len(errors)} gate(s) FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("[validate_v2_gates] all source-side gates PASSED.")
    print("  (page-count gate and DOCX banned-term gate run after the build.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
