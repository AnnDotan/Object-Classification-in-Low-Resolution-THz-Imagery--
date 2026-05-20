"""Build the US-019B v1↔v2 dashboard-thumbnail contact sheet.

Pairs each of the 90 v2-affected cells (Phase B, Phase C noise, Phase C
salt_pepper — see PRD §1) against its snapshotted v1 counterpart so the
operator can spot-check the noise/S&P pipeline move (US-017) with one
click instead of a manual diff hunt.

Inputs:
    artifacts/validation/v1_thumbs/<tag>.png    (snapshotted by US-019B Iter 28)
    artifacts/dashboard_thumbs/<tag>.png        (v2 re-render from US-019B Iter 29)
    artifacts/validation/v1_thumbs_manifest.json (durable v1 hash baseline)

Output:
    artifacts/validation/v1_vs_v2_thumbs.html   (90 pair-rows, static HTML)

The HTML uses relative image paths so it opens cleanly via file:// — no
server needed. Image URLs:
    v1: v1_thumbs/<tag>.png            (sibling directory)
    v2: ../dashboard_thumbs/<tag>.png  (one level up + sibling)

The contact sheet is purely a viewer; it does NOT regenerate either PNG
set. Run `scripts/snapshot_v1_thumbs.py --verify` and the three
`render_cell_thumbs --force --phase ... --axes ...` dispatches BEFORE
opening this. The script fast-fails if any of the 180 expected PNGs is
missing (90 tags × 2 sides), which catches "operator forgot to re-render"
without producing a half-empty contact sheet.

CLI:
    python scripts/build_v1_v2_thumb_contact_sheet.py
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.degradation_levels import PIPELINE_VERSION  # noqa: E402
from src.experiments.cells import iter_cells  # noqa: E402

V1_DIR = REPO_ROOT / "artifacts" / "validation" / "v1_thumbs"
V2_DIR = REPO_ROOT / "artifacts" / "dashboard_thumbs"
MANIFEST = REPO_ROOT / "artifacts" / "validation" / "v1_thumbs_manifest.json"
OUT_HTML = REPO_ROOT / "artifacts" / "validation" / "v1_vs_v2_thumbs.html"

AFFECTED_AXES = {"noise", "salt_pepper"}


def _group_key(c) -> tuple[int, int, int, int]:
    """Sort key: Phase B before noise before salt_pepper, then L, model, dataset."""
    phase_order = 0 if c.phase == "B" else (1 if c.axis == "noise" else 2)
    model_order = {"resnet50": 0, "densenet121": 1, "transnext_tiny": 2}[c.model]
    dataset_order = {"cifar10": 0, "mnist": 1}[c.dataset]
    return (phase_order, c.level or 0, model_order, dataset_order)


def _group_label(c) -> str:
    if c.phase == "B":
        return "Phase B — combined degradation (all axes at L)"
    return f"Phase C — single-axis isolation ({c.axis})"


def affected_cells() -> list:
    cells = [
        c for c in iter_cells()
        if c.phase == "B" or (c.phase == "C" and c.axis in AFFECTED_AXES)
    ]
    cells.sort(key=_group_key)
    return cells


def _check_pngs_present(cells: list) -> None:
    missing: list[str] = []
    for c in cells:
        if not (V1_DIR / f"{c.tag}.png").exists():
            missing.append(f"v1/{c.tag}.png")
        if not (V2_DIR / f"{c.tag}.png").exists():
            missing.append(f"v2/{c.tag}.png")
    if missing:
        raise SystemExit(
            f"contact sheet aborted: {len(missing)} PNG(s) missing on disk; "
            f"first 5 = {missing[:5]}. Run snapshot_v1_thumbs.py (v1 side) "
            "and render_cell_thumbs --force --phase ... (v2 side) first."
        )


_CSS = r"""
:root {
    --bg: #0b0d10;
    --surface: #12151a;
    --surface2: #181c22;
    --border: #2a3040;
    --text: #e8eaf0;
    --text-dim: #8890a0;
    --accent: #ffab40;
    --green: #4caf50;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 24px;
    line-height: 1.5;
}
h1 { font-size: 24px; margin-bottom: 6px; }
h2 {
    font-size: 18px;
    margin: 28px 0 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
    color: var(--accent);
}
.lead { color: var(--text-dim); margin-bottom: 18px; }
.meta {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 18px;
    font-size: 13px;
    color: var(--text-dim);
}
.meta strong { color: var(--text); }
.smoking-gun {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 6px;
    padding: 12px 16px 12px 18px;
    margin-bottom: 24px;
    font-size: 13px;
}
.smoking-gun ul { margin: 6px 0 0 22px; }
.smoking-gun li { margin: 4px 0; color: var(--text-dim); }
.smoking-gun li strong { color: var(--text); }
.pair-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
    gap: 14px;
}
.pair {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px;
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.pair-tag {
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 11px;
    color: var(--text-dim);
    word-break: break-all;
    margin-bottom: 4px;
}
.pair-imgs { display: flex; gap: 8px; }
.pair-imgs figure {
    flex: 1 1 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
}
.pair-imgs img {
    width: 100%;
    height: auto;
    image-rendering: pixelated;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: #000;
}
.pair-imgs figcaption {
    font-size: 11px;
    color: var(--text-dim);
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.pair-imgs figure.v2 figcaption { color: var(--green); }
"""


def _build_pair_html(c) -> str:
    tag_esc = html.escape(c.tag)
    return (
        '<div class="pair">'
        f'<div class="pair-tag">{tag_esc}</div>'
        '<div class="pair-imgs">'
        f'<figure class="v1"><img src="v1_thumbs/{tag_esc}.png" alt="v1 {tag_esc}" loading="lazy"><figcaption>v1 (snapshot)</figcaption></figure>'
        f'<figure class="v2"><img src="../dashboard_thumbs/{tag_esc}.png" alt="v2 {tag_esc}" loading="lazy"><figcaption>v2 (current)</figcaption></figure>'
        '</div>'
        '</div>'
    )


def _build_section_html(label: str, cells: list) -> str:
    pairs = "\n".join(_build_pair_html(c) for c in cells)
    return (
        f'<h2>{html.escape(label)} <span style="color:var(--text-dim);font-weight:400;font-size:13px">'
        f'({len(cells)} cells)</span></h2>\n'
        f'<div class="pair-grid">\n{pairs}\n</div>'
    )


def _smoking_gun_block() -> str:
    return (
        '<div class="smoking-gun">'
        '<strong>Smoking-gun visual expectations (PRD §US-019B):</strong>'
        '<ul>'
        '<li><strong>L5 noise</strong> at low_res=3 → ~74-px blob structure, '
        'NOT 1-px white speckles.</li>'
        '<li><strong>L5 salt_pepper</strong> at low_res=3 → large bright/dark '
        'blobs in ~9 distinct cells, NOT 1-px speckles.</li>'
        '<li><strong>L1 thumbs</strong> nearly indistinguishable from v1 '
        '(noise_std=0.04 at low_res=18 → still fine grain).</li>'
        '<li>Phase B v2 thumbs visibly coarser than v1 on the noise/S&P '
        'contribution at L3+ levels.</li>'
        '<li>Re-running the renderer produces byte-identical PNGs (seed '
        'contract holds — verified separately by US-019B Iter 29 determinism gate).</li>'
        '</ul>'
        '</div>'
    )


def build_html(cells: list) -> str:
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    snapshot_utc = manifest.get("snapshot_utc", "unknown")
    cell_count = manifest.get("cell_count", "unknown")

    by_section: dict[str, list] = {}
    for c in cells:
        by_section.setdefault(_group_label(c), []).append(c)

    sections_html = "\n".join(
        _build_section_html(label, group)
        for label, group in by_section.items()
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>US-019B — v1 vs v2 dashboard thumbnail contact sheet</title>
<style>{_CSS}</style>
</head>
<body>
<h1>US-019B — v1 vs v2 dashboard thumbnail contact sheet</h1>
<p class="lead">Operator visual gate for the 90 v2-affected cells before US-020 GPU dispatch.</p>
<div class="meta">
<strong>PIPELINE_VERSION:</strong> {int(PIPELINE_VERSION)}
&nbsp;·&nbsp; <strong>v1 snapshot UTC:</strong> {html.escape(str(snapshot_utc))}
&nbsp;·&nbsp; <strong>v1 baseline cells:</strong> {html.escape(str(cell_count))}
&nbsp;·&nbsp; <strong>pairs rendered:</strong> {len(cells)}
</div>
{_smoking_gun_block()}
{sections_html}
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_HTML,
        help=f"output HTML path (default: {OUT_HTML.relative_to(REPO_ROOT)})",
    )
    args = parser.parse_args(argv)

    cells = affected_cells()
    assert len(cells) == 90, (
        f"expected 90 v2-affected cells (30 Phase B + 30 noise + 30 salt_pepper); "
        f"iter_cells() yielded {len(cells)} — matrix drift?"
    )

    if not MANIFEST.exists():
        raise SystemExit(
            f"v1 manifest missing at {MANIFEST.relative_to(REPO_ROOT)}; "
            "run scripts/snapshot_v1_thumbs.py first."
        )

    _check_pngs_present(cells)

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_html(cells), encoding="utf-8")
    print(
        f"[contact-sheet] wrote {len(cells)} v1-vs-v2 pairs to "
        f"{out_path.relative_to(REPO_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
