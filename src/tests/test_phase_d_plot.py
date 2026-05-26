"""Tests for `scripts/plot_phase_d_comparison.py` (US-031).

Custom-runner pattern (no pytest) — mirrors `src/tests/test_dashboard.py`
style so the suite stays consistent. Run via:

    PYTHONIOENCODING=utf-8 .venv-gpu/Scripts/python.exe -m src.tests.test_phase_d_plot

Checks:
  - byte-identical PNGs across two consecutive `render()` calls
    (matplotlib determinism pins + `metadata={"CreationDate": None}`).
  - All six `phase_d_recovery_{model}_{dataset}.png` files are emitted.
  - `docs/_autogen/phase_d_recovery_table.tex` has 6 data rows
    (one per (model, dataset)).
  - A corrupted Phase B v2 baseline manifest hash aborts `main()` with
    a non-zero exit code BEFORE writing any plot output (PRD v3 §7
    baseline-comparability invariant).
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Importing the script as a module exposes render() + main() + private helpers.
import scripts.plot_phase_d_comparison as plot_mod  # noqa: E402
from src.experiments.cells import DATASETS, MODELS  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _expected_per_panel_names() -> list[str]:
    return [
        f"phase_d_recovery_{model}_{dataset}.png"
        for model in MODELS
        for dataset in DATASETS
    ]


def _check_byte_identical_across_runs() -> None:
    """Determinism contract — two render() calls must produce byte-identical PNGs.

    Catches: forgotten pdf.fonttype=42 pin, missing metadata={"CreationDate": None},
    datetime.now() overlay smuggled into a future edit, hash-randomized iteration
    over a dict during plot composition.
    """
    with tempfile.TemporaryDirectory() as td:
        out_a = Path(td) / "run_a"
        out_b = Path(td) / "run_b"
        paths_a = plot_mod.render(out_dir=out_a)
        paths_b = plot_mod.render(out_dir=out_b)
        assert len(paths_a) == len(paths_b) == 7, (
            f"expected 7 PNGs per run (1 summary + 6 per-panel); got {len(paths_a)} / {len(paths_b)}"
        )
        # Compare every PNG byte-for-byte via SHA-256.
        mismatches: list[str] = []
        for pa, pb in zip(paths_a, paths_b):
            assert pa.name == pb.name, f"order mismatch: {pa.name} vs {pb.name}"
            ha, hb = _sha256(pa), _sha256(pb)
            if ha != hb:
                mismatches.append(
                    f"{pa.name}: {ha[:12]}... != {hb[:12]}..."
                )
        # Also compare the back-compat alias + the summary PDF, which are
        # written as side-effects of _render_summary() but not in the returned
        # path list.
        for extra in (
            "phase_d_recovery_summary.pdf",
            "phase_d_comparison.png",
            "phase_d_comparison.pdf",
        ):
            pa = out_a / extra
            pb = out_b / extra
            if not pa.exists() or not pb.exists():
                mismatches.append(f"{extra}: missing from one of the two runs")
                continue
            ha, hb = _sha256(pa), _sha256(pb)
            if ha != hb:
                mismatches.append(f"{extra}: {ha[:12]}... != {hb[:12]}...")
        assert not mismatches, (
            "byte-identical determinism violated:\n  - "
            + "\n  - ".join(mismatches)
        )
    print(
        "OK [byte-identical] -- 7 PNGs + summary PDF + back-compat aliases "
        "byte-identical across 2 runs."
    )


def _check_6_per_panel_pngs_emitted() -> None:
    """All six per-panel PNGs land on disk with the canonical filenames."""
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        plot_mod.render(out_dir=out_dir)
        # Canonical per-panel + summary PNGs.
        canonical_pngs = sorted(p.name for p in out_dir.glob("phase_d_recovery_*.png"))
        expected_canonical = sorted(
            ["phase_d_recovery_summary.png"] + _expected_per_panel_names()
        )
        assert canonical_pngs == expected_canonical, (
            "per-panel filename set drift:\n"
            f"  expected: {expected_canonical}\n  got:      {canonical_pngs}"
        )
        # Back-compat alias (US-030 dashboard consumer).
        assert (out_dir / "phase_d_comparison.png").exists(), (
            "missing phase_d_comparison.png back-compat alias"
        )
        assert (out_dir / "phase_d_comparison.pdf").exists(), (
            "missing phase_d_comparison.pdf back-compat alias"
        )
    print(
        "OK [per-panel] -- 6 phase_d_recovery_{m}_{d}.png + 1 summary.png + "
        "back-compat alias emitted."
    )


def _check_latex_table_has_6_rows() -> None:
    """The recovery table must contain 6 data rows (one per (model, dataset)).

    Rows in the tabular body end with `\\\\` and contain `&` separators. The
    header row also contains `&`, so we count lines starting with two-space
    indent + a model name (per the renderer's `f"  {m_pretty} & ..."` format).
    """
    with tempfile.TemporaryDirectory() as td:
        out_tex = Path(td) / "phase_d_recovery_table.tex"
        plot_mod._render_recovery_table(out_tex, level=plot_mod.L3_ANCHOR)
        body = out_tex.read_text(encoding="utf-8")
        # Booktabs hygiene.
        assert r"\toprule" in body, "missing \\toprule"
        assert r"\midrule" in body, "missing \\midrule"
        assert r"\bottomrule" in body, "missing \\bottomrule"
        assert r"\begin{tabular}" in body and r"\end{tabular}" in body, (
            "missing tabular wrapper"
        )
        # No outer \begin{table} — the wrapper lives in Final_Report.tex.
        assert r"\begin{table}" not in body, (
            "table.tex must not include \\begin{table}; the wrapper lives in Final_Report.tex"
        )
        # Six data rows — match a leading "  <Model> & <Dataset> & ..." pattern.
        model_labels = set(plot_mod.MODEL_LABEL.values())
        data_rows = [
            ln for ln in body.splitlines()
            if ln.startswith("  ")
            and any(ln.lstrip().startswith(m + " &") for m in model_labels)
            and ln.rstrip().endswith(r"\\")
        ]
        assert len(data_rows) == 6, (
            f"expected 6 data rows in recovery table; got {len(data_rows)}:\n"
            + "\n".join(data_rows)
        )
        # Every Δ value must be in pp format `{+:.2f}` (or `---` if missing).
        for row in data_rows:
            cells = [c.strip() for c in row.rstrip(r"\\").split("&")]
            assert len(cells) == 5, f"row not 5 cols: {row!r}"
            for cell in cells[2:]:
                if cell == "---":
                    continue
                # Sign + 2 decimal places.
                assert cell[0] in ("+", "-"), f"missing sign in {cell!r} ({row!r})"
                assert "." in cell and len(cell.split(".")[1]) == 2, (
                    f"not 2dp pp format in {cell!r} ({row!r})"
                )
    print(f"OK [latex-table] -- 6 data rows, booktabs hygiene, pp format.")


def _check_baseline_manifest_drift_aborts() -> None:
    """Corrupt one manifest hash; assert main() exits 1 BEFORE writing plots."""
    with tempfile.TemporaryDirectory() as td:
        tmp_root = Path(td)
        tmp_manifest = tmp_root / "manifest.json"
        # Copy the live manifest, then corrupt the first entry's hash.
        live = json.loads(plot_mod.MANIFEST_PATH.read_text(encoding="utf-8"))
        first_tag = sorted(live["entries"].keys())[0]
        live["entries"][first_tag]["metrics_sha256"] = "0" * 64
        tmp_manifest.write_text(json.dumps(live, indent=2), encoding="utf-8")
        # Run the verifier directly — main() then aborts before render().
        ok, drifts = plot_mod._verify_baseline_manifest(
            manifest_path=tmp_manifest, runs_root=plot_mod.RUNS_ROOT,
        )
        assert not ok, "corrupt manifest should fail verification"
        assert any(first_tag in d and "hash drift" in d for d in drifts), (
            f"expected hash-drift report for {first_tag}; got {drifts}"
        )

        # End-to-end: monkeypatch MANIFEST_PATH and run main(); expect rc=1
        # AND no figures written under a sandboxed OUT_DIR.
        tmp_out = tmp_root / "figures"
        tmp_autogen = tmp_root / "_autogen"
        orig_manifest = plot_mod.MANIFEST_PATH
        orig_out = plot_mod.OUT_DIR
        orig_autogen = plot_mod.AUTOGEN_DIR
        try:
            plot_mod.MANIFEST_PATH = tmp_manifest
            plot_mod.OUT_DIR = tmp_out
            plot_mod.AUTOGEN_DIR = tmp_autogen
            rc = plot_mod.main()
        finally:
            plot_mod.MANIFEST_PATH = orig_manifest
            plot_mod.OUT_DIR = orig_out
            plot_mod.AUTOGEN_DIR = orig_autogen
        assert rc == 1, f"main() must exit 1 on drift; got {rc}"
        assert not tmp_out.exists() or not any(tmp_out.iterdir()), (
            "main() wrote plots despite manifest drift"
        )
        assert not tmp_autogen.exists() or not any(tmp_autogen.iterdir()), (
            "main() wrote LaTeX table despite manifest drift"
        )
    print(
        "OK [manifest-drift] -- corrupted hash aborts main() with rc=1, no plots written."
    )


def main() -> int:
    _check_byte_identical_across_runs()
    _check_6_per_panel_pngs_emitted()
    _check_latex_table_has_6_rows()
    _check_baseline_manifest_drift_aborts()
    print("\nAll Phase D plot checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
