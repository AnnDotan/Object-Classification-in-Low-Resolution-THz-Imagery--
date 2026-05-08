"""Phase A tracker-refresh hook integration test (US-005 + US-015).

After US-015 folded the Phase A operator dashboard into the unified Final_Exp.html
(no more Final_Exp_PhaseA.html), the per-row dashboard structure is verified by
test_dashboard.py. What remains specific to Phase A is the runner's hook
integration:

  (a) refresh_all_trackers_hook updates Final_Exp.md AND Final_Exp.html (single
      canonical HTML now) in one invocation.
  (b) Failure in either tracker is isolated — the other still runs, the original
      exception is re-raised so the runner logs a visible WARN.

Run: ``python -m src.tests.test_phase_a_dashboard``
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "scripts_run_phase_a",
        _REPO_ROOT / "scripts" / "run_phase_a.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stage_complete_cell(runs_root: Path, tag: str, *,
                         best_val_acc: float = 0.998,
                         train_at_best: float = 1.0) -> Path:
    rd = runs_root / tag
    rd.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(8):
        rows.append({
            "epoch": i + 1,
            "train_loss": 1.5 - 0.1 * i,
            "train_acc": 0.4 + 0.05 * i if i < 7 else train_at_best,
            "val_loss": 1.4 - 0.1 * i,
            "val_acc": 0.5 + 0.04 * i if i < 7 else best_val_acc,
        })
    with open(rd / "metrics.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc",
                                            "val_loss", "val_acc"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (rd / "metrics.json").write_text(json.dumps({
        "best_val_acc": best_val_acc,
        "best_epoch": 8,
        "last_val_acc": best_val_acc,
        "last_train_acc": train_at_best,
        "epochs_run": 8,
        "hparams": {"head_lr": 1e-3, "backbone_lr": 1e-4, "weight_decay": 1e-4,
                    "label_smoothing": 0.1, "warmup_epochs": 3},
    }), encoding="utf-8")
    (rd / "log.txt").write_text("[lightning] run_name=stub\n", encoding="utf-8")
    (rd / "gate_verdict.json").write_text(json.dumps({
        "tag": tag,
        "decision": "continue",
        "gate_band": "green",
        "reasons": [],
        "best_val_acc": best_val_acc,
        "gap": train_at_best - best_val_acc,
        "epochs_run": 8,
        "best_epoch": 8,
        "gate_explanation": f"GREEN: synthetic for {tag}",
    }), encoding="utf-8")
    return rd


def _check_combined_hook_writes_unified_html() -> None:
    """refresh_all_trackers_hook updates Final_Exp.md AND Final_Exp.html (single)."""
    runner = _load_runner()
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        runs_root = ws / "runs" / "final"
        runs_root.mkdir(parents=True)
        _stage_complete_cell(runs_root, "final_clean_resnet50_mnist")

        old_cwd = os.getcwd()
        os.chdir(ws)
        try:
            (ws / "artifacts").mkdir(exist_ok=True)
            runner.refresh_all_trackers_hook({"tag": "final_clean_resnet50_mnist"})
        finally:
            os.chdir(old_cwd)

        md = ws / "Final_Exp.md"
        html_main = ws / "artifacts" / "Final_Exp.html"
        # The retired sibling MUST NOT be created anymore.
        html_phase_a_legacy = ws / "artifacts" / "Final_Exp_PhaseA.html"

        assert md.exists(), f"Final_Exp.md not written: {md}"
        assert html_main.exists(), f"Final_Exp.html not written: {html_main}"
        assert not html_phase_a_legacy.exists(), (
            "Final_Exp_PhaseA.html should NOT be written after US-015 (folded into Final_Exp.html)"
        )

        body = html_main.read_text(encoding="utf-8")
        assert "final_clean_resnet50_mnist" in body
        assert ">Complete<" in body, "expected Complete badge after staging"
    print("OK [combined-hook-unified] -- both trackers refreshed; no Final_Exp_PhaseA.html sibling.")


def _check_combined_hook_isolates_failures() -> None:
    """If MD refresh fails, HTML refresh still runs; original error re-raised."""
    runner = _load_runner()

    md_calls = {"n": 0}
    html_calls = {"n": 0}

    def failing_md(_v):
        md_calls["n"] += 1
        raise RuntimeError("synthetic md failure")

    def fine_html(_v):
        html_calls["n"] += 1

    original_md = runner.refresh_final_exp_hook
    original_html = runner.refresh_phase_a_dashboard_hook
    runner.refresh_final_exp_hook = failing_md
    runner.refresh_phase_a_dashboard_hook = fine_html
    try:
        runner.refresh_all_trackers_hook({"tag": "final_clean_resnet50_mnist"})
    except RuntimeError as e:
        assert "synthetic md failure" in str(e)
        assert md_calls["n"] == 1
        assert html_calls["n"] == 1, "HTML hook must run even after MD hook failed"
        print("OK [hook-isolation] -- MD failure does not skip HTML; original error re-raised.")
        return
    finally:
        runner.refresh_final_exp_hook = original_md
        runner.refresh_phase_a_dashboard_hook = original_html
    raise AssertionError("expected RuntimeError to bubble up")


def main() -> int:
    _check_combined_hook_writes_unified_html()
    _check_combined_hook_isolates_failures()
    return 0


if __name__ == "__main__":
    sys.exit(main())
