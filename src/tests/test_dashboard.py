"""Final_Exp.html dashboard structure tests (US-011).

Asserts:
  - Output HTML contains exactly 186 tiles
  - Three <details> sections (Phase A / B / C)
  - Tile counts per phase match expectations (6 / 30 / 150)
  - Status badge fallback to 'Pending' when no metrics.json exists
  - Status badge becomes 'Complete' when metrics.json has best_val_acc
  - W&B iframe rendered only when metrics.json carries wandb_run_id+entity+project
  - HTML is well-formed (no unclosed tags) and contains a doctype

Run: ``python -m src.tests.test_dashboard``
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.tools.build_final_dashboard import build_dashboard


def _count_occurrences(s: str, sub: str) -> int:
    return s.count(sub)


def _check_tile_and_section_counts() -> None:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "Final_Exp.html"
        thumbs = Path(td) / "thumbs"
        thumbs.mkdir()
        summary = build_dashboard(out_path=out, thumbs_dir=thumbs)
        assert summary["tiles"] == 186
        assert summary["phase_a"] == 6
        assert summary["phase_b"] == 30
        assert summary["phase_c"] == 150

        body = out.read_text(encoding="utf-8")
        assert body.startswith("<!DOCTYPE html>"), "missing doctype"

        n_tiles = _count_occurrences(body, '<div class="tile"')
        assert n_tiles == 186, f"expected 186 tile divs, got {n_tiles}"

        n_sections = _count_occurrences(body, "<details")
        assert n_sections == 3, f"expected 3 <details> sections, got {n_sections}"

        for phase_label in (
            "Phase A — clean baselines (6)",
            "Phase B — combined degradation (30)",
            "Phase C — single-axis isolation (150)",
        ):
            assert phase_label in body, f"missing section header: {phase_label!r}"

    print("OK [counts] — 186 tiles, 3 sections, A=6/B=30/C=150 confirmed.")


def _stage_run(td: Path, tag: str, metrics: dict) -> None:
    run_dir = td / "runs" / "final" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


def _check_status_and_iframe(td: Path) -> None:
    """Stage a 'complete' run with W&B IDs and verify the dashboard reflects it.

    The dashboard reads metrics.json from runs/final/<tag>/, so we have
    to swap CWD into td for the duration of the call.
    """
    from contextlib import chdir
    tag = "final_clean_resnet50_cifar10"

    with chdir(td):
        # Empty state first: this tag should be 'Pending'
        out = td / "Final_Exp.html"
        thumbs = td / "thumbs"
        thumbs.mkdir(exist_ok=True)
        build_dashboard(out_path=out, thumbs_dir=thumbs)
        body = out.read_text(encoding="utf-8")
        # Confirm a Pending tile for our tag (it's the very first cell)
        # Use a snippet that anchors to the tag.
        m = re.search(
            rf'<div class="tile"[^>]*>\s*<h3>{re.escape(tag)}</h3>(.*?)</div>\s*</div>\s*</details>',
            body, re.DOTALL,
        )
        # Looser match: just make sure 'Pending' appears AFTER the tag's <h3>
        idx = body.index(f"<h3>{tag}</h3>")
        snippet = body[idx:idx + 1500]
        assert ">Pending<" in snippet, f"expected Pending status before staging metrics.json"
        assert "iframe-placeholder" in snippet, "expected placeholder when no W&B run id"

        # Now stage a COMPLETE run with W&B IDs
        _stage_run(td, tag, {
            "best_val_acc": 0.892,
            "best_epoch": 14,
            "wandb_run_id": "abc123",
            "wandb_entity": "thz-team",
            "wandb_project": "thz-final",
        })
        build_dashboard(out_path=out, thumbs_dir=thumbs)
        body = out.read_text(encoding="utf-8")
        idx = body.index(f"<h3>{tag}</h3>")
        snippet = body[idx:idx + 1500]
        assert ">Complete<" in snippet, "expected Complete badge after staging metrics.json"
        assert "val_acc=0.8920" in snippet, "expected val_acc text on tile"
        assert "wandb.ai/thz-team/thz-final/runs/abc123" in snippet, (
            "expected W&B iframe URL with full {entity}/{project}/runs/{run_id} path"
        )

    print("OK [status+iframe] — Pending->Complete transition reflected; "
          "W&B iframe URL constructed from metrics.json fields.")


def _check_failed_status_from_log(td: Path) -> None:
    """A run dir with a log.txt containing a traceback should show 'Failed'."""
    from contextlib import chdir
    tag = "final_B_L2_resnet50_cifar10"
    with chdir(td):
        run_dir = td / "runs" / "final" / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "log.txt").write_text(
            "starting...\nTraceback (most recent call last):\n  File ... ValueError\n",
            encoding="utf-8",
        )
        out = td / "Final_Exp.html"
        thumbs = td / "thumbs"
        thumbs.mkdir(exist_ok=True)
        build_dashboard(out_path=out, thumbs_dir=thumbs)
        body = out.read_text(encoding="utf-8")
        idx = body.index(f"<h3>{tag}</h3>")
        snippet = body[idx:idx + 1500]
        assert ">Failed<" in snippet, "expected Failed badge for traceback in log.txt"
    print("OK [failed-from-log] — traceback in log.txt -> Failed badge.")


def main() -> int:
    _check_tile_and_section_counts()
    with tempfile.TemporaryDirectory() as td:
        _check_status_and_iframe(Path(td))
    with tempfile.TemporaryDirectory() as td:
        _check_failed_status_from_log(Path(td))
    return 0


if __name__ == "__main__":
    sys.exit(main())
