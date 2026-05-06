"""Regression tests for src/tools/build_final_exp_json.py (FINAL_EXP Dashboard, US-003).

Asserts the FinalExpDoc contract emitted by build_final_exp_json:
  - 186 rows split 6 / 30 / 150 across phases A / B / C.
  - Phase C L1 collapse: every C-L1 row's params dict == Phase B L1 row's params.
  - metrics.json round-trip: best_val_acc, last_val_loss, epochs_run, runtime_s
    populate the corresponding row; status flips from Pending to Complete.
  - Weight Privacy: aggregator never opens *.ckpt / *.pt / *.pth or any path
    under artifacts/weights/ (verified via builtins.open patch).
  - Atomic write: no .tmp sibling left behind after success; out_path written.
  - Document-level fields: schema_version == 1, generated_at parses as ISO8601.

Run: ``python -m src.tests.test_build_final_exp_json``
"""
from __future__ import annotations

import builtins
import io
import json
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.tools.build_final_exp_json import build_doc, build_final_exp_json  # noqa: E402
from src.tools.final_exp_schema import SCHEMA_VERSION  # noqa: E402


_FORBIDDEN_PATH_SUBSTRINGS: tuple[str, ...] = (
    ".ckpt", ".pt", ".pth", "artifacts/weights", "artifacts\\weights",
)


def _check_row_counts_and_phase_split() -> None:
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        doc = build_doc(runs_root=runs_root)

    assert doc["counts"]["total"] == 186, doc["counts"]
    assert len(doc["rows"]) == 186, f"got {len(doc['rows'])} rows"
    phases: dict[str, int] = {"A": 0, "B": 0, "C": 0}
    for r in doc["rows"]:
        phases[r["phase"]] += 1
    assert phases == {"A": 6, "B": 30, "C": 150}, phases
    # Empty runs_root -> every row Pending.
    assert doc["counts"]["pending"] == 186, doc["counts"]
    assert doc["counts"]["complete"] == 0
    assert doc["counts"]["running"] == 0
    assert doc["counts"]["failed"] == 0
    print("OK [row-counts] -- 186 rows, phase split 6/30/150, empty -> all Pending.")


def _check_phase_c_l1_collapse() -> None:
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        doc = build_doc(runs_root=runs_root)

    b_l1 = next(r for r in doc["rows"] if r["phase"] == "B" and r["level"] == 1)
    c_l1_rows = [r for r in doc["rows"] if r["phase"] == "C" and r["level"] == 1]
    # 5 axes x 3 models x 2 datasets = 30 Phase C L1 rows
    assert len(c_l1_rows) == 30, f"expected 30 Phase C L1 rows, got {len(c_l1_rows)}"
    for r in c_l1_rows:
        assert r["params"] == b_l1["params"], (
            f"Phase C L1 collapse broken for {r['tag']}: "
            f"{r['params']} != {b_l1['params']}"
        )
    print("OK [collapse] -- every Phase C L1 row's params == Phase B L1 row's params.")


def _check_metrics_roundtrip() -> None:
    """A staged metrics.json must hydrate the corresponding row + flip its status."""
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        tag = "final_clean_resnet50_cifar10"
        cell_dir = runs_root / tag
        cell_dir.mkdir(parents=True)
        (cell_dir / "metrics.json").write_text(
            json.dumps({
                "best_val_acc": 0.42,
                "last_val_loss": 0.31,
                "epochs_run": 12,
                "runtime_s": 345.6,
            }),
            encoding="utf-8",
        )
        doc = build_doc(runs_root=runs_root)

    row = next(r for r in doc["rows"] if r["tag"] == tag)
    assert row["status"] == "Complete", row["status"]
    assert row["val_acc"] == 0.42, row["val_acc"]
    assert row["val_loss"] == 0.31, row["val_loss"]
    assert row["epochs_run"] == 12, row["epochs_run"]
    assert row["runtime_s"] == 345.6, row["runtime_s"]
    # Other 185 rows untouched.
    others = [r for r in doc["rows"] if r["tag"] != tag]
    assert all(r["status"] == "Pending" for r in others), "non-staged rows must stay Pending"
    assert doc["counts"]["complete"] == 1
    assert doc["counts"]["pending"] == 185
    print("OK [roundtrip] -- metrics.json fields hydrate the matching row, status -> Complete.")


def _check_no_weight_path_opened() -> None:
    """Aggregator must NEVER open *.ckpt / *.pt / *.pth or artifacts/weights/* files."""
    opened: list[str] = []
    # `io.open` and `builtins.open` are the same callable, but pathlib looks
    # up `io.open` via its module reference; patching only `builtins.open`
    # leaves pathlib's path unaffected. Patch both to cover all callers.
    real_open = io.open

    def _recording_open(file, *args, **kwargs):  # type: ignore[no-untyped-def]
        try:
            opened.append(str(file))
        except Exception:
            pass
        return real_open(file, *args, **kwargs)

    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        # Plant decoy weight files alongside one real metrics.json so the test
        # would fail if the aggregator ever started recursing or globbing.
        tag = "final_clean_densenet121_cifar10"
        cell_dir = runs_root / tag
        cell_dir.mkdir(parents=True)
        (cell_dir / "metrics.json").write_text(
            json.dumps({"best_val_acc": 0.7, "epochs_run": 5}), encoding="utf-8",
        )
        (cell_dir / "model.ckpt").write_bytes(b"\x00" * 16)
        (cell_dir / "weights.pt").write_bytes(b"\x00" * 16)
        (cell_dir / "snapshot.pth").write_bytes(b"\x00" * 16)

        with patch("builtins.open", side_effect=_recording_open), \
             patch("io.open", side_effect=_recording_open):
            build_doc(runs_root=runs_root)

    bad = [p for p in opened if any(s in p for s in _FORBIDDEN_PATH_SUBSTRINGS)]
    assert not bad, f"aggregator opened forbidden path(s): {bad}"
    # Sanity: at least the staged metrics.json was opened.
    assert any("metrics.json" in p for p in opened), (
        f"expected metrics.json to be opened; got {opened[:5]}..."
    )
    print(f"OK [privacy] -- {len(opened)} open() calls, none touched .ckpt/.pt/.pth/weights.")


def _check_atomic_write_and_no_tmp_leftover() -> None:
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        out = Path(td) / "Final_Exp.json"
        summary = build_final_exp_json(out_path=out, runs_root=runs_root)
        assert out.exists(), "Final_Exp.json was not written"
        # No leftover .tmp files in out's parent.
        leftovers = [p.name for p in out.parent.iterdir() if p.name.startswith("Final_Exp.json.")]
        assert not leftovers, f"leftover tmp files: {leftovers}"
        # Round-trip through json.load to confirm valid output.
        d = json.loads(out.read_text(encoding="utf-8"))
        assert d["counts"]["total"] == 186
        assert summary["rows"] == 186
    print("OK [atomic] -- output written; no .tmp siblings linger.")


def _check_doc_level_fields() -> None:
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        doc = build_doc(runs_root=runs_root)
    assert doc["schema_version"] == SCHEMA_VERSION == 1
    # generated_at must parse as ISO8601 with timezone (Python's fromisoformat
    # accepts the ±HH:MM offset emitted by datetime.isoformat(timespec='seconds')).
    parsed = datetime.fromisoformat(doc["generated_at"])
    assert parsed.tzinfo is not None, f"generated_at lacks tz: {doc['generated_at']}"
    print(f"OK [doc-fields] -- schema_version=1, generated_at parses ({doc['generated_at']}).")


def _check_rows_sorted_by_tag() -> None:
    """Rows must be sorted by tag for byte-stable JSON across runs."""
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        doc = build_doc(runs_root=runs_root)
    tags = [r["tag"] for r in doc["rows"]]
    assert tags == sorted(tags), "rows are not sorted by tag"
    # Sanity: tag prefixes match the canonical scheme.
    pat = re.compile(r"^final_(clean|B_L\d|C_L\d_[a-z_]+)_(resnet50|densenet121|transnext_base)_(cifar10|mnist)$")
    bad = [t for t in tags if not pat.match(t)]
    assert not bad, f"non-canonical tags: {bad[:5]}"
    print(f"OK [sort] -- 186 rows sorted by tag; first={tags[0]}, last={tags[-1]}.")


def main() -> int:
    _check_row_counts_and_phase_split()
    _check_phase_c_l1_collapse()
    _check_metrics_roundtrip()
    _check_no_weight_path_opened()
    _check_atomic_write_and_no_tmp_leftover()
    _check_doc_level_fields()
    _check_rows_sorted_by_tag()
    print("\nAll build_final_exp_json checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
