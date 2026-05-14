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
    # US-014 quarantine: TransNeXt rows are Deferred regardless of disk
    # state (62 rows = 2 Phase A + 10 Phase B + 50 Phase C). Empty runs_root
    # -> remaining 124 are Pending; 62 are Deferred.
    assert doc["counts"]["deferred"] == 62, doc["counts"]
    assert doc["counts"]["pending"] == 124, doc["counts"]
    assert doc["counts"]["complete"] == 0
    assert doc["counts"]["running"] == 0
    assert doc["counts"]["failed"] == 0
    # Counts must sum to total.
    c = doc["counts"]
    assert c["pending"] + c["running"] + c["complete"] + c["failed"] + c["deferred"] == c["total"]
    print("OK [row-counts] -- 186 rows, phase split 6/30/150, 62 Deferred (TransNeXt), 124 Pending.")


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
    assert row["quarantined"] is False, "non-TransNeXt row must not be quarantined"
    # Other 185 rows untouched: 62 Deferred (TransNeXt) + 123 Pending.
    others = [r for r in doc["rows"] if r["tag"] != tag]
    pending_others = [r for r in others if r["status"] == "Pending"]
    deferred_others = [r for r in others if r["status"] == "Deferred"]
    assert len(pending_others) == 123, f"expected 123 Pending; got {len(pending_others)}"
    assert len(deferred_others) == 62, f"expected 62 Deferred; got {len(deferred_others)}"
    assert all("transnext" in r["model"].lower() for r in deferred_others), \
        "all Deferred rows must be TransNeXt"
    assert doc["counts"]["complete"] == 1
    assert doc["counts"]["pending"] == 123
    assert doc["counts"]["deferred"] == 62
    print("OK [roundtrip] -- metrics.json hydrates row; 62 TransNeXt rows stay Deferred.")


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
    pat = re.compile(r"^final_(clean|B_L\d|C_L\d_[a-z_]+)_(resnet50|densenet121|transnext_tiny)_(cifar10|mnist)$")
    bad = [t for t in tags if not pat.match(t)]
    assert not bad, f"non-canonical tags: {bad[:5]}"
    print(f"OK [sort] -- 186 rows sorted by tag; first={tags[0]}, last={tags[-1]}.")


def _check_visual_core_field_round_trip(monkeypatch_dir: bool = True) -> None:
    """US-017: aggregator hydrates `visual_core` from disk PNG presence,
    using `Path.exists()` only (no `open()`). Missing PNG -> None."""
    from src.tools import build_final_exp_json as agg

    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        thumbs = Path(td) / "dashboard_thumbs"
        thumbs.mkdir()
        # Plant a thumb for one specific cell only.
        target_tag = "final_clean_resnet50_cifar10"
        (thumbs / f"{target_tag}.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)

        original_dir = agg._VISUAL_CORE_DIR
        agg._VISUAL_CORE_DIR = thumbs
        try:
            doc = agg.build_doc(runs_root=runs_root)
        finally:
            agg._VISUAL_CORE_DIR = original_dir

    target = next(r for r in doc["rows"] if r["tag"] == target_tag)
    assert target["visual_core"] == f"dashboard_thumbs/{target_tag}.png", target["visual_core"]
    # Every other row's visual_core is None (no PNG planted).
    others = [r for r in doc["rows"] if r["tag"] != target_tag]
    missing = [r for r in others if r["visual_core"] is not None]
    assert not missing, f"unexpected non-None visual_core rows: {[r['tag'] for r in missing[:3]]}"
    print("OK [visual-core] -- aggregator hydrates `visual_core` only when PNG is on disk.")


def _check_visual_core_lookup_does_not_open_png() -> None:
    """US-017 privacy/perf: the aggregator must never `open()` a PNG when
    populating `visual_core` — only `Path.exists()`."""
    from src.tools import build_final_exp_json as agg

    opened: list[str] = []
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
        thumbs = Path(td) / "dashboard_thumbs"
        thumbs.mkdir()
        # Plant a few PNGs the aggregator must NOT open.
        for tag in (
            "final_clean_resnet50_mnist",
            "final_B_L3_densenet121_cifar10",
            "final_C_L2_blur_resnet50_mnist",
        ):
            (thumbs / f"{tag}.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)

        original_dir = agg._VISUAL_CORE_DIR
        agg._VISUAL_CORE_DIR = thumbs
        try:
            with patch("builtins.open", side_effect=_recording_open), \
                 patch("io.open", side_effect=_recording_open):
                agg.build_doc(runs_root=runs_root)
        finally:
            agg._VISUAL_CORE_DIR = original_dir

    bad = [p for p in opened if p.endswith(".png")]
    assert not bad, f"aggregator opened PNG file(s): {bad}"
    print(f"OK [visual-core-privacy] -- {len(opened)} open() calls, none touched a PNG.")


def _check_has_history_field_round_trip() -> None:
    """US-018: aggregator sets `has_history=True` iff `runs/final/<tag>/history.json`
    exists, using `Path.exists()` only (no `open()`). Missing -> False."""
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        # Plant a history.json for one cell.
        target_tag = "final_clean_resnet50_cifar10"
        (runs_root / target_tag).mkdir(parents=True)
        (runs_root / target_tag / "history.json").write_text(
            json.dumps({"schema_version": 1, "tag": target_tag,
                        "history": [{"epoch": 1, "val_acc": 0.5}]}),
            encoding="utf-8",
        )
        from src.tools.build_final_exp_json import build_doc
        doc = build_doc(runs_root=runs_root)

    target = next(r for r in doc["rows"] if r["tag"] == target_tag)
    assert target["has_history"] is True, target.get("has_history")
    others = [r for r in doc["rows"] if r["tag"] != target_tag]
    bad = [r for r in others if r["has_history"]]
    assert not bad, f"unexpected has_history=True rows: {[r['tag'] for r in bad[:3]]}"
    print("OK [has-history] -- aggregator sets has_history from history.json presence.")


def _check_quarantine_override() -> None:
    """US-014: TransNeXt rows must render as Deferred even with on-disk metrics."""
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        # Stage a "Complete" metrics.json under a TransNeXt tag — simulating
        # a leftover from before the quarantine. The aggregator must still
        # render Deferred and must not surface val_acc as if active.
        tag = "final_clean_transnext_tiny_cifar10"
        (runs_root / tag).mkdir(parents=True)
        (runs_root / tag / "metrics.json").write_text(
            json.dumps({"best_val_acc": 0.99, "epochs_run": 60}), encoding="utf-8",
        )
        doc = build_doc(runs_root=runs_root)

    row = next(r for r in doc["rows"] if r["tag"] == tag)
    assert row["status"] == "Deferred", f"expected Deferred, got {row['status']}"
    assert row["quarantined"] is True
    assert row["quarantine_reason"] == "Awaiting Native-Resolution Refactor"
    # Counts: leftover metrics is overridden, so complete stays 0.
    assert doc["counts"]["complete"] == 0, doc["counts"]
    assert doc["counts"]["deferred"] == 62, doc["counts"]
    # No non-TransNeXt row should ever be quarantined.
    cnn_quarantined = [r for r in doc["rows"]
                        if r["quarantined"] and "transnext" not in r["model"].lower()]
    assert not cnn_quarantined, f"CNN rows wrongly quarantined: {cnn_quarantined}"
    print("OK [quarantine] -- TransNeXt row with stale metrics is forced Deferred.")


def _check_update_cell_patches_only_target_row() -> None:
    """US-019: `update_cell(tag)` must update only `tag`'s row + counts +
    generated_at. All other 185 rows must be byte-identical to the prior
    full build (modulo `generated_at` and `counts`)."""
    from src.tools.build_final_exp_json import (
        build_final_exp_json,
        update_cell,
    )

    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        out = Path(td) / "Final_Exp.json"

        # Initial full build with no completed cells.
        build_final_exp_json(out_path=out, runs_root=runs_root)
        before = json.loads(out.read_text(encoding="utf-8"))

        # Stage one cell complete, then call update_cell.
        target_tag = "final_clean_resnet50_cifar10"
        cell_dir = runs_root / target_tag
        cell_dir.mkdir(parents=True)
        (cell_dir / "metrics.json").write_text(
            json.dumps({"best_val_acc": 0.55, "epochs_run": 7}),
            encoding="utf-8",
        )
        summary = update_cell(tag=target_tag, out_path=out, runs_root=runs_root)
        after = json.loads(out.read_text(encoding="utf-8"))

    # Schema, row count, ordering preserved.
    assert after["schema_version"] == before["schema_version"]
    assert len(after["rows"]) == len(before["rows"]) == 186
    assert [r["tag"] for r in after["rows"]] == [r["tag"] for r in before["rows"]]

    # The target row has flipped to Complete.
    target_after = next(r for r in after["rows"] if r["tag"] == target_tag)
    target_before = next(r for r in before["rows"] if r["tag"] == target_tag)
    assert target_before["status"] == "Pending"
    assert target_after["status"] == "Complete"
    assert target_after["val_acc"] == 0.55
    assert target_after["epochs_run"] == 7

    # Every other row is byte-identical to the prior full build.
    for r_after, r_before in zip(after["rows"], before["rows"]):
        if r_after["tag"] == target_tag:
            continue
        assert r_after == r_before, (
            f"row {r_after['tag']} drifted: {r_after} != {r_before}"
        )

    # Counts are recomputed.
    assert after["counts"]["complete"] == before["counts"]["complete"] + 1
    assert after["counts"]["pending"] == before["counts"]["pending"] - 1
    # generated_at is bumped (modulo same-second collisions; assert non-decreasing).
    assert after["generated_at"] >= before["generated_at"]
    assert summary.get("updated_tag") == target_tag
    print("OK [update_cell] -- patches only target row; others byte-identical.")


def _check_update_cell_falls_back_to_full_rebuild() -> None:
    """`update_cell` must rebuild the doc when the on-disk JSON is missing
    or schema-mismatched, instead of crashing."""
    from src.tools.build_final_exp_json import update_cell

    target_tag = "final_clean_resnet50_cifar10"
    # Case 1: missing file → full rebuild produces 186 rows.
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        out = Path(td) / "missing.json"
        summary = update_cell(tag=target_tag, out_path=out, runs_root=runs_root)
        assert out.exists()
        assert summary["counts"]["total"] == 186

    # Case 2: corrupted JSON → full rebuild.
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        out = Path(td) / "Final_Exp.json"
        out.write_text("not-json{", encoding="utf-8")
        update_cell(tag=target_tag, out_path=out, runs_root=runs_root)
        doc = json.loads(out.read_text(encoding="utf-8"))
        assert doc["counts"]["total"] == 186

    # Case 3: schema-mismatched JSON → full rebuild.
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        out = Path(td) / "Final_Exp.json"
        out.write_text(json.dumps({"schema_version": 99, "rows": []}), encoding="utf-8")
        update_cell(tag=target_tag, out_path=out, runs_root=runs_root)
        doc = json.loads(out.read_text(encoding="utf-8"))
        assert doc["schema_version"] == 1
        assert doc["counts"]["total"] == 186
    print("OK [update_cell-fallback] -- missing/corrupt/old-schema JSON triggers full rebuild.")


def _check_update_cell_unknown_tag_raises() -> None:
    """Bad tags must raise — silently ignoring would mask wiring bugs."""
    from src.tools.build_final_exp_json import update_cell

    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        out = Path(td) / "Final_Exp.json"
        try:
            update_cell(tag="not_a_real_tag", out_path=out, runs_root=runs_root)
        except ValueError as e:
            assert "not_a_real_tag" in str(e)
            print("OK [update_cell-bad-tag] -- ValueError names the offending tag.")
            return
        raise AssertionError("update_cell did not raise for unknown tag")


def main() -> int:
    _check_row_counts_and_phase_split()
    _check_phase_c_l1_collapse()
    _check_metrics_roundtrip()
    _check_no_weight_path_opened()
    _check_atomic_write_and_no_tmp_leftover()
    _check_doc_level_fields()
    _check_rows_sorted_by_tag()
    _check_has_history_field_round_trip()
    _check_quarantine_override()
    _check_visual_core_field_round_trip()
    _check_visual_core_lookup_does_not_open_png()
    _check_update_cell_patches_only_target_row()
    _check_update_cell_falls_back_to_full_rebuild()
    _check_update_cell_unknown_tag_raises()
    print("\nAll build_final_exp_json checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
