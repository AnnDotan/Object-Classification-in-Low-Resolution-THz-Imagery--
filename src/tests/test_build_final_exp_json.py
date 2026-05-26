"""Regression tests for src/tools/build_final_exp_json.py (FINAL_EXP Dashboard, US-003).

Asserts the FinalExpDoc contract emitted by build_final_exp_json:
  - 186 rows split 6 / 30 / 150 across phases A / B / C.
  - Phase C L1 collapse: every C-L1 row's params dict == Phase B L1 row's params.
  - metrics.json round-trip: best_val_acc, last_val_loss, epochs_run, runtime_s
    populate the corresponding row; status flips from Pending to Complete.
  - Weight Privacy: aggregator never opens *.ckpt / *.pt / *.pth or any path
    under artifacts/weights/ (verified via builtins.open patch).
  - Atomic write: no .tmp sibling left behind after success; out_path written.
  - Document-level fields: schema_version matches `final_exp_schema.SCHEMA_VERSION`,
    generated_at parses as ISO8601.

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
    assert doc["schema_version"] == SCHEMA_VERSION
    # generated_at must parse as ISO8601 with timezone (Python's fromisoformat
    # accepts the ±HH:MM offset emitted by datetime.isoformat(timespec='seconds')).
    parsed = datetime.fromisoformat(doc["generated_at"])
    assert parsed.tzinfo is not None, f"generated_at lacks tz: {doc['generated_at']}"
    print(f"OK [doc-fields] -- schema_version={SCHEMA_VERSION}, generated_at parses ({doc['generated_at']}).")


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


def _check_history_inlined_onto_row() -> None:
    """The drawer renders learning curves under `file://`, where fetch() of
    cross-origin local files is blocked by every modern browser. The
    aggregator therefore must embed the parsed history.json list directly
    onto the row so the JS can render without a network call.
    Cells without a history.json keep `history=None` so the inline
    contract degrades cleanly."""
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        target_tag = "final_clean_resnet50_cifar10"
        (runs_root / target_tag).mkdir(parents=True)
        history_payload = [
            {"epoch": 1, "train_loss": 1.0, "val_loss": 1.1,
             "train_acc": 0.42, "val_acc": 0.40},
            {"epoch": 2, "train_loss": 0.5, "val_loss": 0.6,
             "train_acc": 0.78, "val_acc": 0.72},
        ]
        (runs_root / target_tag / "history.json").write_text(
            json.dumps({"schema_version": 1, "tag": target_tag,
                        "history": history_payload}),
            encoding="utf-8",
        )
        from src.tools.build_final_exp_json import build_doc
        doc = build_doc(runs_root=runs_root)

    target = next(r for r in doc["rows"] if r["tag"] == target_tag)
    assert target["history"] == history_payload, target.get("history")
    # All other rows must keep history=None — the inline payload is
    # bounded by the number of completed cells.
    nulls = [r for r in doc["rows"]
             if r["tag"] != target_tag and r["history"] is not None]
    assert not nulls, f"unexpected non-null history rows: {[r['tag'] for r in nulls[:3]]}"
    print("OK [history-inlined] -- aggregator embeds history.json list onto the row.")


def _check_image_quality_fields_round_trip() -> None:
    """PSNR/SSIM (US-002): aggregator copies `psnr_mean`/`psnr_std`/`ssim_mean`/
    `ssim_std` from `runs/final/<tag>/image_quality.json` onto the row so the
    dashboard can render them under the Visual Core thumbnail. Cells without
    an image_quality.json get null on all four fields."""
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        # Plant an image_quality.json for one Phase B cell.
        target_tag = "final_B_L3_resnet50_cifar10"
        (runs_root / target_tag).mkdir(parents=True)
        (runs_root / target_tag / "image_quality.json").write_text(
            json.dumps({
                "psnr_mean": 17.42, "psnr_std": 0.61,
                "ssim_mean": 0.4123, "ssim_std": 0.0287,
                "n_samples": 64,
            }),
            encoding="utf-8",
        )
        from src.tools.build_final_exp_json import build_doc
        doc = build_doc(runs_root=runs_root)

    target = next(r for r in doc["rows"] if r["tag"] == target_tag)
    assert target["psnr_mean"] == 17.42, target.get("psnr_mean")
    assert target["psnr_std"] == 0.61, target.get("psnr_std")
    assert abs(target["ssim_mean"] - 0.4123) < 1e-9, target.get("ssim_mean")
    assert abs(target["ssim_std"] - 0.0287) < 1e-9, target.get("ssim_std")

    others = [r for r in doc["rows"] if r["tag"] != target_tag]
    bad = [r for r in others
           if r["psnr_mean"] is not None or r["psnr_std"] is not None
           or r["ssim_mean"] is not None or r["ssim_std"] is not None]
    assert not bad, (
        f"unexpected non-null PSNR/SSIM rows: {[r['tag'] for r in bad[:3]]}"
    )
    print("OK [image-quality] -- aggregator copies PSNR/SSIM from image_quality.json onto rows.")


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
        assert doc["schema_version"] == SCHEMA_VERSION
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


# -----------------------------------------------------------------------------
# US-029.5: Phase D inclusion + schema v2 (treatment field, version bump).
# These tests are written as pytest-discoverable `test_*` functions so the
# user-facing harness `pytest src/tests/test_build_final_exp_json.py -v`
# enumerates them individually. The legacy `_check_*` helpers above run via
# `python -m src.tests.test_build_final_exp_json`.
# -----------------------------------------------------------------------------


# Snapshot of SCHEMA_VERSION's value PRIOR to US-029.5. The strict-inequality
# test below pins this constant so a future revert (or accidental rollback)
# of the v2 bump trips immediately. Updating this constant is intentional —
# do it the same time you bump SCHEMA_VERSION further.
_PREVIOUS_SCHEMA_VERSION: int = 1


def _stage_phase_d_metrics(runs_root: Path, tag: str) -> None:
    """Plant a minimal Phase D `metrics.json` for `tag` under `runs_root`.

    Just enough on-disk presence for `phase_d_present_on_disk` to flip on
    and `build_doc` to hydrate the row to Complete; no checkpoints written
    (the aggregator never reads `*.ckpt` — Weight Privacy invariant).
    """
    cell_dir = runs_root / tag
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "metrics.json").write_text(
        json.dumps({
            "best_val_acc": 0.50,
            "last_val_loss": 0.80,
            "epochs_run": 5,
            "runtime_s": 60.0,
        }),
        encoding="utf-8",
    )


def test_phase_d_absent_returns_186_rows() -> None:
    """No `final_D_*` dirs → default 186-row view is preserved byte-identical
    in row count + phase split. Every row carries `treatment=None`."""
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        # Plant a sentinel Phase A complete cell to exercise the row
        # hydration path; no Phase D directories.
        cell_dir = runs_root / "final_clean_resnet50_cifar10"
        cell_dir.mkdir(parents=True)
        (cell_dir / "metrics.json").write_text(
            json.dumps({"best_val_acc": 0.6, "epochs_run": 4}),
            encoding="utf-8",
        )
        doc = build_doc(runs_root=runs_root)

    assert doc["counts"]["total"] == 186, doc["counts"]
    assert len(doc["rows"]) == 186, len(doc["rows"])
    phase_d_rows = [r for r in doc["rows"] if r["phase"] == "D"]
    assert phase_d_rows == [], f"unexpected Phase D rows: {phase_d_rows[:3]}"
    # Every row's treatment field is None (Phase A/B/C carry treatment=None).
    bad = [r["tag"] for r in doc["rows"] if r["treatment"] is not None]
    assert not bad, f"non-None treatment on A/B/C rows: {bad[:3]}"


def test_phase_d_present_returns_276_rows() -> None:
    """At least one `final_D_*` dir on disk → 276-row view, with 90 Phase D
    rows whose `treatment` ∈ {T1, T2, T3} and 3-way 30/30/30 split."""
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        # One Phase D metrics.json is enough to flip the gate; the other
        # 89 Phase D rows enumerate as Pending. We plant a couple to be
        # explicit about treatment values being preserved.
        _stage_phase_d_metrics(runs_root, "final_D_T1_L3_resnet50_cifar10")
        _stage_phase_d_metrics(runs_root, "final_D_T2_L1_densenet121_mnist")
        doc = build_doc(runs_root=runs_root)

    assert doc["counts"]["total"] == 276, doc["counts"]
    assert len(doc["rows"]) == 276, len(doc["rows"])
    phase_d_rows = [r for r in doc["rows"] if r["phase"] == "D"]
    assert len(phase_d_rows) == 90, len(phase_d_rows)
    treatments = {r["treatment"] for r in phase_d_rows}
    assert treatments == {"T1", "T2", "T3"}, treatments
    # 30 rows per treatment (5 levels × 3 models × 2 datasets).
    by_treatment: dict[str, int] = {"T1": 0, "T2": 0, "T3": 0}
    for r in phase_d_rows:
        by_treatment[r["treatment"]] += 1
    assert by_treatment == {"T1": 30, "T2": 30, "T3": 30}, by_treatment
    # Phase A/B/C still carry treatment=None.
    abc_treatment_bad = [
        r["tag"] for r in doc["rows"]
        if r["phase"] in ("A", "B", "C") and r["treatment"] is not None
    ]
    assert not abc_treatment_bad, abc_treatment_bad[:3]


def test_schema_treatment_field_present() -> None:
    """Every row in build_doc's output carries a 'treatment' key whose
    value is None for A/B/C and a non-empty str for D rows."""
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        # Flip Phase D on so we see both None-bearing and str-bearing rows.
        _stage_phase_d_metrics(runs_root, "final_D_T3_L5_resnet50_mnist")
        doc = build_doc(runs_root=runs_root)

    # The key must be present on every row — not "missing => None".
    missing = [r["tag"] for r in doc["rows"] if "treatment" not in r]
    assert not missing, f"'treatment' key missing on rows: {missing[:3]}"
    for r in doc["rows"]:
        if r["phase"] == "D":
            assert isinstance(r["treatment"], str) and r["treatment"], (
                f"Phase D row {r['tag']!r} has non-str treatment "
                f"{r['treatment']!r}"
            )
        else:
            assert r["treatment"] is None, (
                f"Non-D row {r['tag']!r} has treatment {r['treatment']!r}"
            )


def test_schema_version_bumped() -> None:
    """US-029.5: SCHEMA_VERSION must be strictly greater than the v1
    snapshot. A revert would silently break dashboard cache invalidation,
    so this regression guard is loud."""
    assert SCHEMA_VERSION > _PREVIOUS_SCHEMA_VERSION, (
        f"SCHEMA_VERSION={SCHEMA_VERSION} did not advance past "
        f"v{_PREVIOUS_SCHEMA_VERSION}; US-029.5 requires a bump."
    )
    assert SCHEMA_VERSION >= 2, SCHEMA_VERSION


def main() -> int:
    _check_row_counts_and_phase_split()
    _check_phase_c_l1_collapse()
    _check_metrics_roundtrip()
    _check_no_weight_path_opened()
    _check_atomic_write_and_no_tmp_leftover()
    _check_doc_level_fields()
    _check_rows_sorted_by_tag()
    _check_has_history_field_round_trip()
    _check_history_inlined_onto_row()
    _check_image_quality_fields_round_trip()
    _check_quarantine_override()
    _check_visual_core_field_round_trip()
    _check_visual_core_lookup_does_not_open_png()
    _check_update_cell_patches_only_target_row()
    _check_update_cell_falls_back_to_full_rebuild()
    _check_update_cell_unknown_tag_raises()
    # US-029.5 additions — also discoverable by pytest.
    test_phase_d_absent_returns_186_rows()
    test_phase_d_present_returns_276_rows()
    test_schema_treatment_field_present()
    test_schema_version_bumped()
    print("\nAll build_final_exp_json checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
