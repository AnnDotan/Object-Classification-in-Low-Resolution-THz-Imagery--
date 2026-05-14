"""Torch-free fixture for the RALPH Loop Driver (PRD US-005).

Exercises pathology guard verdicts, sentinel I/O, retry-config construction,
the OOM-hook 3-attempt cap, --skip-existing semantics, the
QUARANTINED_AFTER_RETRY skip on --remediate-only, and the
``--plan final --mode pilot`` assert.

Run: ``python -m pytest src/tests/test_ralph_loop.py -q``
"""
from __future__ import annotations

import io
import json
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

import pytest

# Importing the driver must not require torch / Lightning.
import scripts.run_ralph_loop as ralph
from src.experiments.cells import CellMeta


# ---------------------------------------------------------------------------
# Pathology guard
# ---------------------------------------------------------------------------

def test_pathology_failed_convergence_nan_loss_at_epoch_2() -> None:
    history = [
        {"epoch": 1, "train_loss": 2.3, "val_loss": 2.3, "train_acc": 0.10, "val_acc": 0.10},
        {"epoch": 2, "train_loss": float("nan"), "val_loss": 2.3,
         "train_acc": 0.10, "val_acc": 0.10},
        {"epoch": 3, "train_loss": 2.3, "val_loss": 2.3, "train_acc": 0.11, "val_acc": 0.10},
    ]
    assert ralph.evaluate_pathology(history) == "failed_convergence"


def test_pathology_overfitting_22pp_gap_at_best_epoch() -> None:
    # train 0.92, val 0.70 → gap = 22pp at best-val epoch.
    history = [
        {"epoch": 1, "train_loss": 1.0, "val_loss": 1.2, "train_acc": 0.40, "val_acc": 0.38},
        {"epoch": 5, "train_loss": 0.4, "val_loss": 0.9, "train_acc": 0.85, "val_acc": 0.68},
        {"epoch": 10, "train_loss": 0.2, "val_loss": 0.9,
         "train_acc": 0.92, "val_acc": 0.70},
        {"epoch": 15, "train_loss": 0.1, "val_loss": 1.1,
         "train_acc": 0.95, "val_acc": 0.69},
    ]
    assert ralph.evaluate_pathology(history) == "overfitting"


def test_pathology_healthy_small_gap_no_late_drift() -> None:
    # Gap at best epoch = 9pp < 12pp; val_acc trends up monotonically.
    history = [
        {"epoch": 1, "train_loss": 1.5, "val_loss": 1.5, "train_acc": 0.45, "val_acc": 0.43},
        {"epoch": 10, "train_loss": 0.5, "val_loss": 0.7,
         "train_acc": 0.70, "val_acc": 0.65},
        {"epoch": 20, "train_loss": 0.3, "val_loss": 0.5,
         "train_acc": 0.80, "val_acc": 0.74},
        {"epoch": 30, "train_loss": 0.25, "val_loss": 0.45,
         "train_acc": 0.84, "val_acc": 0.75},
    ]
    assert ralph.evaluate_pathology(history) == "healthy"


def test_pathology_empty_history_is_failed_convergence() -> None:
    assert ralph.evaluate_pathology([]) == "failed_convergence"


def test_pathology_low_best_val_below_failed_conv_threshold() -> None:
    # Best val_acc = 0.11 < 0.13 → failed_convergence.
    history = [
        {"epoch": 1, "train_loss": 2.3, "val_loss": 2.3, "train_acc": 0.10, "val_acc": 0.10},
        {"epoch": 10, "train_loss": 2.3, "val_loss": 2.3,
         "train_acc": 0.11, "val_acc": 0.11},
    ]
    assert ralph.evaluate_pathology(history) == "failed_convergence"


# ---------------------------------------------------------------------------
# Sentinel I/O + skip_existing
# ---------------------------------------------------------------------------

def test_sentinel_write_read_remove(tmp_path: Path) -> None:
    tag = "final_clean_resnet50_cifar10"
    p = ralph.write_sentinel(tag, ralph.NEEDS_FULL_FT, "failed_convergence", base=tmp_path)
    assert p.exists() and p.read_text() == "failed_convergence"
    assert ralph.has_sentinel(tag, ralph.NEEDS_FULL_FT, base=tmp_path) is True
    assert ralph.remove_sentinel(tag, ralph.NEEDS_FULL_FT, base=tmp_path) is True
    assert ralph.has_sentinel(tag, ralph.NEEDS_FULL_FT, base=tmp_path) is False


def test_cell_is_complete_truthy_when_metrics_present(tmp_path: Path) -> None:
    tag = "final_clean_resnet50_cifar10"
    rd = ralph.cell_run_dir(tag, base=tmp_path)
    rd.mkdir(parents=True)
    (rd / "metrics.json").write_text(json.dumps({"best_val_acc": 0.5624}))
    assert ralph.cell_is_complete(tag, base=tmp_path) is True


def test_cell_is_complete_falsey_when_interrupted(tmp_path: Path) -> None:
    tag = "final_clean_resnet50_cifar10"
    rd = ralph.cell_run_dir(tag, base=tmp_path)
    rd.mkdir(parents=True)
    (rd / "metrics.json").write_text(json.dumps({"best_val_acc": 0.5624}))
    (rd / ralph.INTERRUPTED).write_text("")
    assert ralph.cell_is_complete(tag, base=tmp_path) is False


def test_cell_is_complete_falsey_when_missing(tmp_path: Path) -> None:
    assert ralph.cell_is_complete("does_not_exist", base=tmp_path) is False


# ---------------------------------------------------------------------------
# Retry config round-trip (PRD §6.3 + §6.4)
# ---------------------------------------------------------------------------

def test_build_retry_config_failed_convergence_deltas() -> None:
    base = {
        "lr_head": 9e-4,
        "lr_backbone": 9e-5,
        "weight_decay": 1e-4,
        "label_smoothing": 0.05,
        "batch_size": 32,
    }
    cfg = ralph.build_retry_config(base, "failed_convergence")
    assert cfg["full_ft"] is True
    assert cfg["remediation_reason"] == "failed_convergence"
    # §6.4 step 1: lr_backbone = lr_head / 10 (applied first)
    # Then §6.3: lr_head ÷ 3.
    assert cfg["lr_head"] == pytest.approx(9e-4 / 3.0)
    assert cfg["lr_backbone"] == pytest.approx(9e-4 / 10.0)
    assert cfg["weight_decay"] == pytest.approx(1e-4 * 1.5)
    assert cfg["label_smoothing"] == pytest.approx(0.10)


def test_build_retry_config_overfitting_deltas() -> None:
    base = {
        "lr_head": 9e-4,
        "lr_backbone": 9e-5,
        "weight_decay": 1e-4,
        "dropout": 0.0,
    }
    cfg = ralph.build_retry_config(base, "overfitting")
    assert cfg["full_ft"] is True
    assert cfg["remediation_reason"] == "overfitting"
    # §6.4 step 1 first: lr_backbone = lr_head / 10 = 9e-5.
    # §6.3 then ÷ 2: 9e-5 / 2 = 4.5e-5.
    assert cfg["lr_backbone"] == pytest.approx(9e-5 / 2.0)
    assert cfg["weight_decay"] == pytest.approx(2e-4)
    assert cfg["dropout"] == pytest.approx(0.1)


def test_build_retry_config_label_smoothing_cap() -> None:
    base = {"lr_head": 1e-3, "label_smoothing": 0.13}
    cfg = ralph.build_retry_config(base, "failed_convergence")
    assert cfg["label_smoothing"] == pytest.approx(0.15)


def test_retry_config_round_trip(tmp_path: Path) -> None:
    tag = "final_B_L3_resnet50_cifar10"
    base = {"lr_head": 1e-3, "lr_backbone": 1e-4, "weight_decay": 1e-4}
    cfg = ralph.build_retry_config(base, "failed_convergence")
    p = ralph.write_retry_config(tag, cfg, base=tmp_path)
    assert p.exists()
    rt = ralph.read_retry_config(tag, base=tmp_path)
    assert rt == cfg


def test_build_retry_config_rejects_unknown_verdict() -> None:
    with pytest.raises(ValueError, match="unknown verdict"):
        ralph.build_retry_config({"lr_head": 1e-3}, "healthy")


# ---------------------------------------------------------------------------
# OOM hook (PRD acceptance line 395)
# ---------------------------------------------------------------------------

def test_apply_oom_fix_preserves_effective_batch() -> None:
    cfg = ralph.apply_oom_fix({"batch_size": 32, "accumulate_grad_batches": 1})
    assert cfg["batch_size"] == 16
    assert cfg["accumulate_grad_batches"] == 2


def test_apply_oom_fix_compounds_across_attempts() -> None:
    cfg = ralph.apply_oom_fix({"batch_size": 16, "accumulate_grad_batches": 2})
    assert cfg["batch_size"] == 8
    assert cfg["accumulate_grad_batches"] == 4


def test_apply_oom_fix_exhausts_at_batch_1() -> None:
    with pytest.raises(RuntimeError, match="exhausted"):
        ralph.apply_oom_fix({"batch_size": 1, "accumulate_grad_batches": 32})


def test_debugger_log_records_fingerprint_per_attempt(tmp_path: Path) -> None:
    tag = "final_clean_transnext_tiny_cifar10"
    err = RuntimeError("CUDA out of memory. Tried to allocate 1.50 GiB")
    ralph.write_debugger_log(tag, err, attempt=1, base=tmp_path)
    ralph.write_debugger_log(tag, err, attempt=2, base=tmp_path)
    log_text = (tmp_path / tag / ralph.DEBUGGER_LOG).read_text()
    assert "attempt=1" in log_text and "attempt=2" in log_text
    assert "RuntimeError" in log_text


def test_oom_cap_stops_after_three_attempts(tmp_path: Path) -> None:
    tag = "final_clean_resnet50_cifar10"

    class _FakeOOM(RuntimeError):
        pass

    attempts = {"n": 0}

    def always_oom(_tag: str, mode: str = "full", engine: str = "lightning") -> None:
        attempts["n"] += 1
        raise _FakeOOM("simulated OOM")

    cells = [CellMeta(tag=tag, phase="A", model="resnet50", dataset="cifar10",
                      level=None, axis=None)]
    rc = ralph.run_dispatch(
        cells, mode="full", engine="lightning", skip_existing=False,
        base=tmp_path, run_cell_fn=always_oom, oom_exception=_FakeOOM,
    )
    assert rc == 1
    assert attempts["n"] == ralph.MAX_OOM_ATTEMPTS
    log = (tmp_path / tag / ralph.DEBUGGER_LOG).read_text()
    assert "attempt=3" in log


# ---------------------------------------------------------------------------
# Dispatch + post-train pathology integration
# ---------------------------------------------------------------------------

def test_dispatch_writes_needs_full_ft_on_overfitting(tmp_path: Path) -> None:
    tag = "final_B_L3_resnet50_cifar10"

    def fake_train(t: str, mode: str = "full", engine: str = "lightning") -> None:
        rd = tmp_path / t
        rd.mkdir(parents=True, exist_ok=True)
        history = [
            {"epoch": 1, "train_loss": 1.0, "val_loss": 1.2,
             "train_acc": 0.40, "val_acc": 0.38},
            {"epoch": 10, "train_loss": 0.1, "val_loss": 1.0,
             "train_acc": 0.92, "val_acc": 0.70},
        ]
        (rd / "history.json").write_text(json.dumps(history))
        (rd / "metrics.json").write_text(json.dumps({"best_val_acc": 0.70}))

    cells = [CellMeta(tag=tag, phase="B", model="resnet50", dataset="cifar10",
                      level=3, axis=None)]
    rc = ralph.run_dispatch(
        cells, mode="full", engine="lightning", skip_existing=False,
        base=tmp_path, run_cell_fn=fake_train, oom_exception=RuntimeError,
    )
    assert rc == 0
    assert (tmp_path / tag / ralph.NEEDS_FULL_FT).read_text() == "overfitting"


def test_dispatch_skip_existing_honors_complete(tmp_path: Path) -> None:
    tag = "final_clean_resnet50_cifar10"
    rd = tmp_path / tag
    rd.mkdir(parents=True)
    (rd / "metrics.json").write_text(json.dumps({"best_val_acc": 0.56}))

    called = {"n": 0}

    def never_train(t: str, mode: str = "full", engine: str = "lightning") -> None:
        called["n"] += 1

    cells = [CellMeta(tag=tag, phase="A", model="resnet50", dataset="cifar10",
                      level=None, axis=None)]
    rc = ralph.run_dispatch(
        cells, mode="full", engine="lightning", skip_existing=True,
        base=tmp_path, run_cell_fn=never_train, oom_exception=RuntimeError,
    )
    assert rc == 0
    assert called["n"] == 0


def test_dispatch_skip_existing_reattempts_interrupted(tmp_path: Path) -> None:
    tag = "final_clean_resnet50_cifar10"
    rd = tmp_path / tag
    rd.mkdir(parents=True)
    (rd / "metrics.json").write_text(json.dumps({"best_val_acc": 0.56}))
    (rd / ralph.INTERRUPTED).write_text("")

    called = {"n": 0}

    def trains(t: str, mode: str = "full", engine: str = "lightning") -> None:
        called["n"] += 1
        # Healthy history so no NEEDS_FULL_FT.
        (rd / "history.json").write_text(json.dumps([
            {"epoch": 1, "train_loss": 1.0, "val_loss": 1.0,
             "train_acc": 0.6, "val_acc": 0.55},
        ]))

    cells = [CellMeta(tag=tag, phase="A", model="resnet50", dataset="cifar10",
                      level=None, axis=None)]
    ralph.run_dispatch(
        cells, mode="full", engine="lightning", skip_existing=True,
        base=tmp_path, run_cell_fn=trains, oom_exception=RuntimeError,
    )
    assert called["n"] == 1


# ---------------------------------------------------------------------------
# Remediate-only: consumes NEEDS_FULL_FT, respects QUARANTINED_AFTER_RETRY
# ---------------------------------------------------------------------------

def test_remediate_writes_retry_config_and_dispatches(tmp_path: Path) -> None:
    tag = "final_B_L3_resnet50_cifar10"
    ralph.write_sentinel(tag, ralph.NEEDS_FULL_FT, "failed_convergence", base=tmp_path)

    called: dict = {}

    def fake_train(t: str, mode: str = "full", engine: str = "lightning") -> None:
        called["tag"] = t
        rd = tmp_path / t
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "history.json").write_text(json.dumps([
            {"epoch": 1, "train_loss": 1.0, "val_loss": 1.0,
             "train_acc": 0.6, "val_acc": 0.55},
        ]))

    def fake_hparams(_c: CellMeta) -> dict:
        return {"lr_head": 9e-4, "lr_backbone": 9e-5, "weight_decay": 1e-4,
                "label_smoothing": 0.05}

    cells = [CellMeta(tag=tag, phase="B", model="resnet50", dataset="cifar10",
                      level=3, axis=None)]
    rc = ralph.run_remediate(
        cells, mode="full", engine="lightning", base=tmp_path,
        run_cell_fn=fake_train, hparams_loader=fake_hparams,
    )
    assert rc == 0
    assert called["tag"] == tag
    retry = ralph.read_retry_config(tag, base=tmp_path)
    assert retry["full_ft"] is True
    assert retry["remediation_reason"] == "failed_convergence"
    # NEEDS_FULL_FT removed after healthy retry.
    assert not ralph.has_sentinel(tag, ralph.NEEDS_FULL_FT, base=tmp_path)


def test_remediate_skips_quarantined(tmp_path: Path) -> None:
    tag = "final_B_L3_resnet50_cifar10"
    ralph.write_sentinel(tag, ralph.NEEDS_FULL_FT, "overfitting", base=tmp_path)
    ralph.write_sentinel(tag, ralph.QUARANTINED_AFTER_RETRY, "prior_failure",
                         base=tmp_path)

    called = {"n": 0}

    def must_not_train(_t: str, **_kw) -> None:
        called["n"] += 1

    cells = [CellMeta(tag=tag, phase="B", model="resnet50", dataset="cifar10",
                      level=3, axis=None)]
    rc = ralph.run_remediate(
        cells, mode="full", engine="lightning", base=tmp_path,
        run_cell_fn=must_not_train, hparams_loader=lambda c: {},
    )
    assert rc == 0
    assert called["n"] == 0


def test_remediate_quarantines_after_second_failure(tmp_path: Path) -> None:
    tag = "final_B_L3_resnet50_cifar10"
    ralph.write_sentinel(tag, ralph.NEEDS_FULL_FT, "failed_convergence", base=tmp_path)

    def still_failing(t: str, **_kw) -> None:
        rd = tmp_path / t
        rd.mkdir(parents=True, exist_ok=True)
        # NaN loss at epoch 2 → second-failure verdict.
        (rd / "history.json").write_text(json.dumps([
            {"epoch": 1, "train_loss": 2.3, "val_loss": 2.3,
             "train_acc": 0.10, "val_acc": 0.10},
            {"epoch": 2, "train_loss": float("nan"), "val_loss": 2.3,
             "train_acc": 0.10, "val_acc": 0.10},
        ]))

    cells = [CellMeta(tag=tag, phase="B", model="resnet50", dataset="cifar10",
                      level=3, axis=None)]
    rc = ralph.run_remediate(
        cells, mode="full", engine="lightning", base=tmp_path,
        run_cell_fn=still_failing,
        hparams_loader=lambda c: {"lr_head": 1e-3},
    )
    assert rc == 1
    assert ralph.has_sentinel(tag, ralph.QUARANTINED_AFTER_RETRY, base=tmp_path)


# ---------------------------------------------------------------------------
# CLI invariants
# ---------------------------------------------------------------------------

def test_cli_dry_run_lists_filtered_cells() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = ralph.main(["--plan", "final", "--phase", "A", "--model", "resnet50",
                         "--dry-run"])
    out = buf.getvalue()
    assert rc == 0
    assert "final_clean_resnet50_cifar10" in out
    assert "final_clean_resnet50_mnist" in out
    # Phase B/C tags must be filtered out.
    assert "final_B_" not in out
    assert "final_C_" not in out


def test_cli_dry_run_resolves_186_cells_without_filter() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = ralph.main(["--plan", "final", "--dry-run"])
    out = buf.getvalue()
    assert rc == 0
    assert "resolved 186 cell(s)" in out


def test_cli_rejects_plan_final_with_mode_pilot() -> None:
    with pytest.raises(AssertionError, match="incompatible with --mode pilot"):
        ralph.main(["--plan", "final", "--mode", "pilot", "--dry-run"])


def test_filter_cells_by_dataset() -> None:
    from src.experiments.cells import iter_cells
    out = ralph.filter_cells(iter_cells(), dataset="mnist")
    # 3 models × (1 clean + 5 B-levels + 5×5 C-levels) = 3 × 31 = 93 mnist cells.
    assert len(out) == 93
    assert all(c.dataset == "mnist" for c in out)
