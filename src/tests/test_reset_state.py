"""Tests for ``scripts/reset_state.py`` (PRD US-040).

Verifies:

1. ``build_plan()`` enumerates every file that would be touched and
   leaves the filesystem untouched.
2. ``execute_plan()`` moves the Optuna DB to ``.pre-5070.bak``, moves
   ``best_hparams/*.json`` to ``_archive_pre_5070/``, deletes
   ``Final_Exp.{json,html}`` and ``validation/*.json``.
3. The preservation snapshot (priors, lock file) is byte-identical
   pre/post via SHA-256 round-trip.
4. Idempotent backup: a second run with a pre-existing ``.bak`` writes
   ``.bak.1`` rather than overwriting.
5. ``main(['--dry-run'])`` exits 0 and does NOT mutate the filesystem.
6. Empty state (nothing to reset) is handled cleanly (rc=0, empty plan).

Run: ``python -m src.tests.test_reset_state``
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "scripts_reset_state",
        _REPO_ROOT / "scripts" / "reset_state.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod  # dataclasses needs cls.__module__ to be registered
    spec.loader.exec_module(mod)
    return mod


def _stage_workspace(ws: Path) -> dict:
    """Populate ``ws`` with the artifact surfaces the reset operates on.
    Returns a snapshot dict so the test can assert pre/post conditions.
    """
    art = ws / "artifacts"
    art.mkdir(parents=True)

    optuna_db = art / "optuna_thz.db"
    optuna_db.write_bytes(b"FAKE-OPTUNA-DB")

    best = art / "best_hparams"
    best.mkdir()
    winners = {
        "resnet50_cifar10.json": {"best_value": 0.444, "study_name": "resnet50_cifar10_L3"},
        "resnet50_mnist.json": {"best_value": 0.949, "study_name": "resnet50_mnist_L3"},
        "densenet121_cifar10.json": {"best_value": 0.535, "study_name": "densenet121_cifar10_L3"},
        "densenet121_mnist.json": {"best_value": 0.956, "study_name": "densenet121_mnist_L3"},
    }
    for name, body in winners.items():
        (best / name).write_text(json.dumps(body), encoding="utf-8")

    validation = art / "validation"
    validation.mkdir()
    (validation / "resnet50_cifar10_rank0.json").write_text(
        '{"trial_number": 1, "full_val_acc": 0.78}', encoding="utf-8"
    )

    (art / "Final_Exp.json").write_text(
        '{"rows": [], "counts": {"complete": 4}}', encoding="utf-8"
    )
    (art / "Final_Exp.html").write_text(
        "<!doctype html><html><body>stale</body></html>", encoding="utf-8"
    )

    # Preserved surfaces.
    priors = art / "priors"
    priors.mkdir()
    (priors / "resnet50.json").write_text('{"head_lr": {"low": 1e-4}}', encoding="utf-8")
    (priors / "densenet121.json").write_text('{"head_lr": {"low": 1e-4}}', encoding="utf-8")
    (priors / "transnext_base.json").write_text('{"head_lr": {"low": 1e-4}}', encoding="utf-8")
    (priors / "_schema.json").write_text("{}", encoding="utf-8")
    (priors / "PRIORS_SOURCES.md").write_text("# sources\n", encoding="utf-8")

    (ws / "requirements.lock.txt").write_text(
        "torch==2.11.0+cu128\nlightning==2.6.1\n", encoding="utf-8"
    )

    return {"winners": winners}


def _check_build_plan_no_mutation() -> None:
    mod = _load_module()
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        _stage_workspace(ws)
        pre_listing = sorted(p.relative_to(ws) for p in ws.rglob("*") if p.is_file())

        plan = mod.build_plan(repo_root=ws)

        post_listing = sorted(p.relative_to(ws) for p in ws.rglob("*") if p.is_file())
        assert pre_listing == post_listing, "build_plan must not mutate the filesystem"

        kinds = [a.kind for a in plan.actions]
        srcs = [str(a.src.relative_to(ws)).replace("\\", "/") for a in plan.actions]

        # Optuna -> move
        assert "artifacts/optuna_thz.db" in srcs
        # 4 winner JSONs -> move
        for name in (
            "resnet50_cifar10.json",
            "resnet50_mnist.json",
            "densenet121_cifar10.json",
            "densenet121_mnist.json",
        ):
            assert f"artifacts/best_hparams/{name}" in srcs
        # validation cache -> delete
        assert "artifacts/validation/resnet50_cifar10_rank0.json" in srcs
        # Final_Exp.{json,html} -> delete
        assert "artifacts/Final_Exp.json" in srcs
        assert "artifacts/Final_Exp.html" in srcs

        # Action kinds sanity
        assert kinds.count("move") == 5  # optuna + 4 winners
        assert kinds.count("delete") == 3  # validation + Final_Exp.json + html
    print("OK [plan-no-mutation] -- build_plan enumerates 8 actions without touching disk.")


def _check_execute_plan_moves_and_deletes() -> None:
    mod = _load_module()
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        staged = _stage_workspace(ws)
        plan = mod.build_plan(repo_root=ws)

        pre_snapshot = mod.snapshot_preserved(repo_root=ws)
        mod.execute_plan(plan)
        post_snapshot = mod.snapshot_preserved(repo_root=ws)

        art = ws / "artifacts"
        # Optuna moved
        assert not (art / "optuna_thz.db").exists()
        assert (art / "optuna_thz.db.pre-5070.bak").exists()
        assert (art / "optuna_thz.db.pre-5070.bak").read_bytes() == b"FAKE-OPTUNA-DB"
        # best_hparams moved to archive
        archive = art / "best_hparams" / "_archive_pre_5070"
        assert archive.is_dir()
        for name in staged["winners"]:
            assert not (art / "best_hparams" / name).exists()
            assert (archive / name).exists()
            assert json.loads((archive / name).read_text(encoding="utf-8")) == staged["winners"][name]
        # validation cache deleted
        assert list((art / "validation").iterdir()) == []
        # Final_Exp.{json,html} deleted
        assert not (art / "Final_Exp.json").exists()
        assert not (art / "Final_Exp.html").exists()
        # Preserved snapshots identical
        assert pre_snapshot == post_snapshot
        assert len(pre_snapshot) >= 5  # 4 priors + 1 lock at minimum
    print("OK [execute] -- moves + deletes performed; preserved files byte-identical.")


def _check_backup_collision_appends_suffix() -> None:
    mod = _load_module()
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        _stage_workspace(ws)
        # Pre-existing backup -> next backup must get a numeric suffix.
        (ws / "artifacts" / "optuna_thz.db.pre-5070.bak").write_bytes(b"PRIOR-BACKUP")
        plan = mod.build_plan(repo_root=ws)

        # Find the Optuna action and confirm its dst.
        optuna_action = next(
            a for a in plan.actions
            if a.kind == "move" and a.src.name == "optuna_thz.db"
        )
        assert optuna_action.dst is not None
        assert optuna_action.dst.name == "optuna_thz.db.pre-5070.bak.1", optuna_action.dst

        mod.execute_plan(plan)
        # Original backup must survive.
        assert (ws / "artifacts" / "optuna_thz.db.pre-5070.bak").read_bytes() == b"PRIOR-BACKUP"
        # New backup must exist with the staged DB contents.
        assert (ws / "artifacts" / "optuna_thz.db.pre-5070.bak.1").read_bytes() == b"FAKE-OPTUNA-DB"
    print("OK [collision] -- pre-existing .bak preserved; new backup gets .bak.1.")


def _check_dry_run_cli_zero_mutation() -> None:
    mod = _load_module()
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        _stage_workspace(ws)
        pre_listing = sorted(p.relative_to(ws) for p in ws.rglob("*") if p.is_file())

        original_root = mod._REPO_ROOT
        mod._REPO_ROOT = ws  # type: ignore[attr-defined]
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = mod.main(["--dry-run"])
        finally:
            mod._REPO_ROOT = original_root  # type: ignore[attr-defined]

        out = buf.getvalue()
        post_listing = sorted(p.relative_to(ws) for p in ws.rglob("*") if p.is_file())

        assert rc == 0
        assert pre_listing == post_listing, "--dry-run must not mutate filesystem"
        # Plan lines emitted.
        assert "DRY RUN" in out
        assert "optuna_thz.db" in out
        assert "best_hparams" in out
        assert "Final_Exp.json" in out
        assert "Final_Exp.html" in out
    print("OK [dry-run-cli] -- main(['--dry-run']) exits 0 and emits plan; no FS mutation.")


def _check_empty_state_clean_exit() -> None:
    mod = _load_module()
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        # Bare workspace: no artifacts/ at all.
        plan = mod.build_plan(repo_root=ws)
        assert plan.is_empty()

        original_root = mod._REPO_ROOT
        mod._REPO_ROOT = ws  # type: ignore[attr-defined]
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = mod.main(["--yes"])
        finally:
            mod._REPO_ROOT = original_root  # type: ignore[attr-defined]

        assert rc == 0
        assert "Nothing to do" in buf.getvalue()
    print("OK [empty-state] -- bare workspace yields empty plan, rc=0.")


def main() -> int:
    _check_build_plan_no_mutation()
    _check_execute_plan_moves_and_deletes()
    _check_backup_collision_appends_suffix()
    _check_dry_run_cli_zero_mutation()
    _check_empty_state_clean_exit()
    print("All reset_state tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
