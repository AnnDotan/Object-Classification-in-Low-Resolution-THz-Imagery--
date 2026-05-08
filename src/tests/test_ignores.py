"""Weight-privacy ignore-rules contract (US-013).

Mirrors scripts/check_ignores.sh in pure Python so it runs on Windows
where bash isn't standard. Every excluded category must be ignored;
the priors files must NOT be ignored; no binary weight files may be
tracked in the index.

Run: ``python -m src.tests.test_ignores``
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


# Hypothetical paths under each excluded category. They don't need to
# exist on disk — git check-ignore evaluates patterns, not filesystem.
SHOULD_IGNORE = [
    "artifacts/weights/transnext_base_224_1k.pth",
    "artifacts/weights/foo.pt",
    "artifacts/optuna_thz.db",
    "artifacts/dashboard_thumbs/final_clean_resnet50_cifar10.png",
    "artifacts/best_hparams/resnet50_cifar10.json",
    "runs/final/final_clean_resnet50_cifar10/best.ckpt",
    "runs/systematic/legacy_run/best.ckpt",
    "runs/final/x/model_last.pt",
    "wandb/run-20260101_000000-abc123/files/output.log",
    "lightning_logs/version_0/checkpoints/epoch=1.ckpt",
    "checkpoints/foo.pth",
    "weights/foo.pth",
]

SHOULD_TRACK = [
    "artifacts/priors/_schema.json",
    "artifacts/priors/resnet50.json",
    "artifacts/priors/densenet121.json",
    "artifacts/priors/transnext_base.json",
    # US-016: SYNCHRONIZER commits artifacts/Final_Exp.html at every phase
    # boundary so the remote always reflects the latest campaign snapshot.
    # It IS the canonical Gold Standard dashboard — must be tracked.
    "artifacts/Final_Exp.html",
]


def _git(args: list[str]) -> tuple[int, str, str]:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={_REPO_ROOT}"] + args,
        cwd=_REPO_ROOT, capture_output=True, text=True, check=False,
    )
    return result.returncode, result.stdout, result.stderr


def _check_should_ignore() -> None:
    """git check-ignore returns 0 when path IS ignored, 1 when not."""
    failures = []
    for p in SHOULD_IGNORE:
        rc, _, _ = _git(["check-ignore", "-q", p])
        if rc != 0:
            failures.append(p)
    assert not failures, (
        "expected these paths to be ignored, but they are tracked:\n  - "
        + "\n  - ".join(failures)
    )
    print(f"OK [ignored] — {len(SHOULD_IGNORE)} excluded categories all match .gitignore.")


def _check_should_track() -> None:
    failures = []
    for p in SHOULD_TRACK:
        rc, _, _ = _git(["check-ignore", "-q", p])
        if rc == 0:
            failures.append(p)
    assert not failures, (
        "expected these paths to be tracked, but .gitignore excludes them:\n  - "
        + "\n  - ".join(failures)
    )
    print(f"OK [tracked] — {len(SHOULD_TRACK)} priors paths visible to git.")


def _check_no_tracked_weights() -> None:
    rc, out, err = _git(["ls-files"])
    assert rc == 0, f"git ls-files failed: {err}"
    bad = [
        line for line in out.splitlines()
        if line.endswith((".ckpt", ".pt", ".pth", ".bin", ".safetensors",
                          ".onnx", ".h5"))
    ]
    assert not bad, (
        f"binary weight files in index: {bad[:5]} "
        "— never commit these per CLAUDE.md weight-privacy policy."
    )
    print(f"OK [no-binaries] — git index has zero tracked binary weight files.")


def main() -> int:
    _check_should_ignore()
    _check_should_track()
    _check_no_tracked_weights()
    return 0


if __name__ == "__main__":
    sys.exit(main())
