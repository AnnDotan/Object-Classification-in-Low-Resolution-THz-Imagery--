#!/usr/bin/env bash
# Smoke-test the weight-privacy contract from US-013.
#
# For each excluded category, confirm git check-ignore returns true on a
# representative path. For each tracked-on-purpose path (priors), confirm
# check-ignore returns false. Exits non-zero if any expectation fails.
#
# This file's sibling: ../tests/test_ignores.py exercises the same matrix
# from Python (and is invoked by CI on Windows where bash isn't standard).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Paths that MUST be ignored (binary / regenerated outputs).
declare -a SHOULD_IGNORE=(
  "artifacts/weights/transnext_tiny_224_1k.pth"
  "artifacts/weights/anything.pt"
  "artifacts/optuna_thz.db"
  "artifacts/dashboard_thumbs/final_clean_resnet50_cifar10.png"
  "artifacts/best_hparams/resnet50_cifar10.json"
  "runs/final/final_clean_resnet50_cifar10/best.ckpt"
  "runs/systematic/legacy_run/best.ckpt"
  "runs/final/x/model_last.pt"
  "wandb/run-20260101_000000-abc123/files/output.log"
  "lightning_logs/version_0/checkpoints/epoch=1.ckpt"
  "checkpoints/foo.pth"
  "weights/foo.pth"
)

# Paths that MUST be tracked (paper-anchored config + SYNCHRONIZER-committed
# canonical dashboard). artifacts/Final_Exp.html is committed at every phase
# boundary by SYNCHRONIZER per US-016 — the remote always reflects the latest
# campaign snapshot. Mirrors src/tests/test_ignores.py.
declare -a SHOULD_TRACK=(
  "artifacts/priors/_schema.json"
  "artifacts/priors/resnet50.json"
  "artifacts/priors/densenet121.json"
  "artifacts/priors/transnext_tiny.json"
  "artifacts/Final_Exp.html"
)

failures=0

for p in "${SHOULD_IGNORE[@]}"; do
  if git check-ignore -q "$p"; then
    echo "OK    ignored:  $p"
  else
    echo "FAIL  expected ignored but is tracked:  $p"
    failures=$((failures + 1))
  fi
done

for p in "${SHOULD_TRACK[@]}"; do
  if git check-ignore -q "$p"; then
    echo "FAIL  expected tracked but is ignored:  $p"
    failures=$((failures + 1))
  else
    echo "OK    tracked:  $p"
  fi
done

# No checkpoint files should be in the index.
tracked_ckpts="$(git ls-files | grep -E '\.(ckpt|pt|pth|bin|safetensors|onnx|h5)$' || true)"
if [ -n "$tracked_ckpts" ]; then
  echo "FAIL  tracked binary weight files in index:"
  echo "$tracked_ckpts"
  failures=$((failures + 1))
else
  echo "OK    no tracked binary weight files in index."
fi

if [ "$failures" -gt 0 ]; then
  echo
  echo "Privacy check FAILED: $failures issue(s)."
  exit 1
fi
echo
echo "Weight-privacy contract OK."
