# Runbook — Phase B (CNN) Execution

**Audience:** operator on the GPU box. Run these commands sequentially in your training environment (torch + CUDA installed).

**Scope:** 4 CNN (model, dataset) pairs × Optuna tune on L3 + top-3 full-convergence validation + 5-level Phase B sweep = **20 cells**. TransNeXt is quarantined (US-014). See [docs/prds/PHASE_B_VISUAL_CORE.md](../prds/PHASE_B_VISUAL_CORE.md) US-014 → US-019 and [docs/prds/PHASE_B_RUN.md](../prds/PHASE_B_RUN.md) US-020 → US-030.

**Methodology — Option C hybrid (locked 2026-05-07):**
- Stage 1 (`tune_all.py`) is a **fast proxy tune** (5 epochs / 2k train / 1k val per [`src/tune_hyperparams.py`](../../src/tune_hyperparams.py) `run_studies` defaults) — ~3 GPU-h to rank 20 trials per pair.
- Stage 1.5 ([`scripts/validate_top3.py`](../../scripts/validate_top3.py)) re-trains the **top-3 trials per pair** at production protocol (60 epochs / 10k train / 5k val / patience 10) and writes the validated winner with `validated_at_full_convergence: true` — ~15 GPU-h for all 4 pairs.
- Stage 2 (Phase B sweep, `run_all_phases.py --plan final --phase B`) reads only validated winner JSONs.
- Total tune+validate budget: ~18 GPU-h vs. ~100 GPU-h for full-Optuna-at-60-epochs. Defensible methodology in the paper: "Optuna ranked candidates with a 5-epoch / 2k-subset proxy; the top-3 per pair were validated at full convergence."

**Combined orchestration:** [`scripts/run_tune_chain.ps1`](../../scripts/run_tune_chain.ps1) runs Stage 1 + Stage 1.5 sequentially, with skip-existing logic so re-runs are safe.

---

## Stage -1 — GPU environment bootstrap (≈ 5–10 min, first run only)

Run this **once per machine** before anything else. It is a fail-fast, environment-only step (THz Protocol — no experimental logic touched). Driven by [scripts/setup_gpu_env.py](../../scripts/setup_gpu_env.py).

What it does:
1. Audits `nvidia-smi`, driver version, CUDA toolkit, and per-device VRAM.
2. Refuses to continue on Python ≥ 3.13 (no PyTorch wheels yet — the 3.14 incompatibility flagged 2026-05-07).
3. Installs the CUDA-matched `torch / torchvision / torchaudio` wheels via the official `https://download.pytorch.org/whl/cu1XX` index.
4. Installs project deps: `pytorch-lightning, optuna, optuna-integration[pytorch-lightning], plotly, pandas, opencv-python, matplotlib, timm, scikit-learn, torchmetrics, rich, pyyaml, tqdm`.
5. Writes a fresh `requirements.lock.txt` (UTF-8) for diff against the legacy UTF-16 `requirements.txt`.
6. Smoke-tests: `torch.cuda.is_available()`, per-device VRAM ≥ `--min-vram-gib` (default 6 GiB — ResNet50/DenseNet121 batch-32 floor), and a live 64×64 matmul on device 0.

**One-shot bootstrap (recommended):**

```powershell
python scripts/setup_gpu_env.py
```

**Other modes:**

```powershell
python scripts/setup_gpu_env.py --audit-only            # report hardware + Python, no install
python scripts/setup_gpu_env.py --dry-run               # print pip commands without running them
python scripts/setup_gpu_env.py --cuda-override 12.1    # bypass auto-detect
python scripts/setup_gpu_env.py --min-vram-gib 8        # tighter VRAM floor
python scripts/setup_gpu_env.py --make-venv .venv-gpu   # bootstrap a 3.10/3.11 venv first
```

If the audit reports Python ≥ 3.13 (e.g. the current `.venv` running 3.14.2), the script exits with the exact `py -3.11 -m venv` command needed. Re-run the bootstrap from the activated 3.11 venv.

Proceed to the pre-flight only after seeing `Stage -1 complete — environment ready for Stage 1 (Optuna pre-tune)`.

---

## Pre-flight (≈ 30 s)

Sanity-check the codebase before burning GPU hours.

```powershell
python -m src.tests.test_priors                # 8 OK lines
python -m src.tests.test_quarantine_transnext  # 9 PASS lines
python -m src.tests.test_build_final_exp_json  # 10 OK lines
python tune_all.py --validate-only             # "All 3 priors files valid."
```

If any fail, **do not proceed** — flag MASTER for diagnosis.

Verify `nvidia-smi` shows zero stale `python.exe` GPU processes (the US-014 quarantine already wiped TransNeXt dirs; double-check no zombie still holds VRAM).

```powershell
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

---

## Stage 1 — Optuna fast pre-tune at L3 Moderate (≈ 3 GPU-hours, proxy ranking)

> **NOTE (2026-05-07):** This is the **fast proxy tune** (5 ep / 2k train / 1k val) — a low-budget ranking pass, NOT the convergence-quality production training. The validated winners come from Stage 1.5 below. The `~12–14 GPU-h` figure in older revisions of this runbook assumed full-fidelity Optuna; that path was rejected as Option B-full per [PHASE_B_RUN.md](../prds/PHASE_B_RUN.md) §8 O4.

Fires 4 studies × 20 trials, seeded from the paper-anchored priors at [artifacts/priors/](../../artifacts/priors/). TransNeXt is auto-skipped by [tune_all.py](../../tune_all.py) (US-016 quarantine guard). Per-pair fast winners land at `artifacts/best_hparams/{model}_{dataset}.json` with the priors-file SHA-256 round-tripped — **these are overwritten in Stage 1.5**.

**Recommended (combined Stage 1 + 1.5 driver):**

```powershell
$proc = Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile","-WindowStyle","Hidden","-File","scripts\run_tune_chain.ps1" -PassThru
"PID=$($proc.Id)"  # save for later kill if needed
# Watch progress: tail -f .venv-gpu/tune_chain.log
```

**Manual per-pair (foreground):**

```powershell
.venv-gpu\Scripts\python.exe tune_all.py --n-trials 20 --model resnet50 --dataset cifar10
.venv-gpu\Scripts\python.exe tune_all.py --n-trials 20 --model densenet121 --dataset cifar10
.venv-gpu\Scripts\python.exe tune_all.py --n-trials 20 --model resnet50 --dataset mnist
.venv-gpu\Scripts\python.exe tune_all.py --n-trials 20 --model densenet121 --dataset mnist
```

**Resumable.** The Optuna study DB ([artifacts/optuna_thz.db](../../artifacts/optuna_thz.db)) lives on disk; rerun the same command after a crash and trials continue from where they died.

**Verify** before proceeding to Stage 2:

```powershell
python -c "import json; from pathlib import Path; ok = all(Path(f'artifacts/best_hparams/{m}_{d}.json').exists() for m in ('resnet50','densenet121') for d in ('cifar10','mnist')); print('all 4 winners present:', ok)"
```

Expect `all 4 winners present: True`. Spot-check one winner JSON:

```powershell
python -c "import json; print(json.dumps(json.load(open('artifacts/best_hparams/resnet50_cifar10.json')), indent=2))"
```

The `priors_file_hash` field must match `python -c \"import tune_all; print(tune_all.priors_file_hash('resnet50'))\"`.

---

## Stage 1.5 — Top-3 full-convergence validation (≈ 15 GPU-hours)

For each (model, dataset) pair: load the SQLite study, take the top-3 trials by fast val_acc, re-train each at the production protocol (60 ep / 10k train / 5k val / patience=10), pick the winner by full val_acc, and write the validated winner JSON with `validated_at_full_convergence: true`.

```powershell
.venv-gpu\Scripts\python.exe scripts\validate_top3.py
# or per-pair:
.venv-gpu\Scripts\python.exe scripts\validate_top3.py --model resnet50 --dataset cifar10
```

Idempotent at two levels:
- Per-pair: skipped if `artifacts/best_hparams/{m}_{d}.json` already has `validated_at_full_convergence: true`.
- Per-trial: result cached at `artifacts/validation/{m}_{d}_rank{N}.json`.

**Verify all 4 pairs validated before Stage 2:**

```powershell
.venv-gpu\Scripts\python.exe -c "import json; from pathlib import Path; ok=all(json.loads(Path(f'artifacts/best_hparams/{m}_{d}.json').read_text(encoding='utf-8')).get('validated_at_full_convergence') for m in ('resnet50','densenet121') for d in ('cifar10','mnist')); print('all 4 validated:', ok)"
```

Expect `all 4 validated: True`.

---

## Stage 2 — Phase B execution × 20 CNN cells (≈ 10 GPU-hours)

```powershell
python run_all_phases.py --plan final --phase B --skip-existing
```

Flags:
- `--plan final` — 186-cell campaign mode (rejects `--mode pilot`).
- `--phase B` — only the 30 Phase B cells; TransNeXt's 10 are auto-filtered by the US-014 quarantine guard, leaving 20.
- `--skip-existing` — cells whose `runs/final/<tag>/metrics.json` is already complete are no-ops (safe re-run after crash).

**Per-cell side effects (US-016):**
1. Lightning trains to convergence (60 epochs max, early-stop patience 10, monitor `val_acc`).
2. `_merge_metadata_into_metrics_json` round-trips `hparams_source.priors_file_hash` (US-008 + US-015).
3. `_measure_image_quality_for_cell` writes `image_quality.json` with PSNR/SSIM (256-sample subset) — Phase A is intentionally skipped.
4. `refresh_trackers.refresh_all` regenerates [Final_Exp.md](../../Final_Exp.md), [artifacts/Final_Exp.json](../../artifacts/Final_Exp.json), [artifacts/Final_Exp.html](../../artifacts/Final_Exp.html), and visual-core thumbs.

**Ctrl-C semantics:** [run_all_phases.py](../../run_all_phases.py) installs a SIGINT handler that drops `runs/final/<tag>/INTERRUPTED` under the in-flight cell and flushes a tracker refresh before re-raising. The dashboard renders that cell as `Failed`; `--skip-existing` will not re-skip it (no completed metrics) so a follow-up run picks it up.

---

## Stage 3 — Verify and commit

After all 20 cells finish (or you decide to stop early):

```powershell
python scripts/refresh_trackers.py            # final tracker refresh
python -c "import json; d=json.load(open('artifacts/Final_Exp.json')); print(d['counts'])"
```

Expect `complete >= 24` (Phase A 4 + Phase B 20). [Final_Exp.md](../../Final_Exp.md) status summary should read `Phase B (Combined): 20/30` with the `Deferred (TransNeXt — Awaiting Native-Resolution Refactor): 62/186` line intact.

Open [artifacts/Final_Exp.html](../../artifacts/Final_Exp.html) and confirm:
- Visual column shows Original|Degraded thumbnails for every Phase A/B row.
- Status pills: Phase A green, Phase B mostly green, TransNeXt rows dashed-purple `Deferred · Awaiting Native-Resolution Refactor`.
- Best val_acc populates the stat card per phase.

When satisfied:

```powershell
git status
python scripts/sync_trackers_git.py           # commits + pushes tracker artifacts
```

The tracker-sync script only stages `Final_Exp.md`, `artifacts/Final_Exp.json`, `artifacts/Final_Exp.html`, `artifacts/best_hparams/*.json`, and ignores `runs/final/**` per the weight-privacy rule.

---

## Failure recovery cheat-sheet

| Symptom | Action |
|---|---|
| Trial OOM | `tune_all.py` already wraps `Trainer.fit` failures into pruned trials — no action needed; re-run to top up trial count. |
| Ctrl-C mid-cell | Re-run `run_all_phases.py --plan final --phase B --skip-existing`; the INTERRUPTED sentinel marks that cell Failed and the runner moves on. |
| Disk full mid-cell | Free space, delete the offending cell's `runs/final/<tag>/` dir, re-run the orchestrator. |
| `metrics.json` corrupted | Delete the cell dir; re-run. |
| `best_hparams/{m}_{d}.json` missing | Re-run `python tune_all.py --n-trials 20 --model <m> --dataset <d>`. |
| TransNeXt accidentally re-trained | Run `python scripts/quarantine_transnext.py --yes` to wipe + reset rows to Deferred. |

## Commands you should NOT run

- `python tune_all.py --n-trials 20 --model transnext_base` or `--model transnext_small` — neither is the canonical TransNeXt variant any more. Base was swapped to `transnext_small` by US-004 on 2026-05-14; small was swapped to `transnext_tiny` by US-016 on the same day. Use `--model transnext_tiny` for the campaign sweep; `--model transnext_small` and `--model transnext_base` are now ad-hoc smoke-test paths only (and `tune_all.py SUPPORTED_MODELS` no longer lists them — the dispatch will reject these names at the CLI).
- `python run_all_phases.py --plan final --mode pilot` — rejected by assertion (CLAUDE.md "quality over speed").
- `git add runs/`, `git push runs/` — weight privacy violation; `runs/final/**` is gitignored + claudeignored on purpose.
- `python run_all_phases.py --plan final --phase B --skip-existing` **before Stage 1.5 has produced validated winners** — Phase B's `_load_hparams_for_cell` would consume the fast-tune winners (5-ep / 2k subset proxy), violating the convergence-first protocol. Always check `validated_at_full_convergence` is `true` first.

---

## Known issues / gotchas (2026-05-07 handoff notes)

These are not blockers but will save time on a fresh box. Each is also captured in [`progress.txt`](../../progress.txt) under the relevant US.

1. **`setup_gpu_env.py:make_venv` doesn't pick Python 3.12** — only iterates `RECOMMENDED_PY_MINORS = (11, 10)` even though `MAX_SUPPORTED_PY_MINOR = 12`. Workaround: if your box has 3.12 only (no 3.11/3.10), bootstrap manually:
   ```powershell
   py -3.12 -m venv .venv-gpu
   .venv-gpu\Scripts\python.exe scripts\setup_gpu_env.py     # full bootstrap
   ```
   One-line fix candidate: extend the candidate loop to `RECOMMENDED_PY_MINORS + tuple(range(MAX_SUPPORTED_PY_MINOR, 9, -1))`.

2. **`--min-vram-gib 6.0` is strict-fail at exactly 6 GiB cards.** RTX 4050 Laptop reports `5.997 GiB`, fails by ~4 MiB. On any GPU < 6 GiB or strictly equal, append `--min-vram-gib 5.9`. ResNet50 / DenseNet121 mixed-precision actually fit in ~5.5 GiB; the runbook's per-cell `batch_size=16` override remains the OOM-recovery path.

3. **`requirements.txt` is legacy UTF-16-encoded.** `setup_gpu_env.py` warns and writes `requirements.lock.txt` in UTF-8. Re-encode `requirements.txt` to UTF-8 in a separate commit.

4. **`scripts/quarantine_transnext.py:find_running_transnext_processes` does a broad substring match** on `"transnext"` against any python process's full cmdline. VSCode Pylance / Python LSP daemons scan project files including `*transnext*.py`, producing false positives in `test_quarantine_transnext` (PIDs change every run). The substantive check is `nvidia-smi --query-compute-apps` showing zero `python.exe` — that's what gates real GPU work. To tighten: filter for processes whose argv contains a real Python entry point like `tune_all.py --model transnext_base` or `run_systematic.py --model transnext_*`.

5. **PowerShell 5.1 `*>>` writes UTF-16 LE BOM by default + wraps native-exe stderr in ErrorRecord.** This corrupts the tune log so grep treats it as binary, plus prefixes every stderr line with `python.exe :`. [`scripts/run_tune_chain.ps1`](../../scripts/run_tune_chain.ps1) avoids this by using `cmd /c "... >> log 2>&1"` with `PYTHONIOENCODING=utf-8` set in the parent PS process. Result: log is pure UTF-8, grep-clean.

6. **`scripts/validate_top3.py` needed `sys.path.insert(0, REPO_ROOT)` at module top.** Running `python scripts/X.py` only puts `scripts/` on `sys.path`, so `from src.lightning ...` fails. Already fixed in the script. If you add new `scripts/*.py` that import `src.*`, replicate the pattern.

7. **Bash tool background timeout = 10 min max** in this Claude harness. For multi-hour chains, do NOT use `Bash run_in_background`; use `Start-Process powershell -WindowStyle Hidden -File ... -PassThru` to detach OS-level. The launched process survives shell death.

8. **Optuna's `study.optimize(n_trials=20)` ADDS 20 new trials each call** — does not "ensure 20 total". `scripts/run_tune_chain.ps1` guards against accidental doubling by querying SQLite for `t.state.is_finished()` count and skipping the pair if already ≥ 20. If you want fresh trials, delete `artifacts/optuna_thz.db` (and `artifacts/best_hparams/*.json` to invalidate fast winners).
