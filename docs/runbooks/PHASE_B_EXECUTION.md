# Runbook — Phase B (CNN) Execution

**Audience:** operator on the GPU box. Run these commands sequentially in your training environment (torch + CUDA installed).

**Scope:** 4 CNN (model, dataset) pairs × Optuna tune on L3 + 5-level Phase B sweep = **20 cells**. TransNeXt is quarantined (US-014). See [docs/prds/PHASE_B_VISUAL_CORE.md](../prds/PHASE_B_VISUAL_CORE.md) US-014 → US-019.

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

## Stage 1 — Optuna pre-tune at L3 Moderate (≈ 12–14 GPU-hours)

Fires 4 studies × 20 trials, seeded from the paper-anchored priors at [artifacts/priors/](../../artifacts/priors/). TransNeXt is auto-skipped by [tune_all.py](../../tune_all.py) (US-016 quarantine guard). Results land at `artifacts/best_hparams/{model}_{dataset}.json` with the priors-file SHA-256 round-tripped for reproducibility.

```powershell
python tune_all.py --n-trials 20
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

Expect `complete >= 24` (Phase A 4 + Phase B 20). [Final_Exp.md](../../Final_Exp.md) status summary should read `Phase B (Combined): 20/30` with the `Deferred (TransNeXt — Pending Hardware): 62/186` line intact.

Open [artifacts/Final_Exp.html](../../artifacts/Final_Exp.html) and confirm:
- Visual column shows Original|Degraded thumbnails for every Phase A/B row.
- Status pills: Phase A green, Phase B mostly green, TransNeXt rows dashed-purple `Deferred · Pending Hardware`.
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

- `python tune_all.py --n-trials 20 --model transnext_base` — bypasses the US-014 quarantine. Only use when MASTER explicitly reactivates TransNeXt with a hardware tier upgrade.
- `python run_all_phases.py --plan final --mode pilot` — rejected by assertion (CLAUDE.md "quality over speed").
- `git add runs/`, `git push runs/` — weight privacy violation; `runs/final/**` is gitignored + claudeignored on purpose.
