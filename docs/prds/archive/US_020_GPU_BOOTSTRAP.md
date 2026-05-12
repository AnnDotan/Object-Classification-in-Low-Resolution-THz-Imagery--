# PRD Draft — US-020 GPU Bootstrap + US-021/022/023 Roadmap

**Status (2026-05-07):**
- The **US-020 GPU bootstrap** portion of this draft is **superseded** by [PHASE_B_RUN.md](PHASE_B_RUN.md) US-020, which was approved and executed on the local RTX 4050 Laptop on 2026-05-07. [`scripts/setup_gpu_env.py`](../../scripts/setup_gpu_env.py) and `requirements.lock.txt` shipped from that work.
- The **US-021 (Phase C execution), US-022 (results synthesis), US-023 (paper figures)** sections below remain valid as the **post-Phase-B roadmap**. They will be promoted into a new PRD once Phase B (PHASE_B_RUN.md US-022..US-030) completes.
- US-### numbers in this draft conflict with the Phase B PRD (which now occupies US-022..US-030). When this roadmap is promoted, renumber accordingly (suggested: Phase C = US-031, results = US-032, figures = US-033).

**Project:** p-2026-061 — Object Classification in Low-Resolution THz Imagery
**Companion PRD:** [PHASE_B_VISUAL_CORE.md](PHASE_B_VISUAL_CORE.md) (US-014 → US-019, complete).
**Active PRD:** [PHASE_B_RUN.md](PHASE_B_RUN.md) (US-020 → US-030, in flight).
**Deadline pressure:** poster 2026-05-31 (T-24d), presentation 2026-06-21 (T-45d), submission 2026-07-26 (T-80d).

---

## 1. Why this PRD exists

Phase B execution ([docs/runbooks/PHASE_B_EXECUTION.md](../runbooks/PHASE_B_EXECUTION.md)) is fully wired in code but **gated on the GPU box being able to import `torch` + `optuna`**. The current development venv is Python 3.14.2 — no PyTorch wheel exists yet (PyTorch's official wheels lag a Python release by 6–9 months). The runbook already references [scripts/setup_gpu_env.py](../../scripts/setup_gpu_env.py) which doesn't exist; this PRD ships it.

After bootstrap unblocks training, three downstream stories convert raw `runs/final/` outputs into the artifacts needed for the poster + paper.

---

## 2. Goals & Non-Goals

### Goals
- **G1.** Single-command env bootstrap that audits the GPU, refuses on Py ≥ 3.13, installs CUDA-matched torch wheels + project deps, and smoke-tests CUDA availability + VRAM.
- **G2.** Phase C orchestration (150 single-axis isolation cells) — reuses Phase B infrastructure with a `--phase C` flag that's already in [run_all_phases.py](../../run_all_phases.py).
- **G3.** Results synthesis: per-cell `metrics.json` + `image_quality.json` aggregated into a single paper-ready table correlating `val_acc` against `PSNR` and `SSIM`.
- **G4.** Paper-ready figures: degradation curves (val_acc vs. level, per model × dataset), single-axis ablations (Phase C), accuracy ↔ visual-quality scatter.

### Non-Goals
- **NG1.** No new training-engine refactors.
- **NG2.** No TransNeXt re-activation (US-014 quarantine remains).
- **NG3.** No Optuna re-runs once Phase B `best_hparams/*.json` exists — Phase C reuses the same frozen hparams.
- **NG4.** No automated paper writing — figures + tables only; the LaTeX/Markdown body remains a manual artifact.

---

## 3. User Stories

### US-020 — GPU Environment Bootstrap (`scripts/setup_gpu_env.py`)

**Description:** A single fail-fast Python script that brings a fresh Windows / Linux GPU box from "git clone" to "torch.cuda.is_available() == True with the project's deps installed."

**Technical Implementation:**
- New file [scripts/setup_gpu_env.py](../../scripts/setup_gpu_env.py).
- **Audit phase** (no side effects, always runs first):
  - Detect Python version; **exit non-zero on `sys.version_info >= (3, 13)`** with the exact `py -3.11 -m venv .venv-gpu` command in the error message.
  - Run `nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv` and parse driver + per-device VRAM.
  - Detect CUDA toolkit major.minor via `nvidia-smi` → CUDA runtime mapping table (driver → CUDA matrix from NVIDIA docs).
  - Print a colored audit summary (use `rich` only if available; fall back to plain text).
- **Install phase** (`--audit-only` skips this; `--dry-run` prints commands without running them):
  - `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu{NN}` where `cu{NN}` is auto-derived from the CUDA detection.
  - `--cuda-override 12.1` lets the operator force a specific wheel index.
  - Project deps: `pytorch-lightning`, `optuna`, `optuna-integration[pytorch-lightning]`, `plotly`, `pandas`, `opencv-python`, `matplotlib`, `timm`, `scikit-learn`, `torchmetrics`, `rich`, `pyyaml`, `tqdm`, `psutil` (used by `quarantine_transnext.py`).
  - Write `requirements.lock.txt` (UTF-8) via `pip freeze` so future runs can verify drift against the legacy UTF-16 `requirements.txt`.
- **Smoke-test phase** (always runs after install, unless `--audit-only`):
  - `import torch; assert torch.cuda.is_available()` — fail otherwise.
  - For each device: `torch.cuda.get_device_properties(i).total_memory >= --min-vram-gib * 1024**3` (default 6 GiB; ResNet50/DenseNet121 batch-32 OOMs below ~5 GiB).
  - Live 64×64 matmul on device 0 to confirm kernel launch.
- **`--make-venv .venv-gpu`** (optional, runs first if specified): create a Python 3.11 venv via `py -3.11 -m venv` and re-exec the script inside it.
- Exit codes: 0 = ready, 1 = audit failure (Python version, no GPU), 2 = install failure, 3 = smoke-test failure (no CUDA / VRAM too small / matmul broken).

**Acceptance Criteria:**
- [ ] Logic: `python scripts/setup_gpu_env.py --audit-only` reports `OK` on a Py 3.11 + GPU box, `FAIL — Python 3.14.x not supported` on this dev shell, and exits 1.
- [ ] Logic: `--dry-run` emits the pip commands to stdout without invoking pip; matmul smoke-test still runs (it's not destructive).
- [ ] Logic: post-install, `python -c "import torch; print(torch.cuda.is_available())"` prints `True`. `python tune_all.py --validate-only` still prints `All 3 priors files valid.`
- [ ] Privacy: never reads `*.ckpt`/`*.pt`/`*.pth`. `requirements.lock.txt` is committed in this PRD's commit but `.venv-gpu/` is `.gitignore`d (verify in test).
- [ ] Quality: 5-check pytest suite — audit on Py3.13 fails, dry-run is side-effect-free, install/smoke paths are unit-tested behind a `subprocess` fake.
- [ ] Verification: a fresh `git clone` + `python scripts/setup_gpu_env.py --make-venv .venv-gpu` + activation + `python tune_all.py --n-trials 1 --model resnet50 --dataset cifar10` completes 1 trial under 10 minutes on a typical workstation GPU.

---

### US-021 — Phase C Execution (150 single-axis isolation cells, ≈ 50 GPU-h)

**Description:** Run the 150 Phase C cells (5 levels × 5 axes × 3 models × 2 datasets, **minus the 50 TransNeXt cells quarantined per US-014, leaving 100 CNN cells**) using the same frozen Optuna winners from Phase B. Phase C single-axis isolation pins every axis except the named one to L1; at L1 every isolation row collapses to that pair's Phase B L1 row (already enforced in [src/data/degradation_levels.py](../../src/data/degradation_levels.py)).

**Technical Implementation:**
- The wiring is already in place: [run_all_phases.py](../../run_all_phases.py) `--plan final --phase C` filters the matrix correctly; the US-014 quarantine guard skips TransNeXt; `--skip-existing` collapses L1 isolation rows to their Phase B L1 results when present.
- New runbook: `docs/runbooks/PHASE_C_EXECUTION.md` (mirrors PHASE_B_EXECUTION.md structure, no re-tune step — Stage 0 → Stage 1 (execute) → Stage 2 (verify+commit)).
- New tracker check: assert that at L1, every Phase C isolation cell's `metrics.json` round-trips back to its Phase B L1 row's `best_val_acc` (within numerical tolerance — same hparams + same data + same seed = same training trajectory).
- Storage budget: 100 cells × ~30 MB run dirs ≈ 3 GB (gitignored).

**Acceptance Criteria:**
- [ ] Logic: `python run_all_phases.py --plan final --phase C --skip-existing` enumerates exactly 100 CNN cells (50 quarantined).
- [ ] Logic: post-run, `Final_Exp.md` shows `Phase C (Isolation): 100/150` (50 deferred lock the denominator).
- [ ] Logic: L1 collapse — `runs/final/final_C_L1_<axis>_<model>_<dataset>/metrics.json::best_val_acc` matches `runs/final/final_B_L1_<model>_<dataset>/metrics.json::best_val_acc` to 4 decimal places for every (axis, model, dataset).
- [ ] Quality: existing test suites stay green; new `test_phase_c_l1_collapse_invariant` regression test in `src/tests/test_phase_c.py`.
- [ ] Verification: dashboard "All Cells" stat reads 124/186 complete + 62 deferred when both Phases B + C finish on CNN.

---

### US-022 — Results Synthesis (paper-ready CSV + summary table)

**Description:** Aggregate every cell's `metrics.json` + `image_quality.json` into one master CSV (`artifacts/results/master.csv`) and one Markdown summary table (`artifacts/results/summary.md`) suitable for direct inclusion in the poster and paper.

**Technical Implementation:**
- New script `src/tools/build_results_master.py`:
  - For each completed cell (status == `Complete`), emit one row with `tag, phase, model, dataset, level, axis, best_val_acc, last_val_acc, epochs_run, runtime_s, psnr_mean, psnr_std, ssim_mean, ssim_std, hparams_source.priors_file_hash`.
  - Write `master.csv` (pandas) and a pivot table `summary.md` (model × dataset × level → val_acc, with PSNR shown in tooltip-style parens).
- New module `src/analysis/correlations.py`:
  - `corr_acc_vs_quality(df) -> dict` — Pearson + Spearman correlation of `best_val_acc` vs `psnr_mean` and `ssim_mean`, broken down per (model, dataset).
  - Result also written to `summary.md` as a small Markdown table.
- Wire into `refresh_trackers.refresh_all` (soft-fail, lazy-imports pandas).

**Acceptance Criteria:**
- [ ] Logic: `python -m src.tools.build_results_master` emits 124-row `master.csv` after both Phase B + C finish (4 Phase A + 20 Phase B + 100 Phase C); rows for Pending/Deferred cells are excluded.
- [ ] Logic: `summary.md` contains one pivot per phase, plus the correlation table.
- [ ] Privacy: never opens `*.ckpt`. CSV cells never contain absolute paths.
- [ ] Quality: 4-check test suite — 0 cells in / 0 rows out, fixture cell hydration round-trip, correlation symmetry, deferred cells excluded.
- [ ] Verification: `master.csv` opens cleanly in Excel + pandas; `summary.md` renders correctly in GitHub preview.

---

### US-023 — Paper-Ready Figures (Phase B/C visualizations)

**Description:** Generate the three figures the poster and paper need: degradation curves, single-axis ablations, accuracy ↔ visual-quality scatter.

**Technical Implementation:**
- New script `src/analysis/plot_results.py` writing PNG + PDF (vector) to `artifacts/figures/`:
  - **Fig 1 — Degradation curves**: line plot, x = level (L1..L5), y = `best_val_acc`, one line per (model, dataset). Phase B input.
  - **Fig 2 — Single-axis ablations**: 5-panel grid (one per axis), each panel x = level, y = val_acc, one line per (model, dataset). Phase C input.
  - **Fig 3 — Accuracy vs visual quality**: scatter with `ssim_mean` on x-axis, `best_val_acc` on y-axis, color = model, shape = dataset; annotate Pearson r per facet. Combined Phase B + C input.
- Style: tight `matplotlib` defaults, large fonts, no `seaborn` dependency. Each figure has both a colored version (poster) and a grayscale fallback (print).
- Reproducibility: every figure embeds `priors_file_hash + git rev-parse HEAD` in the bottom-left as a small caption — anyone can trace the figure back to the exact run that produced it.

**Acceptance Criteria:**
- [ ] Logic: `python -m src.analysis.plot_results` emits 6 files (3 PNG + 3 PDF) given a populated `master.csv`.
- [ ] Logic: an empty / partial `master.csv` produces a placeholder figure with "incomplete data" text instead of crashing.
- [ ] Privacy: never reads `runs/final/`; pulls everything from `master.csv`.
- [ ] Quality: 3-check test suite — empty input handled, fixture CSV produces non-empty PNGs (file size > 5 KB), grayscale fallback differs in pixel hash from the colored version.
- [ ] Verification: figures pass a manual readability check at 1920×1080 (poster) and 8.5×11 inches at 300 dpi (print).

---

## 4. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **No CUDA wheel for the operator's GPU** (e.g. CUDA 13 driver, only cu118 wheels available) | Med | High | `--cuda-override` lets the operator pick. Audit table maps known driver ranges to wheel indexes; bootstrap exits with the manual `pip install torch ...` command on miss. |
| **Phase C OOM at L5 Extreme + DenseNet121** | Low | Med | The same hparams worked at L5 in Phase B — if not, fall back to `batch_size=16` via `--override` (already supported by `run_systematic._cell_config`). |
| **Disk fills at ~150 GB before Phase C completes** | Med | High | The Lightning checkpoints are the bulk; `LegacyCheckpointCallback` writes only `best.pt` + `model_last.pt` (~80 MB / cell). Phase C 100 cells × 80 MB = 8 GB — within budget. |
| **Poster deadline slip** | Med | High | US-020 must land first; US-021 (Phase C) is 50 GPU-h sequential. **Critical path:** start Stage 1 of Phase B no later than 2026-05-12 to hit the poster deadline. |
| **Optuna trial divergence vs Phase A baselines** | Low | Med | If a winner trial under-performs the CLAUDE.md frozen hparams (`PHASE_A_FROZEN_HPARAMS`), `_load_hparams_for_cell` falls back automatically — Phase B still runs. |

---

## 5. Dependency Map

```
US-020 (env bootstrap)  ──┐
                          ├──► Stage 1+2 of PHASE_B_EXECUTION.md (already in code)
                          │
US-021 (Phase C)  ◄───────┘
                          │
US-022 (results CSV)  ◄───┤  (depends on Phase B+C metrics.json files)
                          │
US-023 (figures)      ◄───┘  (depends on US-022 master.csv)
```

US-020 is the only blocker. US-021 → US-022 → US-023 then run sequentially as data accumulates. US-022 + US-023 *can* both run on partial data (e.g. just Phase B done) and re-run when Phase C lands — they're idempotent.

---

## 6. Open Questions for MASTER

- **O1.** Does US-020 ship `scripts/setup_gpu_env.py` as a standalone script (current proposal) or as a sub-command of `tune_all.py`? Standalone is simpler; folding it in would mean one fewer entry point but couples env management to tuning.
- **O2.** Should US-021's Phase C runbook be a new `PHASE_C_EXECUTION.md`, or an addendum section in `PHASE_B_EXECUTION.md`? Phase C reuses 90% of Phase B's recovery cheat-sheet, so a single doc with a Stage 4 (Phase C) section may be cleaner.
- **O3.** US-023 figure list: is the 3-figure set (degradation, ablations, scatter) sufficient for the poster, or do you also need (a) a sample-image grid showing Original vs Degraded at L1..L5, (b) a runtime-vs-accuracy frontier plot, or (c) a per-class confusion heatmap? More figures = more `src/analysis/` code.
- **O4.** Should `master.csv` (US-022) include `runtime_s`? It's useful for the "compute cost vs accuracy" angle but adds a column most reviewers won't read.

---

**End of draft. Awaiting MASTER decisions on O1–O4 before I implement US-020.**
