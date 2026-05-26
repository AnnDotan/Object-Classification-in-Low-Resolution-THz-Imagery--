# Object Classification in Low-Resolution THz Imagery

> **Final Research Campaign — 276 / 276 cells complete (v1 closed 2026-05-20; pipeline-v2 noise-fix campaign re-ran 90 cells and closed 2026-05-23, `PIPELINE_VERSION = 2`; Phase D regularization sweep added 90 cells and closed 2026-05-26).**
> 5-level degradation curve (L1 Mild → L5 Extreme), deterministic saturation axis, bf16-mixed precision (Blackwell), paper-anchored Optuna pre-tuning, three-treatment Phase D regularization recovery layer.
> Comprehensive research report: [`docs/Final_Exp_Report.md`](docs/Final_Exp_Report.md) / [`artifacts/Final_Exp.pdf`](artifacts/Final_Exp.pdf) — interactive dashboard: `artifacts/Final_Exp.html` — master tracker: [`Final_Exp.md`](Final_Exp.md).

## Research Question

How robust does image classification remain when visual information is severely degraded — low resolution, blur, noise, salt-and-pepper, desaturation — simulating Terahertz (THz) imaging conditions? We evaluate three architectures (one residual CNN, one densely-connected CNN, one aggregated-attention ViT) across a 5-level degradation curve on two datasets (CIFAR-10, MNIST), with single-axis isolation runs to attribute the contribution of each degradation type.

## Implementation Status

| Component | Status |
|---|---|
| 5-level degradation table — [`src/data/degradation_levels.py`](src/data/degradation_levels.py) | ✅ live |
| Saturation axis plumbed through `DataConfig` → `DegradeConfig` | ✅ live (US-001) |
| `degrade_config_for(level, axis)` helper | ✅ live |
| 186-cell matrix generator — [`src/experiments/matrix.py`](src/experiments/matrix.py) | ✅ live (US-007) |
| PyTorch Lightning training loop — [`src/lightning/`](src/lightning/) | ✅ live |
| Offline PSNR/SSIM script — [`src/tools/measure_image_quality.py`](src/tools/measure_image_quality.py) | ✅ live (US-002) |
| Per-model paper-anchored priors — [`artifacts/priors/*.json`](artifacts/priors/) | ✅ live (US-003) |
| `tune_all.py` priors loader + Optuna study runner | ✅ live (US-004 + US-005) |
| TransNeXt `small` + full-FT mode + auto-download (swapped from `base` by US-004 on 2026-05-14) | ✅ live (US-006) |
| `--cell-tag` matrix consumer in [`run_systematic.py`](run_systematic.py) | ✅ live (US-008) |
| `--plan final` orchestration in [`run_all_phases.py`](run_all_phases.py) | ✅ live (US-009) |
| Pre-rendered Original-vs-Degraded thumbs — [`src/tools/render_cell_thumbs.py`](src/tools/render_cell_thumbs.py) | ✅ live (US-010) |
| Final_Exp.html dashboard — [`src/tools/build_final_dashboard.py`](src/tools/build_final_dashboard.py) | ✅ live (US-011, rewritten as the FINAL_EXP Dashboard — pilot-styled tabs / chip filters / 30 s polling — see [`docs/prds/FINAL_EXP_DASHBOARD.md`](docs/prds/FINAL_EXP_DASHBOARD.md)) |
| `artifacts/Final_Exp.json` aggregate + `src/tools/build_final_exp_json.py` | ✅ live (FINAL_EXP US-001..US-003) |
| W&B `run_id` / `entity` / `project` in `metrics.json` | ✅ live (US-012) |
| Weight-privacy hardening (`.gitignore`, `.claudeignore`, `scripts/check_ignores.sh`) | ✅ live (US-013) |
| `scripts/update_final_exp.py` (regenerates `Final_Exp.md` row data after each run) | ✅ live |
| `scripts/fetch_transnext_weights.py` (pre-fetch checkpoint into `artifacts/weights/`) | ✅ live |
| Click-to-toggle learning-curve panels in `Final_Exp.html` (`render_curve_thumbs.py`) | 🗑 deprecated — superseded by the chip-filtered FINAL_EXP Dashboard; thumbs still rendered for ad-hoc inspection. |
| Phase B2 (+ B2-nr) THz-protocol simplification rationale — [`docs/phase_b2.md`](docs/phase_b2.md) | 📐 documented (US-037, 2026-05-26); code wiring lands at US-038, dispatch at US-039 → US-041. |
| Phase C2 THz-protocol single-axis attribution rationale — [`docs/phase_c2.md`](docs/phase_c2.md) | 📐 documented (US-037, 2026-05-26); code wiring lands at US-038, dispatch at US-043 → US-044. |

Legacy 33/36 results in `runs/systematic/` are frozen and kept for reference.

## Campaign Status — RTX 5070 RALPH Loop (branch `5070Ca`, 2026-05-20)

**186-cell campaign:** Phase A (6) + Phase B (30) + Phase C (150). Three models × two datasets across the 5-level degradation curve. Full PRD: [`PRD.md`](PRD.md) · iteration log: [`progress.txt`](progress.txt) (1–26).

| Story (new ID) | Legacy ID | Title | Status |
|---|---|---|---|
| — | US-040 | State reset (Optuna DB + best_hparams archived; trackers rebuilt) | ✅ closed |
| — | US-041 | RTX 5070 cu128 bootstrap + bf16 smoke test | ✅ closed |
| — | US-042 | TransNeXt @ 224×224 un-quarantine | ✅ closed |
| — | US-043 | `resnet50` × {cifar10, mnist} Optuna tune (Stage 1 + 1.5) | ✅ closed |
| **US-001** | US-044 | `densenet121` × {cifar10, mnist} Optuna tune | ✅ **closed 2026-05-14** |
| **US-002** | US-045 | `transnext_tiny` × {cifar10, mnist} Optuna tune *(retargeted from `transnext_small` by US-016 on 2026-05-14; predecessor base→small swap was US-004 same day)* | ✅ **closed 2026-05-14** (cifar10 trial #16, best_value=0.7336; mnist trial #12, best_value=0.9176; both `validated_at_full_convergence: true`) |
| **US-003** | (new) | Phase C single-axis correction (inactive axes → identity) | ✅ **closed 2026-05-14** ([`docs/phase_c.md`](docs/phase_c.md)) |
| **US-004** | (new) | TransNeXt base→small variant swap (in-place repo retargeting) | ✅ **closed 2026-05-14** |
| **US-005** | US-046 (infra) | RALPH Loop Driver Framework — `scripts/run_ralph_loop.py` + pathology guard + retry pass + tests | ✅ **closed** |
| **US-006** | US-046 (split) | Phase A execution — `resnet50` × {cifar10, mnist} (2 cells) | ✅ **closed 2026-05-14** |
| **US-007** | US-046 (split) | Phase A execution — `densenet121` × {cifar10, mnist} (2 cells) | ✅ **closed 2026-05-15** |
| **US-008** | US-046 (split) | Phase A execution — `transnext_tiny` × {cifar10, mnist} (2 cells) | ✅ **closed 2026-05-15** (PARTIAL — VALIDATOR seed=43 deferred) |
| **US-009** | US-046 (split) | Phase B execution — `resnet50` × L1…L5 × {cifar10, mnist} (10 cells) | ✅ **closed 2026-05-15** |
| **US-010** | US-046 (split) | Phase B execution — `densenet121` × L1…L5 × {cifar10, mnist} (10 cells) | ✅ **closed 2026-05-15** |
| **US-011** | US-046 (split) | Phase B execution — `transnext_tiny` × L1…L5 × {cifar10, mnist} (10 cells) | ✅ **closed 2026-05-16** (10/10 healthy first pass) |
| **US-012** | US-046 (split) | Phase C execution — `resnet50` × 5 axes × L1…L5 × {cifar10, mnist} (50 cells) | ✅ **closed 2026-05-17** |
| **US-013** | US-046 (split) | Phase C execution — `densenet121` × 5 axes × L1…L5 × {cifar10, mnist} (50 cells) | ✅ **closed 2026-05-18** |
| **US-014** | US-046 (split) | Phase C execution — `transnext_tiny` × 5 axes × L1…L5 × {cifar10, mnist} (50 cells) | ✅ **closed 2026-05-20** |
| **US-015** | US-047 | End-of-campaign verification + 3 phase-boundary pushes | ⏳ **partial** (Final_Exp.md regen + leak audit + mypy + pytest done 2026-05-20; SYNCHRONIZER tracker pushes + NOTEBOOKLM_SYNC deferred) |
| **US-016** | (new) | TransNeXt small→tiny variant swap (capacity-match to ResNet50; ~310 GPU-h saved across the 62 TransNeXt rows) | ✅ **closed 2026-05-14** |

### Frozen artifacts (6 of 6 winner JSONs)

| `(model, dataset)` | Winner JSON | best_value | Notes |
|---|---|---|---|
| `resnet50 × cifar10` | `artifacts/best_hparams/resnet50_cifar10.json` | 0.5624 | trial #2 (fast-rank-3 → full-rank-1 swap at Stage 1.5) |
| `resnet50 × mnist` | `artifacts/best_hparams/resnet50_mnist.json` | 0.9190 | trial #2 |
| `densenet121 × cifar10` | `artifacts/best_hparams/densenet121_cifar10.json` | 0.6024 | trial #10 (fast-rank-3 → full-rank-1 swap) |
| `densenet121 × mnist` | `artifacts/best_hparams/densenet121_mnist.json` | 0.9192 | trial #2 (no re-ranking — trial #2 won at both fast and full) |
| `transnext_tiny × cifar10` | `artifacts/best_hparams/transnext_tiny_cifar10.json` | **0.7336** | trial #16 (fast-rank-2 → full-rank-1 swap at Stage 1.5) — **highest cifar10 winner across the 3 models** |
| `transnext_tiny × mnist` | `artifacts/best_hparams/transnext_tiny_mnist.json` | 0.9176 | trial #12 (no re-ranking — fast and full both rank #12 first) |

### Final Results — 276 / 276 cells (Phase D added 2026-05-26)

All cells reported below run under **pipeline v2** ([`src.data.degradation_levels.PIPELINE_VERSION`](src/data/degradation_levels.py) `= 2`) — additive Gaussian noise and salt-and-pepper are drawn pre-upsample at `low_res` and then bicubically spread to 224 (sensor-realistic coarse-grain structure). v1 numbers (pre-noise_fix, post-upsample noise) survive only in git history and are not directly comparable on the 90 noise / salt_pepper / Phase B cells re-run during the 2026-05-21 → 2026-05-23 v2 campaign. The remaining 96 cells (Phase A clean + Phase C resolution / blur / saturation) are byte-identical to v1.

Comprehensive long-form report (executive summary + methodology + per-phase tables + 5×5 heatmaps + 3 cross-architecture findings + 5 research conclusions): [`docs/Final_Exp_Report.md`](docs/Final_Exp_Report.md) → [`artifacts/Final_Exp.pdf`](artifacts/Final_Exp.pdf).

#### Phase A — Clean Baselines (6 / 6 cells)

Per-model upper bound at 224×224 under the convergence-first protocol (60 ep / patience 10 / bf16-mixed / AdamW + cosine). All cells `healthy` per the §6.3 pathology guard.

| Model | CIFAR-10 best_val_acc | MNIST best_val_acc | Story |
|---|---:|---:|---|
| `resnet50` | 0.9518 | 0.9914 | US-006 |
| `densenet121` | 0.9356 | **0.9930** | US-007 |
| `transnext_tiny` | **0.9764** | 0.9924 | US-008 |

#### Phase B — Combined Degradation (30 / 30 cells, best_val_acc L1 → L5, pipeline v2)

All five axes active at the same severity per cell. v2 numbers (re-run 2026-05-21 under `PIPELINE_VERSION = 2`); mean Δ across the 30 cells is **−12.49 pp** vs v1, with mid-range levels (L2-L4) dropping hardest. Strictly monotonic across all three architectures on both datasets — no inversions. Bold = best of three architectures per `(dataset, level)` column.

| Model | Dataset | L1 | L2 | L3 | L4 | L5 |
|---|---|---:|---:|---:|---:|---:|
| `resnet50` | cifar10 | 0.7646 | 0.5394 | 0.3586 | 0.2684 | 0.1906 |
| `resnet50` | mnist | **0.9918** | 0.9636 | 0.7886 | 0.5644 | **0.2800** |
| `densenet121` | cifar10 | 0.7848 | 0.5368 | **0.3960** | **0.2852** | 0.1958 |
| `densenet121` | mnist | 0.9874 | **0.9654** | **0.8196** | **0.5808** | 0.2776 |
| `transnext_tiny` | cifar10 | **0.8758** | **0.6422** | 0.3900 | 0.2850 | **0.2010** |
| `transnext_tiny` | mnist | 0.9896 | 0.9600 | 0.7994 | 0.5724 | 0.2748 |

#### Phase C — Single-Axis Isolation Headline (150 / 150 cells, pipeline v2 on noise + salt_pepper)

Only the named axis at level L; the other four axes at identity. Cross-architecture best_val_acc **at L5** (the campaign's worst-case isolation cells). v2 numbers on the `noise` and `salt_pepper` rows (re-run 2026-05-22 / 2026-05-23 under `PIPELINE_VERSION = 2`); `resolution`, `blur`, and `saturation` rows are byte-identical to v1 because those axes were unaffected by US-017. Bold = best of three architectures per `(dataset, axis)` cell.

| Axis @ L5 | resnet50 cifar10 | densenet121 cifar10 | transnext_tiny cifar10 | resnet50 mnist | densenet121 mnist | transnext_tiny mnist |
|---|---:|---:|---:|---:|---:|---:|
| `resolution` | 0.4316 | 0.4432 | **0.4624** | 0.4426 | **0.4540** | 0.4486 |
| `noise` (v2) | 0.6688 | 0.6846 | **0.8008** | 0.9896 | **0.9928** | 0.9914 |
| `blur` | 0.7098 | 0.7142 | **0.8244** | 0.9882 | 0.9846 | **0.9906** |
| `saturation` | 0.8822 | 0.8834 | **0.9534** | 0.9910 | **0.9914** | 0.9904 |
| `salt_pepper` (v2) | 0.8330 | 0.8508 | **0.9422** | 0.9862 | 0.9872 | **0.9894** |

Pipeline-v2 severity ordering on CIFAR-10 at L5: `resolution` (~0.43–0.46) > `noise` (0.67–0.80) > `blur` (0.71–0.82) > `salt_pepper` (0.83–0.94) > `saturation` (0.88–0.95). Noise overtakes blur as the second-worst CIFAR-10 axis under v2; the v1 `noise` row hid this because post-upsample noise was effectively averaged out by the bicubic kernel. MNIST cells on all four perturbation axes stay within the geometric-axis floor (≥ 0.98 at L5) — only `resolution` bites on MNIST. Full per-axis × per-level 5×5 heatmaps per (model, dataset) and the complete 150-row Phase C table live in [`docs/Final_Exp_Report.md`](docs/Final_Exp_Report.md).

#### Phase D — Regularization Recovery Sweep (90 / 90 cells, closed 2026-05-26)

Phase D layers three orthogonal regularization recipes — **T1** architectural dropout/drop-path, **T2** mixup-only label-mixing, **T3** the kitchen-sink combo (T1 ∪ T2 ∪ cutmix α=1.0) — atop the **frozen v2 L3-Optuna winners**, with no re-tune. 3 treatments × 5 levels × 3 models × 2 datasets = **90 cells**; every cell is compared against its Phase B v2 sibling under the SHA-256-locked baseline manifest at [`artifacts/validation/phase_b_v2_baseline_manifest.json`](artifacts/validation/phase_b_v2_baseline_manifest.json). The pixel pipeline is byte-identical to Phase B (`pipeline_version = 2`, `degradation_levels_hash` unchanged); the only axis of variation is the regularization delta layered on top of the hparams blob. Rationale + treatment routing: [`docs/phase_d.md`](docs/phase_d.md).

**L3-anchored recovery table** (auto-derived; canonical LaTeX at [`docs/_autogen/phase_d_recovery_table.tex`](docs/_autogen/phase_d_recovery_table.tex)):

| Model | Dataset | ΔT1@L3 (pp) | ΔT2@L3 (pp) | ΔT3@L3 (pp) |
|---|---|---:|---:|---:|
| `resnet50` | cifar10 | −0.68 | +1.64 | +0.84 |
| `resnet50` | mnist | +0.84 | −0.28 | +0.94 |
| `densenet121` | cifar10 | −0.28 | −0.44 | −0.36 |
| `densenet121` | mnist | −1.22 | −2.12 | −0.24 |
| `transnext_tiny` | cifar10 | **+3.36** | **+3.34** | **+2.98** |
| `transnext_tiny` | mnist | +0.28 | +0.48 | +1.42 |

**Mean Δ across the 30 cells per treatment:** T1 = **+0.17 pp** (median +0.09), T2 = **+0.31 pp** (median +0.18), T3 = **+0.77 pp** (median +0.52) — recovery is monotonic in regularization scope (T3 > T2 > T1). T3 splits +1.15 pp on CIFAR-10 (n=15) vs +0.40 pp on MNIST (n=15); the strongest segment is TransNeXt × CIFAR-10 at **+2.20 pp mean** (T3, across 5 levels), consistent with the v2 Finding #2 attention-softens hypothesis. The CIFAR-10 L5 cells under T3 still average only +0.27 pp — the 3×3-downsample bottleneck (Finding #1) is fundamental information loss that regularization cannot rescue.

**Interpretation.** Regularization recovers a small fraction of the v2 collapse (mean Δ = +0.77 pp under T3 vs the −12.49 pp v2 drop). The collapse is largely a fundamental information bottleneck, not an overfit signal. TransNeXt benefits the most from added regularization — adding mixup + cutmix to attention gives it more recovery than the CNNs — but no treatment closes the v2 gap.

### Headline Research Findings

1. **Universal 3 × 3-downsample bottleneck.** At L5 resolution every architecture lands within ±3 pp on CIFAR-10 (0.4316 / 0.4432 / 0.4624) and ±1 pp on MNIST (0.4426 / 0.4540 / 0.4486). The pretrain receptive-field hierarchy fails uniformly when sub-class geometric structure is destroyed, regardless of whether the backbone is convolutional or attention-based. This is the rate-limiter for the Phase B combined-axes L5 collapse.

2. **TransNeXt softens but does not reverse the v2 CIFAR-10 perturbation collapse** *(re-derived from v2 data 2026-05-23; supersedes the v1 "holds ≥ 0.95" framing).* Under pipeline v2, TransNeXt-tiny retains a single-digit-to-low-double-digit robustness gap over both CNN backbones on every CIFAR-10 perturbation axis at L5 (+13.2 pp on noise, +11.2 pp on blur, +7.1 pp on saturation, +10.0 pp on salt-and-pepper), but the underlying absolute floor is much lower than the v1 noise/S&P numbers suggested: tnx cifar10 noise drops to **0.8008** at L5 (v1 reported 0.9608 under the post-upsample bug) and salt_pepper drops to **0.9422** (v1 reported 0.9602). The v1 "≥ 0.95 across L1→L5" claim now survives only on saturation (axis unchanged by US-017 — `0.9792 → 0.9534` across L1→L5) and partially on salt_pepper (holds at L1-L3, falls below 0.95 at L4-L5). The gap disappears at L5 resolution (all three architectures ≈ 0.46 on CIFAR-10). Read-out: attention's adaptive receptive fields *soften* the coarse-grain perturbation collapse without preventing it — the texture-statistic destruction is too severe for TransNeXt-tiny to fully compensate at low_res = 3.

3. **MNIST geometric-axis robustness floor preserved under pipeline v2.** Across all three architectures the MNIST val_acc on noise / blur / saturation / salt-and-pepper stays flat at **0.984–0.993** across L1 → L5 — including the v2-hardened noise axis (where CIFAR-10 collapses by up to 22.9 pp) and the v2-hardened salt-and-pepper axis (CIFAR-10 drops up to 4.4 pp). The MNIST L5 floor across these four axes under v2 is `noise 0.9896/0.9928/0.9914`, `blur 0.9882/0.9846/0.9906`, `saturation 0.9910/0.9914/0.9904`, `salt_pepper 0.9862/0.9872/0.9894` (r50 / d121 / tnx). Only the resolution axis bites on MNIST (and only at L4-L5). This is now a 60-cell, two-axis empirical confirmation that the THz-like coarse-grain perturbations preserve digit-shape signal in MNIST regardless of severity — the thick digit strokes cover many bicubically-spread perturbation footprints, so the discriminative feature survives. Texture-dominant recognition (CIFAR-10) and geometry-dominant recognition (MNIST) diverge sharply under pipeline v2 in a way the v1 pipeline largely masked.

4. **Regularization partially softens but does not reverse the v2 collapse — confirming the information-bottleneck interpretation.** Across the 90-cell Phase D sweep, mean Δ vs the Phase B v2 baseline is **+0.17 pp** under T1 (architectural dropout / drop-path), **+0.31 pp** under T2 (mixup α=0.2), and **+0.77 pp** under T3 (T1 ∪ T2 ∪ cutmix α=1.0) — monotonic in regularization scope. The per-architecture recovery confirms the v2 attention-softens hypothesis from Finding #2: TransNeXt × CIFAR-10 reaches **+2.20 pp** mean recovery under T3 (across 5 levels), the strongest segment in the sweep, while DenseNet121 × CIFAR-10 sits at just +0.22 pp mean. The L5 ceiling holds across treatments — T3 on CIFAR-10 averages only +0.27 pp at L5, so the 3×3-downsample information bottleneck (Finding #1) remains unrescuable. MNIST T3 averages just +0.40 pp across 15 cells, consistent with Finding #3 — the perturbation-axis floor is already saturated. Operator-locked answer to the §1 research question: **Regularization recovers a small fraction of the v2 collapse (mean Δ = +0.77 pp under T3 vs the −12.49 pp v2 drop). The collapse is largely a fundamental information bottleneck, not an overfit signal.**

### Recent scientific changes

- **2026-05-26 — Phase D regularization sweep closed (90 / 90 cells healthy).** Three orthogonal regularization treatments (T1 architectural dropout/drop-path, T2 mixup α=0.2, T3 the T1 ∪ T2 ∪ cutmix α=1.0 combo) layered atop the frozen v2 L3-Optuna winners — no re-tune, pixel pipeline byte-identical to Phase B (`pipeline_version = 2`, `degradation_levels_hash` unchanged). 90 new cells dispatched against the SHA-256-locked baseline at [`artifacts/validation/phase_b_v2_baseline_manifest.json`](artifacts/validation/phase_b_v2_baseline_manifest.json), run in **56.5 GPU-h** (7.6% over the 52.5 GPU-h budget — acceptable) on the RTX 5070 with **zero pathology-guard sentinels**. Headline scientific outcome: mean Δ vs Phase B v2 = **+0.17 pp (T1) / +0.31 pp (T2) / +0.77 pp (T3)** across all 30 cells per treatment — recovery is monotonic in regularization scope and peaks at **+2.20 pp** on TransNeXt × CIFAR-10 under T3 (mean across 5 levels). The L5 ceiling holds across treatments (CIFAR-10 T3 L5 averages +0.27 pp — the 3×3-downsample bottleneck from Finding #1 is unrescuable). US-026 through US-035 closed end-to-end (PRD v3 → docs/phase_d.md → pilot HARD-GATE → 84-cell sweep → JSON aggregator US-029.5 → dashboard Phase D tab US-030 → recovery plots + LaTeX table US-031 → content sync US-032 → IEEEtran §VII appendix US-033 → NotebookLM 2-pass sync US-034 → ship-readiness audit US-035). Total cells now: **276** (186 v2 + 90 Phase D), all carry `pipeline_version = 2`. Source-of-truth treatments + baseline manifest: [PRD.md §12 (v3 decisions locked)](PRD.md).
- **2026-05-23 — Pipeline v2 noise-fix campaign closed (`PIPELINE_VERSION = 2`).** US-017 moved additive Gaussian noise and salt-and-pepper *before* the bicubic upsample to 224 — the v1 implementation had drawn both at 224×224, so a single noise sample landed on one output pixel and the perceived perturbation at the underlying `low_res` scale was a tiny fraction of the configured `noise_std`. Under v2 the perturbation is drawn at `low_res` and bicubically spread to 224, producing the sensor-realistic coarse-grain structure a true low-resolution detector would record (lag-1 spatial autocorrelation 0.993 at `low_res=8`, vs ~0 in v1). All 90 affected cells (30 Phase B + 30 Phase C noise + 30 Phase C salt_pepper) were re-run 2026-05-21..23 on the RTX 5070 (~50 GPU-h total, under the 53 GPU-h PRD budget). The 96 unaffected cells (Phase A clean + Phase C resolution / blur / saturation) are byte-identical to v1 by `degradation_levels_hash`. **v1 ↔ v2 are not directly comparable** on the 90 re-run cells: the level *values* are unchanged but the order of operations differs. v1 numbers survive only in git history (commit `3e1d089`). Per-cell `metrics.json` now carries `pipeline_version = 2`. Headline scientific shift: under v2 the CIFAR-10 L5 severity ordering becomes `resolution > noise > blur > salt_pepper > saturation` (noise leapfrogs blur — masked in v1 because post-upsample noise was averaged out by the bicubic kernel), and the new 60-cell, two-axis result establishes that coarse-grain stochastic perturbations selectively destroy texture-dependent CIFAR-10 recognition while preserving geometry-dominant MNIST recognition. Phase B mean Δ across the 30 v2 cells is **−12.49 pp** vs v1; Phase C noise mean Δ is **−5.15 pp**; Phase C salt_pepper mean Δ is **−0.81 pp** (severity proportional to perturbed-pixel-count arithmetic: noise samples every `low_res²` pixel, salt_pepper paints only `low_res² × (salt_pepper × 2)` pixels per image at L5 — much fewer at low_res=3). Optuna winners (frozen at v1 L3 Moderate) carry forward unchanged; re-tuning under v2 is deferred to a future US.
- **2026-05-20 — v1 Campaign closed.** All 186 cells complete under the v1 pipeline; comprehensive research report rendered to [`artifacts/Final_Exp.pdf`](artifacts/Final_Exp.pdf) (243 KB) from a new long-form Markdown source [`docs/Final_Exp_Report.md`](docs/Final_Exp_Report.md). Final §6.4 pathology-guard retry tally: 13 cells flagged across the v1 campaign, all 13 declined by operator (5-for-5 across resnet50/densenet121/transnext_tiny Phase B/C dispatches with sentinels). All declined cells were already monotonic with their surrounding L-curve; the §6.4 deltas (`wd × 2` + `dropout = 0.1` + `backbone_lr ÷ 20`) consistently underfit tightly-tuned Optuna winners and improved zero cells across the entire v1 campaign. v1 numbers on the 90 noise/S&P cells were superseded by v2 on 2026-05-23.
- **2026-05-14 — US-003 (Phase C identity).** Inactive axes in Phase C cells now return to **identity** (no degradation) instead of L1-mild values. Pre-US-003 a "blur at L5" cell was contaminated by L1 noise + S&P + resolution + saturation; post-US-003 each Phase C cell isolates exactly one axis. Rationale + identity-value table: [`docs/phase_c.md`](docs/phase_c.md).
- **2026-05-14 — Blur kernel/σ rescale (operator-approved).** The pre-2026-05-14 blur values (K=3..13, σ=0.80..2.30) were native-size kernels invisible after the upsample to 224×224. Rescaled to K=13..91, σ=2.5..18 to produce perceptually meaningful blur on 224×224. See the L1..L5 table below ("Degradation Pipeline & Levels") for the current values. `degradation_levels_hash` rotates; cross-batch comparison against pre-2026-05-14 runs involving blur is invalidated. The 4 already-frozen winner JSONs were tuned at old-L3 blur; operator accepted the residual mismatch (re-tune deferred unless Phase B L3 shows systematic underperformance).
- **2026-05-13 — TransNeXt 224×224 un-quarantine (US-042).** The earlier native-resolution refactor (TransNeXt at 32, MNIST pad-to-32) was rolled back. Every model trains at 224×224 with the same data pipeline; TransNeXt loads its upstream 224-pretrained checkpoint and trains under the same differential-LR full-FT regime as the CNNs.

## Hardware & OS Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| GPU | 8 GB VRAM, CUDA 11.8+ | 24 GB (RTX 3090 / A100) |
| CPU | 8 cores | 16+ cores |
| RAM | 16 GB | 32 GB |
| Disk | 100 GB free | 250 GB SSD |
| OS | Windows 10/11, Ubuntu 20.04+, macOS 13+ | — |
| Python | 3.10+ | 3.12 |
| CUDA toolkit | 11.8 or 12.x | matching `torch` build |

## Hardware Setup — RTX 5070 (Blackwell, 12 GB) — migration target 2026-05-08

The Phase B sweep was paused on an RTX 4050 Laptop (6 GiB VRAM, ≈5.997 GiB usable, sm_89 Ada) after Stage 1 fast tune completed; the campaign will resume on an RTX 5070 12 GB (Blackwell, sm_120). This section captures the hardware-specific knobs.

| Requirement | RTX 5070 setting | Why |
|---|---|---|
| GPU architecture | Blackwell, compute capability `sm_120` | RTX 50-series consumer GPU. PyTorch 2.11.0+cu128 (current pin in [`requirements.lock.txt`](requirements.lock.txt)) ships kernels through `sm_120`. |
| NVIDIA driver | **NVIDIA Studio Driver** (recommended) or current Game Ready Driver, version ≥ 580.xx | Studio Drivers are stability-validated for ML / creative workloads (less frequent releases, fewer regressions). Either works; Studio is the default for unattended overnight training. |
| CUDA runtime | **CUDA 12.x** — specifically **CUDA 12.8** (cu128 wheels) | [`scripts/setup_gpu_env.py:KNOWN_CUDA_INDICES`](scripts/setup_gpu_env.py) tops at `((12, 8), "cu128")`. The "nearest ≤ system" heuristic picks `cu128` for any driver reporting CUDA 12.8 / 12.9 / 13.x runtime, which is the right wheel for Blackwell. |
| VRAM | 12 GiB usable — ResNet50 / DenseNet121 batch 32 mixed-precision fit comfortably with ~6 GiB headroom | Use the default `--min-vram-gib 6.0` (no `--min-vram-gib 5.9` workaround needed on the new box). |
| Mixed precision | `precision="16-mixed"` (already wired) | Blackwell's BF16 / FP16 tensor cores are well-utilised; AMP is enabled by default in [`src/lightning/`](src/lightning/) when CUDA is available. |
| Driver-bundled CUDA | Verify with `nvidia-smi` — should report driver ≥ 580 and CUDA runtime ≥ 12.8 | If the runtime row shows `13.x`, the cu128 wheel is still the right pick — Blackwell's Driver-API is forward-compatible. |
| Power & sleep | AC adapter plugged in, sleep disabled (`powercfg /change standby-timeout-ac 0`) for the multi-hour Stage 1.5 + Stage 2 chains | The whole tune-validate-sweep arc is ~25 GPU-hours; uninterrupted runs are far cheaper than restarts. |
| `setup_gpu_env.py --make-venv` | Manually use `py -3.12 -m venv .venv-gpu` if the new box only has Python 3.12 | The bundled `make_venv` only iterates `RECOMMENDED_PY_MINORS = (11, 10)` even though `MAX_SUPPORTED_PY_MINOR = 12`. See [docs/runbooks/PHASE_B_EXECUTION.md](docs/runbooks/PHASE_B_EXECUTION.md) → Known issues #1. |

**One-line bootstrap on the new box (assumes Python 3.10/3.11/3.12 already installed + Studio Driver + nvidia-smi working):**

```powershell
git clone <repo-url>
Set-Location Object-Classification-in-Low-Resolution-THz-Imagery--
python scripts\setup_gpu_env.py             # full audit + cu128 install + smoke test
.venv-gpu\Scripts\python.exe tune_all.py --validate-only
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

If the smoke test passes, you are ready for Stage 1.5 (`scripts\run_tune_chain.ps1` skips already-finished pairs, then runs `validate_top3.py`). The full handoff runbook is in [docs/runbooks/PHASE_B_EXECUTION.md](docs/runbooks/PHASE_B_EXECUTION.md).

## Step 0 — Environment Setup

**Bash (Linux / macOS / git-bash):**
```bash
git clone <repo-url>
cd Object-Classification-in-Low-Resolution-THz-Imagery--
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m src.tests.test_degradation_determinism   # MSE=0 sanity check
```

**PowerShell (Windows):**
```powershell
git clone <repo-url>
Set-Location Object-Classification-in-Low-Resolution-THz-Imagery--
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
.\venv\Scripts\python.exe -m src.tests.test_degradation_determinism
```

If the determinism test prints `OK [cifar10] ...`, `OK [mnist] ...`, `OK [lightning] ...` you are ready. The first run will download CIFAR-10 / MNIST into `./data/` (≈ 200 MB combined).

## Step 1 — Dataset Setup

CIFAR-10 and MNIST are downloaded automatically into `./data/` on first use by torchvision. To pre-warm the cache without starting training:

**Bash:**
```bash
python -c "from torchvision import datasets; datasets.CIFAR10('./data', train=True, download=True); datasets.MNIST('./data', train=True, download=True)"
```

**PowerShell:**
```powershell
.\venv\Scripts\python.exe -c "from torchvision import datasets; datasets.CIFAR10('./data', train=True, download=True); datasets.MNIST('./data', train=True, download=True)"
```

TransNeXt-Tiny ImageNet-1K weights auto-download to `artifacts/weights/transnext_tiny_224_1k.pth` on first training run. To pre-fetch them upfront (recommended — surfaces network problems before Phase A starts and avoids latency on the first epoch):

**Bash:**
```bash
python scripts/fetch_transnext_weights.py --sizes tiny     # campaign default (post-US-016)
python scripts/fetch_transnext_weights.py --all            # every size
```

**PowerShell:**
```powershell
.\venv\Scripts\python.exe scripts\fetch_transnext_weights.py --sizes tiny
.\venv\Scripts\python.exe scripts\fetch_transnext_weights.py --all
```

Files smaller than 1 MiB (Git-LFS pointers, partial downloads, accidental placeholders) are detected and re-downloaded automatically. Override the source URL via the `THZ_TRANSNEXT_TINY_URL` env var (or `THZ_TRANSNEXT_<SIZE>_URL` for any size) if the default GitHub release URL is unreachable; if both fail the training script raises `FileNotFoundError` with a manual-download instruction.

## Step 2 — Optuna Pre-Tuning (~20 GPU-hours)

Tune hyperparameters on Phase B L3 Moderate for each `(model, dataset)` pair, then freeze them for the 186-cell sweep. Search space is **anchored on paper-derived priors** (`artifacts/priors/{resnet50,densenet121,transnext_tiny}.json`) — every range is held to ≤ 1 decade around the paper anchor.

**Bash:**
```bash
python tune_all.py --validate-only          # dry-run: validates priors files
python tune_all.py --n-trials 20            # 6 pairs × 20 trials
```

**PowerShell:**
```powershell
.\venv\Scripts\python.exe tune_all.py --validate-only
.\venv\Scripts\python.exe tune_all.py --n-trials 20
```

Output: `artifacts/best_hparams/{model}_{dataset}.json` per pair. Each winner JSON carries `priors_file_hash` (SHA-256 of the priors used) so you can audit which search bounds produced the result. Resume on crash is automatic via the `artifacts/optuna_thz.db` SQLite store.

To tune just one pair:

**Bash:**
```bash
python tune_all.py --n-trials 20 --model resnet50 --dataset cifar10
```

**PowerShell:**
```powershell
.\venv\Scripts\python.exe tune_all.py --n-trials 20 --model resnet50 --dataset cifar10
```

## Step 3 — Run the 186-cell Matrix

**Bash:**
```bash
# Phase A — clean baselines (6 runs)
python run_all_phases.py --plan final --phase A

# Phase B — combined degradation (30 runs)
python run_all_phases.py --plan final --phase B

# Phase C — single-axis isolation (150 runs)
python run_all_phases.py --plan final --phase C

# Full pipeline (idempotent — safe to Ctrl-C and resume; auto-tunes if any best_hparams missing)
python run_all_phases.py --plan final --phase all --skip-existing --tune-first
```

**PowerShell:**
```powershell
.\venv\Scripts\python.exe run_all_phases.py --plan final --phase A
.\venv\Scripts\python.exe run_all_phases.py --plan final --phase B
.\venv\Scripts\python.exe run_all_phases.py --plan final --phase C
.\venv\Scripts\python.exe run_all_phases.py --plan final --phase all --skip-existing --tune-first
```

To run a single cell ad-hoc:

**Bash:**
```bash
python run_systematic.py --cell-tag final_B_L3_resnet50_cifar10
```

**PowerShell:**
```powershell
.\venv\Scripts\python.exe run_systematic.py --cell-tag final_B_L3_resnet50_cifar10
```

Outputs land in `runs/final/<tag>/`. Each run writes:
- `metrics.json` — `best_val_acc`, `best_epoch`, `last_*_acc/loss`, `epochs_run`, plus `wandb_run_id` / `wandb_entity` / `wandb_project` (US-012, may be `null` offline), `hparams` + `hparams_source` (US-008), and `cell_tag` / `phase` / `level` / `axis` / `model` / `dataset`.
- `metrics.csv` — per-epoch loss/accuracy
- `best.pt`, `model_last.pt` — Lightning checkpoints (gitignored, claudeignored, **never leave the local machine**)
- `log.txt` — training log

## Step 4 — Inspect Results

| Surface | What it shows |
|---|---|
| `artifacts/Final_Exp.html` | FINAL_EXP Dashboard — pilot-styled, tabbed by Phase A/B/C, with multi-select chip filters (model / dataset / status / axis), graded `L1`–`L5` level badges with parameter tooltips, status pills, `val_acc` / epochs / runtime per row, and 30 s JSON polling for live updates. Initial data is embedded inline so the dashboard works directly via `file://`; polling activates when served over HTTP. |
| `artifacts/Final_Exp.json` | Aggregate of all 186 cells in a stable schema (`src/tools/final_exp_schema.py`) — the data contract the dashboard polls and the source of truth for downstream tools. |
| [`Final_Exp.md`](Final_Exp.md) | Master tracker — Markdown table per cell + `metrics.json` schema reference. |
| `runs/final/<tag>/log.txt` | Live training output. `Get-Content -Wait` (PowerShell) or `tail -f` (Bash). |
| `artifacts/priors/*.json` | Paper-anchored Optuna search bounds (tracked in git). |
| `artifacts/best_hparams/*.json` | Optuna winner per `(model, dataset)` — locally ignored (machine-specific). |

Refresh / regenerate after a run:

**Bash:**
```bash
python scripts/update_final_exp.py                    # regenerate Final_Exp.md row data
python -m src.tools.render_cell_thumbs                # 186 Original|Degraded PNG pairs (idempotent)
python -m src.tools.render_curve_thumbs               # per-cell val_acc / val_loss curves (skips cells with no metrics.csv)
python -m src.tools.build_final_dashboard             # rebuild Final_Exp.html
python -m src.tools.measure_image_quality \
    --cell-tag final_B_L3_resnet50_cifar10 \
    --out runs/final/final_B_L3_resnet50_cifar10/image_quality.json
```

**PowerShell:**
```powershell
.\venv\Scripts\python.exe scripts\update_final_exp.py
.\venv\Scripts\python.exe -m src.tools.render_cell_thumbs
.\venv\Scripts\python.exe -m src.tools.render_curve_thumbs
.\venv\Scripts\python.exe -m src.tools.build_final_dashboard
.\venv\Scripts\python.exe -m src.tools.measure_image_quality `
    --cell-tag final_B_L3_resnet50_cifar10 `
    --out runs/final/final_B_L3_resnet50_cifar10/image_quality.json
```

`scripts/update_final_exp.py --check` exits non-zero if `Final_Exp.md` is out of date — useful for CI / pre-commit hooks.

## Degradation Pipeline & Levels

Identical pipeline for both datasets (saturation applied **before** noise/S&P so additive noise stays color-correct):

```
Original (32×32 CIFAR-10 / 28×28 MNIST → 3-channel)
   ↓ Saturation lerp:  (1−s)·gray + s·img            ← deterministic
   ↓ Downsample → bilinear upsample to 224×224
   ↓ Gaussian blur (separable conv)
   ↓ Additive Gaussian noise
   ↓ Salt-and-pepper noise
   ↓ ImageNet normalization
Model input (224×224×3)
```

5-level degradation table (single source of truth: [`src/data/degradation_levels.py`](src/data/degradation_levels.py)):

| Level | Name | low_res | blur kernel | blur σ | noise std | S&P | saturation |
|---|---|---|---|---|---|---|---|
| L1 | Mild | 18 | 13 | 2.50 | 0.04 | 0.03 | 0.95 |
| L2 | Light | 12 | 25 | 5.00 | 0.08 | 0.06 | 0.65 |
| L3 | Moderate | 8 | 41 | 8.00 | 0.12 | 0.10 | 0.40 |
| L4 | Severe | 6 | 61 | 12.00 | 0.16 | 0.14 | 0.15 |
| L5 | Extreme | 3 | 91 | 18.00 | 0.22 | 0.18 | 0.00 |

Phase C (single-axis isolation) sweeps one axis through L1→L5 with every other axis at **identity** (no degradation). This isolates each axis cleanly so Phase C results reflect the named axis only. Pre-US-003 the inactive axes were pinned at L1 mild values — the change was ratified 2026-05-14; see `docs/phase_c.md`.

## Models

| Model | Type | Strategy | Paper |
|---|---|---|---|
| **ResNet50** | residual CNN | differential LR fine-tuning (head 1e-3, backbone 1e-4) | TResNet (Ridnik et al., 2020) |
| **DenseNet121** | densely-connected CNN | differential LR fine-tuning | DenseNet (Huang et al., 2017) |
| **TransNeXt-Tiny** (default size for the 186-cell campaign; swapped from `small` by US-016 on 2026-05-14 — predecessor base→small swap was US-004 same day) | aggregated-attention ViT | full fine-tuning (no frozen backbone) — `--transnext_size tiny --transnext_mode ft` | TransNeXt (Shi, 2024) §A.3 |

## The 186-cell Matrix

| Phase | Description | Count |
|---|---|---|
| **A** Clean baselines | 3 models × 2 datasets × 1 (no degradation) | **6** |
| **B** Combined degradation | 3 models × 2 datasets × 5 levels (all axes at L) | **30** |
| **C** Single-axis isolation | 3 models × 2 datasets × 5 levels × 5 axes | **150** |
| | | **186** |

Tag scheme: `final_clean_{model}_{dataset}` / `final_B_L{level}_{model}_{dataset}` / `final_C_L{level}_{axis}_{model}_{dataset}`. At L1 every isolation cell collapses to the Phase B L1 row for the same `(model, dataset)`; `--skip-existing` deduplicates these at runtime.

## Training Defaults — Convergence-First

| Parameter | Value | Source |
|---|---|---|
| Optimizer | AdamW (β₁=0.9, β₂=0.999) | TransNeXt paper |
| Scheduler | Cosine LR decay | TransNeXt / EfficientNetV2 |
| Head LR / Backbone LR (CNNs) | 1e-3 / 1e-4 | TResNet, DenseNet |
| **TransNeXt full FT** | **head 5e-4, backbone 5e-5** (10× differential) | TransNeXt §A.3 + CNN differential-LR convention |
| Weight decay | 1e-4 (CNNs) / 5e-2 (TransNeXt) | DenseNet / TransNeXt |
| Label smoothing | 0.1 (CNNs) / **0.1 (TransNeXt FT)** | TransNeXt paper |
| Drop-path rate | 0.0 (CNNs) / **0.1 (TransNeXt FT)** | TransNeXt paper stochastic depth |
| Gradient clipping | max_norm = 1.0 | TransNeXt paper |
| Batch size | 32 | GPU memory |
| **Max epochs** | **60** | quality-over-speed |
| **Early stopping** | **patience=10, min_delta=1e-4, monitor=val_acc, mode=max** | quality-over-speed |
| Precision | **`bf16-mixed`** (Blackwell sm_120) / `32-true` (CPU) | RTX 5070 |
| Seed | `pl.seed_everything(42, workers=True)` | reproducibility lock |

Pilot mode (`--mode pilot`) keeps shorter numbers (5 epochs / patience 2) for smoke tests **only**. `run_all_phases.py --plan final --mode pilot` is rejected with an assertion to prevent contamination of final results.

## Reproducibility

Per-sample local RNG keyed on dataset index (`seed = idx + SEED_OFFSET_VAL`) gives byte-identical validation pixels across every model and run. Two reads of the same val batch satisfy MSE = 0; gated by [`src/tests/test_degradation_determinism.py`](src/tests/test_degradation_determinism.py). PSNR/SSIM are recomputed deterministically on a fixed 256-sample subset and asserted byte-identical across re-runs.

```bash
pytest src/tests/test_degradation_determinism.py -v
```

## Privacy Notes

Binary weight artifacts never leave the local machine: trained checkpoints (`*.ckpt`, `*.pt`, `*.pth`), the Optuna SQLite store, dashboard thumbs, and pretrained TransNeXt weights under `artifacts/weights/` are excluded from Git via [`.gitignore`](.gitignore) and from AI assistant context via [`.claudeignore`](.claudeignore). The contract is gated by [`scripts/check_ignores.sh`](scripts/check_ignores.sh) and its Python sibling [`src/tests/test_ignores.py`](src/tests/test_ignores.py) — both confirm 13 excluded categories ignore correctly, the 4 paper-anchored priors files stay tracked, and zero binary weight files are in the git index. Future AI sessions analyze runs by reading `metrics.json` / `metrics.csv` / the dashboard HTML — never the binary artifacts.

```bash
bash scripts/check_ignores.sh                          # smoke test the contract
python -m src.tests.test_ignores                       # cross-platform variant
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `import torch` fails | venv not activated | activate `venv/` (note: NOT `.venv/`); see Step 0 |
| `FileNotFoundError: missing best_hparams: artifacts/best_hparams/<m>_<d>.json` | tuning never ran for this `(model, dataset)` | `python tune_all.py --n-trials 20 --model M --dataset D` (or `--tune-first` on `run_all_phases.py`) |
| `FileNotFoundError: Pretrained TransNeXt weights missing: artifacts/weights/transnext_tiny_224_1k.pth` | auto-download URL unreachable | set `THZ_TRANSNEXT_TINY_URL` to a working mirror, or manually drop the file at the printed path |
| OOM on TransNeXt-Tiny | batch 32 too large for available VRAM (unexpected — tiny peaks at ~7 GB on the 12 GB RTX 5070) | `--batch-size 16` (with grad-accum 2 to preserve effective batch 32) or fall back to `--transnext_size micro` |
| `--plan final --mode pilot` rejected with `AssertionError` | guardrail prevents short-training contamination | drop `--mode pilot`, or run pilot via `--plan legacy` |
| Optuna trial pruned | normal pruner behavior | no action — trial state still recorded in `artifacts/optuna_thz.db` |
| Determinism test fails (MSE > 0) | data pipeline mutation broke seed-per-index contract | revert recent changes to `src/data/datasets.py` or `src/data/degrade.py` |
| `git check-ignore` returns wrong category | `.gitignore` regression | run `python -m src.tests.test_ignores` to see the failing rule |

## Codebase Structure

```
Final_Exp.md                       — master tracker, 186 rows
src/data/degradation_levels.py     — single-source 5-level table
src/data/degrade.py                — DegradeConfig + degrade_config_for
src/data/datasets.py               — THzLikeCIFAR10 / THzLikeMNIST
src/lightning/                     — Lightning training engine
src/models/                        — model wrappers (TransNeXt + size selector)
src/tools/                         — dashboard generators
src/tune_hyperparams.py            — Optuna single-pair sweep
src/tests/                         — determinism + saturation invariants
tune_all.py                        — Optuna driver across all 6 pairs (planned)
run_all_phases.py                  — final & legacy phase runner
runs/final/                        — 186-cell campaign outputs (gitignored)
runs/systematic/                   — frozen legacy 33-run results (kept for reference)
artifacts/best_hparams/            — Optuna outputs per (model, dataset)
artifacts/Final_Exp.html           — interactive 186-cell dashboard (planned)
artifacts/weights/                 — auto-downloaded pretrained weights (gitignored)
papers/                            — TransNeXt, DenseNet, TResNet, EfficientNetV2, NASNet
agents/                            — sub-agent specs (MASTER + 9 specialists)
```

## Deadlines

| Date | Deliverable |
|---|---|
| 2026-05-31 | Poster & abstract |
| 2026-06-21 | Final presentation |
| 2026-07-26 | Final submission |

## Contributors

Itamar Bahat

## License

Academic and research use.
