# PRD: Phase B Execution + "Visual Core" Dashboard Upgrade

**Project:** p-2026-061 — Object Classification in Low-Resolution THz Imagery
**Scope:** Phase B (30 CNN runs) of the 186-cell Final Research Phase + dashboard enhancements (Visual Core, lazy learning curves, 10-min polling, granular incremental refresh)
**Phase Mapping:** Phase B (Combined Degradation, 30 cells) — `final_B_L{1..5}_{resnet50|densenet121}_{cifar10|mnist}` only. TransNeXt is **deferred** (62 rows quarantined).
**Output artifacts:** updated `runs/final/`, [artifacts/Final_Exp.html](../../artifacts/Final_Exp.html), [artifacts/Final_Exp.json](../../artifacts/Final_Exp.json), `artifacts/visual_core/`, `artifacts/best_hparams/`, [Final_Exp.md](../../Final_Exp.md)
**Companion PRD:** [docs/prds/FINAL_EXP_DASHBOARD.md](FINAL_EXP_DASHBOARD.md) — US-### namespace below is scoped to **this** PRD and continues the project-level numbering (US-014..US-019).

---

## 1. Context & Phase Mapping

Phase A (CNN) is complete: 4/4 (`resnet50`, `densenet121` × `cifar10`, `mnist`). Two TransNeXt rows are stuck in `Running` with no `metrics.json`; per Step 1 they must be hard-reset and TransNeXt deferred for future high-end GPU clusters.

This PRD addresses two coupled deliverables:

1. **Phase B execution path** for the 30 CNN cells: literature-anchored hyperparameters → Optuna refinement on L3 → frozen sweep across L1–L5.
2. **Dashboard upgrade** so each cell exposes its degradation visually (Original vs. Actually Degraded) and its training trajectory (lazy-loaded learning curves), with 10-minute auto-refresh and per-cell incremental aggregation.

**In scope (cells affected):** rows 7–10, 13–16, 19–22, 25–28, 31–34 (the 20 CNN Phase B cells in the matrix — 5 levels × 2 CNN models × 2 datasets). The other 10 Phase B rows (TransNeXt) are quarantined.

> **Correction:** Phase B has 30 rows total (3 models × 2 datasets × 5 levels). With TransNeXt quarantined, 20 CNN rows are executable in this PRD.

## 2. Goals & Non-Goals

### Goals
- **G1.** Hard-quarantine TransNeXt: zero TransNeXt rows in `Running`/`Complete` states across [Final_Exp.json](../../artifacts/Final_Exp.json), [Final_Exp.md](../../Final_Exp.md), and [Final_Exp.html](../../artifacts/Final_Exp.html).
- **G2.** Literature-anchored HPO priors for `resnet50` and `densenet121` × `cifar10` and `mnist`, frozen as JSON in `artifacts/best_hparams/`.
- **G3.** Optuna hybrid search (paper-prior centers, ±1 decade) on **L3 Moderate** for each (model, dataset) pair; winning hparams freeze the entire 5-level Phase B sweep for that pair.
- **G4.** All 20 CNN Phase B cells `Complete` with `metrics.json` carrying convergence stats and `image_quality.json` carrying PSNR/SSIM.
- **G5.** Dashboard renders for every cell a side-by-side Original vs. Actually Degraded sample (deterministic `idx=0` of val set) using parameters from that cell's row.
- **G6.** Clicking a row opens a Plotly.js learning-curves panel (val_loss + val_acc) lazy-loaded from `runs/final/<tag>/history.json`.
- **G7.** Dashboard polling cadence is **10 minutes (600s)**; aggregator supports `--cell <tag>` for incremental updates.

### Non-Goals
- **NG1.** No TransNeXt training, tuning, or HPO in this PRD. Quarantine only.
- **NG2.** No Phase C (single-axis isolation) execution. Phase C uses the same frozen HPO outputs but is a separate PRD.
- **NG3.** No new training-engine refactors — reuse existing [src/lightning/](../../src/lightning/) and [run_all_phases.py](../../run_all_phases.py).
- **NG4.** No filesystem watcher / daemon (rejected option 5C).
- **NG5.** No public hosting of `artifacts/Final_Exp.html` — local-only.

## 3. Technical Constraints & Guardrails (THz Protocol)

- **Deterministic Integrity:** Per-sample seeded degradation (`seed = idx + SEED_OFFSET_VAL`) gives byte-identical val pixels across all models and runs (MSE = 0). Gated by [src/tests/test_degradation_determinism.py](../../src/tests/test_degradation_determinism.py). Visual-core renderer (US-017) **must** use the same `degrade_config_for(...)` path.
- **Weight Isolation:** No `*.ckpt`, `*.pt`, `*.pth`, or files under `artifacts/weights/` may be read or uploaded. Aggregator and visual-core renderer must not `open()` these paths (assertable via `unittest.mock`).
- **5×5 Matrix:** Phase B uses all 5 axes simultaneously at one of 5 levels (no isolation). Pulled from [src/data/degradation_levels.py](../../src/data/degradation_levels.py) — single source of truth.
- **Metric Synthesis:** Each cell must produce `(best_val_acc, PSNR, SSIM)` so Phase B/C downstream analysis can correlate accuracy ↔ visual quality.
- **Git/Sync Hygiene:** [scripts/refresh_trackers.py](../../scripts/refresh_trackers.py) and [scripts/sync_trackers_git.py](../../scripts/sync_trackers_git.py) remain the only writers of tracker outputs. Visual-core PNGs and `history.json` shards belong to the same sync boundary.
- **Reproducibility:** Every config knob round-trips through `metrics.json`. Optuna `priors_file_hash` already required (US-008).
- **Convergence-First:** Max epochs 60, early stop patience 10, `monitor=val_acc` — no shortcuts. `--mode pilot` is rejected for `--plan final`.

## 4. Implementation Plan (User Stories)

Stories are dependency-ordered: **quarantine → priors → tuning → execution → visual core → curves → polling cadence → incremental refresh**. Each story is one Ralph iteration.

---

### US-014: TransNeXt Hard Quarantine (prerequisite)

**Description:** Stop the two `Running` TransNeXt Phase A rows, delete their on-disk artifacts (and `__v2` siblings), and ensure the 62 TransNeXt rows in [artifacts/Final_Exp.json](../../artifacts/Final_Exp.json) report `Pending` and are excluded from any execution-driving iterators.

**Technical Implementation:**
- New utility `scripts/quarantine_transnext.py`:
  1. `pkill`-equivalent: scan running processes for `--model transnext_base` (use `psutil`); if any, log + raise unless `--force`.
  2. `shutil.rmtree` for: `runs/final/final_clean_transnext_base_cifar10`, `runs/final/final_clean_transnext_base_mnist`, and any sibling `*__v2` dirs under `runs/final/` that match `*transnext*`.
  3. Re-run `scripts/refresh_trackers.py` so [Final_Exp.json](../../artifacts/Final_Exp.json) and [Final_Exp.md](../../Final_Exp.md) reflect `Pending` for all 62 TransNeXt rows.
- Add a guard to [run_all_phases.py](../../run_all_phases.py): if `--plan final` and the model token contains `transnext`, log "QUARANTINED — skipped" and continue. Default Phase B model list narrows to `["resnet50", "densenet121"]`.
- Update [Final_Exp.md](../../Final_Exp.md) header note: "TransNeXt rows deferred — see PHASE_B_VISUAL_CORE.md US-014."

**Acceptance Criteria:**
- [ ] Logic: `python scripts/quarantine_transnext.py --force --dry-run` prints a list of 4 dirs to remove and 62 rows to reset, with no side effects.
- [ ] Logic: post-quarantine, `python -c "import json; d=json.load(open('artifacts/Final_Exp.json')); print(sum(1 for r in d['rows'] if 'transnext' in r['model'] and r.get('status','Pending')!='Pending'))"` outputs `0`.
- [ ] Logic: `run_all_phases.py --plan final --phase B --dry-run` enumerates exactly 20 cells, all CNN.
- [ ] Privacy: script never opens `*.ckpt`/`*.pt`/`*.pth` (asserted by mocking `builtins.open`).
- [ ] Quality: `mypy src scripts` passes; full pytest suite green.
- [ ] Verification: `pytest src/tests/test_quarantine_transnext.py -v` (new); manual `git status runs/final/` shows the 4 dirs removed.

---

### US-015: Scientific HPO Priors from Literature ✅ Complete (2026-05-06)

**Description:** Extract paper-derived hyperparameter priors for `resnet50` and `densenet121` and freeze them as JSON. Every hparam cites a paper + section.

**Addendum — adapted to existing infrastructure (2026-05-06):** The repo already shipped a per-model priors system at [`artifacts/priors/`](../../artifacts/priors/) with [`_schema.json`](../../artifacts/priors/_schema.json) (Draft-07), [`tune_all.py`](../../tune_all.py) (`load_priors`, `validate_priors`), and a 3-model test suite ([`src/tests/test_priors.py`](../../src/tests/test_priors.py)). Per CLAUDE.md "minimal changes, no rewrites of working code", US-015 is implemented as small extensions to that system rather than as a new parallel `artifacts/best_hparams/priors.json` file.

Key decisions captured by the addendum:

- **Per-model, not per-(model, dataset).** CLAUDE.md's hparam table is per-model. CIFAR-10 and MNIST share the recipe; Optuna runs a separate study per (model, dataset) pair on L3 Moderate, but seeded from the same prior. The original PRD plan for 4 distinct priors files is superseded.
- **Decade-width interpretation = ±1 decade *from anchor*** (total ratio `high/low ≤ 100`), matching the existing shipped priors and the wording in CLAUDE.md *"narrowed to paper-derived priors ± 1 decade max"*. The PRD's earlier `high/low ≤ 10` criterion was over-strict and would have rejected every existing prior. New constant `tune_all.MAX_LOGUNIFORM_RATIO = 100.0` enforces the convention; `validate_priors` rejects wider bands.
- **TransNeXt prior preserved but unused.** [`artifacts/priors/transnext_base.json`](../../artifacts/priors/transnext_base.json) stays for future reactivation; it is not consumed by Phase B per US-014 quarantine.

**Implementation:**
- [`tune_all.priors_file_hash(model: str) -> str`](../../tune_all.py) — SHA-256 of the on-disk priors file; round-tripped into `metrics.json.hparams_source.priors_file_hash` per US-008.
- [`tune_all.validate_priors`](../../tune_all.py) extended with the `MAX_LOGUNIFORM_RATIO` check.
- [`artifacts/priors/PRIORS_SOURCES.md`](../../artifacts/priors/PRIORS_SOURCES.md) — paper-quote attribution for all 5 hparams × 3 models, with direct citations to [TResNet](../../papers/TResNet.pdf), [DenseNet](../../papers/Densely%20Connected%20Convolutional%20Networks.pdf), and [TransNeXt](../../papers/TransNeXt.pdf), and an explicit decade-width convention block.
- Tests added to [`src/tests/test_priors.py`](../../src/tests/test_priors.py): `_check_decade_width_invariant`, `_check_priors_file_hash_stability`, `_check_priors_sources_md_present`.

**Acceptance Criteria:**
- [x] Logic: every shipped priors file (resnet50, densenet121, transnext_base) loads and validates via `python tune_all.py --validate-only`.
- [x] Logic: `loguniform` bands satisfy `high / low ≤ MAX_LOGUNIFORM_RATIO = 100` (±1 decade from anchor); over-wide bands are rejected by `validate_priors`.
- [x] Logic: `priors_file_hash(model)` returns a stable hex SHA-256, raises `FileNotFoundError` for unknown models, and produces distinct digests across model files.
- [x] Privacy: [`PRIORS_SOURCES.md`](../../artifacts/priors/PRIORS_SOURCES.md) cites paper paths only — no checkpoint paths, weight URIs, or `*.ckpt`/`*.pt`/`*.pth` references.
- [x] Quality: `python -m src.tests.test_priors` green (8 checks: 3 model-validate + malformed + disk + decade-width + hash + sources-md).
- [x] Verification: `python tune_all.py --validate-only` reports `All 3 priors files valid.`; `python -m src.tests.test_priors` reports 8 OK lines.

---

### US-016: Phase B Optuna Tuning + 20-Run Execution ✅ Wiring complete (2026-05-07) — execution awaits GPU

**Description:** For each of the 4 CNN (model, dataset) pairs, run a 20-trial Optuna study at **L3 Moderate** seeded with US-015 priors. Freeze the winning hparams to `artifacts/best_hparams/{model}_{dataset}.json`. Then execute the 20 Phase B cells using those frozen hparams.

**Addendum — adapted to existing infrastructure (2026-05-07):** The repo already shipped a 4-pair tuner ([tune_all.py](../../tune_all.py) → [src/tune_hyperparams.run_studies](../../src/tune_hyperparams.py)) and a per-cell dispatcher ([run_systematic.run_cell](../../run_systematic.py)) with priors-hash round-trip and best_hparams loader. Per CLAUDE.md "minimal changes, no rewrites of working code", US-016 added the missing wiring to that backbone rather than rewriting it.

Wiring delta vs. pre-US-016 state:

- **Quarantine guard in `tune_all.py`** ([tune_all.py](../../tune_all.py)). Without `--model`, the all-models path filters out `transnext_base` via `is_quarantined`. An explicit `--model transnext_base` bypasses the guard — operator intent is never silently overridden.
- **Image-quality auto-wiring in `run_cell`** ([run_systematic.py](../../run_systematic.py)). New `_measure_image_quality_for_cell(spec, run_dir)` calls [src/tools/measure_image_quality.measure](../../src/tools/measure_image_quality.py) (256-sample subset) and writes `runs/final/<tag>/image_quality.json` after the Lightning training returns. Phase A cells are intentionally skipped (PSNR against an identity pipeline is degenerate). Best-effort: failure logs to stderr but does not abort the runner.
- **INTERRUPTED sentinel** ([run_all_phases.py](../../run_all_phases.py) + [src/experiments/run_status.py](../../src/experiments/run_status.py)). The SIGINT handler drops `runs/final/<tag>/INTERRUPTED` under the currently-running cell, then flushes a tracker refresh before re-raising. `detect_status` checks the sentinel **first**, so a Ctrl-C'd cell renders as `Failed` regardless of any partial metrics.json the trainer wrote. New constant `run_status.INTERRUPTED_SENTINEL = "INTERRUPTED"`.
- **Per-cell refresh** stays as-is — the existing post-`Trainer.fit` `refresh_fn()` call in [run_all_phases.py](../../run_all_phases.py) already covers US-016's "after each Trainer.fit" requirement; US-019 will harden it further with `--cell <tag>` incremental mode.
- **Best_hparams loader is unchanged** — already loads `artifacts/best_hparams/{model}_{dataset}.json` and falls back to `PHASE_A_FROZEN_HPARAMS` for clean baselines (Phase B/C still hard-error on missing file).

**Operator runbook:** [docs/runbooks/PHASE_B_EXECUTION.md](../runbooks/PHASE_B_EXECUTION.md) — pre-flight checks, two-stage execution, verification, failure recovery, "do NOT run" list.

**Tests added:**
- [src/tests/test_quarantine_transnext.py](../../src/tests/test_quarantine_transnext.py) — `test_interrupted_sentinel_marks_cell_failed` (sentinel overrides any metrics.json) and `test_tune_all_skips_transnext_when_iterating_all_models` (guard filters TransNeXt by default; explicit `--model transnext_base` bypasses).

**Acceptance Criteria:**
- [x] **Wiring** — quarantine guard in `tune_all.main`, sentinel writer in `run_final_plan` SIGINT path, sentinel detector in `detect_status`, `_measure_image_quality_for_cell` invocation in `run_cell`. Verified by 9-check `test_quarantine_transnext` suite.
- [x] **Logic — `hparams_source` round-trip** already in place via [run_systematic._merge_metadata_into_metrics_json](../../run_systematic.py); `priors_file_hash` is computed by [src/tune_hyperparams._priors_file_hash](../../src/tune_hyperparams.py) and now also by [tune_all.priors_file_hash](../../tune_all.py) (US-015) — both functions are identical SHA-256 of the priors file bytes.
- [x] **Privacy** — `_measure_image_quality_for_cell` opens datasets only via `THzLikeCIFAR10`/`THzLikeMNIST`; never opens `*.ckpt`/`*.pt`/`*.pth`. The aggregator + sentinel detection use `Path.exists()` only.
- [x] **Quality** — full torch-free pytest suite green: `test_priors` (8/8), `test_quarantine_transnext` (9/9), `test_build_final_exp_json` (10/10), `test_dashboard` (10/10), `test_refresh_trackers` (4/4), `test_update_final_exp` (6/6), `test_phase_a_*` (3 suites green), `test_sync_trackers_git` (green), `test_phase_a_gate` (green).
- [ ] **Execution** — `python tune_all.py --n-trials 20` writes 4 winners to `artifacts/best_hparams/`, then `python run_all_phases.py --plan final --phase B --skip-existing` brings `complete >= 24`. **Awaits GPU box; tracked in [docs/runbooks/PHASE_B_EXECUTION.md](../runbooks/PHASE_B_EXECUTION.md).**
- [ ] **Verification** — spot-check `runs/final/final_B_L3_resnet50_cifar10/metrics.json` after Stage 2: `hparams_source.priors_file_hash` must equal `python -c "import tune_all; print(tune_all.priors_file_hash('resnet50'))"`. **Awaits GPU box.**

---

### US-017: Visual Core — Pre-rendered Original vs. Degraded PNGs ✅ Complete (2026-05-07)

**Description:** For every Phase A/B/C cell, pre-render a side-by-side **Original | Degraded** thumbnail using the cell's exact degradation params and a deterministic sample (`idx=0` of val set, `seed = idx + SEED_OFFSET_VAL`). Dashboard rows display the thumbnail in a new "Visual" column.

**Addendum — adapted to existing infrastructure (2026-05-07):** [`src/tools/render_cell_thumbs.py`](../../src/tools/render_cell_thumbs.py) was already shipping a near-identical Visual Core renderer (single side-by-side PNG to `artifacts/dashboard_thumbs/`, deterministic `idx=0`, `SEED_OFFSET_VAL`-stable). Per CLAUDE.md "minimal changes, no rewrites of working code", US-017 reuses it instead of forking a parallel `src/tools/render_visual_core.py`.

Key decisions captured by the addendum:

- **Single side-by-side PNG**, not two separate `__orig.png` + `__degraded.png` files. The dashboard renders one `<img>` per cell so two files would be churn; the L1-collapse invariant still holds because identical params → identical bytes for the whole composite.
- **Output path:** `artifacts/dashboard_thumbs/<tag>.png` (already gitignored + claudeignored).
- **`visual_core` schema field is a string path**, not an object. Aggregator hydrates from `Path.exists()` only — never opens the PNG. Browser fetches lazily (`<img loading="lazy" decoding="async">`).
- **Soft-fail wiring in `refresh_trackers.refresh_all`**. The new third step `refresh_visual_core_thumbs` lazy-imports torch + torchvision; a torch-free environment yields `thumbs_ok=False` with the import error in `errors`, but MD/HTML still publish.
- **All 186 cells rendered** (including TransNeXt rows). Quarantine is about *not training* the model; the Visual Core preview shows what each cell's pixel pipeline produces and remains useful for the deferred set.

**Implementation:**
- [`src/tools/render_cell_thumbs.py`](../../src/tools/render_cell_thumbs.py) — preserved as-is. Idempotent, `--force`, `--tags`, default skip-existing.
- [`src/tools/build_final_exp_json.py`](../../src/tools/build_final_exp_json.py) — new `_visual_core_for(tag)` helper + `visual_core` field on every row; module constants `_VISUAL_CORE_DIR` and `_VISUAL_CORE_REL` are looked up at call time so tests can monkeypatch.
- [`src/tools/final_exp_schema.py`](../../src/tools/final_exp_schema.py) — `visual_core: str | None` added to `FinalExpRow`.
- [`scripts/refresh_trackers.py`](../../scripts/refresh_trackers.py) — new `refresh_visual_core_thumbs()` wrapper; `refresh_all` now reports `thumbs_ok` alongside `md_ok` / `html_ok`.
- [`src/tools/build_final_dashboard.py`](../../src/tools/build_final_dashboard.py) — new `<th>Visual</th>` column (sits between Level and Status), `visualCoreHtml(row)` JS renderer, `.visual-core-thumb` + `.visual-core-missing` CSS classes (96×48 px lazy `<img>`).

**Tests:**
- [`src/tests/test_build_final_exp_json.py`](../../src/tests/test_build_final_exp_json.py) — `_check_visual_core_field_round_trip` (PNG present → relative path; absent → None) and `_check_visual_core_lookup_does_not_open_png` (no `open()` on any `.png`).
- [`src/tests/test_dashboard.py`](../../src/tests/test_dashboard.py) — `_check_visual_core_column_renders` (column position between Level and Status; `loading="lazy"` set; `visual-core-thumb`/`visual-core-missing` classes emitted; `visualCoreHtml` JS function present); `_check_column_headers_in_order` updated to assert 10-column order.
- [`src/tests/test_refresh_trackers.py`](../../src/tests/test_refresh_trackers.py) — all four checks now assert `thumbs_ok` plumbing (invocation count, failure isolation, no-raise contract, end-to-end stub).
- [`src/tests/test_render_thumbs.py`](../../src/tests/test_render_thumbs.py) — pre-existing torch-gated test covers idempotency, `--force`, byte-stable composite shape; still applies.

**Acceptance Criteria:**
- [x] Logic: aggregator hydrates `visual_core` from disk PNG presence — not from any open/read.
- [x] Logic: dashboard initial paint defers PNG fetches via `loading="lazy"`; missing PNGs render a placeholder, no console error.
- [x] Logic: `refresh_all` reports `thumbs_ok` and routes failures into `errors` without raising.
- [x] Privacy: aggregator never opens `*.png`/`*.ckpt`/`*.pt`/`*.pth` (asserted via `unittest.mock.patch` on `builtins.open` and `io.open`).
- [x] Quality: 5 torch-free test suites green (test_build_final_exp_json, test_dashboard, test_refresh_trackers, test_quarantine_transnext, test_update_final_exp); torch-gated `test_render_thumbs` continues to cover the renderer itself.
- [x] Verification: `python -m src.tools.build_final_exp_json` writes 186 rows with `visual_core` populated for every existing PNG; opening [`artifacts/Final_Exp.html`](../../artifacts/Final_Exp.html) shows a Visual column with thumbnails or placeholders.

---

### US-018: Lazy-loaded Interactive Learning Curves ✅ Complete (2026-05-07)

**Description:** Clicking a row opens a fixed right-side drawer showing per-epoch loss + accuracy curves rendered with Plotly. Data is **lazy-loaded** per cell from `runs/final/<tag>/history.json`; the drawer caches payloads in a JS `Map` so re-clicks are instant.

**Addendum — implementation as shipped (2026-05-07):**

- **Callback lives in [src/lightning/callbacks.py](../../src/lightning/callbacks.py)**, not in a new `history_writer.py`. Keeping it next to `LegacyMetricsCSVCallback` reduces import surface and matches the existing convention. The callback writes **on every `on_validation_epoch_end`** (atomic `tmp + os.replace`), not just on `on_fit_end` — that way a SIGINT'd run still produces a usable curve. NaN/inf values are coerced to JSON `null` so `JSON.parse` succeeds in the browser. Wired into the existing callbacks list in [src/lightning/train.py](../../src/lightning/train.py).
- **Backfill script** [scripts/backfill_history_json.py](../../scripts/backfill_history_json.py) parses the canonical `metrics.csv` schema (`epoch,train_loss,train_acc,val_loss,val_acc`) into the new `history.json` schema. Idempotent (`skip-existing` by default; `--force` to overwrite). NaN cells become JSON `null`. Run after the US-014 quarantine to backfill the 4 existing Phase A CNN cells.
- **`has_history` schema field** ([src/tools/final_exp_schema.py](../../src/tools/final_exp_schema.py)) — set by the aggregator via `Path.exists()` only (never opens the file, preserving the lazy-fetch contract). Surfaced as a 📈 indicator on rows; rows without history show ▫.
- **Plotly is loaded with `defer`** from `cdn.plot.ly/plotly-2.35.2.min.js`. Initial paint does not depend on the CDN. **`renderCurvesFallback`** renders a raw-numbers table when Plotly is missing (offline / ad-blocked / file://-without-network). All four traces (`train_loss`, `val_loss`, `train_acc`, `val_acc`) plot in two stacked panels (Loss + Accuracy).
- **Drawer is a fixed right-side aside** with CSS `transform: translateX(...)` for open/close (no inline display toggling). Esc and the × button both close. Clicks anywhere on a `<tr.exp-row>` open the drawer for that row's tag.
- **Cache + dedupe**: `HISTORY_CACHE` (Map) holds parsed docs; `HISTORY_PENDING` (Map) holds in-flight Promises so a quick double-click doesn't fire two `fetch()` calls. On a row whose `has_history` is false, the drawer renders the placeholder "No learning curves yet — cell is `<status>`." without touching the network.
- **One additional table column** ("Curves") to host the 📈 indicator. The clickable area is the entire row, not just the indicator, so the indicator is purely advisory.

**Tests added/updated:**
- [`src/tests/test_build_final_exp_json.py`](../../src/tests/test_build_final_exp_json.py): `_check_has_history_field_round_trip` (planted `history.json` flips `has_history=True`; absence keeps it `False`).
- [`src/tests/test_dashboard.py`](../../src/tests/test_dashboard.py): `_check_curves_drawer_lazy_loaded` (Plotly `defer`, drawer markup, `openCurvesDrawer` + `HISTORY_CACHE` + `renderCurvesFallback` JS hooks, `init()` block contains no `history.json` reference) and updated `_check_column_headers_in_order` for the new 11-column layout.
- [`src/tests/test_backfill_history_json.py`](../../src/tests/test_backfill_history_json.py): 5 checks covering empty roots, CSV→JSON round-trip with NaN-to-null coercion, idempotency + `--force`, missing-csv handling, and the weight-privacy `unittest.mock.patch` assertion.

**Acceptance Criteria:**
- [x] Logic: `HistoryJSONCallback` writes `runs/final/<tag>/history.json` after every validation epoch with `len(history) ≥ 1`. (Verified by callback design + the backfill round-trip test on 3-row CSVs.)
- [x] Logic: dashboard initial paint does NOT fetch any `history.json` — `init()` block grep-checked clean of the substring; `bindCurvesDrawer` only attaches event listeners.
- [x] Logic: clicking a non-`has_history` row opens the drawer with a placeholder, no `fetch()`. (Verified by JS code inspection in `_check_curves_drawer_lazy_loaded`.)
- [x] Logic: re-clicking the same tag hits `HISTORY_CACHE` (no second fetch). The cache is also populated on a successful fetch via `HISTORY_PENDING` chain.
- [x] Privacy: `history.json` contains only `{schema_version, tag, history: [{epoch, train_loss, val_loss, train_acc, val_acc}]}` — no checkpoint paths, no host paths. Backfill `unittest.mock` assertion confirms `*.ckpt`/`*.pt`/`*.pth` are never opened.
- [x] Quality: 7 torch-free suites green: `test_build_final_exp_json` (14/14), `test_dashboard` (13/13), `test_refresh_trackers` (7/7), `test_quarantine_transnext` (9/9), `test_priors` (8/8), `test_update_final_exp` (6/6), `test_backfill_history_json` (5/5). Plus `test_phase_a_*`, `test_sync_trackers_git`, `test_ignores`.
- [x] Verification: `python scripts/backfill_history_json.py` produces `runs/final/final_clean_resnet50_cifar10/history.json` (and 3 siblings). `python scripts/refresh_trackers.py` writes `has_history=true` for those 4 rows in `Final_Exp.json`. Open [artifacts/Final_Exp.html](../../artifacts/Final_Exp.html), click a Phase A row → drawer opens with both Plotly panels populated.

---

### US-019: Granular Real-Time Polling — `--cell <tag>` Incremental Aggregator + 10-Minute Client Polling ✅ Complete (2026-05-07)

**Description:** Trigger refresh **after each** experiment completion or interruption. Aggregator gains `--cell <tag>` to update only one row in `Final_Exp.json` instead of the full 186-row scan. Dashboard polling cadence drops from 30s to **600s (10 minutes)**.

**Addendum — implementation as shipped (2026-05-07):**

- **`update_cell(tag)` aggregator helper** ([src/tools/build_final_exp_json.py](../../src/tools/build_final_exp_json.py)). Loads the on-disk `Final_Exp.json`, validates the tag against the 186-cell matrix, replaces only that row's hydration, recomputes `counts`, bumps `generated_at`, writes atomically. Falls back to a full rebuild when the file is missing, JSON-corrupt, or schema-mismatched. New CLI flag: `python -m src.tools.build_final_exp_json --cell <tag>`.
- **`refresh_trackers.py` is now also a CLI** ([scripts/refresh_trackers.py](../../scripts/refresh_trackers.py)) with `--cell <tag>`, `--no-md`, `--no-thumbs`. The `refresh_all(cell=..., skip_md=..., skip_thumbs=...)` Python API exposes the same behavior in-process. Incremental mode runs JSON cell-update + HTML rebuild (+ thumbs unless `--no-thumbs`); MD is skipped by default in incremental mode because the dashboard reads JSON, not MD.
- **In-process per-cell refresh in [run_all_phases.py](../../run_all_phases.py)**. After every `Trainer.fit` returns (success OR exception), the runner calls `refresh_fn(cell=spec.tag)` instead of the full `refresh_fn()`. The SIGINT path also passes the active cell so dashboards surface a `Failed` status without a 186-cell rescan. A `TypeError` fallback preserves compatibility with test stubs that don't accept the kwarg. The PRD's "detached subprocess" choice was unnecessary because Lightning has already returned by the time `refresh_fn` runs — the runner is single-threaded and synchronous between cells.
- **Dashboard 600 s cadence** ([src/tools/build_final_dashboard.py](../../src/tools/build_final_dashboard.py)): `const POLL_INTERVAL_MS = 600000`. The visibility-pause stale threshold also moved from 2 min to 30 min so a hidden tab doesn't flag the dashboard as stale just because the next poll is still pending.
- **Countdown pill**: header now reads `Generated: ... · Last polled: ... · Next poll in: M:SS`. The `ts-countdown` element ticks once per second (reusing the existing 1 s `relTimer` interval) and shows `(paused)` when the visibility handler stops the polling timer.
- **Manual "🔄 Refresh now" button** in the header. Calls `loadJsonOnce()` directly — does NOT touch `setInterval`, so an impatient operator can't accidentally create a parallel polling loop. Disables itself for the duration of the fetch and shows `⟳ refreshing…`.

**Tests added/updated:**
- [`src/tests/test_build_final_exp_json.py`](../../src/tests/test_build_final_exp_json.py): `_check_update_cell_patches_only_target_row` (185 sibling rows byte-identical), `_check_update_cell_falls_back_to_full_rebuild` (missing/corrupt/old-schema paths), `_check_update_cell_unknown_tag_raises` (typo guard).
- [`src/tests/test_dashboard.py`](../../src/tests/test_dashboard.py): `_check_polling_cadence_is_10_minutes` (static-grep `POLL_INTERVAL_MS = 600000`, assert `30000` is gone), `_check_manual_refresh_and_countdown_pill_present` (button + countdown + `loadJsonOnce` wired without `setInterval`).
- [`src/tests/test_refresh_trackers.py`](../../src/tests/test_refresh_trackers.py): `_check_incremental_mode_skips_md_and_calls_json_cell` (per-cell path), `_check_skip_thumbs_flag_skips_thumb_render`, `_check_cli_main_routes_cell_argument` (`--cell` / `--no-md` / `--no-thumbs` plumbing).
- [`src/tests/test_quarantine_transnext.py`](../../src/tests/test_quarantine_transnext.py): `test_interrupted_sentinel_marks_cell_failed` (US-016 sentinel detector flips a Ctrl-C'd cell to Failed even with stale metrics — already in place from US-016, still load-bearing here).

**Acceptance Criteria:**
- [x] Logic: `python -m src.tools.build_final_exp_json --cell final_B_L3_resnet50_cifar10` updates only that row; the other 185 rows are byte-identical (verified by zip-and-equal in the unit test).
- [x] Logic: SIGINT writes `runs/final/<tag>/INTERRUPTED` and the incremental refresh marks the row `Failed` (covered by `test_interrupted_sentinel_marks_cell_failed` + the SIGINT handler in `run_all_phases.py`).
- [x] Logic: `POLL_INTERVAL_MS === 600000` — static-grepped by `_check_polling_cadence_is_10_minutes`.
- [x] Logic: countdown pill ticks once per second via `relTimer`; "🔄 Refresh now" forces a fetch without altering the timer.
- [x] Privacy: `update_cell` only opens `Final_Exp.json` and `runs/final/<tag>/metrics.json` (transitively via `read_metrics`); never opens `*.ckpt`/`*.pt`/`*.pth` (existing weight-privacy assertion in `_check_no_weight_path_opened` covers all aggregator paths).
- [x] Quality: 6 torch-free suites green: `test_build_final_exp_json` (13/13), `test_dashboard` (12/12), `test_refresh_trackers` (7/7), `test_quarantine_transnext` (9/9), `test_priors` (8/8), `test_update_final_exp` (6/6). Plus `test_phase_a_*`, `test_sync_trackers_git`, `test_ignores`.
- [x] Verification: `python scripts/refresh_trackers.py --cell <tag>` writes a single-row patch in <1 s on the live JSON; a sample run produced `final_exp_json[--cell=final_clean_resnet50_cifar10]: 186 rows ... -> artifacts\Final_Exp.json` and the 185 untouched rows hashed identically.

---

## 5. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Optuna trial OOM on CIFAR-10 + ResNet50** | Med | High | Priors include `batch_size` choice list capped at 64; trial wraps `Trainer.fit()` in try/except `torch.cuda.OutOfMemoryError` → trial pruned, not crashed. |
| **Cell crash mid-Phase-B (CUDA, disk full)** | Med | Med | `--skip-existing` + `INTERRUPTED` sentinel = safe re-run idempotency. `metrics.json` written atomically. |
| **`history.json` corrupted by partial write on Ctrl-C** | Low | Low | `HistoryJSONCallback` uses `tmp + os.replace` atomic write on every `on_validation_epoch_end`, not just `on_fit_end`. |
| **Visual Core PNG drift due to seed offset change** | Low | High | `test_visual_core.py` includes byte-hash regression on the L3 reference cell. Any pipeline change that shifts pixels fails this test. |
| **TransNeXt zombie process still holds GPU** | Med | High | US-014 `quarantine_transnext.py --force` does `psutil` scan + kill before `rmtree`. CI-style sanity check: `nvidia-smi` shows zero `python.exe` PIDs before Phase B kicks off. |
| **Aggregator partial-write breaks dashboard mid-poll** | Low | Med | Atomic `tmp + os.replace`. Dashboard fetch retries once on JSON parse error; if 2nd attempt also fails, shows last-known-good cached payload. |
| **Priors decade-width violation slips into PR** | Low | Med | Unit test in US-015 enforces `high/low ≤ 10` per numeric field; CI fails the merge. |
| **10-minute polling masks fast-failing cells** | Low | Low | Per-cell `--cell` push from `run_all_phases.py` writes JSON immediately; the 10-min interval is only the dashboard's auto-pull. Manual "🔄 Refresh now" available. |

### Fail-Soft Contract

1. `run_all_phases.py` skips any cell whose `runs/final/<tag>/metrics.json` shows `status == "Done"`.
2. Aggregator preserves last-known-good `Final_Exp.json` if any single cell read raises.
3. Dashboard `fetch` failure → shows cached state with a red "stale" badge in the header pill.
4. Optuna study resumes from `artifacts/optuna_thz.db` on crash.
5. Visual-core renderer skips a cell if its degraded PNG already exists and matches the expected byte hash for the cell's params (so re-running is cheap).

### Batching Strategy

- Phase B is 20 CNN cells (≈10 GPU-hours). Batch as 4 sub-batches of 5 cells each (one per `(model, dataset)` pair, all 5 levels), so a crashed batch loses ≤ 5 cells of work.
- Optuna pre-tune is 4 studies × 20 trials ≈ 12–14 GPU-hours. Run sequentially before Phase B execution.
- Phase C (separate PRD) reuses the same frozen hparams — no re-tune.

---

## 6. Dependency-Ordered Story Map

```
US-014 ──┬──► US-015 ──► US-016 ──► US-019
         │                  │
         └──► US-017 ◄──────┘
                  │
                  └──► US-018
```

US-014 (quarantine) gates everything. US-017 (visual core) can render plan rows before US-016 finishes any cells, so it parallelizes with HPO + execution. US-018 (curves) requires US-016 cells writing `history.json`. US-019 (incremental refresh) wraps US-016's outer loop.

---

## 7. Definition-of-Done (PRD-level)

- [ ] All 8 acceptance-criteria sets above check green.
- [ ] [Final_Exp.md](../../Final_Exp.md) status summary reads `Phase A: 4/6 (TransNeXt deferred)`, `Phase B: 20/30 (TransNeXt deferred)`, total `24/186` (or `24/124` if we re-base the denominator post-quarantine — see open question O1).
- [ ] [artifacts/Final_Exp.html](../../artifacts/Final_Exp.html) renders all 4 features (visual-core column, lazy curves on click, 10-min auto-poll, manual refresh).
- [ ] `pytest src/tests` green (≥ 84 + new tests added in US-014/015/017/018/019).
- [ ] `mypy src scripts` green.
- [ ] No `*.ckpt`/`*.pt`/`*.pth` paths leaked into `Final_Exp.json` / `Final_Exp.md` / `priors.json` / `best_hparams/*.json` / `history.json`.
- [ ] Determinism gate ([src/tests/test_degradation_determinism.py](../../src/tests/test_degradation_determinism.py)) green (MSE = 0).

## 8. Open Questions for MASTER

- **O1.** Should the Phase B/Phase A row counts in [Final_Exp.md](../../Final_Exp.md) and the dashboard be re-based to `124` (TransNeXt rows hidden entirely) or remain `186` with a "Deferred" badge on quarantined rows? Decision affects US-014 and the dashboard "counts" pill.
- **O2.** When TransNeXt eventually returns (high-end GPU cluster), do we re-use the same priors mechanism (US-015) under a `transnext_*.json` priors block, or does TransNeXt's linear-probe protocol warrant its own PRD?
- **O3.** Should the visual-core `idx=0` choice be promoted to a configurable knob (`--sample-idx`) in case `idx=0` happens to be a degenerate / unrepresentative sample for one of the datasets?

---

**End of PRD. Awaiting MASTER approval to proceed with US-014 execution.**
