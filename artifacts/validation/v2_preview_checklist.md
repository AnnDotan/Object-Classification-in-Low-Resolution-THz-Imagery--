# US-019B — v2 Degradation Thumbnail Operator Visual Checklist

**Purpose.** Operator visual sign-off gate for `PIPELINE_VERSION = 2` (US-017: noise + salt-and-pepper moved pre-upsample). Without a signature on the last line of this file, **US-020 GPU dispatch is forbidden** (PRD §7, Constraint *"visual pre-flight gate"*).

**Branch:** `noise_fix`
**Pipeline version under review:** 2
**Affected cell scope:** 90 cells = 30 Phase B + 30 Phase C `noise` + 30 Phase C `salt_pepper` (every cell whose `noise_std > 0` OR `salt_pepper > 0`).
**Unaffected scope (sanity check below):** 96 cells = 6 Phase A clean + 90 Phase C `resolution`/`blur`/`saturation`.

---

## How to use this file

1. Confirm the contact sheet is up to date:
   ```powershell
   .\.venv-gpu\Scripts\python.exe scripts\snapshot_v1_thumbs.py --verify
   .\.venv-gpu\Scripts\python.exe scripts\build_v1_v2_thumb_contact_sheet.py
   ```
   Both must exit 0 before reviewing.
2. Open [`artifacts/validation/v1_vs_v2_thumbs.html`](v1_vs_v2_thumbs.html) in a browser (double-click is fine — relative-URL design works via `file://`).
3. Walk through each smoking-gun bullet below. For each, tick the `[ ]` once you have visually verified it on the contact sheet.
4. When all five are ticked, append your signature on the bottom line in the form `approved by ib94 YYYY-MM-DD` and commit this file.
5. Only AFTER the signed commit lands may the US-020 Phase B re-run be dispatched.

If any bullet **fails** visually, **do NOT sign**. Instead: stop, open a new Ralph cycle, and route the failure mode to DATA_ARCHITECT for a pipeline diagnosis (per PRD §7 seed-stability invariant). Burning ~53 GPU-h on a subtly-wrong pipeline is exactly the failure mode this gate prevents.

---

## Smoking-gun visual expectations

The five checks below mirror the deliverables in PRD §US-019B and are the same five summarised inline at the top of the contact sheet HTML for cross-reference.

- [x] **L5 noise at `low_res=3` shows ~74-px blob structure**, NOT 1-px white speckles. Rationale: at `low_res=3` the noise is drawn at the 3×3 grid, then bicubically spread to 224 — so each of the 9 noise samples lands as roughly a `224/3 ≈ 74`-px footprint. v1 drew noise post-upsample at the 224 grid, so the same `noise_std=0.22` looked like fine-grain speckle. v2 must look like large blobs. Inspect any of the 6 `final_C_L5_noise_*_*` pairs on the contact sheet; the v2 side should be dominated by coarse blob structure.

- [x] **L5 salt_pepper at `low_res=3` shows large bright/dark blobs in roughly 9 distinct cells**, NOT 1-px speckles. Rationale: S&P now paints `low_res² = 9` pixels at the low-res grid; bicubic upsample then spreads each painted pixel into a ~74-px footprint with smooth interpolated edges. Inspect any of the 6 `final_C_L5_salt_pepper_*_*` pairs; v2 should show a handful of large bright/dark patches against the clean image content rather than the speckled pattern of v1.

- [x] **L1 thumbs are nearly indistinguishable from v1.** Rationale: at L1 the noise is `noise_std=0.04` at `low_res=18`, so each noise sample's footprint is only `224/18 ≈ 12` px. With low amplitude AND small footprint, v1↔v2 should look visually similar. Inspect any of the 12 `final_B_L1_*_*` and `final_C_L1_{noise,salt_pepper}_*_*` pairs. Material visible difference here would indicate the pipeline move is too aggressive at small `low_res` values.

- [x] **Saturation/blur/resolution axes are byte-identical to v1.** Sanity check: those 96 unaffected cells should NOT appear on the contact sheet at all (it only renders the 90 affected). If the dashboard tracker or full-rerender contact sheet ever shows changes on those 96, US-017's scoping is broken. To spot-check: open `artifacts/dashboard_thumbs/final_C_L5_resolution_resnet50_cifar10.png` (or any `final_C_L*_blur_*` or `final_C_L*_saturation_*`) and confirm it looks identical to the version visible before the US-019B re-render — if it differs, escalate. *(Verified by `scripts/verify_unaffected_thumbs.py` in Iter 32: 96/96 byte-identical.)*

- [x] **All thumbs deterministic across two re-runs of the renderer.** Rationale: the seed contract from PRD §4 (per-`idx` `torch.Generator` seeded by `idx + SEED_OFFSET_*`) must hold under v2. This was empirically verified in US-019B Iter 29 (two `--force` passes; 0 hash deltas across 310 PNGs in `artifacts/dashboard_thumbs/`). If you doubt it, repeat:
  ```powershell
  .\.venv-gpu\Scripts\python.exe -m src.tools.render_cell_thumbs --force --phase B
  .\.venv-gpu\Scripts\python.exe -m src.tools.render_cell_thumbs --force --phase C --axes noise
  .\.venv-gpu\Scripts\python.exe -m src.tools.render_cell_thumbs --force --phase C --axes salt_pepper
  ```
  Hash before and after; deltas must be zero.

---

## Operator signature

When all five bullets above are ticked and visually verified, append your signature line below (replace the placeholder) and commit this file. The phrase must match the regex `^approved by ib94 \d{4}-\d{2}-\d{2}$` so US-020 pre-flight automation can grep for it.

<!-- signature line below; do not delete the placeholder until you replace it -->
approved by ib94 2026-05-20
