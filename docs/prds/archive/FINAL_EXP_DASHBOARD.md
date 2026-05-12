# PRD: FINAL_EXP Dashboard (`artifacts/Final_Exp.html`)

**Project:** p-2026-061 — Object Classification in Low-Resolution THz Imagery
**Scope:** Interactive dashboard for the 186-cell Final Research Phase
**Output artifact:** [artifacts/Final_Exp.html](../../artifacts/Final_Exp.html)
**Replaces:** N/A — legacy [artifacts/dashboard_experiment_plan.html](../../artifacts/dashboard_experiment_plan.html) is preserved as historical reference.
**US-### namespace:** scoped to this PRD only (does not collide with project-level US-### numbering in [PRD.md](../../PRD.md)).

---

## 1. Introduction

The legacy "🔬 Experiment Plan Dashboard" served the 36-cell pilot and is now insufficient for the 186-cell Final Research Phase (Phases A/B/C, five degradation levels, five isolation axes). We are building a successor — `Final_Exp.html` — that preserves the pilot dashboard's look-and-feel and filter ergonomics while adding:

- Phase-segregated navigation (A / B / C as top-level tabs in a single HTML file)
- A first-class Degradation column rendering the precise level + parameter dict per row
- Live polling of a build-time aggregate (`Final_Exp.json`) so completed cells appear without manual reload
- Multi-select filters, free-text search, and named filter presets persisted in `localStorage`

The dashboard consumes the automation pipeline established in project-level US-016 (`refresh_trackers.py` + `sync_trackers_git.py`). Build is a static HTML file; "real-time" is achieved via client-side polling of a sibling JSON.

## 2. Goals

- **G1.** Visual + interaction parity with the pilot dashboard's header, stats bar, filter row, and table styling.
- **G2.** All 186 cells navigable via 3 tabs: **Phase A** (6), **Phase B** (30), **Phase C** (150) — single HTML file, sticky tab bar, per-tab filter state.
- **G3.** Every row exposes the full degradation spec at a glance: a level badge (`L1`–`L5`) with an on-hover tooltip listing `low_res`, `blur (kernel, σ)`, `noise std`, `S&P`, `saturation`.
- **G4.** New rows / status flips appear within ≤30s of `refresh_trackers.py` completing, with a 3-second highlight animation on changed cells.
- **G5.** Filter superset of the pilot: model, dataset, status, sort + Phase, Level, Axis (Phase C only), multi-select chips, free-text tag search, and saved presets in `localStorage`.
- **G6.** Data layer: primary source is `artifacts/Final_Exp.json` (built from per-run `runs/final/<tag>/metrics.json`); `Final_Exp.md` remains a human-readable mirror and is used as a fallback for plan rows that have no completed run yet.
- **G7.** Zero secrets in `Final_Exp.json` — only metric scalars and config keys; no checkpoint paths, no weight blobs (respects the Weight Privacy rule in [CLAUDE.md](../../CLAUDE.md)).

## 3. User Stories

Stories are dependency-ordered: schema → aggregator → wiring → tests → HTML scaffold → data binding → tabs → degradation column → filters → search → persistence → polling → diff highlights. Each story fits one Ralph iteration.

---

### US-001: Define `Final_Exp.json` schema

**Description:** As a dashboard renderer, I want a stable JSON contract so the HTML and the aggregator can evolve independently.

**Acceptance Criteria:**
- [ ] New file `src/tools/final_exp_schema.py` exports a `FinalExpRow` TypedDict and a `FinalExpDoc` TypedDict.
- [ ] `FinalExpRow` fields: `tag` (str), `phase` (`"A" | "B" | "C"`), `model` (str), `dataset` (str), `level` (`"L1"..."L5" | "clean"`), `axis` (`"none" | "all" | "resolution" | "noise" | "blur" | "saturation" | "salt_pepper"`), `params` (dict: `low_res:int, blur_kernel:int, blur_sigma:float, noise_std:float, salt_pepper:float, saturation:float`), `status` (`"Pending" | "Running" | "Done" | "Failed"`), `val_acc` (float \| null), `val_loss` (float \| null), `epochs_run` (int \| null), `runtime_s` (float \| null), `started_at` (ISO8601 \| null), `finished_at` (ISO8601 \| null).
- [ ] `FinalExpDoc` fields: `schema_version` (literal `1`), `generated_at` (ISO8601 UTC), `rows` (list[FinalExpRow]), `counts` (dict: `total, pending, running, done, failed`).
- [ ] No checkpoint paths, no weight URIs, no host paths — schema enforces metric-only fields.
- [ ] Typecheck passes.

---

### US-002: Implement `build_final_exp_json.py` aggregator

**Description:** As a build pipeline, I want a script that emits `artifacts/Final_Exp.json` so the dashboard has a single, validated source.

**Acceptance Criteria:**
- [ ] New script `src/tools/build_final_exp_json.py` with a `main()` entry point.
- [ ] Plan rows: enumerated from [src/experiments/matrix.py](../../src/experiments/matrix.py) and [src/data/degradation_levels.py](../../src/data/degradation_levels.py) — exactly 186 rows produced.
- [ ] Completed rows: hydrate `val_acc, val_loss, epochs_run, runtime_s, started_at, finished_at, status` by reading `runs/final/<tag>/metrics.json`. Missing files → row stays `Pending`.
- [ ] `params` per row populated from `level_params(level)` with axis pinning honoring Phase C semantics (named axis at level L, others pinned to L1).
- [ ] Output written atomically (`tmp + os.replace`) to `artifacts/Final_Exp.json`.
- [ ] Script never opens `*.ckpt`, `*.pt`, `*.pth`, or files under `runs/**/*.ckpt` / `artifacts/weights/`.
- [ ] Typecheck passes.

---

### US-003: Wire aggregator into `refresh_trackers.py`

**Description:** As an operator, I want `refresh_trackers.py` to regenerate `Final_Exp.json` whenever it regenerates `Final_Exp.md`, so both stay in lockstep.

**Acceptance Criteria:**
- [ ] [scripts/refresh_trackers.py](../../scripts/refresh_trackers.py) calls `build_final_exp_json.main()` after `update_final_exp.py`.
- [ ] On aggregator failure, the script logs the error but does not delete an existing `Final_Exp.json` (last-known-good preserved).
- [ ] [scripts/sync_trackers_git.py](../../scripts/sync_trackers_git.py) `add`s `artifacts/Final_Exp.json` alongside `Final_Exp.md`.
- [ ] Typecheck passes.

---

### US-004: Test the aggregator

**Description:** As a maintainer, I want regression tests so schema drift, row count drift, and forbidden-path reads are caught in CI.

**Acceptance Criteria:**
- [ ] New test `src/tests/test_build_final_exp_json.py`.
- [ ] Test: emitted doc has exactly 186 rows split 6 / 30 / 150 across phases A / B / C.
- [ ] Test: at L1, every Phase C isolation row has `params` equal to the Phase B L1 row's `params` (collapse invariant).
- [ ] Test: a fixture `metrics.json` with `val_acc=0.42, status="Done"` round-trips into the corresponding row.
- [ ] Test: aggregator does not call `open()` on any path matching `*.ckpt`, `*.pt`, `*.pth` (assert via `unittest.mock` patch on `builtins.open`).
- [ ] Typecheck passes. All tests pass.

---

### US-005: HTML scaffold + CSS parity with pilot

**Description:** As a user, I want the new dashboard to feel identical to the pilot's chrome (header, stats bar, filter row, table) so the team's muscle memory carries over.

**Acceptance Criteria:**
- [ ] New file [artifacts/Final_Exp.html](../../artifacts/Final_Exp.html) — static, no server.
- [ ] CSS variables block (`--bg, --surface, --surface2, --surface3, --border, --text, --accent, --green, --orange, --red, --blue`) copied verbatim from [artifacts/dashboard_experiment_plan.html](../../artifacts/dashboard_experiment_plan.html).
- [ ] Header reads "🔬 Final Experiment Dashboard — 186-cell Research Phase" with project subtitle.
- [ ] Stats bar with cards: `Total`, `Pending`, `Running`, `Done`, `Failed`, `Best val_acc` — same `.stat-card` styling as pilot.
- [ ] Sticky tab bar below stats with three tabs: `Phase A · 6`, `Phase B · 30`, `Phase C · 150`.
- [ ] Empty `<table>` shell with column headers: `Tag | Model | Dataset | Phase | Level | Status | val_acc | Epochs | Runtime`.
- [ ] No JS data binding yet — table renders empty `<tbody>`.
- [ ] Typecheck passes (HTML lints clean — `tidy -e` exits 0).
- [ ] Verify changes work in browser.

---

### US-006: Load `Final_Exp.json` and render the table

**Description:** As a user, I want the table populated from the aggregate JSON so I can see all 186 cells.

**Acceptance Criteria:**
- [ ] Inline `<script>` in `Final_Exp.html` `fetch`es `./Final_Exp.json` on `DOMContentLoaded`.
- [ ] On success, every row is appended to `<tbody>` with text cells matching the schema fields.
- [ ] Stats bar values populate from `doc.counts` (`Total`, `Pending`, `Running`, `Done`, `Failed`) and a max-over-rows for `Best val_acc`.
- [ ] On fetch failure, a banner reads "Could not load Final_Exp.json — run scripts/refresh_trackers.py" with no JS errors in console.
- [ ] Status cell uses color tokens: `Done → --green`, `Running → --blue`, `Pending → --orange`, `Failed → --red`.
- [ ] Typecheck passes.
- [ ] Verify changes work in browser.

---

### US-007: Three-tab phase navigation

**Description:** As a user, I want to switch between Phase A / B / C without losing my place so I can drill into one phase at a time.

**Acceptance Criteria:**
- [ ] Clicking a tab sets `data-active-phase` on the tab bar and hides rows whose `phase` does not match (CSS class toggle, no DOM rebuild).
- [ ] Active tab visually distinct (border-bottom accent matching pilot's tab styling if present, otherwise `--accent` underline).
- [ ] Active phase persisted to `localStorage` key `final_exp.active_phase` and restored on next load.
- [ ] Stats bar updates to reflect counts for the active phase only; an `All 186` chip in the stats bar shows the global total for cross-phase context.
- [ ] Default tab on first ever load: `Phase A`.
- [ ] Typecheck passes.
- [ ] Verify changes work in browser.

---

### US-008: Level badge column with parameter tooltip

**Description:** As a user, I want to see at a glance which degradation level a row uses and inspect the exact parameters without leaving the table.

**Acceptance Criteria:**
- [ ] `Level` column renders a pill badge (`clean`, `L1`, `L2`, `L3`, `L4`, `L5`) with intensity-graded background (e.g. `L1` light → `L5` saturated).
- [ ] Native `title` attribute on the badge contains a multi-line string: `low_res=N | blur=KxK σ=S | noise=N | S&P=P | saturation=S` derived from `row.params`.
- [ ] Phase C rows additionally show the active axis as a small caption under the badge (e.g. `axis: noise`); Phase A rows show `clean`; Phase B rows show `all`.
- [ ] Badge styling consistent across tabs (no per-phase overrides).
- [ ] Typecheck passes.
- [ ] Verify changes work in browser.

---

### US-009: Port pilot filters (model / dataset / status + sort)

**Description:** As a user, I want the same filtering and sorting controls I had in the pilot dashboard.

**Acceptance Criteria:**
- [ ] Filter row above the table contains four `<select>` controls: `Model` (All / ResNet50 / DenseNet121 / TransNeXt), `Dataset` (All / CIFAR-10 / MNIST), `Status` (All / Pending / Running / Done / Failed), `Sort by` (val_acc desc / val_acc asc / epochs desc / runtime desc / tag asc).
- [ ] Filtering re-applies in pure JS over the in-memory rows array; no refetch.
- [ ] Sort reorders `<tbody>` children stably; ties broken by `tag asc`.
- [ ] Filter values are scoped per active phase tab (switching tabs restores that tab's last filter set).
- [ ] Typecheck passes.
- [ ] Verify changes work in browser.

---

### US-010: Phase / Level / Axis context filters

**Description:** As a researcher, I want to filter by Level (L1–L5) globally and by Axis when on the Phase C tab so I can isolate a specific degradation slice.

**Acceptance Criteria:**
- [ ] `Level` select added with options `All / clean / L1 / L2 / L3 / L4 / L5`.
- [ ] `Axis` select added with options `All / resolution / noise / blur / saturation / salt_pepper` — visible only when active tab is `Phase C`; hidden (not just disabled) otherwise.
- [ ] At Phase C + Level=L1, every isolation row collapses to identical `params` — dashboard does not deduplicate (visual confirmation of the collapse invariant).
- [ ] Selections persist per-tab in `localStorage` under `final_exp.filters.<phase>`.
- [ ] Typecheck passes.
- [ ] Verify changes work in browser.

---

### US-011: Convert filter selects to multi-select chips

**Description:** As a user, I want to compare multiple models or datasets simultaneously without picking "All".

**Acceptance Criteria:**
- [ ] `Model`, `Dataset`, `Status`, `Level`, `Axis` filters render as a row of clickable chips (one per option) with a multi-select toggle.
- [ ] Empty selection ≡ "All" (no filtering on that axis).
- [ ] Active chips visually distinct (filled `--accent` background, dimmed text for inactive).
- [ ] A "Clear all" link resets all chip filters on the active tab.
- [ ] Sort remains a single-value `<select>`.
- [ ] Typecheck passes.
- [ ] Verify changes work in browser.

---

### US-012: Free-text tag search

**Description:** As a user, I want to find a specific run by typing part of its tag.

**Acceptance Criteria:**
- [ ] Search `<input type="text" placeholder="Search tag…">` added to the filter row.
- [ ] Substring match (case-insensitive) against `row.tag`; updates on every `input` event.
- [ ] Search query is AND-combined with chip filters.
- [ ] Search query persisted per-tab in `localStorage` under `final_exp.filters.<phase>.search`.
- [ ] An `Esc` keypress while focused clears the search box.
- [ ] Typecheck passes.
- [ ] Verify changes work in browser.

---

### US-013: Named filter presets in `localStorage`

**Description:** As a power user, I want to save filter combinations I use repeatedly so I don't have to reconstruct them.

**Acceptance Criteria:**
- [ ] "Presets" dropdown next to "Clear all": `Save current as…`, `<list of saved presets>`, `Delete preset…`.
- [ ] `Save current as…` prompts for a name and writes `{filters, search, sort}` for the active phase to `localStorage` key `final_exp.presets.<phase>.<name>`.
- [ ] Selecting a saved preset restores filters, search, and sort.
- [ ] Presets scoped per phase (Phase A presets do not appear when on Phase B).
- [ ] No backend, no network — pure `localStorage`.
- [ ] Typecheck passes.
- [ ] Verify changes work in browser.

---

### US-014: Poll `Final_Exp.json` every 30 s

**Description:** As a user watching a long sweep, I want the table to update without manual reloads.

**Acceptance Criteria:**
- [ ] After initial load, a `setInterval(poll, 30_000)` fetches `./Final_Exp.json?t=<epoch_ms>` (cache-bust query string).
- [ ] On poll, rows are reconciled by `tag` key — existing `<tr>` elements are mutated in place; new tags are appended; deleted tags (should never happen — assert) are removed.
- [ ] Stats bar recomputes from the new doc.
- [ ] If `doc.generated_at` is unchanged, the poll is a no-op (cheap early-exit).
- [ ] Polling pauses while the browser tab is hidden (`document.visibilitychange`) and resumes on visibility.
- [ ] Typecheck passes.
- [ ] Verify changes work in browser.

---

### US-015: "Last refreshed" timestamp in header

**Description:** As a user, I want to know how fresh the data is so I can trust what I'm seeing.

**Acceptance Criteria:**
- [ ] Header shows two timestamps: `Generated: <doc.generated_at, ISO>` and `Last polled: <relative, e.g. "5s ago">`.
- [ ] Relative timestamp updates every 1 s without a re-fetch.
- [ ] If polling has been paused (tab hidden) for >2 minutes, the relative timestamp turns `--orange`.
- [ ] If the last fetch errored, the relative timestamp turns `--red` and shows `Last poll failed`.
- [ ] Typecheck passes.
- [ ] Verify changes work in browser.

---

### US-016: Visual diff highlight on changed cells

**Description:** As a user, I want completed cells to flash so I notice them without scanning the whole table.

**Acceptance Criteria:**
- [ ] On reconcile, any `<tr>` whose `status` transitioned from `Pending`/`Running` to `Done` or `Failed` gets class `row-flash`.
- [ ] `.row-flash` runs a 3-second CSS animation (background pulse `--accent-dim → transparent`) and is removed on `animationend`.
- [ ] Animation respects `prefers-reduced-motion: reduce` (replaced with a static 3-second border highlight, no movement).
- [ ] No flash on the initial load (only on subsequent polls).
- [ ] Typecheck passes.
- [ ] Verify changes work in browser.

---

## 4. Non-Goals

- **Server-side rendering or live websockets.** The dashboard remains a static file + JSON polling. No Flask, no FastAPI, no SSE.
- **Editing runs from the dashboard.** Read-only — no buttons that re-trigger training, kill jobs, or change `metrics.json`.
- **Replacing the pilot dashboard.** [artifacts/dashboard_experiment_plan.html](../../artifacts/dashboard_experiment_plan.html) is preserved untouched as historical reference.
- **W&B / Optuna integration in the dashboard UI.** Links out are fine; embedding iframes or syncing live W&B state is out of scope.
- **Inline plotting of training curves.** The pilot's row-level plot rendering (if any) is not ported in this PRD; thumbnails come from `src/tools/render_curve_thumbs.py` and are linked, not embedded.
- **Mobile-first responsive design.** Desktop-only, ≥1280 px target. No grid collapse below that width.
- **Authentication or access control.** Local artifact, opened via `file://` or a static server.
- **Schema versioning beyond v1.** The schema includes a `schema_version` field, but migration tooling is deferred until v2 is needed.

## 5. Technical Notes

### 5.1 Reuse from existing infrastructure
- **Plan enumeration:** [src/experiments/matrix.py](../../src/experiments/matrix.py) is the single source of truth for the 186 cells. Do not re-enumerate in JS.
- **Level table:** [src/data/degradation_levels.py](../../src/data/degradation_levels.py) — call `level_params(level)` server-side, ship the resolved dict to the client.
- **Tracker pipeline:** [scripts/refresh_trackers.py](../../scripts/refresh_trackers.py) and [scripts/sync_trackers_git.py](../../scripts/sync_trackers_git.py) (project US-016) are the only legitimate triggers for regenerating `Final_Exp.json`.
- **CSS:** Copy verbatim from the pilot — do not redesign tokens. The "look identical" requirement is binding.

### 5.2 Constraints
- **Weight Privacy:** No code path may read `*.ckpt`, `*.pt`, `*.pth`, or `runs/**/*.ckpt`. Enforced by US-004.
- **Determinism:** `Final_Exp.json` must be byte-identical for identical run state — sort rows by `tag` ascending before serialization.
- **Atomic writes:** Always write `Final_Exp.json` via `tmp + os.replace` so partial files never reach the browser mid-poll.
- **No CDN dependencies for new behavior.** Chart.js is already loaded in the pilot for curve thumbs; do not add new external deps. All multi-select / preset logic is hand-rolled vanilla JS.

### 5.3 Open questions deferred to implementation
- Exact placement of the `Best val_acc` stat card (per-tab vs global) — defer to UI iteration; default to per-tab.
- Whether to surface `axis` as a separate column on Phase C or only inside the Level badge caption — current spec keeps it in the badge caption (US-008); a dedicated column can be added later as a non-breaking story.
- Color ramp for L1→L5 badge intensity — match the existing degradation thumbnail palette in `src/tools/render_cell_thumbs.py` if a palette is already defined there; otherwise pick a perceptually uniform ramp.

---

## 6. Story Dependency Graph

```
US-001 (schema)
  └── US-002 (aggregator)
        ├── US-003 (refresh_trackers wiring)
        └── US-004 (tests)
              └── US-005 (HTML scaffold)
                    └── US-006 (load + render)
                          └── US-007 (tabs)
                                └── US-008 (level badge)
                                      └── US-009 (pilot filters)
                                            └── US-010 (phase/level/axis)
                                                  └── US-011 (multi-select chips)
                                                        └── US-012 (search)
                                                              └── US-013 (presets)
                                                                    └── US-014 (polling)
                                                                          └── US-015 (last refreshed)
                                                                                └── US-016 (diff highlight)
```

Strict linear chain after US-004 — each UI story builds on the previous one's DOM/state contract.
