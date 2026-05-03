# Agent Hierarchy — THz Image Classification Project

> Orchestration layer for the **186-cell Final Research Phase**. Every agent has a scoped file-system, a scoped tool-set, and must submit a plan for MASTER approval before any code change.

## Structure

```
MASTER (orchestrator, no write access to code)
├── EXECUTOR         — runs training jobs
├── OPTIMIZER        — model/training surgical improvements
├── DESIGNER         — experiment grids + dashboards
├── REPORTER         — poster / presentation / paper drafts
├── SECURITY         — .gitignore / .claudeignore hygiene
├── LIBRARIAN        — README / AGENTS / docs knowledge capture
├── DATA_ARCHITECT   — src/data/ and degradation pipeline integrity
├── VALIDATOR        — diagnostic experiments + reproducibility gating
├── SYNCHRONIZER     — context integrity + source alignment
└── NOTEBOOKLM_SYNC  — NotebookLM source mirror + git lockstep
```

## Index

| Agent | Role | Primary Scope | Current Active Task |
|-------|------|----------------|---------------------|
| [MASTER](agents/MASTER.md) | Orchestrator | `agents/`, `AGENTS.md`, `TODO.md`, `Final_Exp.md` | Drive 186-cell plan implementation across sprints 1–7 |
| [EXECUTOR](agents/EXECUTOR.md) | Experiment runner | `runs/final/`, `artifacts/`, `logs/` | Idle — awaiting Sprint 5 (final phase runner) before any 186-cell launch |
| [OPTIMIZER](agents/OPTIMIZER.md) | Model/training | `src/models/`, `src/lightning/` | Sprint 3: TransNeXt size selector + auto-download |
| [DESIGNER](agents/DESIGNER.md) | Exp. & dashboard design | `src/tools/`, `artifacts/`, `docs/` | Sprint 6: `generate_final_dashboard.py` + cell pre-rendering |
| [REPORTER](agents/REPORTER.md) | Results synthesis | `artifacts/reports/`, `README.md` | Idle — awaiting Phase A completion |
| [SECURITY](agents/SECURITY.md) | Repo hygiene | `.gitignore`, `.claudeignore`, `.gitattributes` | Sprint 7: extend ignore lists for `runs/final/`, weights, Optuna DB |
| [LIBRARIAN](agents/LIBRARIAN.md) | Knowledge mgmt | `README.md`, `AGENTS.md`, `CLAUDE.md`, `docs/` | Maintain `Final_Exp.md` ↔ code consistency |
| [DATA_ARCHITECT](agents/DATA_ARCHITECT.md) | Pipeline integrity | `src/data/` | Sprint 1: plumb saturation through DataConfig + PSNR/SSIM hook |
| [VALIDATOR](agents/VALIDATOR.md) | Validation & diagnostics | `scripts/validation/`, `runs/validation/` | Sprint 1: extend determinism test for saturation + PSNR/SSIM |
| [SYNCHRONIZER](agents/SYNCHRONIZER.md) | Context integrity | repo-wide read; `AGENTS.md`, `TODO.md` write | Pre-flight check before any `EnterPlanMode` |
| [NOTEBOOKLM_SYNC](agents/NOTEBOOKLM_SYNC.md) | NotebookLM mirror | `scripts/notebooklm_*` write; `notebooklm` CLI | Mirror curated repo subset → `THz Project` notebook; git-commit in lockstep |

## Coordination Protocol

1. **Plan first, code second.** Every code change begins with a plan submitted to MASTER. MASTER reviews and explicitly approves before any agent writes code.
2. **Fair-comparison invariant.** All models share identical `DegradeConfig` parameters and preprocessing logic. DATA_ARCHITECT owns and enforces this invariant — single source of truth lives in [`src/data/degradation_levels.py`](src/data/degradation_levels.py).
3. **Convergence-first.** No speed caps. `max_epochs=60`, `EarlyStopping(patience=10, min_delta=1e-4)` for all 186 final runs.
4. **Weight privacy.** Trained checkpoints never leave `runs/final/`; gitignored, claudeignored. Future Claude analysis is JSON/CSV/HTML only.
5. **Knowledge capture.** LIBRARIAN updates `README.md` + `Final_Exp.md` after every validated run.

## Current Deliverable Queue

> See [`TODO.md`](TODO.md) for the full Sprint 1–7 backlog.

- [x] **Sprint 1 (DATA_ARCHITECT)** — `degradation_levels.py`, saturation axis in `degrade.py`, `degrade_config_for` helper, `Final_Exp.md` initialized (2026-05-02)
- [ ] **Sprint 1 remaining** — saturation through `DataConfig`, PSNR/SSIM in `setup()`, determinism test extension
- [ ] **Sprint 2 (OPTIMIZER)** — FP16 in main Trainer, convergence-first defaults, ModelCheckpoint pinned to `runs/final/`
- [ ] **Sprint 3 (OPTIMIZER)** — TransNeXt size selector + SHA256-verified auto-download
- [ ] **Sprint 4 (OPTIMIZER)** — paper-anchored Optuna search + `tune_all.py`
- [ ] **Sprint 5 (EXECUTOR + DESIGNER)** — `--plan final` switch + three-loop matrix in `run_all_phases.py`
- [ ] **Sprint 6 (DESIGNER)** — `generate_final_dashboard.py` + `update_final_exp.py` + `DashboardRefreshCallback`
- [ ] **Sprint 7 (SECURITY)** — `.gitignore` / `.claudeignore` / `setup.sh` / `verify_env.py` / pre-commit guard

## Upcoming Milestones

| Date | Deliverable | Lead Agent |
|------|-------------|------------|
| 2026-05-31 | Poster & abstract | REPORTER |
| 2026-06-21 | Final presentation | REPORTER |
| 2026-07-26 | Final submission | REPORTER |

## Activity Log

| Date | Agent | Action | Outcome |
|------|-------|--------|---------|
| 2026-04-11 | MASTER | Initialized agent hierarchy (8 sub-agents + index) | ✅ |
| 2026-04-11 | SECURITY | Extended `.gitignore` / `.claudeignore` for weights + caches | ✅ |
| 2026-04-?? | MASTER | Added SYNCHRONIZER (context-integrity guardian) — 9 sub-agents total | ✅ |
| 2026-05-02 | MASTER | Final Research Plan approved — pivot from legacy 36-exp to 186-cell campaign | ✅ |
| 2026-05-02 | DATA_ARCHITECT | Sprint 1 batch 1: `degradation_levels.py`, saturation axis, `degrade_config_for`, `Final_Exp.md` | ✅ |
| 2026-05-02 | LIBRARIAN | Repo-wide MD cleanup: removed 8 legacy docs (EXPERIMENT_PLAN, EXPERIMENTS_SUMMARY, docs/{IMPLEMENTATION_NOTES,VISUALIZATION_GUIDE,DASHBOARD_README,RESULTS_SUMMARY,.agent}, artifacts/summaries/experiment_journal); rewrote README, TODO, AGENTS, CLAUDE for 186-cell campaign | ✅ |
| 2026-05-03 | MASTER | Added NOTEBOOKLM_SYNC sub-agent (10 sub-agents total) — owns `scripts/notebooklm_sync.py` + `notebooklm_manifest.yml`; mirrors curated subset into `THz Project` notebook | ✅ |
