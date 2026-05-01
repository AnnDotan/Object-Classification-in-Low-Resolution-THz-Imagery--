# Agent Hierarchy — THz Image Classification Project

> Orchestration layer for the 36-experiment research plan. Every agent has a scoped file-system, a scoped tool-set, and must submit a plan for MASTER approval before any code change.

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
└── VALIDATOR        — diagnostic experiments + reproducibility gating
```

## Index

| Agent | Role | Primary Scope | Current Active Task |
|-------|------|----------------|---------------------|
| [MASTER](agents/MASTER.md) | Orchestrator | `agents/`, `AGENTS.md`, `TODO.md` | Enforce Plan-Mode protocol across all sub-agents |
| [EXECUTOR](agents/EXECUTOR.md) | Experiment runner | `runs/`, `artifacts/`, `logs/` | Idle — awaiting approved Phase B L3 plan |
| [OPTIMIZER](agents/OPTIMIZER.md) | Model/training | `src/models/`, `src/runner.py` | Analyze `papers/TransNeXt.pdf` for positional-bias interpolation + resolution discrepancy |
| [DESIGNER](agents/DESIGNER.md) | Exp. & dashboard design | `src/tools/`, `artifacts/`, `docs/` | Idle — awaiting Phase C/D grids |
| [REPORTER](agents/REPORTER.md) | Results synthesis | `docs/`, `artifacts/reports/`, `README.md` | Idle — awaiting Phase B L3 completion |
| [SECURITY](agents/SECURITY.md) | Repo hygiene | `.gitignore`, `.claudeignore`, `.gitattributes` | Audit & extend ignore lists for weights/datasets |
| [LIBRARIAN](agents/LIBRARIAN.md) | Knowledge mgmt | `README.md`, `AGENTS.md`, `CLAUDE.md`, `docs/` | Update `AGENTS.md` + `README.md` after every validated run |
| [DATA_ARCHITECT](agents/DATA_ARCHITECT.md) | Pipeline integrity | `src/data/` | Verify degradation consistency CIFAR-10 vs MNIST at L1/L2/L3 |
| [VALIDATOR](agents/VALIDATOR.md) | Validation & diagnostics | `scripts/validation/`, `runs/validation/` | Design TransNeXt 32×32 head-only linear probe |

## Coordination Protocol

1. **Plan first, code second.** Every experiment series — whether a new run, a code change, or a diagnostic — begins with an implementation plan submitted to MASTER. MASTER reviews and explicitly approves before any agent writes code.
2. **Fair comparison invariant.** All models must share identical degradation parameters and preprocessing logic. DATA_ARCHITECT owns and enforces this invariant.
3. **Minimal changes.** Per `CLAUDE.md`: "Minimal changes, no rewrites. Keep pipeline intact, log everything."
4. **Validation before promotion.** No run moves from `runs/systematic/` → `runs/official/` without VALIDATOR ✅ and MASTER sign-off.
5. **Knowledge capture.** LIBRARIAN updates `AGENTS.md` + `README.md` after every validated run so findings compound across sessions.

## Current Deliverable Queue (queued, awaiting MASTER plan approval)

- [ ] OPTIMIZER: TransNeXt positional-bias interpolation analysis (from `papers/TransNeXt.pdf`)
- [ ] OPTIMIZER: Resolution discrepancy mitigation proposal
- [ ] VALIDATOR: TransNeXt 32×32 head-only linear probe design
- [ ] DATA_ARCHITECT: CIFAR-10 vs MNIST degradation pipeline consistency report
- [ ] SECURITY: `.gitignore` / `.claudeignore` audit (in progress — see diff)

## Upcoming Milestones

| Date | Deliverable | Lead Agent |
|------|-------------|------------|
| 2026-05-31 | Poster | REPORTER |
| 2026-06-21 | Presentation | REPORTER |
| 2026-07-26 | Submission | REPORTER |

## Activity Log

| Date | Agent | Action | Outcome |
|------|-------|--------|---------|
| 2026-04-11 | MASTER | Initialized agent hierarchy (8 sub-agents + index) | ✅ |
| 2026-04-11 | SECURITY | Extended `.gitignore` / `.claudeignore` for weights + caches | ✅ |
