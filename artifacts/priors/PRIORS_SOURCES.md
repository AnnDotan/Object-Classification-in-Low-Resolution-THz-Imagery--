# Priors — Source Attribution

Companion to [`artifacts/priors/{model}.json`](.) (and `_schema.json`). Authored per [`docs/prds/PHASE_B_VISUAL_CORE.md`](../../docs/prds/PHASE_B_VISUAL_CORE.md) US-015 to make every Optuna search range traceable to a peer-reviewed source.

> **Decade-width convention.** The project enforces *paper anchor ± 1 decade* per [`CLAUDE.md`](../../CLAUDE.md). For `loguniform` hparams that means the band may extend at most one decade above and one decade below the anchor — total span up to two decades, or `high / low ≤ 100`. Validator: [`tune_all.py`](../../tune_all.py) `MAX_LOGUNIFORM_RATIO`. Wider bands are rejected as "blind exploration".

> **Per-(model, dataset) granularity.** The project runs CIFAR-10 and MNIST through the **same** training recipe (CLAUDE.md hparam table is per-model, not per-dataset). The 4 CNN Phase B (model, dataset) pairs therefore share their model's priors file; Optuna runs a separate study per pair on L3 Moderate, but seeded from the same prior. The PRD's earlier per-(model, dataset) priors plan is superseded by this addendum.

> **TransNeXt is un-quarantined.** The legacy US-014 hold (waiting on Blackwell hardware) was lifted when the RTX 5070 came online; all 3 models — ResNet50, DenseNet121, TransNeXt — train at **224×224** in the campaign.

## File index

- [`_schema.json`](\_schema.json) — JSON Schema (Draft-07) for prior file shape
- [`resnet50.json`](resnet50.json) — ResNet50 prior, used for `resnet50_cifar10` + `resnet50_mnist`
- [`densenet121.json`](densenet121.json) — DenseNet121 prior, used for `densenet121_cifar10` + `densenet121_mnist`
- [`transnext_tiny.json`](transnext_tiny.json) — TransNeXt-tiny prior (canonical campaign variant; swapped from `transnext_small` by US-016 on 2026-05-14, which itself swapped from `transnext_base` by US-004 same day; trained at 224×224 with full-FT + differential LR)
- this file — paper-quote attribution

## Hyperparameter sources

The five hparams enforced by the schema (`head_lr`, `backbone_lr`, `weight_decay`, `label_smoothing`, `warmup_epochs`) are anchored as follows. Each row's `citation` field in the prior JSON points to the paper file shown here.

### ResNet50 ([artifacts/priors/resnet50.json](resnet50.json))

| Hparam | Anchor | Range | Source |
|---|---|---|---|
| `head_lr` | `1e-3` | `[1e-4, 1e-2]` (loguniform) | [papers/TResNet.pdf](../../papers/TResNet.pdf) §3.3 — *"AdamW with one-cycle policy. Initial LR 1e-3 for the classification head"* |
| `backbone_lr` | `1e-4` | `[1e-5, 1e-3]` (loguniform) | [papers/TResNet.pdf](../../papers/TResNet.pdf) §3.3 — *"differential learning rate, backbone LR ~10× smaller than head"* |
| `weight_decay` | `1e-4` | `[1e-5, 1e-3]` (loguniform) | [papers/TResNet.pdf](../../papers/TResNet.pdf) §3.3 — weight decay `1e-4` |
| `label_smoothing` | `0.1` | `[0.0, 0.2]` (uniform) | [papers/TResNet.pdf](../../papers/TResNet.pdf) §3.3 — label smoothing `0.1` |
| `warmup_epochs` | `0` | `[0, 5]` (uniform → int) | [papers/TResNet.pdf](../../papers/TResNet.pdf) §3.3 — cosine schedules typically use 0–5 warmup epochs |

### DenseNet121 ([artifacts/priors/densenet121.json](densenet121.json))

| Hparam | Anchor | Range | Source |
|---|---|---|---|
| `head_lr` | `1e-3` | `[1e-4, 1e-2]` (loguniform) | [papers/Densely Connected Convolutional Networks.pdf](../../papers/Densely%20Connected%20Convolutional%20Networks.pdf) §4 — *"initial LR 0.1 with SGD"* (the project adapts that to AdamW head LR `1e-3` per CLAUDE.md CNN row) |
| `backbone_lr` | `1e-4` | `[1e-5, 1e-3]` (loguniform) | [papers/Densely Connected Convolutional Networks.pdf](../../papers/Densely%20Connected%20Convolutional%20Networks.pdf) §4 — fine-tuning LRs an order of magnitude below head |
| `weight_decay` | `1e-4` | `[1e-5, 1e-3]` (loguniform) | [papers/Densely Connected Convolutional Networks.pdf](../../papers/Densely%20Connected%20Convolutional%20Networks.pdf) §4 — weight decay `1e-4` |
| `label_smoothing` | `0.1` | `[0.0, 0.2]` (uniform) | [papers/TResNet.pdf](../../papers/TResNet.pdf) §3.3 — DenseNet predates label smoothing; we cite TResNet as the modern CNN training reference |
| `warmup_epochs` | `0` | `[0, 5]` (uniform → int) | [papers/Densely Connected Convolutional Networks.pdf](../../papers/Densely%20Connected%20Convolutional%20Networks.pdf) §4 — short warmup is standard for cosine fine-tuning |

### TransNeXt — Tiny ([artifacts/priors/transnext_tiny.json](transnext_tiny.json)) — canonical 2026-05-14, 224×224

The campaign canonical variant was swapped twice on 2026-05-14: first `transnext_base` (~89M params) → `transnext_small` (~50M params) (US-004) for capacity match to ResNet50 / DenseNet121 and dataset size; then `transnext_small` → `transnext_tiny` (~28M params) (US-016) for an even tighter capacity-match to ResNet50 (~25M) and ~310 GPU-h savings across the 62 TransNeXt rows. Priors are unchanged across both swaps — they are LR/WD/warmup, paper-derived, and not architecture-dependent.


| Hparam | Anchor | Range | Source |
|---|---|---|---|
| `head_lr` | `1e-3` | `[1e-4, 1e-2]` (loguniform) | [papers/TransNeXt.pdf](../../papers/TransNeXt.pdf) §A.3 — head LR `1e-3` (linear probe), inherited as FT head anchor |
| `backbone_lr` | `5e-5` | `[5e-6, 5e-4]` (loguniform) | [papers/TransNeXt.pdf](../../papers/TransNeXt.pdf) §A.3 — full FT typically `1e-5`–`1e-4` for ImageNet-pretrained ViTs |
| `weight_decay` | `5e-2` | `[5e-3, 5e-1]` (loguniform) | [papers/TransNeXt.pdf](../../papers/TransNeXt.pdf) §A.3 — weight decay `5e-2` |
| `label_smoothing` | `0.0` | `[0.0, 0.1]` (uniform) | [papers/TransNeXt.pdf](../../papers/TransNeXt.pdf) §A.3 — LS `0.0` for linear probe; FT can tolerate small LS |
| `warmup_epochs` | `5` | `[0, 10]` (uniform → int) | [papers/TransNeXt.pdf](../../papers/TransNeXt.pdf) §A.3 — cosine + linear warmup is standard for ViTs |

## Reproducibility

The hash of each priors file is round-tripped into `runs/final/<tag>/metrics.json` as `hparams_source.priors_file_hash` ([tune_all.py](../../tune_all.py) `priors_file_hash()`). Any change to a priors JSON — even a trailing newline — produces a fresh SHA-256 and invalidates downstream `best_hparams/{model}_{dataset}.json` records. Re-tune is required when a priors file is edited.

## How to extend

When adding a new model:

1. Author `artifacts/priors/<model>.json` (validate against `_schema.json`, all five required hparams must cite a paper).
2. Add `<model>` to `tune_all.SUPPORTED_MODELS`.
3. Append a paper-quote row to this file under a new section.
4. Run `python tune_all.py --validate-only` and `python -m src.tests.test_priors`.
