# Priors — Source Attribution

Companion to [`artifacts/priors/{model}.json`](.) (and `_schema.json`). Authored per [`docs/prds/PHASE_B_VISUAL_CORE.md`](../../docs/prds/PHASE_B_VISUAL_CORE.md) US-015 to make every Optuna search range traceable to a peer-reviewed source.

> **Decade-width convention.** The project enforces *paper anchor ± 1 decade* per [`CLAUDE.md`](../../CLAUDE.md). For `loguniform` hparams that means the band may extend at most one decade above and one decade below the anchor — total span up to two decades, or `high / low ≤ 100`. Validator: [`tune_all.py`](../../tune_all.py) `MAX_LOGUNIFORM_RATIO`. Wider bands are rejected as "blind exploration".

> **Per-(model, dataset) granularity.** The project runs CIFAR-10 and MNIST through the **same** training recipe (CLAUDE.md hparam table is per-model, not per-dataset). The 4 CNN Phase B (model, dataset) pairs therefore share their model's priors file; Optuna runs a separate study per pair on L3 Moderate, but seeded from the same prior. The PRD's earlier per-(model, dataset) priors plan is superseded by this addendum.

> **TransNeXt is quarantined** ([US-014](../../docs/prds/PHASE_B_VISUAL_CORE.md#us-014-transnext-hard-quarantine-prerequisite)). [`transnext_base.json`](transnext_base.json) is preserved for future reactivation but is not consumed by the Phase B sweep.

## File index

- [`_schema.json`](\_schema.json) — JSON Schema (Draft-07) for prior file shape
- [`resnet50.json`](resnet50.json) — ResNet50 prior, used for `resnet50_cifar10` + `resnet50_mnist`
- [`densenet121.json`](densenet121.json) — DenseNet121 prior, used for `densenet121_cifar10` + `densenet121_mnist`
- [`transnext_base.json`](transnext_base.json) — TransNeXt prior (V3 un-quarantined; bare name routes via `_v3_cell_settings`)
- [`transnext_small_native.json`](transnext_small_native.json) — TransNeXt native-32 small variant (PRD US-042 alias)
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

### TransNeXt — Base ([artifacts/priors/transnext_base.json](transnext_base.json)) — V3 un-quarantined

| Hparam | Anchor | Range | Source |
|---|---|---|---|
| `head_lr` | `1e-3` | `[1e-4, 1e-2]` (loguniform) | [papers/TransNeXt.pdf](../../papers/TransNeXt.pdf) §A.3 — head LR `1e-3` (linear probe), inherited as FT head anchor |
| `backbone_lr` | `5e-5` | `[5e-6, 5e-4]` (loguniform) | [papers/TransNeXt.pdf](../../papers/TransNeXt.pdf) §A.3 — full FT typically `1e-5`–`1e-4` for ImageNet-pretrained ViTs |
| `weight_decay` | `5e-2` | `[5e-3, 5e-1]` (loguniform) | [papers/TransNeXt.pdf](../../papers/TransNeXt.pdf) §A.3 — weight decay `5e-2` |
| `label_smoothing` | `0.0` | `[0.0, 0.1]` (uniform) | [papers/TransNeXt.pdf](../../papers/TransNeXt.pdf) §A.3 — LS `0.0` for linear probe; FT can tolerate small LS |
| `warmup_epochs` | `5` | `[0, 10]` (uniform → int) | [papers/TransNeXt.pdf](../../papers/TransNeXt.pdf) §A.3 — cosine + linear warmup is standard for ViTs |

### TransNeXt — Small Native ([artifacts/priors/transnext_small_native.json](transnext_small_native.json)) — PRD US-042 native-32 alias

Same paper source as `transnext_base` (TransNeXt §A.3). The `_native` alias is architecturally `transnext_small` (embed_dims `[72,144,288,576]`, depths `[5,5,22,5]`) but pre-baked for the V3 native path (`img_size=32`, `patch_size=2`). Search ranges mirror `transnext_base.json` so the decade-bound contract carries over without re-derivation.

| Hparam | Anchor | Range | Source |
|---|---|---|---|
| `head_lr` | `5e-4` (V3 FT row, CLAUDE.md) | `[1e-4, 1e-2]` (loguniform) | [papers/TransNeXt.pdf](../../papers/TransNeXt.pdf) §A.3 — head LR `1e-3` (LP), inherited as FT head anchor; CLAUDE.md V3 row anchors `5e-4` |
| `backbone_lr` | `5e-5` (V3 FT row) | `[5e-6, 5e-4]` (loguniform) | [papers/TransNeXt.pdf](../../papers/TransNeXt.pdf) §A.3 — full FT typically `1e-5`–`1e-4`; CLAUDE.md V3 row anchors `5e-5` |
| `weight_decay` | `5e-2` | `[5e-3, 5e-1]` (loguniform) | [papers/TransNeXt.pdf](../../papers/TransNeXt.pdf) §A.3 — weight decay `5e-2` |
| `label_smoothing` | `0.1` (V3 FT row) | `[0.0, 0.1]` (uniform) | [papers/TransNeXt.pdf](../../papers/TransNeXt.pdf) §A.3 — LS `0.0` for LP; CLAUDE.md V3 row anchors `0.1` for FT |
| `warmup_epochs` | `5` | `[0, 10]` (uniform → int) | [papers/TransNeXt.pdf](../../papers/TransNeXt.pdf) §A.3 — cosine + linear warmup is standard for ViTs |

## Reproducibility

The hash of each priors file is round-tripped into `runs/final/<tag>/metrics.json` as `hparams_source.priors_file_hash` ([tune_all.py](../../tune_all.py) `priors_file_hash()`). Any change to a priors JSON — even a trailing newline — produces a fresh SHA-256 and invalidates downstream `best_hparams/{model}_{dataset}.json` records. Re-tune is required when a priors file is edited.

## How to extend

When adding a new model:

1. Author `artifacts/priors/<model>.json` (validate against `_schema.json`, all five required hparams must cite a paper).
2. Add `<model>` to `tune_all.SUPPORTED_MODELS`.
3. Append a paper-quote row to this file under a new section.
4. Run `python tune_all.py --validate-only` and `python -m src.tests.test_priors`.
