from __future__ import annotations

import os
import sys
from functools import partial
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


# Vendored upstream TransNeXt sits under src/models/transnext_official/.
# It uses bare imports (e.g. `from transnext import ...`, `from attention_native ...`),
# so its parent directory must be on sys.path before any sub-import.
_THIS_DIR = Path(__file__).resolve().parent
_TRANSNEXT_DIR = _THIS_DIR / "transnext_official"
if str(_TRANSNEXT_DIR) not in sys.path:
    sys.path.insert(0, str(_TRANSNEXT_DIR))


def _maybe_force_native_attention() -> None:
    """Block the optional `swattention` CUDA extension so transnext.py falls
    back to `attention_native`.

    Required for `torch.compile`: the swattention path uses a custom CUDA op
    that Inductor cannot graph-capture. Setting sys.modules['swattention']=None
    makes `__import__('swattention')` raise ImportError, which is what
    transnext.py's `is_installed` helper catches.

    Idempotent — safe to call before every model build.
    """
    if os.environ.get("TRANSNEXT_FORCE_NATIVE", "0") == "1":
        # Intentional block-import idiom: setting sys.modules[name] = None makes
        # `import name` raise ImportError, which is what transnext.py's
        # `is_installed` helper catches. mypy types sys.modules as
        # dict[str, ModuleType] so the None is a deliberate violation.
        sys.modules.setdefault("swattention", None)  # type: ignore[arg-type]


# Per-variant model specs lifted from src/models/transnext_official/transnext.py.
# Kept here so the wrapper can override img_size / patch_size / pretrain_size
# without monkey-patching the upstream factory functions (which hardcode
# `patch_size=4` as a positional kwarg, blocking override via **kwargs).
#
# `_native` variants (PRD US-042): same architecture as their base counterpart
# but with native-32 defaults pre-baked into the spec dict. Functionally
# equivalent to passing the V3 native overrides at call time
# (img_size=32, patch_size=2). The matrix uses `_v3_cell_settings` on the bare
# `transnext_*` names; the `_native` aliases exist for the PRD's explicit
# `--model transnext_small_native` operator surface and for the Optuna
# priors file naming. Both routes converge on the same architecture.
_TRANSNEXT_SPECS: dict[str, dict] = {
    "transnext_micro": dict(embed_dims=[48, 96, 192, 384],  num_heads=[2, 4, 8, 16],  depths=[2, 2, 15, 2]),
    "transnext_tiny":  dict(embed_dims=[72, 144, 288, 576], num_heads=[3, 6, 12, 24], depths=[2, 2, 15, 2]),
    "transnext_small": dict(embed_dims=[72, 144, 288, 576], num_heads=[3, 6, 12, 24], depths=[5, 5, 22, 5]),
    "transnext_base":  dict(embed_dims=[96, 192, 384, 768], num_heads=[4, 8, 16, 32], depths=[5, 5, 23, 5]),
    # PRD US-042 native-resolution aliases. `default_img_size`, `default_patch_size`,
    # `default_pretrain_size` are consumed by `create_transnext_model` only when
    # the caller does NOT override them at call time.
    "transnext_micro_native": dict(
        embed_dims=[48, 96, 192, 384],  num_heads=[2, 4, 8, 16],  depths=[2, 2, 15, 2],
        default_img_size=32, default_patch_size=2, default_pretrain_size=None,
    ),
    "transnext_small_native": dict(
        embed_dims=[72, 144, 288, 576], num_heads=[3, 6, 12, 24], depths=[5, 5, 22, 5],
        default_img_size=32, default_patch_size=2, default_pretrain_size=None,
    ),
    "transnext_base_native": dict(
        embed_dims=[96, 192, 384, 768], num_heads=[4, 8, 16, 32], depths=[5, 5, 23, 5],
        default_img_size=32, default_patch_size=2, default_pretrain_size=None,
    ),
}

# Module-level sentinel — callers can introspect to check what's resolvable.
TRANSNEXT_NATIVE_VARIANTS: tuple[str, ...] = (
    "transnext_micro_native",
    "transnext_small_native",
    "transnext_base_native",
)
_TRANSNEXT_COMMON = dict(
    window_size=[3, 3, 3, None],
    mlp_ratios=[8, 8, 4, 4],
    qkv_bias=True,
    sr_ratios=[8, 4, 2, 1],
)


def create_transnext_model(
    model_name: str,
    num_classes: int = 10,
    pretrained: bool = False,
    checkpoint_path: Optional[str] = None,
    drop_path_rate: float = 0.0,
    img_size: Optional[int] = None,
    patch_size: Optional[int] = None,
    pretrain_size: Optional[int] = None,
) -> nn.Module:
    """Build a TransNeXt model with project-level overrides for native-res FT.

    V3 native path: `img_size=32, patch_size=2` for both CIFAR-10 and
    32x32-padded MNIST. Stage-1 grid 16x16, stage-4 grid 2x2 — every stage
    retains >=4 tokens. CPB MLPs and AggregatedAttention pool sizes are
    regenerated at construct time from (img_size, sr_ratio, pretrain_size).

    Defaults are resolved with this precedence (highest first):
      1. Explicit caller kwarg (non-None).
      2. `default_*` keys on the model's spec dict (PRD US-042 `_native` aliases).
      3. Wrapper-level fallback: img_size=224, patch_size=4.

    Args:
        img_size: spatial dim the model expects. None -> spec default or 224.
        patch_size: stage-1 stride. None -> spec default or 4.
        pretrain_size: scale used by `get_relative_position_cpb` to normalise
            relative coords for the CPB MLP. When loading 224-pretrained
            weights into a native-32 model, MUST be 224 so the CPB MLP stays
            in its training distribution. Defaults to 224 if `pretrained`
            else `img_size`.
    """
    if model_name not in _TRANSNEXT_SPECS:
        raise ValueError(f"Unsupported TransNeXt model: {model_name}")

    _maybe_force_native_attention()
    import transnext as transnext_module  # late import; sys.path must be ready

    TransNeXt = transnext_module.TransNeXt
    spec = _TRANSNEXT_SPECS[model_name]

    # Resolve defaults from spec when caller left them None (PRD US-042).
    if img_size is None:
        img_size = spec.get("default_img_size", 224)
    if patch_size is None:
        patch_size = spec.get("default_patch_size", 4)
    if pretrain_size is None:
        spec_default = spec.get("default_pretrain_size", "__unset__")
        if spec_default == "__unset__":
            pretrain_size = 224 if pretrained else img_size
        else:
            pretrain_size = spec_default if spec_default is not None else (
                224 if pretrained else img_size
            )

    model = TransNeXt(
        img_size=img_size,
        pretrain_size=pretrain_size,
        patch_size=patch_size,
        in_chans=3,
        num_classes=num_classes,
        embed_dims=spec["embed_dims"],
        num_heads=spec["num_heads"],
        depths=spec["depths"],
        drop_path_rate=drop_path_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **_TRANSNEXT_COMMON,
    )  # `default_img_size`/`default_patch_size`/`default_pretrain_size` are
    # intentionally NOT forwarded — they're spec-level metadata for the
    # wrapper's default resolution, never upstream constructor kwargs.

    if pretrained:
        if checkpoint_path is None:
            raise ValueError("pretrained=True requires checkpoint_path for TransNeXt")

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                state = ckpt["state_dict"]
            elif "model" in ckpt:
                state = ckpt["model"]
            else:
                state = ckpt
        else:
            state = ckpt

        clean_state: dict[str, torch.Tensor] = {}
        for k, v in state.items():
            nk = k[7:] if k.startswith("module.") else k
            clean_state[nk] = v

        model_state = model.state_dict()
        filtered_state: dict[str, torch.Tensor] = {}
        skipped_head: list[str] = []
        skipped_shape: list[tuple[str, tuple, tuple]] = []

        for k, v in clean_state.items():
            if k.startswith("head."):
                skipped_head.append(k)
                continue
            if k not in model_state:
                continue  # surfaces as `unexpected` from load_state_dict below
            target_shape = tuple(model_state[k].shape)
            source_shape = tuple(v.shape)
            if target_shape != source_shape:
                skipped_shape.append((k, source_shape, target_shape))
                continue
            filtered_state[k] = v

        missing, unexpected = model.load_state_dict(filtered_state, strict=False)

        print(
            f"[TransNeXt] {model_name} img_size={img_size} "
            f"patch_size={patch_size} pretrain_size={pretrain_size}"
        )
        print(f"[TransNeXt] skipped head keys ({len(skipped_head)}): {skipped_head}")
        if skipped_shape:
            print(
                f"[TransNeXt] shape-mismatched keys RANDOM-INITIALISED "
                f"({len(skipped_shape)}):"
            )
            for k, src, tgt in skipped_shape:
                print(f"  {k}: ckpt {src} -> model {tgt}")
        print(f"[TransNeXt] missing keys ({len(missing)}): {missing}")
        print(f"[TransNeXt] unexpected keys ({len(unexpected)}): {unexpected}")

    return model
