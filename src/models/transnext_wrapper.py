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
_TRANSNEXT_SPECS: dict[str, dict] = {
    "transnext_micro": dict(embed_dims=[48, 96, 192, 384],  num_heads=[2, 4, 8, 16],  depths=[2, 2, 15, 2]),
    "transnext_tiny":  dict(embed_dims=[72, 144, 288, 576], num_heads=[3, 6, 12, 24], depths=[2, 2, 15, 2]),
    "transnext_small": dict(embed_dims=[72, 144, 288, 576], num_heads=[3, 6, 12, 24], depths=[5, 5, 22, 5]),
    "transnext_base":  dict(embed_dims=[96, 192, 384, 768], num_heads=[4, 8, 16, 32], depths=[5, 5, 23, 5]),
}

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
    """Build a TransNeXt model at the upstream 224x224 default.

    The 224 path keeps the upstream paper's CPB MLP coord distribution intact
    and matches the pretrained checkpoints distributed at img_size=224,
    patch_size=4. CIFAR-10 (32) and MNIST (28) inputs are upsampled to 224
    by the data pipeline before reaching the model.

    Args:
        img_size: spatial dim the model expects. Default 224.
        patch_size: stage-1 stride. Default 4.
        pretrain_size: scale used by `get_relative_position_cpb` to normalise
            relative coords for the CPB MLP. Defaults to 224 if `pretrained`
            else `img_size`.
    """
    if model_name not in _TRANSNEXT_SPECS:
        raise ValueError(f"Unsupported TransNeXt model: {model_name}")

    _maybe_force_native_attention()
    import transnext as transnext_module  # late import; sys.path must be ready

    TransNeXt = transnext_module.TransNeXt
    spec = _TRANSNEXT_SPECS[model_name]

    if img_size is None:
        img_size = 224
    if patch_size is None:
        patch_size = 4
    if pretrain_size is None:
        pretrain_size = 224 if pretrained else img_size

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
    )

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
