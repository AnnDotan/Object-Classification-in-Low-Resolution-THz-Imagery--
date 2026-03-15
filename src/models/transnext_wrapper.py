from __future__ import annotations

from pathlib import Path
import sys
import torch
import torch.nn as nn


# מוסיף את תיקיית ה-classification של TransNeXt ל-PYTHONPATH
_THIS_DIR = Path(__file__).resolve().parent
_TRANSNEXT_DIR = _THIS_DIR / "transnext_official"

if str(_TRANSNEXT_DIR) not in sys.path:
    sys.path.insert(0, str(_TRANSNEXT_DIR))


def create_transnext_model(
    model_name: str,
    num_classes: int = 10,
    pretrained: bool = False,
    checkpoint_path: str | None = None,
) -> nn.Module:
    """
    מחזיר מודל TransNeXt מתוך הקוד הרשמי.

    model_name אפשריים בשלב ראשון:
    - transnext_micro
    - transnext_tiny
    - transnext_small
    - transnext_base
    """

    # חשוב: ה-import אחרי עדכון sys.path
    import transnext as transnext_module

    # מיפוי שמות מהפרויקט שלך לפונקציות/בנאים של הריפו הרשמי
    name_map = {
        "transnext_micro": "transnext_micro",
        "transnext_tiny": "transnext_tiny",
        "transnext_small": "transnext_small",
        "transnext_base": "transnext_base",
    }

    if model_name not in name_map:
        raise ValueError(f"Unsupported TransNeXt model: {model_name}")

    ctor_name = name_map[model_name]
    if not hasattr(transnext_module, ctor_name):
        raise AttributeError(
            f"Could not find constructor '{ctor_name}' inside transnext.py. "
            f"Open transnext.py and verify the exact function names."
        )

    ctor = getattr(transnext_module, ctor_name)

    # ניסיון בסיסי: כמו מודלי timm - num_classes ניתן להחלפה
    try:
        model = ctor(num_classes=num_classes)
    except TypeError:
        # fallback אם החתימה שונה
        model = ctor()
        # מנסים להחליף ראש סיווג ידנית
        if hasattr(model, "head") and isinstance(model.head, nn.Module):
            in_features = getattr(model.head, "in_features", None)
            if in_features is not None:
                model.head = nn.Linear(in_features, num_classes)
            else:
                raise RuntimeError(
                    "Found model.head but could not read in_features."
                )
        else:
            raise RuntimeError(
                "Could not replace classification head automatically. "
                "Open transnext.py and locate the classifier layer."
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

        clean_state = {}
        for k, v in state.items():
            nk = k[7:] if k.startswith("module.") else k
            clean_state[nk] = v

        filtered_state = {
            k: v for k, v in clean_state.items()
            if not k.startswith("head.")
        }

        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        print("[TransNeXt] missing keys:", missing)
        print("[TransNeXt] unexpected keys:", unexpected)

    return model