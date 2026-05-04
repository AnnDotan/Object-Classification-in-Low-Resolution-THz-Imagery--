"""TransNeXt LP vs FT mode invariants (US-006 regression guard).

Two contracts:
  (a) lp mode (freeze_backbone=True): every backbone parameter has
      requires_grad=False, every head parameter has requires_grad=True.
  (b) ft mode (freeze_backbone=False): every parameter has requires_grad=True.

Uses pretrained=False so the test runs without network or real
checkpoints — it exercises only the freeze toggle, which is what US-006
actually changed. Auto-download is covered separately by an import smoke
test (the helper module loads cleanly).

Run: ``python -m src.tests.test_transnext_modes``
"""
from __future__ import annotations

import sys

import torch  # noqa: F401  — ensure torch is importable in this env

from src.lightning.module import THzClassifier


def _build(size: str, mode: str) -> THzClassifier:
    return THzClassifier(
        model_name=f"transnext_{size}",
        num_classes=10,
        pretrained=False,
        freeze_backbone=(mode == "lp"),
        scheduler_type="none",
        epochs=1,
    )


def _split_params(model) -> tuple[list, list]:
    """(backbone_params, head_params) from any model exposing `head`."""
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if "head" in name:
            head_params.append((name, p))
        else:
            backbone_params.append((name, p))
    assert head_params, "model has no 'head' params — cannot test LP/FT split"
    return backbone_params, head_params


def _check_lp_freezes_backbone() -> None:
    cls = _build("micro", "lp")
    backbone, head = _split_params(cls.model)
    bad_b = [n for n, p in backbone if p.requires_grad]
    bad_h = [n for n, p in head if not p.requires_grad]
    assert not bad_b, f"lp mode: backbone params still trainable: {bad_b[:3]}"
    assert not bad_h, f"lp mode: head params not trainable: {bad_h[:3]}"
    print(f"OK [lp] — backbone frozen ({len(backbone)} params), "
          f"head trainable ({len(head)} params).")


def _check_ft_unfreezes_all() -> None:
    cls = _build("micro", "ft")
    backbone, head = _split_params(cls.model)
    bad = [n for n, p in (backbone + head) if not p.requires_grad]
    assert not bad, f"ft mode: some params still frozen: {bad[:3]}"
    print(f"OK [ft] — all {len(backbone) + len(head)} params trainable.")


def _check_auto_download_helper_imports() -> None:
    """The auto-download helper must import cleanly and define every TransNeXt
    size's URL fallback. Network is not exercised — that path is environment-
    dependent and out of scope for unit tests."""
    from src.models import transnext_weights as tw

    for size in ("micro", "tiny", "small", "base"):
        url = tw._resolve_url(f"transnext_{size}")
        assert url, f"no fallback URL for transnext_{size}"
    print("OK [download-helper] — module imports, all 4 sizes have a URL fallback.")


def main() -> int:
    _check_auto_download_helper_imports()
    _check_lp_freezes_backbone()
    _check_ft_unfreezes_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
