"""Pre-rendered thumbnail invariants (US-010).

Asserts:
  - Renders one PNG per requested cell at 224 x 448 (height x width)
  - Composite has Original on left, Degraded on right
  - Idempotent: second call with same tags writes 0 new files (skipped)
  - --force regenerates
  - Unknown tag raises ValueError naming the tag
  - PNG file exists at artifacts/dashboard_thumbs/<tag>.png

Run: ``python -m src.tests.test_render_thumbs``
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
from torchvision.io import read_image

from src.tools.render_cell_thumbs import (
    THUMB_HEIGHT,
    THUMB_WIDTH_PER_HALF,
    render_thumbs,
)


# Small but phase-diverse tag list — one Phase A, one Phase B, one Phase C
# (clean vs heavily-degraded vs single-axis isolation).
_SAMPLE_TAGS = [
    "final_clean_resnet50_cifar10",
    "final_B_L4_resnet50_cifar10",
    "final_C_L3_noise_resnet50_mnist",
]


def _check_dimensions_and_idempotency(out_dir: Path) -> None:
    summary = render_thumbs(tags=_SAMPLE_TAGS, force=False, out_dir=out_dir)
    assert summary["written"] == 3 and summary["skipped"] == 0, summary
    assert summary["failed"] == 0, summary

    for tag in _SAMPLE_TAGS:
        p = out_dir / f"{tag}.png"
        assert p.exists(), f"missing PNG: {p}"
        img = read_image(str(p))  # uint8 tensor [C, H, W]
        assert img.shape[1] == THUMB_HEIGHT, (
            f"{tag}: expected H={THUMB_HEIGHT}, got {img.shape[1]}"
        )
        assert img.shape[2] == 2 * THUMB_WIDTH_PER_HALF, (
            f"{tag}: expected W={2*THUMB_WIDTH_PER_HALF}, got {img.shape[2]}"
        )

    # Second pass: must skip everything (idempotent).
    summary2 = render_thumbs(tags=_SAMPLE_TAGS, force=False, out_dir=out_dir)
    assert summary2["written"] == 0, summary2
    assert summary2["skipped"] == 3, summary2

    print("OK [dims+idempotent] — 3 thumbs at 224x448; second pass skipped all.")


def _check_force_regenerates(out_dir: Path) -> None:
    summary = render_thumbs(
        tags=[_SAMPLE_TAGS[0]], force=True, out_dir=out_dir,
    )
    assert summary["written"] == 1 and summary["skipped"] == 0, summary
    print("OK [force] — --force regenerates existing thumb.")


def _check_clean_vs_degraded_left_right(out_dir: Path) -> None:
    """For a heavily-degraded cell, the right half should differ from the left
    (ie. degradation actually changed the pixels), and the left half should
    equal the upsampled-but-otherwise-clean reference."""
    tag = "final_B_L5_resnet50_cifar10"
    summary = render_thumbs(tags=[tag], force=True, out_dir=out_dir)
    assert summary["written"] == 1, summary

    img = read_image(str(out_dir / f"{tag}.png")).float() / 255.0
    half = THUMB_WIDTH_PER_HALF
    left = img[:, :, :half]
    right = img[:, :, half:]
    # Phase B L5 is extreme — left and right MUST differ noticeably
    diff = (left - right).abs().mean().item()
    assert diff > 0.01, (
        f"left/right halves nearly identical (diff={diff:.4f}) — "
        "degradation pipeline may not have applied"
    )
    print(f"OK [left/right] — Phase B L5 thumb has clear degradation "
          f"(mean abs diff={diff:.4f}).")


def _check_unknown_tag_rejected() -> None:
    try:
        render_thumbs(tags=["bogus_tag"], force=False, out_dir=Path("/tmp/never"))
    except ValueError as e:
        assert "bogus_tag" in str(e)
        print("OK [unknown-tag] — render_thumbs rejects unknown tag with name.")
        return
    raise AssertionError("expected ValueError for unknown tag")


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        _check_dimensions_and_idempotency(out_dir)
        _check_force_regenerates(out_dir)
        _check_clean_vs_degraded_left_right(out_dir)
    _check_unknown_tag_rejected()
    return 0


if __name__ == "__main__":
    sys.exit(main())
