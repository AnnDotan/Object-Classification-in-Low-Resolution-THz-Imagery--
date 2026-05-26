"""Pre-render Original vs Degraded image pairs for the 186-cell dashboard (US-010).

For each cell in build_final_matrix(), saves a side-by-side PNG (224 x 448)
to artifacts/dashboard_thumbs/<tag>.png — the dashboard (US-011) loads
these without on-the-fly degradation, keeping page-load instant.

Sample index is fixed per dataset so the same source image is used across
every cell of that dataset, making visual comparison easy. CIFAR-10 uses
val index 13 (a horse — user asked 2026-05-13 for a visually distinct
class, not val[0] cat); MNIST stays at val index 0.

CLI:
    python -m src.tools.render_cell_thumbs                # all 186, skip existing
    python -m src.tools.render_cell_thumbs --force        # all 186, regenerate
    python -m src.tools.render_cell_thumbs --tags T1 T2   # subset
    python -m src.tools.render_cell_thumbs --phase B      # 30 Phase B tags
    python -m src.tools.render_cell_thumbs --phase C --axes noise          # 30 tags
    python -m src.tools.render_cell_thumbs --phase C --axes noise,salt_pepper  # 60 tags

Output dir is gitignored + claudeignored (US-013).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

from src.data.degrade import degrade_image
from src.data.degradation_levels import AXES
from src.experiments.matrix import build_final_matrix, cells_by_tag


THUMB_DIR = Path("artifacts/dashboard_thumbs")
THUMB_HEIGHT = 224
THUMB_WIDTH_PER_HALF = 224
# 2026-05-13: user asked for a different CIFAR-10 sample than val[0] (cat).
# val[13] is a horse — visually distinct, deterministic, and matches the
# "e.g. 7 = horse" hint the user gave in the planning round (it was the
# class they had in mind, just at a different index in the test split).
# MNIST val[0] is a clean "7" — kept.
SAMPLE_INDEX_PER_DATASET: dict[str, int] = {
    "cifar10": 13,
    "mnist": 0,
}


# Cache the val split per dataset so we don't re-instantiate (and re-download)
# for every one of 186 cells.
_VAL_DATASET_CACHE: dict[str, object] = {}
# One-shot per-process audit log so the user sees which class each
# dataset's sample resolved to. Keyed by dataset name.
_SAMPLE_AUDIT_LOGGED: set[str] = set()


def _val_dataset(name: str, root: str = "./data"):
    if name in _VAL_DATASET_CACHE:
        return _VAL_DATASET_CACHE[name]
    tf = transforms.ToTensor()
    if name == "cifar10":
        ds = datasets.CIFAR10(root=root, train=False, download=True, transform=tf)
    elif name == "mnist":
        ds = datasets.MNIST(root=root, train=False, download=True, transform=tf)
    else:
        raise ValueError(f"unknown dataset {name!r}")
    _VAL_DATASET_CACHE[name] = ds
    return ds


def _to_3channel_at(img: torch.Tensor, size: int) -> torch.Tensor:
    """MNIST -> 3 channels; resize to (3, size, size); clamp to [0,1]."""
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    if img.shape[-1] != size:
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return img.clamp(0, 1)


def _build_pair(spec) -> torch.Tensor:
    """Return a (3, 224, 448) tensor: original on left, degraded on right."""
    ds = _val_dataset(spec.dataset)
    sample_idx = SAMPLE_INDEX_PER_DATASET[spec.dataset]
    img, label = ds[sample_idx]

    if spec.dataset not in _SAMPLE_AUDIT_LOGGED:
        classes = getattr(ds, "classes", None)
        class_name = classes[label] if classes else str(label)
        print(
            f"[render_cell_thumbs] {spec.dataset}: sampled idx={sample_idx} "
            f"class={class_name}"
        )
        _SAMPLE_AUDIT_LOGGED.add(spec.dataset)

    clean = _to_3channel_at(img, THUMB_HEIGHT)

    # Match the val seed offset used in training: degraded pixels are
    # what the model actually sees. The exact seed isn't critical here
    # (any one image; the dashboard is qualitative) but using the val
    # offset keeps the thumbs consistent with the val stream.
    if img.shape[0] == 1:
        img_3 = img.repeat(3, 1, 1)
    else:
        img_3 = img
    from src.data.degrade import SEED_OFFSET_VAL
    seed = sample_idx + SEED_OFFSET_VAL
    degraded = degrade_image(img_3, spec.degrade_config, seed=seed).clamp(0, 1)

    pair = torch.cat([clean, degraded], dim=2)  # along width
    assert pair.shape == (3, THUMB_HEIGHT, 2 * THUMB_WIDTH_PER_HALF), pair.shape
    return pair


def _parse_axes_arg(value: str) -> tuple[str, ...]:
    """argparse type= callable: comma-separated axis names validated against AXES.

    Mirrors ``scripts/run_ralph_loop.py:_parse_axes_arg`` (US-019) so the
    --axes flag accepts the same input shape on both the runner and the
    thumb renderer (US-019B dispatch chain).
    """
    parts = tuple(p.strip() for p in value.split(",") if p.strip())
    if not parts:
        raise argparse.ArgumentTypeError(
            f"--axes requires at least one axis; expected subset of {list(AXES)}"
        )
    invalid = [a for a in parts if a not in AXES]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unknown axis/axes {invalid!r}; expected subset of {list(AXES)}"
        )
    return parts


def render_thumbs(
    tags: Optional[Iterable[str]] = None,
    force: bool = False,
    out_dir: Path = THUMB_DIR,
    phase: Optional[str] = None,
    axes: Optional[Iterable[str]] = None,
) -> dict:
    """Render thumbnails for the given tags (or all 186 if tags is None).

    Filters compose: `tags` (explicit subset), `phase` (one of A/B/C), and
    `axes` (subset of `src.data.degradation_levels.AXES`, Phase C only) all
    AND together. Passing `axes` with a non-C phase is rejected to match
    the run_ralph_loop.py --axes contract (US-019).

    Returns a {written, skipped, failed} count summary.
    """
    if axes is not None and phase is not None and phase != "C":
        raise ValueError(
            f"axes filter is only valid with phase='C'; got phase={phase!r} "
            "(axes are a Phase C single-axis-isolation concept)"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix = build_final_matrix(include_phase_d=True)
    if tags is not None:
        tag_set = set(tags)
        by_tag = cells_by_tag(matrix)
        unknown = tag_set - by_tag.keys()
        if unknown:
            raise ValueError(f"unknown tags: {sorted(unknown)[:3]}")
        matrix = [by_tag[t] for t in tag_set]
    if phase is not None:
        matrix = [c for c in matrix if c.phase == phase]
    if axes is not None:
        axes_set = set(axes)
        matrix = [c for c in matrix if c.axis in axes_set]

    written = skipped = failed = 0
    for spec in matrix:
        target = out_dir / f"{spec.tag}.png"
        if target.exists() and not force:
            skipped += 1
            continue
        try:
            pair = _build_pair(spec)
            save_image(pair, target)
            written += 1
        except Exception as e:
            print(f"[FAIL] {spec.tag}: {type(e).__name__}: {e}", file=sys.stderr)
            failed += 1

    return {"written": written, "skipped": skipped, "failed": failed,
            "total": len(matrix), "out_dir": str(out_dir)}


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pre-render Original vs Degraded thumbs for the 186-cell dashboard."
    )
    p.add_argument(
        "--tags", nargs="+", default=None,
        help="Render only these cell tags (default: all 186).",
    )
    p.add_argument(
        "--phase", default=None, choices=["A", "B", "C", "D"],
        help="Restrict rendering to cells of this phase (default: all phases — 276 with Phase D included).",
    )
    p.add_argument(
        "--axes", default=None, type=_parse_axes_arg,
        help="Comma-separated Phase C axes to render (subset of "
             f"{list(AXES)}). Only valid with --phase C.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-render even if the .png already exists.",
    )
    p.add_argument(
        "--out-dir", type=str, default=str(THUMB_DIR),
        help=f"Output directory (default: {THUMB_DIR}).",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    if args.axes is not None and args.phase != "C":
        phase_label = args.phase if args.phase is not None else "all"
        parser.error(
            f"--axes is only valid with --phase C; got --phase {phase_label} "
            "(axes are a Phase C single-axis-isolation concept)"
        )
    summary = render_thumbs(
        tags=args.tags,
        force=args.force,
        out_dir=Path(args.out_dir),
        phase=args.phase,
        axes=args.axes,
    )
    print(
        f"thumbs: {summary['written']} written, {summary['skipped']} skipped, "
        f"{summary['failed']} failed (of {summary['total']}) -> {summary['out_dir']}"
    )
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
