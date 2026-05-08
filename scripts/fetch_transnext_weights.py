"""Pre-fetch official TransNeXt pretrained weights for the campaign.

Run this once before Phase A to populate `artifacts/weights/`. The
training pipeline also auto-downloads on first use, but doing it
upfront keeps the first epoch latency-free and surfaces network
problems early.

Replaces tiny placeholder files (e.g. the 136-byte `transnext_micro_*.pth`
LFS pointer) since `find_or_download_weights()` now treats sub-1 MiB
files as invalid.

CLI:
    python scripts/fetch_transnext_weights.py                       # base only (campaign default)
    python scripts/fetch_transnext_weights.py --sizes base small    # multiple
    python scripts/fetch_transnext_weights.py --all                 # every size
    python scripts/fetch_transnext_weights.py --force               # re-download
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.transnext_weights import (  # noqa: E402
    MIN_VALID_BYTES,
    _expected_path,
    _looks_valid,
    _resolve_url,
    _try_download,
)

_KNOWN_SIZES: tuple[str, ...] = ("micro", "tiny", "small", "base")
_DEFAULT_WEIGHTS_DIR = Path("artifacts/weights")


def fetch(size: str, weights_dir: Path, force: bool) -> str:
    model_name = f"transnext_{size}"
    target = _expected_path(model_name, weights_dir)

    if _looks_valid(target) and not force:
        return f"skip   {target.name} ({target.stat().st_size:,} bytes, ok)"

    if target.exists() and not _looks_valid(target):
        size_b = target.stat().st_size
        print(
            f"[fetch] {target} is {size_b} bytes — below the {MIN_VALID_BYTES:,} "
            f"byte threshold; will overwrite with a real download.",
            file=sys.stderr,
        )

    url = _resolve_url(model_name)
    if not url:
        return f"FAIL   {model_name}: no URL configured"

    print(f"[fetch] {model_name}: downloading from {url}")
    if not _try_download(url, target):
        return f"FAIL   {model_name}: download error (see stderr)"
    if not _looks_valid(target):
        return (
            f"FAIL   {model_name}: downloaded file is only "
            f"{target.stat().st_size} bytes (URL returned a 404 page?)"
        )
    return f"ok     {target.name} ({target.stat().st_size:,} bytes)"


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pre-fetch TransNeXt pretrained weights into artifacts/weights/."
    )
    p.add_argument("--sizes", nargs="+", default=["base"], choices=list(_KNOWN_SIZES),
                   help="Sizes to fetch (default: base — the campaign default).")
    p.add_argument("--all", action="store_true",
                   help="Fetch every known size, overrides --sizes.")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if the file already passes the size check.")
    p.add_argument("--weights-dir", type=str, default=str(_DEFAULT_WEIGHTS_DIR),
                   help=f"Destination directory (default: {_DEFAULT_WEIGHTS_DIR}).")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    sizes = list(_KNOWN_SIZES) if args.all else args.sizes
    weights_dir = Path(args.weights_dir)

    print(f"[fetch] target dir: {weights_dir.resolve()}")
    results: list[str] = []
    for s in sizes:
        results.append(fetch(s, weights_dir, args.force))

    print()
    for r in results:
        print(r)
    return 1 if any(r.startswith("FAIL") for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
