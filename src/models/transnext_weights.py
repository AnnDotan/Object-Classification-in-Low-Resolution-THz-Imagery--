"""Pretrained-weight discovery + auto-download for TransNeXt.

Looks under `artifacts/weights/{model_name}_224_1k.pth`. If the file is
missing and `pretrained=True`, attempts a best-effort HTTP download from
the official GitHub release. Each size's URL can be overridden via the
`THZ_TRANSNEXT_<SIZE>_URL` env var (e.g. `THZ_TRANSNEXT_BASE_URL=...`).

If both the local file and the download fail, raises a `FileNotFoundError`
with the exact path and a manual-download instruction. The campaign never
silently trains on randomly-initialized weights when `pretrained=True`.
"""
from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Best-effort default URLs (TransNeXt official GitHub release).
# Override per-size via env var if these paths change upstream.
_DEFAULT_URLS: dict[str, str] = {
    "transnext_micro": "https://github.com/DaiShiResearch/TransNeXt/releases/download/checkpoint/transnext_micro_224_1k.pth",
    "transnext_tiny":  "https://github.com/DaiShiResearch/TransNeXt/releases/download/checkpoint/transnext_tiny_224_1k.pth",
    "transnext_small": "https://github.com/DaiShiResearch/TransNeXt/releases/download/checkpoint/transnext_small_224_1k.pth",
    "transnext_base":  "https://github.com/DaiShiResearch/TransNeXt/releases/download/checkpoint/transnext_base_224_1k.pth",
}


def _expected_path(model_name: str, weights_dir: Path) -> Path:
    return weights_dir / f"{model_name}_224_1k.pth"


def _resolve_url(model_name: str) -> str | None:
    suffix = model_name.removeprefix("transnext_").upper()
    return os.environ.get(
        f"THZ_TRANSNEXT_{suffix}_URL",
        _DEFAULT_URLS.get(model_name),
    )


def _try_download(url: str, target: Path) -> bool:
    """Download `url` to `target`. Returns True on success, False on failure."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            with open(tmp, "wb") as f:
                # 1 MiB chunks — TransNeXt-Base is ~360 MiB so streaming matters
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
        tmp.replace(target)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        print(f"[transnext_weights] download failed: {type(e).__name__}: {e}",
              file=sys.stderr)
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        return False


def find_or_download_weights(
    model_name: str,
    weights_dir: Path | None = None,
) -> Path:
    """Return a Path to the pretrained checkpoint, downloading if needed.

    Raises FileNotFoundError if the file is absent and the download fails;
    the exception message includes the expected on-disk path and a clear
    instruction so the user can drop the file in manually.
    """
    weights_dir = weights_dir or Path("artifacts/weights")
    target = _expected_path(model_name, weights_dir)
    if target.exists():
        return target

    url = _resolve_url(model_name)
    if url:
        print(f"[transnext_weights] downloading {model_name} weights from {url}",
              file=sys.stderr)
        if _try_download(url, target):
            return target

    suffix = model_name.removeprefix("transnext_").upper()
    raise FileNotFoundError(
        f"Pretrained TransNeXt weights missing: {target}\n"
        f"  Auto-download URL was unreachable. To proceed, either:\n"
        f"    1) Download the official checkpoint from "
        f"https://github.com/DaiShiResearch/TransNeXt and place it at the path above, OR\n"
        f"    2) Set THZ_TRANSNEXT_{suffix}_URL to a working mirror, OR\n"
        f"    3) Pass pretrained=False (random init — only valid for sanity checks)."
    )
