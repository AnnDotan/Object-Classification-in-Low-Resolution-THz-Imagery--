"""Stage -1 — GPU environment bootstrap for the 186-cell campaign.

Scope (THz Protocol):
    Environment preparation ONLY. This script never touches experimental
    logic, degradation tables, or `runs/`. It audits hardware/drivers,
    builds the correct PyTorch install command for the detected CUDA
    toolkit, syncs project dependencies, and runs a deterministic smoke
    test before Stage 1 (Optuna pre-tune) is allowed to begin.

Fail-fast contract:
    Any of the following stops the script with a non-zero exit and a
    human-readable explanation — never a silent fallback:
      * `nvidia-smi` not on PATH or returns no devices.
      * Active interpreter is Python >= 3.13 (no PyTorch wheels yet —
        the 3.14 incompatibility flagged by MASTER on 2026-05-07).
      * `pip install` fails for torch/torchvision or any project dep.
      * `torch.cuda.is_available()` is False after install.
      * Detected GPU has < `--min-vram-gib` (default 6 GiB) usable VRAM.

CLI:
    python scripts/setup_gpu_env.py                     # full bootstrap
    python scripts/setup_gpu_env.py --audit-only        # report only, no install
    python scripts/setup_gpu_env.py --skip-install      # audit + smoke test
    python scripts/setup_gpu_env.py --cuda-override 12.1
    python scripts/setup_gpu_env.py --min-vram-gib 8
    python scripts/setup_gpu_env.py --make-venv .venv-gpu  # bootstrap a 3.11 venv

Notes:
    * Stdlib only at import time — torch is loaded inside the smoke test
      after install completes.
    * Idempotent: re-running on a healthy box exits 0 after the audit.
"""
from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

# Highest Python minor version with PyTorch wheels at the time of writing.
# Bump when upstream publishes 3.13/3.14 wheels.
MAX_SUPPORTED_PY_MINOR = 12
RECOMMENDED_PY_MINORS = (11, 10)

# Known CUDA -> PyTorch wheel-index mapping. Order matters (highest first)
# for the "fall back to nearest <= system" heuristic in `pytorch_index_for_cuda`.
KNOWN_CUDA_INDICES: tuple[tuple[tuple[int, int], str], ...] = (
    ((12, 8), "cu128"),
    ((12, 6), "cu126"),
    ((12, 4), "cu124"),
    ((12, 1), "cu121"),
    ((11, 8), "cu118"),
)

PROJECT_DEPS = (
    "pytorch-lightning",
    "optuna",
    "optuna-integration[pytorch-lightning]",
    "plotly",
    "pandas",
    "opencv-python",
    "matplotlib",
    "timm",
    "scikit-learn",
    "torchmetrics",
    "rich",
    "pyyaml",
    "tqdm",
)

REQUIRED_PATHS = (
    "src",
    "src/lightning",
    "src/data",
    "src/models",
    "scripts",
    "artifacts",
    "docs/runbooks/PHASE_B_EXECUTION.md",
)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _section(title: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n{title}\n{bar}", flush=True)


def _ok(msg: str) -> None:
    print(f"  [OK]    {msg}", flush=True)


def _warn(msg: str) -> None:
    print(f"  [WARN]  {msg}", flush=True)


def _fatal(msg: str, code: int = 1) -> None:
    print(f"\n[FATAL] {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def _run(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    pretty = " ".join(cmd)
    print(f"  $ {pretty}", flush=True)
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=capture,
    )


# ---------------------------------------------------------------------------
# 1. hardware + driver audit
# ---------------------------------------------------------------------------
@dataclass
class GpuAudit:
    nvidia_smi: Path
    driver_version: str
    cuda_version: tuple[int, int]
    devices: list[str]


def audit_gpu(cuda_override: Optional[str]) -> GpuAudit:
    smi = shutil.which("nvidia-smi")
    if smi is None:
        _fatal(
            "nvidia-smi not found on PATH. Install the NVIDIA driver "
            "(https://www.nvidia.com/Download/index.aspx) and re-open the shell."
        )

    proc = subprocess.run(
        [smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        _fatal(
            "nvidia-smi reported no devices. Verify the driver is loaded "
            f"(stderr: {proc.stderr.strip() or '<empty>'})."
        )
    devices = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    driver_version = devices[0].split(",")[1].strip() if devices else "unknown"

    if cuda_override:
        cuda_tuple = _parse_cuda_string(cuda_override)
        _ok(f"CUDA version overridden via --cuda-override: {cuda_tuple[0]}.{cuda_tuple[1]}")
    else:
        cuda_tuple = _detect_cuda_version(smi)

    _ok(f"nvidia-smi found at {smi}")
    _ok(f"Driver version: {driver_version}")
    _ok(f"CUDA toolkit (runtime): {cuda_tuple[0]}.{cuda_tuple[1]}")
    for dev in devices:
        _ok(f"Device: {dev}")
    return GpuAudit(Path(smi), driver_version, cuda_tuple, devices)


def _detect_cuda_version(smi_path: str) -> tuple[int, int]:
    proc = subprocess.run([smi_path], text=True, capture_output=True)
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", proc.stdout)
    if match:
        return int(match.group(1)), int(match.group(2))

    # Fallback: nvcc, if present.
    nvcc = shutil.which("nvcc")
    if nvcc:
        proc = subprocess.run([nvcc, "--version"], text=True, capture_output=True)
        match = re.search(r"release\s+(\d+)\.(\d+)", proc.stdout)
        if match:
            return int(match.group(1)), int(match.group(2))

    _fatal(
        "Could not parse a CUDA version from nvidia-smi or nvcc. "
        "Re-run with --cuda-override <X.Y> after confirming the toolkit version."
    )
    return (0, 0)  # unreachable


def _parse_cuda_string(raw: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", raw.strip())
    if not match:
        _fatal(f"--cuda-override value '{raw}' is not in 'X.Y' form.")
    return int(match.group(1)), int(match.group(2))


def pytorch_index_for_cuda(cuda: tuple[int, int]) -> str:
    for known, tag in KNOWN_CUDA_INDICES:
        if cuda >= known:
            return tag
    _fatal(
        f"Detected CUDA {cuda[0]}.{cuda[1]} is older than the minimum "
        f"supported {KNOWN_CUDA_INDICES[-1][0][0]}.{KNOWN_CUDA_INDICES[-1][0][1]}. "
        "Upgrade the driver or pin a legacy torch wheel manually."
    )
    return ""  # unreachable


# ---------------------------------------------------------------------------
# 2. python audit
# ---------------------------------------------------------------------------
def audit_python() -> None:
    info = sys.version_info
    print(f"  Interpreter: {sys.executable}")
    print(f"  Version    : {info.major}.{info.minor}.{info.micro}")
    print(f"  Platform   : {platform.platform()}")

    if info.major != 3:
        _fatal(f"Python 3 required; found {info.major}.{info.minor}.")

    if info.minor > MAX_SUPPORTED_PY_MINOR:
        rec = ", ".join(f"3.{m}" for m in RECOMMENDED_PY_MINORS)
        _fatal(
            f"Python 3.{info.minor} has no PyTorch wheels yet "
            f"(highest supported: 3.{MAX_SUPPORTED_PY_MINOR}). "
            f"Create a venv with one of: {rec}.\n"
            "  Example (Windows, py launcher):\n"
            "    py -3.11 -m venv .venv-gpu\n"
            "    .venv-gpu\\Scripts\\activate\n"
            f"    python {Path(__file__).as_posix()}\n"
            "  Or have this script bootstrap one for you:\n"
            f"    python {Path(__file__).as_posix()} --make-venv .venv-gpu"
        )

    if info.minor < min(RECOMMENDED_PY_MINORS):
        _warn(
            f"Python 3.{info.minor} is older than the recommended 3.10/3.11. "
            "Build wheels exist but Lightning 2.5+ may drop support soon."
        )
    _ok(f"Python 3.{info.minor} is within the supported PyTorch wheel range.")


# ---------------------------------------------------------------------------
# 3. pytorch install
# ---------------------------------------------------------------------------
def install_pytorch(cuda: tuple[int, int], dry_run: bool) -> None:
    tag = pytorch_index_for_cuda(cuda)
    index_url = f"https://download.pytorch.org/whl/{tag}"
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--upgrade",
        "torch", "torchvision", "torchaudio",
        "--extra-index-url", index_url,
    ]
    print(f"  PyTorch wheel index: {index_url}")
    if dry_run:
        print(f"  [dry-run] would execute: {' '.join(cmd)}")
        return
    _run(cmd)
    _ok("torch / torchvision / torchaudio installed.")


# ---------------------------------------------------------------------------
# 4. project deps + requirements.txt sync
# ---------------------------------------------------------------------------
def install_project_deps(dry_run: bool) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", *PROJECT_DEPS]
    if dry_run:
        print(f"  [dry-run] would execute: {' '.join(cmd)}")
        return
    _run(cmd)
    _ok(f"Installed/updated {len(PROJECT_DEPS)} project dependencies.")


def sync_requirements(dry_run: bool) -> None:
    target = REPO_ROOT / "requirements.txt"
    if not target.exists():
        _warn(f"requirements.txt not found at {target}; skipping sync.")
        return
    if dry_run:
        print("  [dry-run] would refresh requirements.txt via pip freeze")
        return
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        text=True,
        capture_output=True,
        check=True,
    )
    # Write UTF-8 to a sibling file rather than overwriting the existing
    # UTF-16 file in-place — preserves the historical artifact and gives
    # the operator a clean diff to review.
    refreshed = REPO_ROOT / "requirements.lock.txt"
    refreshed.write_text(proc.stdout, encoding="utf-8")
    _ok(f"Wrote refreshed lockfile to {refreshed.relative_to(REPO_ROOT)}.")
    _warn(
        "requirements.txt is UTF-16-encoded (legacy). Compare against "
        "requirements.lock.txt and re-encode as UTF-8 in a separate commit."
    )


# ---------------------------------------------------------------------------
# 5. smoke test
# ---------------------------------------------------------------------------
def smoke_test(min_vram_gib: float) -> None:
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:
        _fatal(f"Failed to import torch after install: {exc}")

    print(f"  torch version : {torch.__version__}")
    print(f"  CUDA build    : {torch.version.cuda}")
    print(f"  cuDNN         : {torch.backends.cudnn.version()}")

    if not torch.cuda.is_available():
        _fatal("torch.cuda.is_available() is False — driver/CUDA mismatch.")

    n = torch.cuda.device_count()
    _ok(f"torch reports {n} CUDA device(s).")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        gib = props.total_memory / (1024 ** 3)
        print(f"    [{i}] {props.name} — {gib:.2f} GiB, sm_{props.major}{props.minor}")
        if gib < min_vram_gib:
            _fatal(
                f"GPU {i} has {gib:.2f} GiB < required {min_vram_gib:.1f} GiB. "
                "ResNet50 / DenseNet121 at batch 32 will OOM. "
                "Lower --min-vram-gib only after MASTER signs off on a smaller batch."
            )

    # Tiny end-to-end allocation: confirms the runtime can actually launch
    # a kernel, not just enumerate devices.
    x = torch.randn(64, 64, device="cuda")
    y = (x @ x.t()).sum().item()
    assert y == y, "NaN sentinel from CUDA matmul"  # NaN != NaN; trips on driver bugs
    _ok("Allocated + multiplied a 64x64 tensor on device 0.")


def audit_paths() -> None:
    missing = [p for p in REQUIRED_PATHS if not (REPO_ROOT / p).exists()]
    if missing:
        _fatal(
            "Required project paths are missing — are you running from the "
            f"wrong checkout?\n  Repo root: {REPO_ROOT}\n  Missing: {missing}"
        )
    _ok(f"All {len(REQUIRED_PATHS)} required project paths exist under {REPO_ROOT}.")


# ---------------------------------------------------------------------------
# optional venv bootstrap
# ---------------------------------------------------------------------------
def make_venv(target: Path) -> None:
    if target.exists():
        _fatal(f"--make-venv target {target} already exists; refusing to overwrite.")

    candidates: list[list[str]] = []
    if os.name == "nt":
        # Prefer the py launcher if present.
        py = shutil.which("py")
        if py:
            for minor in RECOMMENDED_PY_MINORS:
                candidates.append([py, f"-3.{minor}", "-m", "venv", str(target)])
    for minor in RECOMMENDED_PY_MINORS:
        exe = shutil.which(f"python3.{minor}") or shutil.which(f"python{minor}")
        if exe:
            candidates.append([exe, "-m", "venv", str(target)])

    if not candidates:
        rec = ", ".join(f"3.{m}" for m in RECOMMENDED_PY_MINORS)
        _fatal(
            f"No Python in {rec} found on PATH. Install one from "
            "https://www.python.org/downloads/ and retry."
        )

    last_err: Optional[Exception] = None
    for cmd in candidates:
        try:
            _run(cmd)
            break
        except subprocess.CalledProcessError as exc:
            last_err = exc
            continue
    else:
        _fatal(f"All venv bootstrap attempts failed. Last error: {last_err}")

    activate_hint = (
        f"{target}\\Scripts\\activate" if os.name == "nt"
        else f"source {target}/bin/activate"
    )
    print(
        f"\n  Venv created at {target}.\n"
        f"  Activate it and re-run this script:\n"
        f"    {activate_hint}\n"
        f"    python {Path(__file__).as_posix()}\n"
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage -1 GPU environment bootstrap (THz Protocol).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--audit-only", action="store_true",
                   help="Report hardware + Python audit, then exit (no install, no smoke test).")
    p.add_argument("--skip-install", action="store_true",
                   help="Skip pip install steps; run audit + smoke test only.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print pip commands without executing them.")
    p.add_argument("--cuda-override", default=None,
                   help="Force CUDA version (X.Y) instead of auto-detect.")
    p.add_argument("--min-vram-gib", type=float, default=6.0,
                   help="Fail-fast threshold for per-GPU VRAM (CNN batch-32 floor).")
    p.add_argument("--make-venv", default=None, type=Path,
                   help="Bootstrap a Python 3.10/3.11 venv at this path and exit.")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if args.make_venv is not None:
        _section("Bootstrap Python 3.10/3.11 venv")
        make_venv(args.make_venv)
        return 0

    _section("Stage -1 / Step 1 — Hardware & driver audit")
    gpu = audit_gpu(args.cuda_override)

    _section("Stage -1 / Step 2 — Python interpreter audit")
    audit_python()

    _section("Stage -1 / Step 3 — Project paths audit")
    audit_paths()

    if args.audit_only:
        _section("Audit-only mode — exiting before install / smoke test")
        return 0

    if not args.skip_install:
        _section("Stage -1 / Step 4 — PyTorch install (CUDA-matched wheels)")
        install_pytorch(gpu.cuda_version, dry_run=args.dry_run)

        _section("Stage -1 / Step 5 — Project dependency sync")
        install_project_deps(dry_run=args.dry_run)
        sync_requirements(dry_run=args.dry_run)
    else:
        _warn("--skip-install set; not invoking pip.")

    if args.dry_run:
        _section("Dry-run complete — smoke test skipped")
        return 0

    _section("Stage -1 / Step 6 — Deterministic smoke test")
    smoke_test(min_vram_gib=args.min_vram_gib)

    _section("Stage -1 complete — environment ready for Stage 1 (Optuna pre-tune)")
    print("  Next: python tune_all.py --n-trials 20")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
