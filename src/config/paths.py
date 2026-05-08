"""Local-only path configuration.

All paths are resolved relative to the repository root. There is no cloud
sync, no Drive split, and no hybrid logic — the project lives in a single
local checkout.

Override the defaults with the ``THZ_DATA_PATH`` and ``THZ_CHECKPOINTS_PATH``
environment variables if your install keeps these directories outside the
repo.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_PATH: Path = Path(os.environ.get("THZ_DATA_PATH", REPO_ROOT / "data"))
CHECKPOINTS_PATH: Path = Path(
    os.environ.get("THZ_CHECKPOINTS_PATH", REPO_ROOT / "checkpoints")
)
