"""Hybrid Cloud-Local path configuration.

The repository is split between two physical locations:

* **Drive (synced to NotebookLM)** — code + docs + lightweight run artefacts
  (.json, .csv, .html). Lives under ``BASE_DRIVE_PATH``.
* **Local fast storage (PC only)** — raw datasets, model checkpoints, virtualenv.
  Lives under ``LOCAL_STORAGE_PATH``.

Override the defaults with environment variables ``THZ_DRIVE_PATH`` and
``THZ_LOCAL_PATH`` if your install is non-standard.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Roots ----------------------------------------------------------------

BASE_DRIVE_PATH: Path = Path(
    os.environ.get(
        "THZ_DRIVE_PATH",
        r"G:/My Drive/Object-Classification-in-Low-Resolution-THz-Imagery",
    )
)

LOCAL_STORAGE_PATH: Path = Path(
    os.environ.get("THZ_LOCAL_PATH", r"C:/Users/ib94/Documents/Object-Classification-in-Low-Resolution-THz-Imagery--")
)

# --- Drive-synced subtrees (code + lightweight artefacts) -----------------

DRIVE_SYNCED_DIRS: tuple[Path, ...] = (
    BASE_DRIVE_PATH / "src",
    BASE_DRIVE_PATH / "docs",
    BASE_DRIVE_PATH / "agents",
)

RUNS_DRIVE_PATH: Path = BASE_DRIVE_PATH / "runs"
RUNS_DRIVE_EXTENSIONS: frozenset[str] = frozenset({".json", ".csv", ".html"})

# --- Local-only subtrees (heavy binaries, never in Drive or Git) ----------

DATA_PATH: Path = LOCAL_STORAGE_PATH / "data"
CHECKPOINTS_PATH: Path = LOCAL_STORAGE_PATH / "checkpoints"
VENV_PATH: Path = LOCAL_STORAGE_PATH / "venv"

LOCAL_ONLY_DIRS: tuple[Path, ...] = (DATA_PATH, CHECKPOINTS_PATH, VENV_PATH)

# --- Helpers --------------------------------------------------------------


def is_drive_synced(path: os.PathLike[str] | str) -> bool:
    """Return True if ``path`` lives under a Drive-synced root.

    A file inside ``runs/`` only counts as synced when its extension is in
    :data:`RUNS_DRIVE_EXTENSIONS` — heavy artefacts in ``runs/`` stay local.
    """
    p = Path(path).resolve()
    if any(p == d or d in p.parents for d in DRIVE_SYNCED_DIRS):
        return True
    if RUNS_DRIVE_PATH in p.parents and p.suffix.lower() in RUNS_DRIVE_EXTENSIONS:
        return True
    return False


def is_local_only(path: os.PathLike[str] | str) -> bool:
    """Return True if ``path`` lives under a local-only root."""
    p = Path(path).resolve()
    return any(p == d or d in p.parents for d in LOCAL_ONLY_DIRS)
