"""Capture SHA-256 manifest of Phase B v2 metrics.json baselines.

US-028 pre-flight helper (PRD v3 §7 baseline-comparability invariant).
Produces artifacts/validation/phase_b_v2_baseline_manifest.json with
{tag: sha256(metrics.json)} for all 30 Phase B v2 cells. The manifest
is re-verified at US-031 plot rendering to prevent silent baseline
drift during the Phase D campaign.

Usage:
    python scripts/snapshot_phase_b_v2_baseline.py            # capture + write
    python scripts/snapshot_phase_b_v2_baseline.py --verify   # re-check hashes

Exit codes:
    0  manifest written / verification passed
    1  one or more Phase B metrics.json files missing
    2  hash mismatch on --verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.experiments.cells import iter_cells  # noqa: E402

RUNS_ROOT = REPO_ROOT / "runs" / "final"
MANIFEST_PATH = REPO_ROOT / "artifacts" / "validation" / "phase_b_v2_baseline_manifest.json"
SCHEMA_VERSION = 1
EXPECTED_PHASE_B_COUNT = 30


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _phase_b_metrics_paths() -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for meta in iter_cells():
        if meta.phase != "B":
            continue
        out.append((meta.tag, RUNS_ROOT / meta.tag / "metrics.json"))
    return out


def capture() -> int:
    entries: dict[str, dict[str, str]] = {}
    missing: list[str] = []
    for tag, metrics_path in _phase_b_metrics_paths():
        if not metrics_path.exists():
            missing.append(tag)
            continue
        entries[tag] = {
            "metrics_sha256": sha256_of(metrics_path),
            "metrics_path": metrics_path.relative_to(REPO_ROOT).as_posix(),
        }

    if missing:
        print(f"ERROR: {len(missing)} Phase B cells missing metrics.json:", file=sys.stderr)
        for t in missing:
            print(f"  - {t}", file=sys.stderr)
        return 1

    if len(entries) != EXPECTED_PHASE_B_COUNT:
        print(
            f"ERROR: expected {EXPECTED_PHASE_B_COUNT} Phase B v2 entries; got {len(entries)}",
            file=sys.stderr,
        )
        return 1

    payload = {
        "schema_version": SCHEMA_VERSION,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase_b_count": len(entries),
        "purpose": (
            "Phase B v2 baseline lock for the Phase D campaign "
            "(PRD v3 §7 baseline-comparability invariant)."
        ),
        "entries": entries,
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {MANIFEST_PATH.relative_to(REPO_ROOT)} with {len(entries)} entries.")
    return 0


def verify() -> int:
    if not MANIFEST_PATH.exists():
        print(f"ERROR: manifest not found at {MANIFEST_PATH}", file=sys.stderr)
        return 2

    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries: dict[str, dict[str, str]] = payload["entries"]

    drifts: list[str] = []
    for tag, metrics_path in _phase_b_metrics_paths():
        expected = entries.get(tag, {}).get("metrics_sha256")
        if expected is None:
            drifts.append(f"{tag}: missing from manifest")
            continue
        if not metrics_path.exists():
            drifts.append(f"{tag}: metrics.json deleted")
            continue
        actual = sha256_of(metrics_path)
        if actual != expected:
            drifts.append(f"{tag}: hash changed ({expected[:12]}... -> {actual[:12]}...)")

    if drifts:
        print(f"ERROR: {len(drifts)} baseline drift(s) detected:", file=sys.stderr)
        for d in drifts:
            print(f"  - {d}", file=sys.stderr)
        return 2

    print(f"OK: {len(entries)} Phase B v2 baselines unchanged since manifest capture.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--verify",
        action="store_true",
        help="Re-check on-disk metrics.json hashes against the manifest.",
    )
    args = ap.parse_args()
    return verify() if args.verify else capture()


if __name__ == "__main__":
    raise SystemExit(main())
