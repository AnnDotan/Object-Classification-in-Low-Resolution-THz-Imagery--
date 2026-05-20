"""Snapshot v1 dashboard thumbnails of the 90 v2-affected cells (US-019B).

The 90 cells whose `noise_std > 0` OR `salt_pepper > 0` (PRD §1) are the
Phase B (30) + Phase C `noise` (30) + Phase C `salt_pepper` (30) tags.
Their `degrade_image` output changes under PIPELINE_VERSION=2 (US-017) so
the operator visual gate (US-019B) needs a v1 baseline to diff against.

PRD §3 ruled out a v1 *metrics* archive, but the v1 *thumbnails* are
load-bearing for the US-019B contact sheet and have no other home:
`artifacts/dashboard_thumbs/` is gitignored from the start (US-013) so
the PRD's `git show 3e1d089:...` recovery path doesn't actually work.

This script copies each v1 thumb to `artifacts/validation/v1_thumbs/<tag>.png`
and writes a SHA-256 manifest to `artifacts/validation/v1_thumbs_manifest.json`.
The PNG blobs themselves are gitignored (operator visual gate is a local
ritual); the manifest is committed as durable evidence of which v1 byte
sequence was the baseline at re-render time.

Idempotent: re-running compares hashes against the manifest. Mismatch =>
exit 1 with a diff list (proves nothing silently overwrote the baseline).

CLI:
    python scripts/snapshot_v1_thumbs.py            # snapshot + write manifest
    python scripts/snapshot_v1_thumbs.py --verify   # verify only, exit 1 on drift
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.degradation_levels import PIPELINE_VERSION  # noqa: E402
from src.experiments.cells import iter_cells  # noqa: E402

SRC_DIR = REPO_ROOT / "artifacts" / "dashboard_thumbs"
DST_DIR = REPO_ROOT / "artifacts" / "validation" / "v1_thumbs"
MANIFEST = REPO_ROOT / "artifacts" / "validation" / "v1_thumbs_manifest.json"

# The 90 v2-affected cells: Phase B (all 30) + Phase C noise/salt_pepper (60).
AFFECTED_AXES = {"noise", "salt_pepper"}


def affected_tags() -> list[str]:
    """Return the 90 cell tags whose v2 noise/S&P pipeline changes their pixels."""
    tags: list[str] = []
    for c in iter_cells():
        if c.phase == "B":
            tags.append(c.tag)
        elif c.phase == "C" and c.axis in AFFECTED_AXES:
            tags.append(c.tag)
    return tags


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def snapshot(tags: list[str]) -> dict[str, str]:
    """Copy v1 thumbs to v1_thumbs/ and return {tag: sha256} for the manifest."""
    DST_DIR.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    missing: list[str] = []
    for tag in tags:
        src = SRC_DIR / f"{tag}.png"
        if not src.exists():
            missing.append(tag)
            continue
        dst = DST_DIR / f"{tag}.png"
        shutil.copy2(src, dst)
        hashes[tag] = sha256_of(dst)
    if missing:
        raise SystemExit(
            f"v1 thumbs missing on disk for {len(missing)} tag(s); "
            f"first 3 = {missing[:3]}. Re-render v1 BEFORE bumping "
            "PIPELINE_VERSION, or this baseline is unrecoverable."
        )
    return hashes


def verify(tags: list[str], manifest: dict) -> int:
    """Hash each snapshot file and compare against the manifest."""
    recorded: dict[str, str] = manifest["sha256"]
    missing: list[str] = []
    drift: list[tuple[str, str, str]] = []
    for tag in tags:
        p = DST_DIR / f"{tag}.png"
        if not p.exists():
            missing.append(tag)
            continue
        actual = sha256_of(p)
        expected = recorded.get(tag)
        if expected != actual:
            drift.append((tag, expected or "<missing-from-manifest>", actual))
    if missing or drift:
        print(f"[verify] FAIL — {len(missing)} missing PNG(s), "
              f"{len(drift)} hash mismatch(es).", file=sys.stderr)
        for tag in missing[:5]:
            print(f"  missing: {tag}", file=sys.stderr)
        for tag, exp, got in drift[:5]:
            print(f"  drift:   {tag}\n    expected={exp}\n    actual  ={got}",
                  file=sys.stderr)
        return 1
    print(f"[verify] OK — {len(tags)} v1 thumbs match manifest sha256s.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing snapshot against the manifest "
                             "(no copy); exit 1 on drift.")
    args = parser.parse_args(argv)

    tags = affected_tags()
    assert len(tags) == 90, (
        f"expected 90 v2-affected cells (30 Phase B + 30 noise + 30 salt_pepper); "
        f"iter_cells() yielded {len(tags)} — matrix drift?"
    )

    if args.verify:
        if not MANIFEST.exists():
            print(f"[verify] no manifest at {MANIFEST.relative_to(REPO_ROOT)}; "
                  "run without --verify to write one.", file=sys.stderr)
            return 1
        manifest = json.loads(MANIFEST.read_text())
        return verify(tags, manifest)

    hashes = snapshot(tags)
    manifest_payload = {
        "schema_version": 1,
        "purpose": (
            "v1 dashboard thumbnail SHA-256 baseline for the 90 cells affected "
            "by PIPELINE_VERSION=2 noise/S&P pipeline move (US-017 / US-019B). "
            "v1 thumbs are gitignored; this manifest is the durable evidence "
            "of which v1 pixels were the comparison baseline."
        ),
        "pipeline_version_at_snapshot": int(PIPELINE_VERSION),
        "snapshot_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "src_dir": str(SRC_DIR.relative_to(REPO_ROOT)),
        "dst_dir": str(DST_DIR.relative_to(REPO_ROOT)),
        "cell_count": len(hashes),
        "sha256": dict(sorted(hashes.items())),
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest_payload, indent=2) + "\n")
    print(
        f"[snapshot] wrote {len(hashes)} v1 thumbs to "
        f"{DST_DIR.relative_to(REPO_ROOT)}/ and a SHA-256 manifest to "
        f"{MANIFEST.relative_to(REPO_ROOT)} (pipeline_version_at_snapshot="
        f"{PIPELINE_VERSION})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
