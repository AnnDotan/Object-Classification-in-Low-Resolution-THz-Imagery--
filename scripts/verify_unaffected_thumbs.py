"""Verify the 96 v2-unaffected dashboard thumbnails are byte-identical to v1.

PRD §US-019B acceptance criterion #2: prove that the 96 cells whose
`noise_std == 0 AND salt_pepper == 0` (Phase A clean + Phase C
resolution/blur/saturation) produce byte-identical PNGs under
PIPELINE_VERSION=2 as they did under v1. The US-017 noise/S&P pipeline
move is a no-op at the identity values used by these cells, so v1==v2
is the load-bearing invariant — if it fails, US-017's scoping is broken
and the 90-cell re-run scope is too narrow.

Approach (idempotent, durable manifest):
  1. Hash every on-disk PNG for the 96 unaffected tags. These bytes are
     STILL v1 bytes today because the US-019B Iter 29 `--force` re-render
     only touched the 90 affected tags. The hashes captured here ARE
     the v1 baseline.
  2. Write `artifacts/validation/unaffected_thumbs_manifest.json` with
     {tag: sha256, pipeline_version_at_snapshot, snapshot_utc, cell_count=96}.
     If a manifest already exists, do NOT overwrite — re-use it (the
     original baseline is load-bearing; clobbering it would destroy
     audit trail).
  3. Re-render all 96 under v2 by invoking `render_thumbs(force=True, ...)`
     for `--phase A` and `--phase C --axes resolution,blur,saturation`.
  4. Re-hash every output PNG; compare against the manifest. Any delta
     fails the test (exit 1 with the diff list).

Modes:
  default (no flag) — full pass: snapshot + re-render + verify.
  --verify          — verify-only: re-hash current PNGs vs manifest, no
                      re-render. Use for later audits.

CLI:
    python scripts/verify_unaffected_thumbs.py              # full pass
    python scripts/verify_unaffected_thumbs.py --verify     # audit only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.degradation_levels import PIPELINE_VERSION  # noqa: E402
from src.experiments.cells import iter_cells  # noqa: E402

THUMB_DIR = REPO_ROOT / "artifacts" / "dashboard_thumbs"
MANIFEST = REPO_ROOT / "artifacts" / "validation" / "unaffected_thumbs_manifest.json"

AFFECTED_AXES = {"noise", "salt_pepper"}
UNAFFECTED_AXES = ("resolution", "blur", "saturation")


def unaffected_tags() -> list[str]:
    """Return the 96 cell tags whose v2 pipeline output is identical to v1.

    These are exactly the cells whose `noise_std == 0 AND salt_pepper == 0`:
      - Phase A (6 clean cells: 3 models x 2 datasets), AND
      - Phase C with axis in {resolution, blur, saturation} (90 cells:
        3 axes x 5 levels x 3 models x 2 datasets).
    Total: 6 + 90 = 96.
    """
    tags: list[str] = []
    for c in iter_cells():
        if c.phase == "A":
            tags.append(c.tag)
        elif c.phase == "C" and c.axis not in AFFECTED_AXES:
            tags.append(c.tag)
    return tags


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def hash_all(tags: list[str]) -> tuple[dict[str, str], list[str]]:
    hashes: dict[str, str] = {}
    missing: list[str] = []
    for tag in tags:
        p = THUMB_DIR / f"{tag}.png"
        if not p.exists():
            missing.append(tag)
            continue
        hashes[tag] = sha256_of(p)
    return hashes, missing


def write_manifest(hashes: dict[str, str]) -> None:
    payload = {
        "schema_version": 1,
        "purpose": (
            "Byte-level SHA-256 baseline for the 96 dashboard thumbnails whose "
            "v2 pipeline output should be identical to v1 (cells with "
            "noise_std == 0 AND salt_pepper == 0). Captured BEFORE re-rendering "
            "under PIPELINE_VERSION=2 so the post-re-render hash comparison "
            "proves US-017's noise/S&P pipeline move is a no-op for these cells."
        ),
        "pipeline_version_at_snapshot": int(PIPELINE_VERSION),
        "snapshot_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "src_dir": str(THUMB_DIR.relative_to(REPO_ROOT)),
        "cell_count": len(hashes),
        "sha256": dict(sorted(hashes.items())),
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n")


def compare(actual: dict[str, str], manifest: dict, tags: list[str],
            missing: list[str]) -> int:
    expected: dict[str, str] = manifest["sha256"]
    drift: list[tuple[str, str, str]] = []
    for tag in tags:
        if tag in missing:
            continue
        a = actual.get(tag)
        e = expected.get(tag)
        if a != e:
            drift.append((tag, e or "<missing-from-manifest>", a or "<missing-on-disk>"))
    if missing or drift:
        print(f"[verify] FAIL - {len(missing)} missing PNG(s), "
              f"{len(drift)} hash mismatch(es).", file=sys.stderr)
        for tag in missing[:5]:
            print(f"  missing: {tag}", file=sys.stderr)
        for tag, exp, got in drift[:5]:
            print(f"  drift:   {tag}\n    expected={exp}\n    actual  ={got}",
                  file=sys.stderr)
        return 1
    print(f"[verify] OK - {len(tags)} unaffected thumbs match manifest sha256s.")
    return 0


def rerender_unaffected() -> None:
    """Force-re-render all 96 unaffected thumbs under the current pipeline.

    Calls into the renderer's Python API (not the CLI) so this script is a
    single process - no shell quoting headaches, no dataset re-downloads
    between the two phases.
    """
    from src.tools.render_cell_thumbs import render_thumbs  # local import: torch

    phase_a = render_thumbs(force=True, phase="A")
    print(f"[rerender] phase A: written={phase_a['written']} "
          f"failed={phase_a['failed']} total={phase_a['total']}")
    if phase_a["failed"] > 0:
        raise SystemExit(f"phase A re-render had {phase_a['failed']} failures")

    phase_c = render_thumbs(force=True, phase="C", axes=UNAFFECTED_AXES)
    print(f"[rerender] phase C axes={list(UNAFFECTED_AXES)}: "
          f"written={phase_c['written']} failed={phase_c['failed']} "
          f"total={phase_c['total']}")
    if phase_c["failed"] > 0:
        raise SystemExit(f"phase C re-render had {phase_c['failed']} failures")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify-only: re-hash current PNGs vs the existing manifest "
             "(no re-render). Exit 1 on drift.",
    )
    args = parser.parse_args(argv)

    tags = unaffected_tags()
    assert len(tags) == 96, (
        f"expected 96 v2-unaffected cells (6 Phase A + 90 Phase C "
        f"resolution/blur/saturation); iter_cells() yielded {len(tags)} - "
        "matrix drift?"
    )

    if args.verify:
        if not MANIFEST.exists():
            print(f"[verify] no manifest at {MANIFEST.relative_to(REPO_ROOT)}; "
                  "run without --verify to write one and re-render.",
                  file=sys.stderr)
            return 1
        actual, missing = hash_all(tags)
        manifest = json.loads(MANIFEST.read_text())
        return compare(actual, manifest, tags, missing)

    # Full pass: snapshot (if needed) -> re-render -> verify.
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
        print(f"[snapshot] re-using existing manifest at "
              f"{MANIFEST.relative_to(REPO_ROOT)} "
              f"(snapshot_utc={manifest.get('snapshot_utc')}, "
              f"cell_count={manifest.get('cell_count')}).")
    else:
        pre_hashes, pre_missing = hash_all(tags)
        if pre_missing:
            raise SystemExit(
                f"cannot snapshot: {len(pre_missing)} unaffected PNG(s) missing "
                f"on disk; first 3 = {pre_missing[:3]}. Run "
                "`python -m src.tools.render_cell_thumbs` (without --force) to "
                "render the unaffected cells first."
            )
        write_manifest(pre_hashes)
        manifest = json.loads(MANIFEST.read_text())
        print(f"[snapshot] wrote {len(pre_hashes)} v1 unaffected-thumb hashes "
              f"to {MANIFEST.relative_to(REPO_ROOT)} "
              f"(pipeline_version_at_snapshot={PIPELINE_VERSION}).")

    rerender_unaffected()

    actual, missing = hash_all(tags)
    return compare(actual, manifest, tags, missing)


if __name__ == "__main__":
    sys.exit(main())
