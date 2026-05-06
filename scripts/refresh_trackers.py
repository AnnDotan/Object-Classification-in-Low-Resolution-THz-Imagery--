"""Shared tracker-refresh helper (US-016 + US-019 CLI).

Single source of truth for the post-cell tracker refresh used by both
`scripts/run_phase_a.py` and `run_all_phases.py`. Calls
`scripts/update_final_exp.py` (Markdown tracker) and
`src.tools.build_final_dashboard` (HTML dashboard) in sequence; failures in
one do NOT skip the other.

Per PRD US-005 / US-015 / US-016: tracker-write failure must not abort the
runner. This module returns a result dict; callers may log warnings or
escalate as policy dictates.

US-019 CLI:
    python scripts/refresh_trackers.py                            # full refresh (md + json + html + thumbs)
    python scripts/refresh_trackers.py --cell <tag>               # incremental: patch only one row + html
    python scripts/refresh_trackers.py --cell <tag> --no-md       # skip the (slower) MD regen
    python scripts/refresh_trackers.py --no-thumbs                # skip torch-required visual core
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Optional

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_update_final_exp():
    src = _SCRIPTS_DIR / "update_final_exp.py"
    spec = importlib.util.spec_from_file_location("scripts_update_final_exp", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def refresh_final_exp_md() -> tuple[bool, str | None]:
    """Run scripts/update_final_exp.py with default args. Returns (ok, error)."""
    try:
        mod = _load_update_final_exp()
        rc = mod.main([])
        if rc != 0:
            return False, f"update_final_exp.main returned rc={rc}"
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def refresh_final_exp_html() -> tuple[bool, str | None]:
    """Run src.tools.build_final_dashboard.main with default args. Returns (ok, error)."""
    try:
        from src.tools.build_final_dashboard import main as _build  # lazy import
        rc = _build([])
        if rc != 0:
            return False, f"build_final_dashboard.main returned rc={rc}"
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def refresh_final_exp_json_cell(tag: str) -> tuple[bool, str | None]:
    """US-019 incremental: patch only `tag`'s row in `Final_Exp.json`.

    Falls back to a full rebuild if the on-disk JSON is missing or
    schema-mismatched (handled inside `update_cell`).
    """
    try:
        from src.tools.build_final_exp_json import update_cell  # lazy
        update_cell(tag=tag)
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def refresh_visual_core_thumbs() -> tuple[bool, str | None]:
    """Render any missing Visual Core PNGs (US-017). Returns (ok, error).

    Idempotent: `render_thumbs` skips PNGs that already exist on disk. Only
    new cells (or `--force`) trigger work. The renderer transitively pulls
    in torch + torchvision; a missing dependency surfaces as a soft error
    via the (ok, error) contract — it does NOT crash the wider refresh.
    """
    try:
        from src.tools.render_cell_thumbs import render_thumbs  # lazy
        summary = render_thumbs()  # all 186, skip-existing
        if summary.get("failed", 0) > 0:
            return False, f"{summary['failed']} thumbnail(s) failed to render"
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def refresh_all(
    *,
    cell: Optional[str] = None,
    skip_md: bool = False,
    skip_thumbs: bool = False,
) -> dict:
    """Refresh trackers. Failures isolated.

    Modes (US-019):
      - **Full** (default, `cell=None`): run MD + HTML + thumbs. Each step
        always runs regardless of whether the previous one succeeded.
      - **Incremental** (`cell=<tag>`): patch only that row's hydration in
        `Final_Exp.json`, then rebuild the HTML so the inline-data block
        reflects the change. MD is skipped by default in this mode (it's
        the slowest step) — pass `skip_md=False` explicitly to include it.

    Returns:
        {
          "md_ok":     bool | None,   # None when MD step was skipped
          "html_ok":   bool,
          "json_ok":   bool | None,   # None unless incremental (cell != None)
          "thumbs_ok": bool | None,   # None when thumbs step was skipped
          "mode":      "full" | "incremental",
          "cell":      tag | None,
          "errors":    list[str],
        }
    """
    is_incremental = cell is not None
    mode = "incremental" if is_incremental else "full"

    # Full mode regenerates the MD unless `--no-md` was passed.
    # Incremental mode skips the MD by default — it's the slowest step and
    # the dashboard reads JSON, not MD. `--no-md` is honored either way.
    md_should_run = (not is_incremental) and (not skip_md)

    errors: list[str] = []
    md_ok: Optional[bool] = None
    html_ok: bool
    json_ok: Optional[bool] = None
    thumbs_ok: Optional[bool] = None

    if md_should_run:
        md_ok, md_err = refresh_final_exp_md()
        if md_err:
            errors.append(f"final_exp_md: {md_err}")

    if is_incremental:
        json_ok, json_err = refresh_final_exp_json_cell(cell)  # type: ignore[arg-type]
        if json_err:
            errors.append(f"final_exp_json[{cell}]: {json_err}")

    html_ok, html_err = refresh_final_exp_html()
    if html_err:
        errors.append(f"final_exp_html: {html_err}")

    if not skip_thumbs:
        thumbs_ok, thumbs_err = refresh_visual_core_thumbs()
        if thumbs_err:
            errors.append(f"visual_core_thumbs: {thumbs_err}")

    return {
        "md_ok": md_ok,
        "html_ok": html_ok,
        "json_ok": json_ok,
        "thumbs_ok": thumbs_ok,
        "mode": mode,
        "cell": cell,
        "errors": errors,
    }


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Refresh Final_Exp.{md,json,html} + Visual Core thumbs (US-019).",
    )
    p.add_argument(
        "--cell", default=None,
        help="Incremental: patch only this single tag's row in Final_Exp.json "
             "(skips MD by default; HTML still rebuilds).",
    )
    p.add_argument(
        "--no-md", action="store_true",
        help="Skip Markdown regeneration (faster; useful with --cell).",
    )
    p.add_argument(
        "--no-thumbs", action="store_true",
        help="Skip Visual Core PNG render (faster; useful in torch-free envs).",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    result = refresh_all(
        cell=args.cell,
        skip_md=args.no_md,
        skip_thumbs=args.no_thumbs,
    )
    for err in result["errors"]:
        print(f"[refresh][WARN] {err}", file=sys.stderr)
    cell_suffix = f" cell={result['cell']}" if result["cell"] else ""
    print(
        f"refresh: mode={result['mode']} "
        f"md_ok={result['md_ok']} json_ok={result['json_ok']} "
        f"html_ok={result['html_ok']} thumbs_ok={result['thumbs_ok']}"
        f"{cell_suffix}"
    )
    # Exit 0 unless the html step (the only always-running, always-required step) failed.
    return 0 if result["html_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
