#!/usr/bin/env python3
"""NotebookLM source synchronization for THz Project.

Mirrors a curated subset of the local Git tree into the `THz Project`
notebook on NotebookLM. Driven by `scripts/notebooklm_manifest.yml`.

Subcommands
-----------
status   diff repo (manifest-matched files) vs notebook (read-only).
push     upload files matched by manifest but missing from notebook.
prune    delete sources that match an `include` glob but no longer exist
         on disk. Sources in `preserve_titles` are never deleted.
refresh  re-upload sources whose local mtime is newer than the recorded
         upload mtime in `scripts/notebooklm_sync_state.json`.
full     push + prune + refresh + git commit (if --apply).

All mutating subcommands require --apply; otherwise they dry-run.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required: `python -m pip install pyyaml`")

REPO = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO / "scripts" / "notebooklm_manifest.yml"
LOG_PATH = REPO / "scripts" / "notebooklm_sync.log"
STATE_PATH = REPO / "scripts" / "notebooklm_sync_state.json"

REQUIRED_DENY = {
    "**/*.ckpt",
    "**/*.pt",
    "**/*.pth",
    "artifacts/weights/**",
    "runs/**",
}


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        sys.exit(f"manifest not found: {MANIFEST_PATH}")
    with MANIFEST_PATH.open(encoding="utf-8") as f:
        m = yaml.safe_load(f)
    deny = set(m.get("deny", []))
    missing = REQUIRED_DENY - deny
    if missing:
        sys.exit(
            "[deny-glob violation] manifest is missing required weight-privacy "
            f"deny globs: {sorted(missing)}. Refusing to run."
        )
    return m


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    with STATE_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def save_state(state: dict) -> None:
    with STATE_PATH.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.write("\n")


def expand_includes(manifest: dict) -> list[Path]:
    """Return repo-absolute paths matched by any `include` glob, minus deny."""
    matched: set[Path] = set()
    for pat in manifest["include"]:
        for p in REPO.glob(pat):
            if p.is_file():
                matched.add(p.resolve())
    keep: list[Path] = []
    for p in matched:
        rel = p.relative_to(REPO).as_posix()
        if any(fnmatch.fnmatch(rel, d) for d in manifest["deny"]):
            continue
        keep.append(p)
    return sorted(keep, key=lambda x: x.relative_to(REPO).as_posix())


def nbcli(args: list[str], storage: str, check: bool = True) -> subprocess.CompletedProcess:
    cmd = ["notebooklm", "--storage", storage, *args]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", env=env
    )
    if check and r.returncode != 0:
        raise RuntimeError(
            f"notebooklm CLI failed (exit {r.returncode}):\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stderr: {r.stderr.strip()}"
        )
    return r


def first_json_object(text: str) -> dict:
    """Extract the first top-level JSON object from CLI output (skipping warnings)."""
    start = text.find("{")
    if start < 0:
        raise RuntimeError(f"no JSON object in output:\n{text}")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == "\\" and in_str:
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise RuntimeError(f"unterminated JSON object in output:\n{text[start:]}")


def get_notebook_metadata(storage: str, notebook_id: str) -> dict:
    r = nbcli(["use", notebook_id], storage, check=False)
    if r.returncode != 0:
        sys.exit(
            f"[notebook-not-found] cannot select '{notebook_id}': {r.stderr.strip()}"
        )
    r = nbcli(["metadata", "--json"], storage)
    return first_json_object(r.stdout)


TEXT_EXTS = {".md", ".txt", ".py", ".yml", ".yaml", ".json", ".toml", ".cfg", ".ini", ".csv", ".rst"}


def upload_args_for(rel_path: str) -> tuple[list[str], bool]:
    """(extra_cli_args, needs_rename_after_upload).

    NotebookLM accepts only specific source kinds. Empirically:
      - `--type text` reads any path's content and stores as markdown-typed text.
      - `--type file --mime-type application/pdf` for PDFs.
      - Other binary mime types (text/plain, octet-stream) are rejected with HTTP 400.
    """
    ext = Path(rel_path).suffix.lower()
    if ext == ".pdf":
        return ["--type", "file", "--mime-type", "application/pdf"], True
    if ext in TEXT_EXTS:
        return ["--type", "text", "--title", rel_path], False
    # Unknown extension — try as text (safest default for source code).
    return ["--type", "text", "--title", rel_path], False


def _extract_source_id(data: dict) -> str | None:
    """Pull the source UUID out of `source add --json` output (CLI 0.3.x: nested under `source`)."""
    if not isinstance(data, dict):
        return None
    if "source" in data and isinstance(data["source"], dict):
        return data["source"].get("id")
    return data.get("id") or data.get("source_id")


def upload_file(local_path: Path, rel_path: str, storage: str) -> str | None:
    extra, needs_rename = upload_args_for(rel_path)
    args = ["source", "add", str(local_path), *extra, "--json"]
    r = nbcli(args, storage)
    try:
        data = first_json_object(r.stdout)
    except RuntimeError:
        return None
    src_id = _extract_source_id(data)
    if needs_rename and src_id:
        # PDFs upload async — wait for processing to complete before renaming,
        # otherwise the rename hits the source while it is still in flight and
        # silently no-ops, leaving the bare filename as the title.
        nbcli(["source", "wait", src_id], storage, check=False)
        nbcli(["source", "rename", src_id, rel_path], storage, check=False)
    return src_id


def log_action(action: str, title: str, source_id: str | None = None) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"{ts} {action} {title}"
    if source_id:
        line += f" {source_id}"
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def diff(manifest: dict, storage: str) -> tuple[list[str], list[str], list[Path], dict]:
    """Return (missing_in_notebook, only_in_notebook_not_preserved, file_paths, nb_meta)."""
    nb = get_notebook_metadata(storage, manifest["notebook_id"])
    nb_titles = {s["title"] for s in nb.get("sources", [])}
    files = expand_includes(manifest)
    file_titles = {p.relative_to(REPO).as_posix() for p in files}
    preserve = set(manifest.get("preserve_titles", []))

    missing = sorted(file_titles - nb_titles)
    only_nb = sorted(nb_titles - file_titles - preserve)
    return missing, only_nb, files, nb


def cmd_status(manifest: dict, storage: str, _apply: bool = False) -> int:
    missing, only_nb, files, nb = diff(manifest, storage)
    file_titles = {p.relative_to(REPO).as_posix() for p in files}
    nb_titles = {s["title"] for s in nb.get("sources", [])}
    aligned = sorted(file_titles & nb_titles)
    preserve = set(manifest.get("preserve_titles", []))

    print(f"Notebook : {nb['title']} ({nb['id']})")
    print(f"Sources  : {len(nb.get('sources', []))} in notebook, "
          f"{len(files)} matched by manifest, {len(preserve)} preserved")
    print()
    print(f"  [aligned] {len(aligned)} file(s)")
    for t in aligned:
        print(f"      = {t}")
    print()
    print(f"  [missing in notebook] {len(missing)} file(s) (push will add)")
    for t in missing:
        print(f"      + {t}")
    print()
    print(f"  [only in notebook] {len(only_nb)} source(s) (prune will delete)")
    for t in only_nb:
        print(f"      - {t}")
    print()
    print(f"  [preserved] {len(preserve)} (never touched)")
    return 0


def cmd_push(manifest: dict, storage: str, apply_: bool) -> int:
    missing, _, files, _ = diff(manifest, storage)
    if not missing:
        print("Nothing to push — all manifest files already in notebook.")
        return 0
    by_title = {p.relative_to(REPO).as_posix(): p for p in files}
    print(f"Files to upload: {len(missing)}")
    for t in missing:
        print(f"  + {t}")
    if not apply_:
        print("\nDRY RUN. Pass --apply to upload.")
        return 0

    state = load_state()
    print(f"\nUploading {len(missing)} file(s)...")
    ok = 0
    for i, title in enumerate(missing, 1):
        path = by_title[title]
        print(f"  [{i}/{len(missing)}] {title}", flush=True)
        try:
            src_id = upload_file(path, title, storage)
            log_action("ADD", title, src_id)
            state[title] = {
                "mtime": path.stat().st_mtime,
                "source_id": src_id,
                "uploaded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            save_state(state)
            ok += 1
        except Exception as e:
            print(f"      FAILED: {e}", file=sys.stderr)
            log_action("ADD-FAIL", title)
    print(f"\nPushed {ok}/{len(missing)}.")
    return 0 if ok == len(missing) else 1


def cmd_prune(manifest: dict, storage: str, apply_: bool) -> int:
    _, only_nb, _, _ = diff(manifest, storage)
    if not only_nb:
        print("Nothing to prune.")
        return 0
    print(f"Sources to delete: {len(only_nb)}")
    for t in only_nb:
        print(f"  - {t}")
    if not apply_:
        print("\nDRY RUN. Pass --apply to delete.")
        return 0

    state = load_state()
    print(f"\nDeleting {len(only_nb)} source(s)...")
    ok = 0
    for i, title in enumerate(only_nb, 1):
        print(f"  [{i}/{len(only_nb)}] {title}", flush=True)
        try:
            nbcli(["source", "delete-by-title", title, "-y"], storage)
            log_action("DELETE", title)
            state.pop(title, None)
            save_state(state)
            ok += 1
        except Exception as e:
            print(f"      FAILED: {e}", file=sys.stderr)
            log_action("DELETE-FAIL", title)
    print(f"\nPruned {ok}/{len(only_nb)}.")
    return 0 if ok == len(only_nb) else 1


def cmd_refresh(manifest: dict, storage: str, apply_: bool) -> int:
    nb = get_notebook_metadata(storage, manifest["notebook_id"])
    nb_titles = {s["title"] for s in nb.get("sources", [])}
    files = expand_includes(manifest)
    state = load_state()

    drifted: list[tuple[str, Path]] = []
    untracked: list[tuple[str, Path]] = []
    for p in files:
        title = p.relative_to(REPO).as_posix()
        if title not in nb_titles:
            continue
        rec = state.get(title)
        if rec is None:
            untracked.append((title, p))
            continue
        if p.stat().st_mtime > rec.get("mtime", 0) + 1:  # 1s tolerance
            drifted.append((title, p))

    if not drifted and not untracked:
        print("Nothing to refresh — all uploaded files are tracked and unchanged.")
        return 0

    if drifted:
        print(f"Drifted (mtime newer than upload): {len(drifted)}")
        for title, _ in drifted:
            print(f"  ~ {title}")
    if untracked:
        print(f"Untracked (in notebook but no upload state): {len(untracked)}")
        for title, _ in untracked:
            print(f"  ? {title}")

    if not apply_:
        print("\nDRY RUN. Pass --apply to re-upload drifted (untracked are NOT auto-refreshed).")
        return 0

    targets = drifted  # untracked stays untouched until manifest cleanup
    print(f"\nRe-uploading {len(targets)} drifted source(s)...")
    ok = 0
    for i, (title, path) in enumerate(targets, 1):
        print(f"  [{i}/{len(targets)}] {title}", flush=True)
        try:
            nbcli(["source", "delete-by-title", title, "-y"], storage)
            src_id = upload_file(path, title, storage)
            log_action("REFRESH", title, src_id)
            state[title] = {
                "mtime": path.stat().st_mtime,
                "source_id": src_id,
                "uploaded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            save_state(state)
            ok += 1
        except Exception as e:
            print(f"      FAILED: {e}", file=sys.stderr)
            log_action("REFRESH-FAIL", title)
    print(f"\nRefreshed {ok}/{len(targets)}.")
    return 0 if ok == len(targets) else 1


def cmd_full(manifest: dict, storage: str, apply_: bool) -> int:
    rc = 0
    print("=== push ===")
    rc |= cmd_push(manifest, storage, apply_)
    print("\n=== prune ===")
    rc |= cmd_prune(manifest, storage, apply_)
    print("\n=== refresh ===")
    rc |= cmd_refresh(manifest, storage, apply_)
    if apply_:
        print("\n=== git commit ===")
        commit_to_git()
    return rc


def commit_to_git() -> None:
    paths = [
        "scripts/notebooklm_manifest.yml",
        "scripts/notebooklm_sync.log",
        "scripts/notebooklm_sync_state.json",
    ]
    existing = [p for p in paths if (REPO / p).exists()]
    if not existing:
        print("No artifact files to commit.")
        return
    subprocess.run(["git", "add", *existing], cwd=REPO, check=False)
    diff = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=REPO
    )
    if diff.returncode == 0:
        print("Nothing staged for commit.")
        return
    msg = f"[notebooklm-sync] mirror sources @ {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    r = subprocess.run(
        ["git", "commit", "-m", msg], cwd=REPO, capture_output=True, text=True
    )
    if r.returncode == 0:
        print(r.stdout.strip())
    else:
        print(f"Git commit failed: {r.stderr.strip()}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="notebooklm_sync",
        description="Sync curated repo files into the THz Project NotebookLM notebook.",
    )
    ap.add_argument(
        "command",
        choices=["status", "push", "prune", "refresh", "full"],
        help="action to perform",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="execute mutations (without it: dry run)",
    )
    ap.add_argument(
        "--storage",
        default=None,
        help="path to storage_state.json (default: manifest.storage_state)",
    )
    args = ap.parse_args()

    manifest = load_manifest()
    storage = (
        args.storage
        or os.environ.get("NOTEBOOKLM_AUTH_JSON")
        or manifest.get("storage_state")
    )
    if not storage or not Path(storage).exists():
        sys.exit(
            f"[auth-missing] storage_state not found at {storage!r}. "
            "Run `notebooklm login` or set NOTEBOOKLM_AUTH_JSON."
        )

    handlers = {
        "status": cmd_status,
        "push": cmd_push,
        "prune": cmd_prune,
        "refresh": cmd_refresh,
        "full": cmd_full,
    }
    return handlers[args.command](manifest, storage, args.apply)


if __name__ == "__main__":
    sys.exit(main() or 0)
