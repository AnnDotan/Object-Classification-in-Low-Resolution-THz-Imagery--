from __future__ import annotations

import csv
from pathlib import Path


def parse_key_value_file(path: Path) -> dict:
    data = {}
    if not path.exists():
        return data

    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()

    return data


def extract_best_val_acc(log_path: Path) -> str:
    if not log_path.exists():
        return ""

    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in reversed(lines):
        if "best_val_acc=" in line:
            start = line.find("best_val_acc=") + len("best_val_acc=")
            end = line.find(")", start)
            if end == -1:
                return line[start:].strip()
            return line[start:end].strip()
    return ""


def extract_total_time(log_path: Path) -> str:
    if not log_path.exists():
        return ""

    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in reversed(lines):
        if line.startswith("Total time:"):
            return line.replace("Total time:", "").strip()
    return ""


def is_run_dir(path: Path) -> bool:
    if not path.is_dir():
        return False

    has_config = (path / "run_config.txt").exists()
    has_log = (path / "log.txt").exists()
    has_metrics = (path / "metrics.csv").exists()

    return has_config or has_log or has_metrics


def collect_runs_recursive(runs_root: Path) -> list[dict]:
    rows = []

    for path in runs_root.rglob("*"):
        if not is_run_dir(path):
            continue

        run_config_path = path / "run_config.txt"
        log_path = path / "log.txt"

        config = parse_key_value_file(run_config_path)

        try:
            group = path.relative_to(runs_root).parts[0]
        except Exception:
            group = "unknown"

        row = {
            "group": group,
            "run_name": path.name,
            "model_name": config.get("model_name", ""),
            "pretrained": config.get("pretrained", ""),
            "out_size": config.get("out_size", ""),
            "low_res": config.get("low_res", ""),
            "epochs": config.get("epochs", ""),
            "batch_size": config.get("batch_size", ""),
            "train_subset": config.get("train_subset", ""),
            "val_subset": config.get("val_subset", ""),
            "lr": config.get("lr", ""),
            "tag": config.get("tag", ""),
            "best_val_acc": extract_best_val_acc(log_path),
            "total_time": extract_total_time(log_path),
            "run_dir": str(path),
        }
        rows.append(row)

    return sorted(rows, key=lambda x: (x["group"].lower(), x["run_name"].lower()))


def save_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "group",
        "run_name",
        "model_name",
        "pretrained",
        "out_size",
        "low_res",
        "epochs",
        "batch_size",
        "train_subset",
        "val_subset",
        "lr",
        "tag",
        "best_val_acc",
        "total_time",
        "run_dir",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    runs_root = Path("runs")
    if not runs_root.exists():
        print("No runs directory found.")
        return

    rows = collect_runs_recursive(runs_root)
    out_path = Path("artifacts/tables/run_summary.csv")
    save_csv(rows, out_path)

    print(f"Saved summary to: {out_path}")
    print(f"Total runs summarized: {len(rows)}")


if __name__ == "__main__":
    main()