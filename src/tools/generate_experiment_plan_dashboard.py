#!/usr/bin/env python3
"""
Generate Experiment Plan Dashboard
====================================

Creates an interactive HTML dashboard showing ALL 36 planned experiments
with results (when available), degradation pipeline samples, comparative
charts per degradation level, dates, and durations.

Dark theme with white text and green accents.
"""

import csv
import json
import base64
import io
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torchvision import datasets, transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.data.degrade import DegradeConfig, degrade_image

# ──────────────────────────────────────────────────────────────────────
# EXPERIMENT PLAN DEFINITION — mirrors EXPERIMENT_PLAN.md
# ──────────────────────────────────────────────────────────────────────

DEGRADATION_LEVELS = {
    1: {"name": "Mild",     "low_res": 16, "blur_kernel": 3, "blur_sigma": 0.5,
        "gaussian_noise_std": 0.04, "salt_pepper_amount": 0.02, "p_grayscale": 0.3},
    2: {"name": "Moderate", "low_res": 16, "blur_kernel": 5, "blur_sigma": 1.0,
        "gaussian_noise_std": 0.08, "salt_pepper_amount": 0.05, "p_grayscale": 0.3},
    3: {"name": "Severe",   "low_res": 8,  "blur_kernel": 7, "blur_sigma": 1.5,
        "gaussian_noise_std": 0.12, "salt_pepper_amount": 0.08, "p_grayscale": 0.3},
}

MODELS = ["resnet50", "densenet121", "transnext_micro"]

MODEL_LABELS = {
    "resnet50":       "ResNet-50",
    "densenet121":    "DenseNet-121",
    "transnext_micro": "TransNeXt Micro",
}

MODEL_COLORS = {
    "resnet50":       "#4fc3f7",
    "densenet121":    "#81d4fa",
    "transnext_micro": "#b3e5fc",
}

MODEL_COLORS_CHART = {
    "resnet50":       "rgba(79, 195, 247, 0.85)",
    "densenet121":    "rgba(129, 212, 250, 0.85)",
    "transnext_micro": "rgba(179, 229, 252, 0.85)",
}

MODEL_BORDERS_CHART = {
    "resnet50":       "rgba(79, 195, 247, 1)",
    "densenet121":    "rgba(129, 212, 250, 1)",
    "transnext_micro": "rgba(179, 229, 252, 1)",
}

DATASETS = ["cifar10", "mnist"]

ISOLATION_TYPES = ["downsampling", "blur", "noise", "salt_pepper"]

# ──────────────────────────────────────────────────────────────────────
# Build planned experiments
# ──────────────────────────────────────────────────────────────────────

def build_experiment_plan():
    """Return list of all 36 planned experiments."""
    experiments = []

    # Phase A: CIFAR-10 systematic
    for level_id, level in DEGRADATION_LEVELS.items():
        for model in MODELS:
            experiments.append({
                "exp_id": f"C{level_id}-{MODELS.index(model)+1}",
                "phase": "A",
                "phase_name": "CIFAR-10 Systematic",
                "dataset": "cifar10",
                "level": level_id,
                "level_name": level["name"],
                "model": model,
                "degradation_type": "all",
                "tag_pattern": f"sys_L{level_id}_{level['name'].lower()}_{model}",
                "deg_params": level,
            })

    # Phase B: MNIST systematic
    for level_id, level in DEGRADATION_LEVELS.items():
        for model in MODELS:
            experiments.append({
                "exp_id": f"M{level_id}-{MODELS.index(model)+1}",
                "phase": "B",
                "phase_name": "MNIST Systematic",
                "dataset": "mnist",
                "level": level_id,
                "level_name": level["name"],
                "model": model,
                "degradation_type": "all",
                "tag_pattern": f"sys_L{level_id}_{level['name'].lower()}_{model}_mnist",
                "deg_params": level,
            })

    # Phase C: Single-degradation isolation (CIFAR-10 only, Level 2 params)
    iso_params = {
        "downsampling": {"low_res": 16, "blur_kernel": 1, "blur_sigma": 0.0,
                         "gaussian_noise_std": 0.0, "salt_pepper_amount": 0.0, "p_grayscale": 0.3},
        "blur":         {"low_res": 16, "blur_kernel": 5, "blur_sigma": 1.0,
                         "gaussian_noise_std": 0.0, "salt_pepper_amount": 0.0, "p_grayscale": 0.0},
        "noise":        {"low_res": 16, "blur_kernel": 1, "blur_sigma": 0.0,
                         "gaussian_noise_std": 0.08, "salt_pepper_amount": 0.0, "p_grayscale": 0.0},
        "salt_pepper":  {"low_res": 16, "blur_kernel": 1, "blur_sigma": 0.0,
                         "gaussian_noise_std": 0.0, "salt_pepper_amount": 0.05, "p_grayscale": 0.0},
    }
    for iso_type in ISOLATION_TYPES:
        for i, model in enumerate(MODELS):
            experiments.append({
                "exp_id": f"I-{ISOLATION_TYPES.index(iso_type)*3 + i + 1}",
                "phase": "C",
                "phase_name": "Single-Degradation Isolation",
                "dataset": "cifar10",
                "level": 2,
                "level_name": f"{iso_type} only",
                "model": model,
                "degradation_type": iso_type,
                "tag_pattern": f"{iso_type}_{model}",
                "deg_params": iso_params[iso_type],
            })

    # Phase D: Clean baselines
    clean_params = {"low_res": 224, "blur_kernel": 1, "blur_sigma": 0.0,
                    "gaussian_noise_std": 0.0, "salt_pepper_amount": 0.0, "p_grayscale": 0.0}
    for ds in DATASETS:
        for i, model in enumerate(MODELS):
            suffix = "_mnist" if ds == "mnist" else ""
            experiments.append({
                "exp_id": f"B-{DATASETS.index(ds)*3 + i + 1}",
                "phase": "D",
                "phase_name": "Clean Baseline",
                "dataset": ds,
                "level": 0,
                "level_name": "Clean (no degradation)",
                "model": model,
                "degradation_type": "clean",
                "tag_pattern": f"clean_{model}{suffix}",
                "deg_params": clean_params,
            })

    return experiments


# ──────────────────────────────────────────────────────────────────────
# Scan existing runs and match to planned experiments
# ──────────────────────────────────────────────────────────────────────

def parse_config(run_dir: Path) -> dict:
    config_path = run_dir / "run_config.txt"
    data = {}
    if config_path.exists():
        for line in config_path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                data[k.strip()] = v.strip()
    return data


def get_metrics(run_dir: Path) -> list[dict] | None:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return None
    rows = []
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("epoch"):
                    rows.append({
                        "epoch": int(row["epoch"]),
                        "train_loss": float(row["train_loss"]),
                        "train_acc": float(row["train_acc"]),
                        "val_loss": float(row["val_loss"]),
                        "val_acc": float(row["val_acc"]),
                    })
    except Exception:
        return None
    return rows if rows else None


def extract_total_time(log_path: Path) -> str:
    if not log_path.exists():
        return ""
    for line in reversed(log_path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if line.startswith("Total time:"):
            return line.replace("Total time:", "").strip()
    return ""


def get_run_date(run_dir: Path) -> str:
    """Try to extract date from run dir or file modification time."""
    # Try from run_config run_name (contains timestamp like _20260310-191712)
    cfg = parse_config(run_dir)
    run_name = cfg.get("run_name", "")
    import re
    m = re.search(r"(\d{8})-(\d{6})", run_name)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d").strftime("%d/%m/%Y")
        except ValueError:
            pass
    # Fall back to file modification time
    for fname in ["metrics.csv", "log.txt", "run_config.txt"]:
        fp = run_dir / fname
        if fp.exists():
            mtime = os.path.getmtime(fp)
            return datetime.fromtimestamp(mtime).strftime("%d/%m/%Y")
    return ""


def scan_runs(runs_root: Path) -> list[dict]:
    """Scan only runs/systematic/ for plan-related runs."""
    results = []
    sys_dir = runs_root / "systematic"
    if not sys_dir.exists():
        return results
    for run_dir in sorted(sys_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        config = parse_config(run_dir)
        metrics = get_metrics(run_dir)
        if not metrics:
            continue
        best_val_acc = max(m["val_acc"] for m in metrics)
        log_path = run_dir / "log.txt"
        results.append({
            "group": "systematic",
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "model_name": config.get("model_name", "unknown"),
            "tag": config.get("tag", ""),
            "low_res": config.get("low_res", "16"),
            "out_size": config.get("out_size", "224"),
            "epochs": len(metrics),
            "best_val_acc": best_val_acc,
            "degradation_type": config.get("degradation_type", "all"),
            "blur_kernel": config.get("blur_kernel", ""),
            "blur_sigma": config.get("blur_sigma", ""),
            "gaussian_noise_std": config.get("gaussian_noise_std", ""),
            "salt_pepper_amount": config.get("salt_pepper_amount", ""),
            "p_grayscale": config.get("p_grayscale", ""),
            "freeze_backbone": config.get("freeze_backbone", "False"),
            "total_time": extract_total_time(log_path),
            "date": get_run_date(run_dir),
            "metrics": metrics,
        })
    return results


def match_experiment_to_run(exp: dict, runs: list[dict]) -> dict | None:
    """Find the matching systematic run for a planned experiment (by tag)."""
    tag_pat = exp["tag_pattern"].lower()
    for run in runs:
        tag = run.get("tag", "").lower()
        if tag and tag == tag_pat:
            return run
        # Partial match: tag starts with the pattern
        if tag and tag.startswith(tag_pat):
            return run
    return None


# ──────────────────────────────────────────────────────────────────────
# Image generation
# ──────────────────────────────────────────────────────────────────────

SAMPLE_IDX = 7  # horse from CIFAR-10
MNIST_SAMPLE_IDX = 3  # sample digit from MNIST


def get_cifar10_sample(idx=SAMPLE_IDX):
    ds = datasets.CIFAR10(root="./data", train=False, download=True,
                          transform=transforms.ToTensor())
    img, label = ds[idx]
    return img, ds.classes[label]


def get_mnist_sample(idx=MNIST_SAMPLE_IDX):
    ds = datasets.MNIST(root="./data", train=False, download=True,
                        transform=transforms.ToTensor())
    img, label = ds[idx]
    # Convert 1-channel to 3-channel for consistency
    img = img.repeat(3, 1, 1)
    return img, str(label)


def tensor_to_base64(t: torch.Tensor, size: int = 112) -> str:
    t = t.clamp(0, 1).unsqueeze(0)
    t = torch.nn.functional.interpolate(t, size=(size, size), mode="nearest")
    t = t.squeeze(0)
    arr = (t.permute(1, 2, 0).numpy() * 255).astype("uint8")
    if arr.shape[2] == 1:
        arr = arr.squeeze(2)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def generate_degraded_image(deg_params: dict, original_img: torch.Tensor) -> str:
    """Generate base64 degraded image for given params."""
    kwargs = dict(
        low_res=deg_params.get("low_res", 16),
        out_size=224,
        p_grayscale=0.0,  # deterministic for display
        degradation_type="all",
    )
    bk = deg_params.get("blur_kernel", 5)
    if bk and int(bk) > 1:
        kwargs["blur_kernel"] = int(bk)
    bs = deg_params.get("blur_sigma", 1.0)
    if bs and float(bs) > 0:
        kwargs["blur_sigma"] = float(bs)
    gn = deg_params.get("gaussian_noise_std", 0.08)
    if gn:
        kwargs["gaussian_noise_std"] = float(gn)
    sp = deg_params.get("salt_pepper_amount", 0.05)
    if sp:
        kwargs["salt_pepper_amount"] = float(sp)

    deg_cfg = DegradeConfig(**kwargs)
    degraded = degrade_image(original_img.clone(), deg_cfg, seed=42)
    return tensor_to_base64(degraded, size=112)


# ──────────────────────────────────────────────────────────────────────
# HTML Generation
# ──────────────────────────────────────────────────────────────────────

def generate_html(experiments: list[dict], runs: list[dict],
                  original_images: dict, degraded_images: dict):
    """Generate the full dashboard HTML.
    
    original_images: dict mapping dataset name ('cifar10', 'mnist') to base64 string
    """

    # Match experiments to runs
    for exp in experiments:
        matched = match_experiment_to_run(exp, runs)
        exp["run"] = matched

    # Stats
    total = len(experiments)
    completed = sum(1 for e in experiments if e["run"] is not None)
    pending = total - completed

    best_acc_overall = 0
    best_model_overall = ""
    for e in experiments:
        if e["run"] and e["run"]["best_val_acc"] > best_acc_overall:
            best_acc_overall = e["run"]["best_val_acc"]
            best_model_overall = MODEL_LABELS.get(e["model"], e["model"])

    # Collect metrics for charts
    all_metrics_json = {}
    for exp in experiments:
        if exp["run"] and exp["run"].get("metrics"):
            run_id = exp["exp_id"]
            m = exp["run"]["metrics"]
            all_metrics_json[run_id] = {
                "epochs": [x["epoch"] for x in m],
                "train_acc": [x["train_acc"] for x in m],
                "val_acc": [x["val_acc"] for x in m],
                "train_loss": [x["train_loss"] for x in m],
                "val_loss": [x["val_loss"] for x in m],
            }

    # Build comparative chart data per phase/level
    # Phase A: CIFAR-10 levels 1,2,3
    # Phase B: MNIST levels 1,2,3
    # Phase C: isolation types
    # Phase D: clean baselines
    comparison_data = {}

    # Phase A comparisons
    for level_id in [1, 2, 3]:
        key = f"cifar10_L{level_id}"
        level_exps = [e for e in experiments if e["phase"] == "A" and e["level"] == level_id]
        comparison_data[key] = {
            "label": f"CIFAR-10 — Level {level_id} ({DEGRADATION_LEVELS[level_id]['name']})",
            "models": [],
            "accuracies": [],
            "colors": [],
            "borders": [],
        }
        for e in level_exps:
            comparison_data[key]["models"].append(MODEL_LABELS.get(e["model"], e["model"]))
            acc = e["run"]["best_val_acc"] * 100 if e["run"] else 0
            comparison_data[key]["accuracies"].append(round(acc, 1))
            comparison_data[key]["colors"].append(MODEL_COLORS_CHART.get(e["model"], "rgba(200,200,200,0.8)"))
            comparison_data[key]["borders"].append(MODEL_BORDERS_CHART.get(e["model"], "rgba(200,200,200,1)"))

    # Phase B comparisons
    for level_id in [1, 2, 3]:
        key = f"mnist_L{level_id}"
        level_exps = [e for e in experiments if e["phase"] == "B" and e["level"] == level_id]
        comparison_data[key] = {
            "label": f"MNIST — Level {level_id} ({DEGRADATION_LEVELS[level_id]['name']})",
            "models": [],
            "accuracies": [],
            "colors": [],
            "borders": [],
        }
        for e in level_exps:
            comparison_data[key]["models"].append(MODEL_LABELS.get(e["model"], e["model"]))
            acc = e["run"]["best_val_acc"] * 100 if e["run"] else 0
            comparison_data[key]["accuracies"].append(round(acc, 1))
            comparison_data[key]["colors"].append(MODEL_COLORS_CHART.get(e["model"], "rgba(200,200,200,0.8)"))
            comparison_data[key]["borders"].append(MODEL_BORDERS_CHART.get(e["model"], "rgba(200,200,200,1)"))

    # Phase C comparisons - by isolation type
    for iso_type in ISOLATION_TYPES:
        key = f"iso_{iso_type}"
        iso_exps = [e for e in experiments if e["phase"] == "C" and e["degradation_type"] == iso_type]
        comparison_data[key] = {
            "label": f"Isolation — {iso_type.replace('_', ' ').title()} Only",
            "models": [],
            "accuracies": [],
            "colors": [],
            "borders": [],
        }
        for e in iso_exps:
            comparison_data[key]["models"].append(MODEL_LABELS.get(e["model"], e["model"]))
            acc = e["run"]["best_val_acc"] * 100 if e["run"] else 0
            comparison_data[key]["accuracies"].append(round(acc, 1))
            comparison_data[key]["colors"].append(MODEL_COLORS_CHART.get(e["model"], "rgba(200,200,200,0.8)"))
            comparison_data[key]["borders"].append(MODEL_BORDERS_CHART.get(e["model"], "rgba(200,200,200,1)"))

    # Phase D comparisons — clean per dataset
    for ds in DATASETS:
        key = f"clean_{ds}"
        clean_exps = [e for e in experiments if e["phase"] == "D" and e["dataset"] == ds]
        ds_label = "CIFAR-10" if ds == "cifar10" else "MNIST"
        comparison_data[key] = {
            "label": f"Clean Baseline — {ds_label}",
            "models": [],
            "accuracies": [],
            "colors": [],
            "borders": [],
        }
        for e in clean_exps:
            comparison_data[key]["models"].append(MODEL_LABELS.get(e["model"], e["model"]))
            acc = e["run"]["best_val_acc"] * 100 if e["run"] else 0
            comparison_data[key]["accuracies"].append(round(acc, 1))
            comparison_data[key]["colors"].append(MODEL_COLORS_CHART.get(e["model"], "rgba(200,200,200,0.8)"))
            comparison_data[key]["borders"].append(MODEL_BORDERS_CHART.get(e["model"], "rgba(200,200,200,1)"))

    # Cross-level comparison (all models across all levels for CIFAR-10)
    cross_level_data = {"labels": [], "datasets": []}
    for model in MODELS:
        model_accs = []
        for level_id in [1, 2, 3]:
            exps = [e for e in experiments if e["phase"] == "A"
                    and e["level"] == level_id and e["model"] == model]
            if exps and exps[0]["run"]:
                model_accs.append(round(exps[0]["run"]["best_val_acc"] * 100, 1))
            else:
                model_accs.append(0)
        cross_level_data["datasets"].append({
            "label": MODEL_LABELS[model],
            "data": model_accs,
            "borderColor": MODEL_BORDERS_CHART[model],
            "backgroundColor": MODEL_COLORS_CHART[model],
            "tension": 0.3,
            "pointRadius": 6,
            "pointHoverRadius": 9,
            "borderWidth": 3,
        })
    cross_level_data["labels"] = ["Level 1 (Mild)", "Level 2 (Moderate)", "Level 3 (Severe)"]

    # ── Build experiment table rows ──
    def build_experiment_rows(phase_exps):
        rows_html = ""
        for exp in phase_exps:
            run = exp.get("run")
            status = "completed" if run else "pending"
            status_icon = "✅" if run else "⏳"
            acc_val = run["best_val_acc"] * 100 if run else 0
            acc_display = f"{acc_val:.1f}%" if run else "—"
            acc_class = ""
            if run:
                acc_class = "acc-high" if acc_val >= 70 else ("acc-mid" if acc_val >= 50 else "acc-low")
            epochs_display = str(run["epochs"]) if run else "—"
            date_display = run.get("date", "—") if run else "—"
            duration = run.get("total_time", "") if run else ""
            if duration:
                try:
                    secs = float(duration.replace("s", ""))
                    mins = secs / 60
                    duration = f"{mins:.0f}m" if mins > 1 else f"{secs:.0f}s"
                except ValueError:
                    pass
            if not duration:
                duration = "—"

            model_color = MODEL_COLORS.get(exp["model"], "#aaa")
            model_label = MODEL_LABELS.get(exp["model"], exp["model"])
            ds_label = "CIFAR-10" if exp["dataset"] == "cifar10" else "MNIST"

            deg_img_key = f"{exp['phase']}_{exp.get('level', 0)}_{exp.get('degradation_type', 'all')}"
            deg_b64 = degraded_images.get(deg_img_key, "")
            orig_b64 = original_images.get(exp["dataset"], "")

            # Clickable for learning curve
            click_attr = ""
            if run:
                click_attr = f'onclick="showLearningCurve(\'{exp["exp_id"]}\')" style="cursor:pointer;"'

            # Build sample images HTML: original → degraded
            sample_html = ""
            if orig_b64 or deg_b64:
                orig_tag = f"<img src='data:image/png;base64,{orig_b64}' class='sample-thumb' title='Original'>" if orig_b64 else ""
                deg_tag = f"<img src='data:image/png;base64,{deg_b64}' class='sample-thumb' title='Degraded'>" if deg_b64 else ""
                arrow = "<span class='sample-arrow'>→</span>" if orig_b64 and deg_b64 else ""
                sample_html = f"{orig_tag}{arrow}{deg_tag}"
            else:
                sample_html = "—"

            rows_html += f"""
            <tr class="exp-row {status}" data-phase="{exp['phase']}" data-model="{exp['model']}"
                data-dataset="{exp['dataset']}" data-status="{status}" data-expid="{exp['exp_id']}" {click_attr}>
                <td class="id-cell">{exp['exp_id']}</td>
                <td>{status_icon}</td>
                <td><span class="model-dot" style="background:{model_color}"></span>{model_label}</td>
                <td>{ds_label}</td>
                <td>{exp['level_name']}</td>
                <td class="{acc_class}">{acc_display}</td>
                <td>{epochs_display}</td>
                <td>{date_display}</td>
                <td>{duration}</td>
                <td class="sample-cell">
                    {sample_html}
                </td>
            </tr>"""
        return rows_html

    # Build phase sections
    phases = [
        ("A", "Phase A — CIFAR-10 Systematic", "3 degradation levels × 3 models",
         [e for e in experiments if e["phase"] == "A"]),
        ("B", "Phase B — MNIST Systematic", "3 degradation levels × 3 models",
         [e for e in experiments if e["phase"] == "B"]),
        ("C", "Phase C — Single-Degradation Isolation", "4 degradation types × 3 models (CIFAR-10)",
         [e for e in experiments if e["phase"] == "C"]),
        ("D", "Phase D — Clean Baselines", "No degradation, 3 models × 2 datasets",
         [e for e in experiments if e["phase"] == "D"]),
    ]

    phase_sections_html = ""
    for phase_id, phase_title, phase_desc, phase_exps in phases:
        phase_completed = sum(1 for e in phase_exps if e.get("run"))
        phase_total = len(phase_exps)
        pct = (phase_completed / phase_total * 100) if phase_total else 0

        rows = build_experiment_rows(phase_exps)

        # Determine which comparison charts belong to this phase
        chart_keys = []
        if phase_id == "A":
            chart_keys = [f"cifar10_L{i}" for i in [1, 2, 3]]
        elif phase_id == "B":
            chart_keys = [f"mnist_L{i}" for i in [1, 2, 3]]
        elif phase_id == "C":
            chart_keys = [f"iso_{t}" for t in ISOLATION_TYPES]
        elif phase_id == "D":
            chart_keys = [f"clean_{ds}" for ds in DATASETS]

        charts_html = ""
        for ck in chart_keys:
            charts_html += f"""
            <div class="comparison-chart-box">
                <canvas id="chart-{ck}" height="220"></canvas>
            </div>"""

        phase_sections_html += f"""
        <div class="phase-section" id="phase-{phase_id}">
            <div class="phase-header">
                <div>
                    <h2 class="phase-title">{phase_title}</h2>
                    <div class="phase-desc">{phase_desc}</div>
                </div>
                <div class="phase-progress">
                    <div class="progress-bar-outer">
                        <div class="progress-bar-inner" style="width:{pct:.0f}%"></div>
                    </div>
                    <span class="progress-text">{phase_completed}/{phase_total} completed</span>
                </div>
            </div>

            <table class="exp-table">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Status</th>
                        <th>Model</th>
                        <th>Dataset</th>
                        <th>Degradation</th>
                        <th>Best Acc</th>
                        <th>Epochs</th>
                        <th>Date</th>
                        <th>Duration</th>
                        <th>Sample</th>
                    </tr>
                </thead>
                <tbody>{rows}
                </tbody>
            </table>

            <div class="comparison-charts-row">
                {charts_html}
            </div>
        </div>"""

    # Cross-level chart for Phase A
    cross_level_chart_html = """
    <div class="cross-level-section">
        <h2 class="section-title">Cross-Level Robustness — CIFAR-10</h2>
        <p class="section-desc">Accuracy vs. degradation severity for each model (Phase A)</p>
        <div class="cross-chart-container">
            <canvas id="chart-cross-level" height="300"></canvas>
        </div>
    </div>"""

    # Learning curve modal
    learning_curve_html = """
    <div id="lc-modal" class="lc-modal" style="display:none;">
        <div class="lc-modal-content">
            <div class="lc-modal-header">
                <span id="lc-title">Learning Curve</span>
                <button onclick="closeLCModal()" class="lc-close">&times;</button>
            </div>
            <canvas id="lc-canvas" height="300"></canvas>
        </div>
    </div>"""

    now_str = datetime.now().strftime("%d/%m/%Y %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Experiment Plan Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root {{
    --bg:        #0b0d10;
    --surface:   #12151a;
    --surface2:  #181c22;
    --surface3:  #1e232b;
    --border:    #2a3040;
    --border-hi: #3a4560;
    --text:      #e8eaf0;
    --text-dim:  #8890a0;
    --accent:    #ffffff;
    --accent2:   #d0d4dc;
    --accent-dim: rgba(255,255,255,0.08);
    --green:     #4caf50;
    --green2:    #81c784;
    --orange:    #ffab40;
    --red:       #ff5252;
    --blue:      #4fc3f7;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 24px;
    line-height: 1.6;
}}

/* Header */
.header {{
    text-align: center;
    margin-bottom: 32px;
    padding: 32px 20px;
    background: linear-gradient(135deg, var(--surface) 0%, var(--surface3) 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
}}
.header h1 {{
    font-size: 2em;
    color: var(--accent);
    margin-bottom: 6px;
    letter-spacing: -0.5px;
}}
.header .subtitle {{
    color: var(--text-dim);
    font-size: 0.95em;
}}

/* Stats bar */
.stats-bar {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 16px;
    margin-bottom: 28px;
}}
.stat-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    transition: border-color 0.2s;
}}
.stat-card:hover {{ border-color: var(--accent); }}
.stat-value {{
    font-size: 2em;
    font-weight: 700;
    color: var(--accent);
    line-height: 1.2;
}}
.stat-value.pending {{ color: var(--orange); }}
.stat-label {{
    font-size: 0.78em;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

/* Filters */
.filters {{
    display: flex;
    gap: 14px;
    margin-bottom: 24px;
    flex-wrap: wrap;
    align-items: flex-end;
}}
.filter-group label {{
    display: block;
    font-size: 0.75em;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}}
.filter-group select {{
    background: var(--surface2);
    color: var(--text);
    border: 1px solid var(--border);
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 0.88em;
    cursor: pointer;
    transition: border-color 0.2s;
}}
.filter-group select:hover {{ border-color: var(--accent); }}
.filter-group select:focus {{ outline: none; border-color: var(--accent); }}

/* Phase sections */
.phase-section {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    margin-bottom: 28px;
    overflow: hidden;
}}
.phase-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 24px;
    background: var(--surface2);
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
    gap: 16px;
}}
.phase-title {{
    color: var(--accent);
    font-size: 1.25em;
    font-weight: 600;
}}
.phase-desc {{
    color: var(--text-dim);
    font-size: 0.85em;
}}
.phase-progress {{
    display: flex;
    align-items: center;
    gap: 12px;
}}
.progress-bar-outer {{
    width: 120px;
    height: 8px;
    background: var(--surface3);
    border-radius: 4px;
    overflow: hidden;
}}
.progress-bar-inner {{
    height: 100%;
    background: linear-gradient(90deg, var(--green), var(--green2));
    border-radius: 4px;
    transition: width 0.5s ease;
}}
.progress-text {{
    font-size: 0.82em;
    color: var(--text-dim);
    white-space: nowrap;
}}

/* Experiment table */
.exp-table {{
    width: 100%;
    border-collapse: collapse;
}}
.exp-table th {{
    background: var(--surface3);
    padding: 10px 14px;
    text-align: left;
    font-size: 0.75em;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    z-index: 1;
}}
.exp-table td {{
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    font-size: 0.88em;
}}
.exp-table tr.exp-row:hover {{
    background: var(--accent-dim);
}}
.exp-table tr.exp-row.pending td {{
    opacity: 0.5;
}}
.exp-table tr.exp-row.completed td {{
    opacity: 1;
}}
.id-cell {{
    font-family: 'Consolas', 'Monaco', monospace;
    font-weight: 600;
    color: var(--blue);
    font-size: 0.85em !important;
}}
.model-dot {{
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 8px;
    vertical-align: middle;
}}
.acc-high {{ color: var(--green2); font-weight: 700; }}
.acc-mid {{ color: var(--orange); font-weight: 600; }}
.acc-low {{ color: var(--red); font-weight: 600; }}
.sample-thumb {{
    width: 48px;
    height: 48px;
    border-radius: 6px;
    border: 1px solid var(--border);
    image-rendering: pixelated;
    vertical-align: middle;
}}
.sample-arrow {{
    color: var(--text2);
    font-size: 16px;
    margin: 0 4px;
    vertical-align: middle;
}}
.sample-cell {{
    padding: 4px 10px !important;
    white-space: nowrap;
}}

/* Comparison charts */
.comparison-charts-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px;
    padding: 20px 24px;
    background: var(--surface2);
    border-top: 1px solid var(--border);
}}
.comparison-chart-box {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
}}

/* Cross-level section */
.cross-level-section {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 28px;
}}
.section-title {{
    color: var(--accent);
    font-size: 1.2em;
    margin-bottom: 4px;
}}
.section-desc {{
    color: var(--text-dim);
    font-size: 0.85em;
    margin-bottom: 16px;
}}
.cross-chart-container {{
    max-width: 700px;
    margin: 0 auto;
}}

/* Learning curve modal */
.lc-modal {{
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.8);
    z-index: 1000;
    display: flex;
    align-items: center;
    justify-content: center;
}}
.lc-modal-content {{
    background: var(--surface);
    border: 1px solid var(--accent);
    border-radius: 14px;
    padding: 24px;
    width: 90%;
    max-width: 750px;
}}
.lc-modal-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
}}
.lc-modal-header span {{
    color: var(--accent);
    font-size: 1.1em;
    font-weight: 600;
}}
.lc-close {{
    background: var(--surface2);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 4px 12px;
    cursor: pointer;
    font-size: 1.3em;
}}
.lc-close:hover {{ border-color: var(--red); color: var(--red); }}

/* Footer */
footer {{
    text-align: center;
    color: var(--text-dim);
    font-size: 0.78em;
    margin-top: 32px;
    padding: 16px;
    border-top: 1px solid var(--border);
}}
footer a {{ color: var(--accent); text-decoration: none; }}

/* Scrollbar */
::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-track {{ background: var(--bg); }}
::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--text-dim); }}
</style>
</head>
<body>

<div class="header">
    <h1>🔬 Experiment Plan Dashboard</h1>
    <div class="subtitle">
        Object Classification in Low-Resolution THz-like Imagery — 36 Experiments, 4 Phases<br>
        Generated: {now_str}
    </div>
</div>

<div class="stats-bar">
    <div class="stat-card">
        <div class="stat-value">{total}</div>
        <div class="stat-label">Total Experiments</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{completed}</div>
        <div class="stat-label">Completed</div>
    </div>
    <div class="stat-card">
        <div class="stat-value pending">{pending}</div>
        <div class="stat-label">Pending</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{best_acc_overall*100:.1f}%</div>
        <div class="stat-label">Best Accuracy</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{best_model_overall or '—'}</div>
        <div class="stat-label">Top Model</div>
    </div>
</div>

<div class="filters">
    <div class="filter-group">
        <label>Phase</label>
        <select id="filterPhase" onchange="applyFilters()">
            <option value="all">All Phases</option>
            <option value="A">A — CIFAR-10 Systematic</option>
            <option value="B">B — MNIST Systematic</option>
            <option value="C">C — Isolation</option>
            <option value="D">D — Clean Baseline</option>
        </select>
    </div>
    <div class="filter-group">
        <label>Model</label>
        <select id="filterModel" onchange="applyFilters()">
            <option value="all">All Models</option>
            <option value="resnet50">ResNet-50</option>
            <option value="densenet121">DenseNet-121</option>
            <option value="transnext_micro">TransNeXt Micro</option>
        </select>
    </div>
    <div class="filter-group">
        <label>Dataset</label>
        <select id="filterDataset" onchange="applyFilters()">
            <option value="all">All Datasets</option>
            <option value="cifar10">CIFAR-10</option>
            <option value="mnist">MNIST</option>
        </select>
    </div>
    <div class="filter-group">
        <label>Status</label>
        <select id="filterStatus" onchange="applyFilters()">
            <option value="all">All</option>
            <option value="completed">Completed ✅</option>
            <option value="pending">Pending ⏳</option>
        </select>
    </div>
</div>

{phase_sections_html}

{cross_level_chart_html}

{learning_curve_html}

<footer>
    Experiment Plan Dashboard — THz-like Image Classification Project<br>
    Full plan: <a href="#">EXPERIMENT_PLAN.md</a> |
    36 experiments across 4 phases, 3 models, 2 datasets
</footer>

<script>
// ── Data ──
const ALL_METRICS = {json.dumps(all_metrics_json)};
const COMPARISON_DATA = {json.dumps(comparison_data)};
const CROSS_LEVEL_DATA = {json.dumps(cross_level_data)};

// ── Chart.js defaults ──
Chart.defaults.color = '#8890a0';
Chart.defaults.borderColor = '#2a3040';
Chart.defaults.font.family = "'Segoe UI', sans-serif";

// ── Comparison bar charts ──
const compCharts = {{}};
for (const [key, data] of Object.entries(COMPARISON_DATA)) {{
    const canvas = document.getElementById('chart-' + key);
    if (!canvas) continue;
    const hasData = data.accuracies.some(v => v > 0);
    compCharts[key] = new Chart(canvas, {{
        type: 'bar',
        data: {{
            labels: data.models,
            datasets: [{{
                label: 'Best Val Accuracy (%)',
                data: data.accuracies,
                backgroundColor: data.colors,
                borderColor: data.borders,
                borderWidth: 2,
                borderRadius: 6,
                barPercentage: 0.7,
            }}]
        }},
        options: {{
            responsive: true,
            plugins: {{
                title: {{
                    display: true,
                    text: data.label + (hasData ? '' : ' (pending)'),
                    color: hasData ? '#ffffff' : '#8890a0',
                    font: {{ size: 13, weight: '600' }}
                }},
                legend: {{ display: false }},
                tooltip: {{
                    callbacks: {{
                        label: ctx => ctx.parsed.y > 0 ? ctx.parsed.y.toFixed(1) + '%' : 'Not yet run'
                    }}
                }}
            }},
            scales: {{
                y: {{
                    beginAtZero: true,
                    max: 100,
                    title: {{ display: true, text: 'Accuracy (%)', color: '#8890a0' }},
                    grid: {{ color: '#1e232b' }},
                    ticks: {{ color: '#8890a0' }},
                }},
                x: {{
                    grid: {{ display: false }},
                    ticks: {{ color: '#e8eaf0', font: {{ size: 11 }} }},
                }}
            }}
        }}
    }});
}}

// ── Cross-level line chart ──
const crossCanvas = document.getElementById('chart-cross-level');
if (crossCanvas && CROSS_LEVEL_DATA.datasets) {{
    new Chart(crossCanvas, {{
        type: 'line',
        data: {{
            labels: CROSS_LEVEL_DATA.labels,
            datasets: CROSS_LEVEL_DATA.datasets,
        }},
        options: {{
            responsive: true,
            plugins: {{
                title: {{
                    display: true,
                    text: 'Accuracy vs. Degradation Level (CIFAR-10)',
                    color: '#ffffff',
                    font: {{ size: 14, weight: '600' }}
                }},
                legend: {{
                    labels: {{ color: '#e8eaf0', usePointStyle: true, pointStyle: 'circle' }}
                }},
                tooltip: {{
                    callbacks: {{
                        label: ctx => ctx.dataset.label + ': ' + (ctx.parsed.y > 0 ? ctx.parsed.y.toFixed(1) + '%' : 'pending')
                    }}
                }}
            }},
            scales: {{
                y: {{
                    beginAtZero: true,
                    max: 100,
                    title: {{ display: true, text: 'Best Val Accuracy (%)', color: '#8890a0' }},
                    grid: {{ color: '#1e232b' }},
                    ticks: {{ color: '#8890a0' }},
                }},
                x: {{
                    grid: {{ color: '#1e232b' }},
                    ticks: {{ color: '#e8eaf0' }},
                }}
            }}
        }}
    }});
}}

// ── Learning curve modal ──
let lcChart = null;
function showLearningCurve(expId) {{
    const data = ALL_METRICS[expId];
    if (!data) return;
    const modal = document.getElementById('lc-modal');
    const canvas = document.getElementById('lc-canvas');
    const title = document.getElementById('lc-title');
    title.textContent = 'Learning Curve — ' + expId;
    modal.style.display = 'flex';

    if (lcChart) lcChart.destroy();
    lcChart = new Chart(canvas, {{
        type: 'line',
        data: {{
            labels: data.epochs,
            datasets: [
                {{
                    label: 'Train Accuracy',
                    data: data.train_acc.map(v => v * 100),
                    borderColor: '#4fc3f7',
                    backgroundColor: 'rgba(79,195,247,0.1)',
                    tension: 0.3,
                    pointRadius: 3,
                    borderWidth: 2,
                }},
                {{
                    label: 'Val Accuracy',
                    data: data.val_acc.map(v => v * 100),
                    borderColor: '#81d4fa',
                    backgroundColor: 'rgba(129,212,250,0.1)',
                    tension: 0.3,
                    pointRadius: 3,
                    borderWidth: 2,
                    borderDash: [5, 3],
                }},
                {{
                    label: 'Train Loss',
                    data: data.train_loss,
                    borderColor: '#ff5252',
                    tension: 0.3,
                    pointRadius: 2,
                    borderWidth: 1.5,
                    yAxisID: 'yLoss',
                    hidden: true,
                }},
                {{
                    label: 'Val Loss',
                    data: data.val_loss,
                    borderColor: '#ffab40',
                    tension: 0.3,
                    pointRadius: 2,
                    borderWidth: 1.5,
                    borderDash: [5, 3],
                    yAxisID: 'yLoss',
                    hidden: true,
                }},
            ]
        }},
        options: {{
            responsive: true,
            interaction: {{ mode: 'index', intersect: false }},
            plugins: {{
                legend: {{ labels: {{ color: '#e8eaf0', usePointStyle: true }} }},
            }},
            scales: {{
                x: {{
                    title: {{ display: true, text: 'Epoch', color: '#8890a0' }},
                    grid: {{ color: '#1e232b' }},
                    ticks: {{ color: '#8890a0' }},
                }},
                y: {{
                    position: 'left',
                    title: {{ display: true, text: 'Accuracy (%)', color: '#8890a0' }},
                    grid: {{ color: '#1e232b' }},
                    ticks: {{ color: '#8890a0' }},
                    min: 0, max: 100,
                }},
                yLoss: {{
                    position: 'right',
                    title: {{ display: true, text: 'Loss', color: '#8890a0' }},
                    grid: {{ drawOnChartArea: false }},
                    ticks: {{ color: '#8890a0' }},
                }},
            }},
        }},
    }});
}}

function closeLCModal() {{
    document.getElementById('lc-modal').style.display = 'none';
    if (lcChart) {{ lcChart.destroy(); lcChart = null; }}
}}

document.getElementById('lc-modal').addEventListener('click', function(e) {{
    if (e.target === this) closeLCModal();
}});

// ── Filters ──
function applyFilters() {{
    const phase = document.getElementById('filterPhase').value;
    const model = document.getElementById('filterModel').value;
    const dataset = document.getElementById('filterDataset').value;
    const status = document.getElementById('filterStatus').value;

    // Show/hide phase sections
    document.querySelectorAll('.phase-section').forEach(sec => {{
        const secPhase = sec.id.replace('phase-', '');
        if (phase === 'all' || secPhase === phase) {{
            sec.style.display = '';
        }} else {{
            sec.style.display = 'none';
        }}
    }});

    // Show/hide rows
    document.querySelectorAll('.exp-row').forEach(row => {{
        let show = true;
        if (phase !== 'all' && row.dataset.phase !== phase) show = false;
        if (model !== 'all' && row.dataset.model !== model) show = false;
        if (dataset !== 'all' && row.dataset.dataset !== dataset) show = false;
        if (status !== 'all' && row.dataset.status !== status) show = false;
        row.style.display = show ? '' : 'none';
    }});
}}

// ── Keyboard ──
document.addEventListener('keydown', function(e) {{
    if (e.key === 'Escape') closeLCModal();
}});
</script>

</body>
</html>"""

    return html


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    runs_root = Path("runs")

    print("[PLAN] Building experiment plan (36 experiments)...")
    experiments = build_experiment_plan()
    print(f"       {len(experiments)} experiments defined")

    print("[SCAN] Scanning existing runs...")
    runs = scan_runs(runs_root)
    print(f"       {len(runs)} completed runs found")

    # Generate sample images for both datasets
    print("[IMAGES] Generating degradation samples...")
    cifar_img, cifar_label = get_cifar10_sample()
    mnist_img, mnist_label = get_mnist_sample()

    original_images = {
        "cifar10": tensor_to_base64(cifar_img, size=112),
        "mnist": tensor_to_base64(mnist_img, size=112),
    }
    # Map dataset to its original image tensor
    dataset_originals = {"cifar10": cifar_img, "mnist": mnist_img}

    degraded_images = {}
    # For each unique degradation config
    seen = set()
    for exp in experiments:
        deg_key = f"{exp['phase']}_{exp.get('level', 0)}_{exp.get('degradation_type', 'all')}"
        if deg_key in seen:
            continue
        seen.add(deg_key)
        params = exp["deg_params"]
        if params.get("low_res", 224) < 200:  # Only degrade if there's actual degradation
            try:
                src_img = dataset_originals.get(exp["dataset"], cifar_img)
                degraded_images[deg_key] = generate_degraded_image(params, src_img)
                print(f"       {deg_key}: generated ({exp['dataset']})")
            except Exception as e:
                print(f"       {deg_key}: FAILED ({e})")

    print("[HTML] Generating dashboard...")
    html = generate_html(experiments, runs, original_images, degraded_images)

    out_path = Path("artifacts") / "dashboard_experiment_plan.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[OK] Dashboard saved to: {out_path}")
    matched = sum(1 for e in experiments
                  if match_experiment_to_run(e, runs) is not None)
    print(f"     {matched}/{len(experiments)} experiments have results")


if __name__ == "__main__":
    main()
