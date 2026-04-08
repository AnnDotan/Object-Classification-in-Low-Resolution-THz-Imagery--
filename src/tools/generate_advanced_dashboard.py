#!/usr/bin/env python3
"""
Generate an advanced interactive HTML dashboard with:
- Full-pipeline model comparison (main project goal)
- Learning curve viewer for full-pipeline runs
- Separate section for single-degradation experiments (extension)
- Detailed results tables
"""

import csv
from pathlib import Path
import json
import sys


SINGLE_DEG_TAGS = {
    "blur": "Gaussian Blur",
    "downsampling": "Downsampling",
    "salt_pepper": "Salt & Pepper",
    "noise": "Gaussian Noise",
}

MODELS = ["resnet50", "densenet121", "transnext_micro"]
DEGRADATIONS = ["downsampling", "blur", "salt_pepper"]

DEG_COLORS = {
    "downsampling": "#2196F3",
    "blur": "#FF9800",
    "salt_pepper": "#E91E63",
}
DEG_LABELS = {
    "downsampling": "Downsampling Only",
    "blur": "Gaussian Blur Only",
    "salt_pepper": "Salt & Pepper Only",
}
MODEL_LABELS = {
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
    "transnext_micro": "TransNeXt Micro",
}


def load_run_summary(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("best_val_acc"):
                try:
                    row["best_val_acc"] = float(row["best_val_acc"])
                except ValueError:
                    row["best_val_acc"] = None
            else:
                row["best_val_acc"] = None
            rows.append(row)
    return rows


def get_metrics_csv(run_dir: str) -> list[dict] | None:
    metrics_path = Path(run_dir) / "metrics.csv"
    if not metrics_path.exists():
        return None
    rows = []
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("epoch"):
                    try:
                        rows.append({
                            "epoch": int(row["epoch"]),
                            "train_loss": float(row["train_loss"]),
                            "train_acc": float(row["train_acc"]),
                            "val_loss": float(row["val_loss"]),
                            "val_acc": float(row["val_acc"]),
                        })
                    except (ValueError, KeyError):
                        continue
    except Exception:
        return None
    return rows if rows else None


def get_run_config(run_dir: str) -> dict:
    config_path = Path(run_dir) / "run_config.txt"
    data = {}
    if config_path.exists():
        for line in config_path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                data[k.strip()] = v.strip()
    return data


def classify_single_degradation(run: dict) -> str | None:
    """Return degradation label if single-degradation run, else None."""
    tag = run.get("tag", "")
    for keyword, label in SINGLE_DEG_TAGS.items():
        # Match tags like "blur_resnet50", "downsampling_quick", "noise_quick",
        # "salt_pepper_densenet121", "exp1_resnet50_downsampling"
        if tag.startswith(keyword + "_") or tag.startswith(f"exp1_resnet50_{keyword}"):
            return label
        if tag == keyword:
            return label
        # Match "downsampling_robustness"
        if keyword == "downsampling" and tag == "downsampling_robustness":
            return label
    return None


def find_systematic_experiments(runs: list[dict]) -> dict:
    """Find the 9 systematic experiments (3 models x 3 degradation types)."""
    experiments = {}
    for model in MODELS:
        for deg in DEGRADATIONS:
            key = f"{deg}_{model}"
            for run in runs:
                tag = run.get("tag", "")
                model_name = run.get("model_name", "")
                if model_name == model and tag == key and run.get("group") == "official":
                    experiments[key] = run
                    break
            if key not in experiments:
                for run in runs:
                    if key in run.get("run_name", "") and run.get("model_name") == model:
                        experiments[key] = run
                        break
    return experiments


def _acc_class(val):
    if val is None:
        return "acc-pending"
    if val >= 0.7:
        return "acc-good"
    if val >= 0.5:
        return "acc-warn"
    return "acc-poor"


def _acc_fmt(val):
    if val is None:
        return "pending..."
    return f"{val*100:.2f}%"


def load_sample_images() -> dict:
    """Load pre-generated sample images from JSON."""
    path = Path("artifacts/tables/sample_images.json")
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def make_config_key(low_res, out_size, deg_type):
    return f"lr{low_res}_out{out_size}_deg{deg_type}"


def generate_advanced_html(runs: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_images = load_sample_images()

    models = sorted(set(r.get("model_name", "") for r in runs if r.get("model_name")))
    completed_runs = [r for r in runs if r.get("best_val_acc") is not None]

    # Enrich runs with degradation config key and strength details
    for r in runs:
        run_dir = r.get("run_dir", "")
        low_res = r.get("low_res", "16")
        out_size = r.get("out_size", "32")
        cfg_data = get_run_config(run_dir)
        deg_type = cfg_data.get("degradation_type", "all")
        r["_config_key"] = make_config_key(low_res, out_size, deg_type) if low_res and out_size else ""
        r["_deg_type"] = deg_type
        # Build degradation strength details based on active degradation type
        # Defaults from DegradeConfig
        lr_val = int(low_res) if low_res and low_res.isdigit() else 0
        details = {}
        if deg_type in ("all", "downsampling"):
            details["low_res"] = lr_val
        if deg_type in ("all", "blur"):
            details["blur_kernel"] = 5
            details["blur_sigma"] = 1.0
        if deg_type in ("all", "noise"):
            details["noise_std"] = 0.08
        if deg_type in ("all", "salt_pepper"):
            details["salt_pepper"] = 0.05
        if deg_type == "all":
            details["p_grayscale"] = 0.3
        r["_deg_details"] = details
        # Compute a single "degradation severity" score for sorting
        # Lower low_res = more severe; more active degradations = more severe
        severity = 0.0
        if "low_res" in details and details["low_res"] > 0:
            severity += (32.0 / details["low_res"])  # lr=8->4, lr=16->2, lr=32->1
        if "blur_kernel" in details:
            severity += details["blur_kernel"] * details["blur_sigma"]
        if "noise_std" in details:
            severity += details["noise_std"] * 50  # 0.08*50=4
        if "salt_pepper" in details:
            severity += details["salt_pepper"] * 50  # 0.05*50=2.5
        if "p_grayscale" in details:
            severity += details["p_grayscale"] * 3  # 0.3*3=0.9
        r["_deg_severity"] = round(severity, 2)

    # Classify runs into full-pipeline vs single-degradation
    full_pipeline_runs = []
    single_deg_runs = []
    for r in runs:
        deg_label = classify_single_degradation(r)
        if deg_label:
            r["_deg_label"] = deg_label
            single_deg_runs.append(r)
        else:
            full_pipeline_runs.append(r)

    fp_completed = [r for r in full_pipeline_runs if r.get("best_val_acc") is not None]
    sd_completed = [r for r in single_deg_runs if r.get("best_val_acc") is not None]

    # --- Full-pipeline model comparison (main goal) ---
    fp_model_comparison = []
    for model in models:
        model_runs = [r for r in fp_completed if r.get("model_name") == model]
        if model_runs:
            accs = [r.get("best_val_acc", 0) for r in model_runs]
            fp_model_comparison.append({
                "model": model, "avg_acc": sum(accs)/len(accs),
                "max_acc": max(accs), "count": len(model_runs),
            })

    # Top full-pipeline runs for learning curve viewer
    top_runs = sorted(fp_completed, key=lambda x: x.get("best_val_acc", 0), reverse=True)[:10]
    learning_curves = {}
    for run in top_runs:
        metrics = get_metrics_csv(run["run_dir"])
        if metrics:
            learning_curves[run["run_name"]] = metrics
    top_runs_data = [
        {"name": r["run_name"], "model": r["model_name"],
         "acc": r["best_val_acc"], "deg": r["low_res"]}
        for r in top_runs
    ]

    # --- Single-degradation systematic experiments ---
    systematic = find_systematic_experiments(runs)
    systematic_curves = {}
    for key, run in systematic.items():
        metrics = get_metrics_csv(run["run_dir"])
        if metrics:
            systematic_curves[key] = metrics

    systematic_configs = {}
    for key, run in systematic.items():
        cfg = get_run_config(run["run_dir"])
        systematic_configs[key] = cfg

    systematic_table = []
    for model in MODELS:
        for deg in DEGRADATIONS:
            key = f"{deg}_{model}"
            run = systematic.get(key, {})
            cfg = systematic_configs.get(key, {})
            acc = run.get("best_val_acc")
            systematic_table.append({
                "model": model, "degradation": deg,
                "best_val_acc": acc,
                "status": "completed" if acc is not None else "pending",
            })

    per_model_curves = {}
    for model in MODELS:
        per_model_curves[model] = {}
        for deg in DEGRADATIONS:
            key = f"{deg}_{model}"
            if key in systematic_curves:
                per_model_curves[model][deg] = systematic_curves[key]

    # --- Pre-build HTML fragments ---
    # 3x3 summary table
    summary_rows_html = ""
    for m in MODELS:
        cells = ""
        for d in DEGRADATIONS:
            key = f"{d}_{m}"
            run_data = systematic.get(key, {})
            acc_val = run_data.get("best_val_acc") if isinstance(run_data, dict) else None
            cells += f'<td class="acc-cell {_acc_class(acc_val)}">{_acc_fmt(acc_val)}</td>'
        summary_rows_html += f'<tr><td><strong>{MODEL_LABELS.get(m, m)}</strong></td>{cells}</tr>\n'

    # Single-degradation detailed table (all sd runs with degradation type column)
    sd_sorted = sorted(sd_completed, key=lambda x: x.get("best_val_acc", 0), reverse=True)
    sd_table_rows = ""
    for r in sd_sorted:
        acc_val = r.get("best_val_acc")
        deg_label = r.get("_deg_label", "?")
        ck = r.get("_config_key", "")
        sample_btn = (
            f'<button class="btn-sample" onclick="showSampleModal(\'{ck}\')">View</button>'
            if ck and ck in sample_images else '-'
        )
        sd_table_rows += (
            f'<tr>'
            f'<td><span class="badge badge-{r.get("group", "")}">{r.get("group", "")}</span></td>'
            f'<td><small title="{r.get("run_name", "")}">{r.get("run_name", "")[:50]}</small></td>'
            f'<td><strong>{r.get("model_name", "-")}</strong></td>'
            f'<td><span class="deg-badge">{deg_label}</span></td>'
            f'<td class="acc-cell {_acc_class(acc_val)}">{_acc_fmt(acc_val)}</td>'
            f'<td>{r.get("low_res", "-")}</td>'
            f'<td>{r.get("out_size", "-")}</td>'
            f'<td>{r.get("epochs", "-")}</td>'
            f'<td>{sample_btn}</td>'
            f'</tr>\n'
        )

    # Model robustness chart divs (for single-deg section)
    model_chart_divs = ""
    for m in MODELS:
        model_chart_divs += (
            f'<div class="chart-wrapper"><h3 class="chart-title">'
            f'{MODEL_LABELS.get(m, m)}</h3>'
            f'<div id="modelRobust_{m}" style="height:380px;"></div></div>\n'
        )

    # Train vs val chart divs (for single-deg section)
    trainval_chart_divs = ""
    for m in MODELS:
        trainval_chart_divs += (
            f'<div class="chart-wrapper"><h3 class="chart-title">'
            f'{MODEL_LABELS.get(m, m)} - Train vs Val</h3>'
            f'<div id="trainVal_{m}" style="height:380px;"></div></div>\n'
        )

    model_options_html = "".join(f'<option value="{m}">{m}</option>' for m in models)

    best_fp_acc = max([r.get('best_val_acc', 0) for r in fp_completed], default=0)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Results Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 20px;
            text-align: center;
        }}
        h1 {{ font-size: 32px; margin-bottom: 8px; }}
        .subtitle {{ font-size: 14px; opacity: 0.9; margin-bottom: 5px; }}
        .content {{ padding: 40px; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 15px;
            margin-bottom: 40px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 18px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{ font-size: 28px; font-weight: bold; margin-bottom: 5px; }}
        .stat-label {{ font-size: 11px; opacity: 0.9; }}
        .section {{ margin-bottom: 50px; }}
        .section-title {{
            font-size: 22px;
            font-weight: 700;
            margin-bottom: 20px;
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        @media (max-width: 1100px) {{
            .charts-grid {{ grid-template-columns: 1fr; }}
        }}
        .chart-wrapper {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            border: 1px solid #e0e0e0;
        }}
        .chart-title {{
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 10px;
            color: #333;
        }}
        .params-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            margin-bottom: 20px;
        }}
        .params-table th {{
            background: #667eea;
            color: white;
            padding: 10px;
            text-align: left;
        }}
        .params-table td {{
            padding: 8px 10px;
            border-bottom: 1px solid #eee;
        }}
        .params-table tr:hover {{ background: #f5f5ff; }}
        .acc-cell {{
            font-weight: bold;
            font-size: 14px;
        }}
        .acc-good {{ color: #2e7d32; }}
        .acc-warn {{ color: #f57c00; }}
        .acc-poor {{ color: #c62828; }}
        .acc-pending {{ color: #999; font-style: italic; }}
        .top-runs-list {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 12px;
            margin-bottom: 20px;
        }}
        .run-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 12px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .run-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        .run-card.active {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        .run-card-name {{
            font-weight: 600;
            font-size: 13px;
            margin-bottom: 6px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .run-card-stats {{
            font-size: 11px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 5px;
        }}
        .table-container {{ overflow-x: auto; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        thead {{ background: #f0f0f0; font-weight: 600; }}
        th {{ padding: 10px; text-align: left; border-bottom: 2px solid #ddd; }}
        td {{ padding: 10px; border-bottom: 1px solid #eee; }}
        tbody tr:hover {{ background: #f9f9f9; }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 500;
        }}
        .badge-official {{ background: #e3f2fd; color: #1976d2; }}
        .badge-pilot {{ background: #f3e5f5; color: #7b1fa2; }}
        .badge-archive {{ background: #f5f5f5; color: #666; }}
        .deg-badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            background: #fff3e0;
            color: #e65100;
        }}
        .filter-section {{
            margin-bottom: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .filter-group {{
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            align-items: center;
        }}
        input[type="text"], select {{
            padding: 7px 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 13px;
        }}
        button {{
            padding: 7px 14px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
        }}
        button:hover {{ background: #764ba2; }}
        .extension-header {{
            background: linear-gradient(135deg, #f5f5f5 0%, #e0e0e0 100%);
            border: 2px solid #bdbdbd;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 25px;
            text-align: center;
        }}
        .extension-header h2 {{
            font-size: 20px;
            color: #555;
            margin-bottom: 8px;
        }}
        .extension-header p {{
            font-size: 13px;
            color: #888;
        }}
        .divider {{
            border: none;
            border-top: 3px dashed #ccc;
            margin: 60px 0 40px 0;
        }}
        footer {{
            background: #f0f0f0;
            padding: 15px;
            text-align: center;
            font-size: 12px;
            color: #666;
        }}
        /* Sample image modal */
        .modal-overlay {{
            display: none;
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(0,0,0,0.6);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }}
        .modal-overlay.active {{ display: flex; }}
        .modal-content {{
            background: white;
            border-radius: 12px;
            padding: 30px;
            max-width: 500px;
            width: 90%;
            text-align: center;
            box-shadow: 0 20px 60px rgba(0,0,0,0.4);
            position: relative;
        }}
        .modal-close {{
            position: absolute;
            top: 10px; right: 15px;
            font-size: 24px;
            cursor: pointer;
            color: #999;
            background: none;
            border: none;
            padding: 5px;
        }}
        .modal-close:hover {{ color: #333; }}
        .modal-title {{
            font-size: 16px;
            font-weight: 700;
            margin-bottom: 15px;
            color: #333;
        }}
        .sample-images {{
            display: flex;
            gap: 20px;
            justify-content: center;
            align-items: flex-start;
            margin-bottom: 15px;
        }}
        .sample-box {{
            text-align: center;
        }}
        .sample-box img {{
            border: 2px solid #ddd;
            border-radius: 6px;
            image-rendering: pixelated;
        }}
        .sample-box .sample-label {{
            font-size: 12px;
            font-weight: 600;
            margin-top: 6px;
            color: #555;
        }}
        .sample-info {{
            font-size: 12px;
            color: #888;
            margin-top: 8px;
        }}
        .btn-sample {{
            padding: 3px 8px;
            font-size: 11px;
            background: #e3f2fd;
            color: #1976d2;
            border: 1px solid #90caf9;
            border-radius: 4px;
            cursor: pointer;
        }}
        .btn-sample:hover {{ background: #bbdefb; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Advanced Results Dashboard</h1>
            <p class="subtitle">Low-Resolution Image Classification - Full Degradation Pipeline</p>
            <p class="subtitle">Comparing model robustness under combined degradation (downsampling + blur + noise + grayscale)</p>
        </header>

        <div class="content">
            <!-- Stats -->
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{len(full_pipeline_runs)}</div>
                    <div class="stat-label">Full-Pipeline Runs</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(fp_completed)}</div>
                    <div class="stat-label">Completed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{best_fp_acc:.1%}</div>
                    <div class="stat-label">Best Accuracy (Full Pipeline)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(models)}</div>
                    <div class="stat-label">Models Tested</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(sd_completed)}</div>
                    <div class="stat-label">Single-Degradation Runs</div>
                </div>
            </div>

            <!-- ============================================================ -->
            <!-- MAIN: Full Degradation Pipeline - Model Comparison            -->
            <!-- ============================================================ -->

            <!-- SECTION 1: Model Comparison Chart (Full Pipeline Only) -->
            <div class="section">
                <h2 class="section-title">Model Comparison - Full Degradation Pipeline</h2>
                <p style="margin-bottom:20px; color:#666; font-size:14px;">
                    Models compared on data that underwent the <strong>complete degradation pipeline</strong>
                    (downsampling + blur + noise + grayscale). This is the core project objective.
                </p>
                <div class="charts-grid">
                    <div class="chart-wrapper">
                        <h3 class="chart-title">Max &amp; Average Accuracy per Model (Full Pipeline Only)</h3>
                        <div id="modelChart" style="height:350px;"></div>
                    </div>
                </div>
            </div>

            <!-- SECTION 2: Learning Curves (Full Pipeline Only) -->
            <div class="section">
                <h2 class="section-title">Learning Curves - Full Pipeline Runs</h2>
                <p style="margin-bottom:15px; color:#666; font-size:14px;">
                    Click a run card to view its full learning curve. Only full-pipeline runs shown.
                </p>
                <div class="top-runs-list" id="topRunsList"></div>
                <div class="chart-wrapper">
                    <div id="learningCurveChart" style="height:420px;"></div>
                </div>
            </div>

            <!-- SECTION 3: Full Pipeline Results Table -->
            <div class="section">
                <h2 class="section-title">Full Pipeline - All Experiment Results</h2>
                <div class="filter-section">
                    <div class="filter-group">
                        <input type="text" id="searchBox" placeholder="Search..." style="flex:1; max-width:250px;">
                        <select id="groupFilter">
                            <option value="">All Groups</option>
                            <option value="official">Official</option>
                            <option value="pilot">Pilot</option>
                            <option value="archive">Archive</option>
                        </select>
                        <select id="modelFilter">
                            <option value="">All Models</option>
                            {model_options_html}
                        </select>
                        <button onclick="filterTable()">Filter</button>
                        <button onclick="resetFilters()" style="background:#999;">Reset</button>
                        <span style="margin-left:15px; font-size:12px; color:#666;">Sort by:</span>
                        <button onclick="sortTable('config')" style="background:#43a047; font-size:11px; padding:5px 10px;">Config Group</button>
                        <button onclick="sortTable('severity')" style="background:#e65100; font-size:11px; padding:5px 10px;">Severity</button>
                        <button onclick="sortTable('accuracy')" style="background:#1565c0; font-size:11px; padding:5px 10px;">Accuracy</button>
                    </div>
                </div>
                <div class="table-container">
                    <table id="resultsTable">
                        <thead>
                            <tr>
                                <th>Group</th>
                                <th>Run Name</th>
                                <th>Model</th>
                                <th>Degradation Config</th>
                                <th>Degradation Details</th>
                                <th>Severity</th>
                                <th>Best Val Acc</th>
                                <th>Epochs</th>
                                <th>Sample</th>
                            </tr>
                        </thead>
                        <tbody id="tableBody"></tbody>
                    </table>
                </div>
            </div>

            <!-- ============================================================ -->
            <!-- DIVIDER                                                       -->
            <!-- ============================================================ -->
            <hr class="divider">

            <!-- ============================================================ -->
            <!-- EXTENSION: Single-Degradation Experiments                     -->
            <!-- ============================================================ -->
            <div class="extension-header">
                <h2>Extension: Single-Degradation Type Isolation</h2>
                <p>The experiments below tested models with only <strong>one degradation type at a time</strong>
                   (not the full pipeline). This is an additional analysis, not the main project objective.</p>
            </div>

            <!-- SECTION 4: Single-Degradation Summary Table -->
            <div class="section">
                <h2 class="section-title">Single-Degradation - All Runs</h2>
                <p style="margin-bottom:15px; color:#666; font-size:14px;">
                    Each row shows an experiment that used only one degradation type.
                    The "Degradation Type" column indicates which single degradation was applied.
                </p>
                <div class="table-container">
                    <table class="params-table">
                        <thead>
                            <tr>
                                <th>Group</th>
                                <th>Run Name</th>
                                <th>Model</th>
                                <th>Degradation Type</th>
                                <th>Best Val Acc</th>
                                <th>Low Res</th>
                                <th>Out Size</th>
                                <th>Epochs</th>
                                <th>Sample</th>
                            </tr>
                        </thead>
                        <tbody>
                            {sd_table_rows}
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- SECTION 5: 3x3 Systematic Grid -->
            <div class="section">
                <h2 class="section-title">Single-Degradation - Systematic 3x3 Grid</h2>
                <p style="margin-bottom:15px; color:#666; font-size:14px;">
                    3 models tested on 3 isolated degradation types (official runs only, 20 epochs).
                </p>
                <table class="params-table">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Downsampling Only</th>
                            <th>Gaussian Blur Only</th>
                            <th>Salt &amp; Pepper Only</th>
                        </tr>
                    </thead>
                    <tbody>
                        {summary_rows_html}
                    </tbody>
                </table>
            </div>

            <!-- SECTION 6: Single-Degradation Learning Curves -->
            <div class="section">
                <h2 class="section-title">Single-Degradation - Learning Curves by Model</h2>
                <p style="margin-bottom:20px; color:#666; font-size:14px;">
                    Each graph shows one model tested on three different single degradation types.
                </p>
                <div class="charts-grid">
                    {model_chart_divs}
                </div>
            </div>

            <!-- SECTION 7: Train vs Val (Single-Degradation) -->
            <div class="section">
                <h2 class="section-title">Single-Degradation - Train vs Validation</h2>
                <p style="margin-bottom:20px; color:#666; font-size:14px;">
                    Compare training accuracy vs validation accuracy per degradation type.
                    Large gaps indicate overfitting.
                </p>
                <div class="charts-grid">
                    {trainval_chart_divs}
                </div>
            </div>
        </div>

        <footer>
            <p>Advanced Results Dashboard | {len(fp_completed)} full-pipeline + {len(sd_completed)} single-degradation completed | Generated by generate_advanced_dashboard.py</p>
        </footer>
    </div>

    <!-- Sample Image Modal -->
    <div class="modal-overlay" id="sampleModal">
        <div class="modal-content">
            <button class="modal-close" onclick="closeSampleModal()">&times;</button>
            <div class="modal-title" id="modalTitle">Degradation Example</div>
            <div class="sample-images">
                <div class="sample-box">
                    <img id="modalOriginal" src="" width="128" height="128" alt="Original">
                    <div class="sample-label">Original</div>
                </div>
                <div class="sample-box">
                    <img id="modalDegraded" src="" width="128" height="128" alt="Degraded">
                    <div class="sample-label">After Degradation</div>
                </div>
            </div>
            <div class="sample-info" id="modalInfo"></div>
        </div>
    </div>

    <script>
        // Sample images data
        const sampleImages = {json.dumps(sample_images)};

        // Full-pipeline runs only (for main sections)
        const fpRunsData = {json.dumps(full_pipeline_runs)};
        const learningCurvesData = {json.dumps(learning_curves)};
        const topRunsData = {json.dumps(top_runs_data)};

        // Single-degradation data (for extension section)
        const perModelCurves = {json.dumps(per_model_curves)};
        const systematicTable = {json.dumps(systematic_table)};

        const DEG_COLORS = {json.dumps(DEG_COLORS)};
        const DEG_LABELS = {json.dumps(DEG_LABELS)};
        const MODEL_LABELS = {json.dumps(MODEL_LABELS)};
        const MODELS = {json.dumps(MODELS)};
        const DEGRADATIONS = {json.dumps(DEGRADATIONS)};

        // =====================================================
        // Model Comparison Chart (Full Pipeline Only)
        // =====================================================
        function createModelChart() {{
            const modelComparison = {json.dumps(fp_model_comparison)};
            if (!modelComparison.length) return;
            Plotly.newPlot('modelChart', [
                {{
                    x: modelComparison.map(m => m.model),
                    y: modelComparison.map(m => m.max_acc),
                    name: 'Max Accuracy',
                    type: 'bar',
                    marker: {{color: '#667eea'}},
                    text: modelComparison.map(m => (m.max_acc*100).toFixed(1)+'%'),
                    textposition: 'outside'
                }},
                {{
                    x: modelComparison.map(m => m.model),
                    y: modelComparison.map(m => m.avg_acc),
                    name: 'Avg Accuracy',
                    type: 'bar',
                    marker: {{color: '#764ba2'}},
                    text: modelComparison.map(m => (m.avg_acc*100).toFixed(1)+'%'),
                    textposition: 'outside'
                }}
            ], {{
                xaxis: {{title: 'Model'}},
                yaxis: {{title: 'Accuracy', range: [0, 1]}},
                barmode: 'group',
                hovermode: 'closest'
            }}, {{responsive: true}});
        }}

        // =====================================================
        // Interactive Learning Curve Viewer (Full Pipeline)
        // =====================================================
        let selectedRunName = null;

        function populateTopRunsList() {{
            const container = document.getElementById('topRunsList');
            container.innerHTML = '';
            topRunsData.forEach((run, idx) => {{
                const card = document.createElement('div');
                card.className = 'run-card' + (idx === 0 ? ' active' : '');
                if (idx === 0) selectedRunName = run.name;
                card.innerHTML = `
                    <div class="run-card-name" title="${{run.name}}">#${{idx+1}} ${{run.model}}</div>
                    <div class="run-card-stats">
                        <div><small>Acc:</small> <strong>${{(run.acc*100).toFixed(1)}}%</strong></div>
                        <div><small>LR:</small> <strong>${{run.deg}}</strong></div>
                    </div>`;
                card.onclick = () => {{
                    document.querySelectorAll('.run-card').forEach(e => e.classList.remove('active'));
                    card.classList.add('active');
                    selectedRunName = run.name;
                    displayLearningCurve(run.name);
                }};
                container.appendChild(card);
            }});
            if (topRunsData.length > 0) displayLearningCurve(topRunsData[0].name);
        }}

        function displayLearningCurve(runName) {{
            const metrics = learningCurvesData[runName];
            if (!metrics || !metrics.length) {{
                document.getElementById('learningCurveChart').innerHTML = '<p style="padding:20px;text-align:center;color:#999;">No metrics data</p>';
                return;
            }}
            Plotly.newPlot('learningCurveChart', [
                {{ x: metrics.map(m=>m.epoch), y: metrics.map(m=>m.train_acc), name: 'Train Acc', type: 'scatter', mode: 'lines+markers', line: {{color: '#667eea', width: 2}}, marker: {{size: 4}} }},
                {{ x: metrics.map(m=>m.epoch), y: metrics.map(m=>m.val_acc), name: 'Val Acc', type: 'scatter', mode: 'lines+markers', line: {{color: '#764ba2', width: 2}}, marker: {{size: 4}} }},
                {{ x: metrics.map(m=>m.epoch), y: metrics.map(m=>m.train_loss), name: 'Train Loss', type: 'scatter', mode: 'lines', line: {{color: '#ff9999', width: 1, dash: 'dot'}}, yaxis: 'y2', visible: 'legendonly' }},
                {{ x: metrics.map(m=>m.epoch), y: metrics.map(m=>m.val_loss), name: 'Val Loss', type: 'scatter', mode: 'lines', line: {{color: '#ff6666', width: 1, dash: 'dot'}}, yaxis: 'y2', visible: 'legendonly' }}
            ], {{
                title: runName.substring(0,60),
                hovermode: 'x unified',
                xaxis: {{title: 'Epoch'}},
                yaxis: {{title: 'Accuracy'}},
                yaxis2: {{title: 'Loss', overlaying: 'y', side: 'right'}}
            }}, {{responsive: true}});
        }}

        // =====================================================
        // Full Pipeline Results Table (color-coded by degradation config)
        // =====================================================
        const CONFIG_GROUP_COLORS = [
            'rgba(33,150,243,0.10)',   // blue
            'rgba(76,175,80,0.10)',    // green
            'rgba(255,152,0,0.10)',    // orange
            'rgba(156,39,176,0.10)',   // purple
            'rgba(0,150,136,0.10)',    // teal
            'rgba(244,67,54,0.10)',    // red
            'rgba(121,85,72,0.10)',    // brown
            'rgba(63,81,181,0.10)',    // indigo
        ];
        const CONFIG_GROUP_BORDERS = [
            'rgba(33,150,243,0.35)',
            'rgba(76,175,80,0.35)',
            'rgba(255,152,0,0.35)',
            'rgba(156,39,176,0.35)',
            'rgba(0,150,136,0.35)',
            'rgba(244,67,54,0.35)',
            'rgba(121,85,72,0.35)',
            'rgba(63,81,181,0.35)',
        ];

        function getConfigGroupColor(configKey, keyMap) {{
            if (!configKey) return '';
            if (!(configKey in keyMap)) {{
                keyMap[configKey] = Object.keys(keyMap).length;
            }}
            return CONFIG_GROUP_COLORS[keyMap[configKey] % CONFIG_GROUP_COLORS.length];
        }}

        function getConfigGroupBorder(configKey, keyMap) {{
            if (!configKey) return '';
            if (!(configKey in keyMap)) {{
                keyMap[configKey] = Object.keys(keyMap).length;
            }}
            return CONFIG_GROUP_BORDERS[keyMap[configKey] % CONFIG_GROUP_BORDERS.length];
        }}

        function configKeyToLabel(ck) {{
            if (!ck) return '-';
            const m = ck.match(/lr(\\d+)_out(\\d+)_deg(\\w+)/);
            if (!m) return ck;
            const deg = m[3] === 'all' ? 'Full Pipeline' : m[3];
            return `lr=${{m[1]}} out=${{m[2]}} ${{deg}}`;
        }}

        function degDetailsToHtml(d) {{
            if (!d || Object.keys(d).length === 0) return '-';
            const parts = [];
            if (d.low_res !== undefined) parts.push(`<span title="Downsampling resolution">DS=${{d.low_res}}</span>`);
            if (d.blur_kernel !== undefined) parts.push(`<span title="Blur kernel=${{d.blur_kernel}} sigma=${{d.blur_sigma}}">Blur=${{d.blur_kernel}}/${{d.blur_sigma}}</span>`);
            if (d.noise_std !== undefined) parts.push(`<span title="Gaussian noise std=${{d.noise_std}}">Noise=${{d.noise_std}}</span>`);
            if (d.salt_pepper !== undefined) parts.push(`<span title="Salt & pepper amount=${{d.salt_pepper}}">S&P=${{d.salt_pepper}}</span>`);
            if (d.p_grayscale !== undefined) parts.push(`<span title="Grayscale probability=${{d.p_grayscale}}">Gray=${{d.p_grayscale}}</span>`);
            return parts.join(' <small style="color:#ccc">|</small> ');
        }}

        let currentSortMode = 'config';

        function sortTable(mode) {{
            currentSortMode = mode;
            filterTable();
        }}

        function populateTable(data) {{
            const tbody = document.getElementById('tableBody');
            tbody.innerHTML = '';
            const sorted = [...data].sort((a, b) => {{
                if (currentSortMode === 'severity') {{
                    const sa = a._deg_severity || 0;
                    const sb = b._deg_severity || 0;
                    if (sa !== sb) return sb - sa;
                    return (b.best_val_acc || 0) - (a.best_val_acc || 0);
                }} else if (currentSortMode === 'accuracy') {{
                    return (b.best_val_acc || 0) - (a.best_val_acc || 0);
                }} else {{
                    // config group
                    const ka = a._config_key || 'zzz';
                    const kb = b._config_key || 'zzz';
                    if (ka !== kb) return ka.localeCompare(kb);
                    return (b.best_val_acc || 0) - (a.best_val_acc || 0);
                }}
            }});
            const keyMap = {{}};
            sorted.forEach(run => {{
                const tr = document.createElement('tr');
                const accColor = run.best_val_acc === null ? '' :
                    (run.best_val_acc >= 0.6 ? 'acc-good' : run.best_val_acc >= 0.5 ? 'acc-warn' : 'acc-poor');
                const ck = run._config_key || '';
                const bgColor = getConfigGroupColor(ck, keyMap);
                const bdColor = getConfigGroupBorder(ck, keyMap);
                if (bgColor) {{
                    tr.style.background = bgColor;
                    tr.style.borderLeft = '4px solid ' + bdColor;
                }}
                const sampleBtn = (ck && sampleImages[ck])
                    ? `<button class="btn-sample" onclick="showSampleModal('${{ck}}')">View</button>`
                    : '-';
                const sev = run._deg_severity || 0;
                const sevColor = sev >= 12 ? '#c62828' : sev >= 8 ? '#f57c00' : sev >= 4 ? '#1976d2' : '#666';
                tr.innerHTML = `
                    <td><span class="badge badge-${{run.group}}">${{run.group}}</span></td>
                    <td><small title="${{run.run_name}}">${{run.run_name.substring(0,45)}}</small></td>
                    <td><strong>${{run.model_name || '-'}}</strong></td>
                    <td><code>${{configKeyToLabel(ck)}}</code></td>
                    <td style="font-size:11px;">${{degDetailsToHtml(run._deg_details)}}</td>
                    <td style="font-weight:bold; color:${{sevColor}}">${{sev.toFixed(1)}}</td>
                    <td class="acc-cell ${{accColor}}">${{run.best_val_acc !== null ? (run.best_val_acc*100).toFixed(2)+'%' : '-'}}</td>
                    <td>${{run.epochs || '-'}}</td>
                    <td>${{sampleBtn}}</td>`;
                tbody.appendChild(tr);
            }});
        }}

        function filterTable() {{
            const search = document.getElementById('searchBox').value.toLowerCase();
            const group = document.getElementById('groupFilter').value;
            const model = document.getElementById('modelFilter').value;
            populateTable(fpRunsData.filter(r =>
                (!search || r.run_name.toLowerCase().includes(search)) &&
                (!group || r.group === group) &&
                (!model || r.model_name === model)
            ));
        }}

        function resetFilters() {{
            document.getElementById('searchBox').value = '';
            document.getElementById('groupFilter').value = '';
            document.getElementById('modelFilter').value = '';
            populateTable(fpRunsData);
        }}

        // =====================================================
        // Single-Degradation: Per-Model Robustness Charts
        // =====================================================
        function createModelRobustnessCharts() {{
            MODELS.forEach(model => {{
                const traces = [];
                DEGRADATIONS.forEach(deg => {{
                    const curveData = perModelCurves[model]?.[deg];
                    if (curveData && curveData.length > 0) {{
                        traces.push({{
                            x: curveData.map(d => d.epoch),
                            y: curveData.map(d => d.val_acc),
                            name: DEG_LABELS[deg] || deg,
                            type: 'scatter',
                            mode: 'lines+markers',
                            line: {{color: DEG_COLORS[deg], width: 2.5}},
                            marker: {{size: 5}}
                        }});
                    }}
                }});

                const layout = {{
                    hovermode: 'x unified',
                    xaxis: {{title: 'Epoch'}},
                    yaxis: {{title: 'Validation Accuracy', range: [0, 1]}},
                    legend: {{x: 0.01, y: 0.99, bgcolor: 'rgba(255,255,255,0.8)'}},
                    margin: {{t: 10, r: 20}}
                }};

                Plotly.newPlot('modelRobust_' + model, traces, layout, {{responsive: true}});
            }});
        }}

        // =====================================================
        // Single-Degradation: Train vs Validation
        // =====================================================
        function createTrainValCharts() {{
            MODELS.forEach(model => {{
                const traces = [];
                DEGRADATIONS.forEach(deg => {{
                    const curveData = perModelCurves[model]?.[deg];
                    if (curveData && curveData.length > 0) {{
                        traces.push({{
                            x: curveData.map(d => d.epoch),
                            y: curveData.map(d => d.train_acc),
                            name: DEG_LABELS[deg] + ' (Train)',
                            type: 'scatter',
                            mode: 'lines',
                            line: {{color: DEG_COLORS[deg], width: 1.5, dash: 'dash'}},
                            legendgroup: deg
                        }});
                        traces.push({{
                            x: curveData.map(d => d.epoch),
                            y: curveData.map(d => d.val_acc),
                            name: DEG_LABELS[deg] + ' (Val)',
                            type: 'scatter',
                            mode: 'lines+markers',
                            line: {{color: DEG_COLORS[deg], width: 2.5}},
                            marker: {{size: 4}},
                            legendgroup: deg
                        }});
                    }}
                }});

                const layout = {{
                    hovermode: 'x unified',
                    xaxis: {{title: 'Epoch'}},
                    yaxis: {{title: 'Accuracy', range: [0, 1]}},
                    legend: {{x: 0.01, y: 0.99, bgcolor: 'rgba(255,255,255,0.8)', font: {{size: 10}}}},
                    margin: {{t: 10, r: 20}}
                }};

                Plotly.newPlot('trainVal_' + model, traces, layout, {{responsive: true}});
            }});
        }}

        // =====================================================
        // Sample Image Modal
        // =====================================================
        function showSampleModal(configKey) {{
            const data = sampleImages[configKey];
            if (!data) {{
                alert('No sample image available for this configuration.');
                return;
            }}
            const degTypeLabel = data.degradation_type === 'all'
                ? 'Full Pipeline (all degradations)'
                : data.degradation_type.charAt(0).toUpperCase() + data.degradation_type.slice(1) + ' Only';
            document.getElementById('modalTitle').textContent =
                'Degradation Example: ' + degTypeLabel;
            document.getElementById('modalOriginal').src =
                'data:image/png;base64,' + data.original_b64;
            document.getElementById('modalDegraded').src =
                'data:image/png;base64,' + data.degraded_b64;
            document.getElementById('modalInfo').textContent =
                'CIFAR-10 "' + data.label + '" | low_res=' + data.low_res +
                ' | out_size=' + data.out_size + ' | degradation=' + data.degradation_type;
            document.getElementById('sampleModal').classList.add('active');
        }}

        function closeSampleModal() {{
            document.getElementById('sampleModal').classList.remove('active');
        }}

        document.getElementById('sampleModal').addEventListener('click', function(e) {{
            if (e.target === this) closeSampleModal();
        }});

        // =====================================================
        // Initialize
        // =====================================================
        document.addEventListener('DOMContentLoaded', function() {{
            createModelChart();
            populateTopRunsList();
            populateTable(fpRunsData);
            createModelRobustnessCharts();
            createTrainValCharts();
            document.getElementById('searchBox').addEventListener('keypress', e => {{
                if (e.key === 'Enter') filterTable();
            }});
        }});
    </script>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[OK] Advanced dashboard generated: {output_path}")
    print(f"     Full-pipeline runs: {len(fp_completed)} completed / {len(full_pipeline_runs)} total")
    print(f"     Single-degradation runs: {len(sd_completed)} completed / {len(single_deg_runs)} total")


def main():
    csv_path = Path("artifacts/tables/run_summary.csv")
    if not csv_path.exists():
        print(f"[ERROR] Run summary not found: {csv_path}")
        return
    print("[GENERATE] Generating advanced dashboard...")
    runs = load_run_summary(csv_path)
    # Filter: minimum 10 epochs, exclude incomplete/pilot runs
    runs = [r for r in runs if r.get("epochs") and int(r["epochs"]) >= 10]
    output = Path("artifacts/dashboard_advanced.html")
    generate_advanced_html(runs, output)


if __name__ == "__main__":
    main()
