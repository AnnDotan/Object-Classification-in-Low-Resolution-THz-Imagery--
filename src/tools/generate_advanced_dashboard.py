#!/usr/bin/env python3
"""
Generate an advanced interactive HTML dashboard with:
- Latest experiment results with hyperparameters
- Per-model robustness analysis (3 graphs x 3 curves)
- Learning curve viewer
- Train vs Validation comparison
- Detailed results table
"""

import csv
from pathlib import Path
import json
import re


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


def find_systematic_experiments(runs: list[dict]) -> dict:
    """Find the 9 systematic experiments (3 models x 3 degradation types)."""
    MODELS = ["resnet50", "densenet121", "transnext_micro"]
    DEGRADATIONS = ["downsampling", "blur", "salt_pepper"]

    experiments = {}
    for model in MODELS:
        for deg in DEGRADATIONS:
            key = f"{deg}_{model}"
            # Find matching run
            for run in runs:
                tag = run.get("tag", "")
                run_name = run.get("run_name", "")
                model_name = run.get("model_name", "")
                if model_name == model and tag == key and run.get("group") == "official":
                    experiments[key] = run
                    break
            # Fallback: search by run_name pattern
            if key not in experiments:
                for run in runs:
                    if key in run.get("run_name", "") and run.get("model_name") == model:
                        experiments[key] = run
                        break

    return experiments


def generate_advanced_html(runs: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    models = sorted(set(r.get("model_name", "") for r in runs if r.get("model_name")))
    completed_runs = [r for r in runs if r.get("best_val_acc") is not None]

    # Find the 9 systematic experiments
    systematic = find_systematic_experiments(runs)

    # Load learning curves for systematic experiments
    systematic_curves = {}
    for key, run in systematic.items():
        metrics = get_metrics_csv(run["run_dir"])
        if metrics:
            systematic_curves[key] = metrics

    # Load configs for systematic experiments
    systematic_configs = {}
    for key, run in systematic.items():
        cfg = get_run_config(run["run_dir"])
        systematic_configs[key] = cfg

    # Top runs for general learning curve viewer
    top_runs = sorted(completed_runs, key=lambda x: x.get("best_val_acc", 0), reverse=True)[:10]
    learning_curves = {}
    for run in top_runs:
        metrics = get_metrics_csv(run["run_dir"])
        if metrics:
            learning_curves[run["run_name"]] = metrics
    top_runs_data = [{"name": r["run_name"], "model": r["model_name"], "acc": r["best_val_acc"], "deg": r["low_res"]} for r in top_runs]

    # Model comparison
    model_comparison = []
    for model in models:
        model_runs = [r for r in completed_runs if r.get("model_name") == model]
        if model_runs:
            accs = [r.get("best_val_acc", 0) for r in model_runs]
            model_comparison.append({"model": model, "avg_acc": sum(accs)/len(accs), "max_acc": max(accs), "count": len(model_runs)})

    # Build systematic experiment summary table data
    MODELS = ["resnet50", "densenet121", "transnext_micro"]
    DEGRADATIONS = ["downsampling", "blur", "salt_pepper"]

    systematic_table = []
    for model in MODELS:
        for deg in DEGRADATIONS:
            key = f"{deg}_{model}"
            run = systematic.get(key, {})
            cfg = systematic_configs.get(key, {})
            acc = run.get("best_val_acc")
            systematic_table.append({
                "model": model,
                "degradation": deg,
                "best_val_acc": acc,
                "epochs": cfg.get("epochs", run.get("epochs", "")),
                "batch_size": cfg.get("batch_size", run.get("batch_size", "")),
                "lr": cfg.get("lr", run.get("lr", "")),
                "low_res": cfg.get("low_res", run.get("low_res", "")),
                "out_size": cfg.get("out_size", run.get("out_size", "")),
                "pretrained": cfg.get("pretrained", run.get("pretrained", "")),
                "degradation_type": cfg.get("degradation_type", deg),
                "total_time": run.get("total_time", ""),
                "status": "completed" if acc is not None else "pending",
            })

    # Build per-model curve data for the 3 comparison charts
    per_model_curves = {}
    for model in MODELS:
        per_model_curves[model] = {}
        for deg in DEGRADATIONS:
            key = f"{deg}_{model}"
            if key in systematic_curves:
                per_model_curves[model][deg] = systematic_curves[key]

    DEG_COLORS = {
        "downsampling": "#2196F3",
        "blur": "#FF9800",
        "salt_pepper": "#E91E63",
    }
    DEG_LABELS = {
        "downsampling": "Downsampling",
        "blur": "Gaussian Blur",
        "salt_pepper": "Salt & Pepper Noise",
    }
    MODEL_LABELS = {
        "resnet50": "ResNet-50",
        "densenet121": "DenseNet-121",
        "transnext_micro": "TransNeXt Micro",
    }

    # Pre-build summary table rows (avoid f-string brace issues)
    summary_rows_html = ""
    for m in MODELS:
        cells = ""
        for d in DEGRADATIONS:
            key = f"{d}_{m}"
            run_data = systematic.get(key, {})
            acc_val = run_data.get("best_val_acc") if isinstance(run_data, dict) else None
            cells += f'<td class="acc-cell {_acc_class(acc_val)}">{_acc_fmt(acc_val)}</td>'
        summary_rows_html += f'<tr><td><strong>{MODEL_LABELS.get(m, m)}</strong></td>{cells}</tr>\n'

    # Pre-build model robustness chart divs
    model_chart_divs = ""
    for m in MODELS:
        model_chart_divs += f'<div class="chart-wrapper"><h3 class="chart-title">{MODEL_LABELS.get(m, m)}</h3><div id="modelRobust_{m}" style="height:380px;"></div></div>\n'

    # Pre-build train vs val chart divs
    trainval_chart_divs = ""
    for m in MODELS:
        trainval_chart_divs += f'<div class="chart-wrapper"><h3 class="chart-title">{MODEL_LABELS.get(m, m)} - Train vs Val</h3><div id="trainVal_{m}" style="height:380px;"></div></div>\n'

    # Pre-build model filter options
    model_options_html = "".join(f'<option value="{m}">{m}</option>' for m in models)

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
        .hyperparams-box {{
            background: #f0f4ff;
            border: 1px solid #c5cae9;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 25px;
        }}
        .hyperparams-title {{
            font-weight: 600;
            font-size: 15px;
            margin-bottom: 12px;
            color: #333;
        }}
        .hyperparams-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 10px;
        }}
        .hp-item {{
            display: flex;
            justify-content: space-between;
            padding: 6px 10px;
            background: white;
            border-radius: 4px;
            font-size: 13px;
        }}
        .hp-label {{ color: #666; }}
        .hp-value {{ font-weight: 600; color: #333; }}
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
        footer {{
            background: #f0f0f0;
            padding: 15px;
            text-align: center;
            font-size: 12px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Advanced Results Dashboard</h1>
            <p class="subtitle">Low-Resolution Image Classification - Degradation Robustness Analysis</p>
            <p class="subtitle">3 Models x 3 Degradation Types = 9 Systematic Experiments</p>
        </header>

        <div class="content">
            <!-- Stats -->
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{len(runs)}</div>
                    <div class="stat-label">Total Runs</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(completed_runs)}</div>
                    <div class="stat-label">Completed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(systematic)}</div>
                    <div class="stat-label">Systematic (9)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{max([r.get('best_val_acc', 0) for r in completed_runs], default=0):.1%}</div>
                    <div class="stat-label">Best Accuracy</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(models)}</div>
                    <div class="stat-label">Models</div>
                </div>
            </div>

            <!-- SECTION 1: Latest Systematic Experiments -->
            <div class="section">
                <h2 class="section-title">Latest Systematic Experiments (3x3)</h2>

                <!-- Hyperparameters -->
                <div class="hyperparams-box">
                    <div class="hyperparams-title">Common Hyperparameters (Identical Across All 9 Runs)</div>
                    <div class="hyperparams-grid">
                        <div class="hp-item"><span class="hp-label">Epochs:</span><span class="hp-value">20</span></div>
                        <div class="hp-item"><span class="hp-label">Batch Size:</span><span class="hp-value">32</span></div>
                        <div class="hp-item"><span class="hp-label">Learning Rate:</span><span class="hp-value">0.001</span></div>
                        <div class="hp-item"><span class="hp-label">Low Res:</span><span class="hp-value">16</span></div>
                        <div class="hp-item"><span class="hp-label">Output Size:</span><span class="hp-value">224</span></div>
                        <div class="hp-item"><span class="hp-label">Pretrained:</span><span class="hp-value">True</span></div>
                        <div class="hp-item"><span class="hp-label">Train Subset:</span><span class="hp-value">5,000</span></div>
                        <div class="hp-item"><span class="hp-label">Val Subset:</span><span class="hp-value">2,000</span></div>
                        <div class="hp-item"><span class="hp-label">Dataset:</span><span class="hp-value">CIFAR-10</span></div>
                        <div class="hp-item"><span class="hp-label">Optimizer:</span><span class="hp-value">AdamW</span></div>
                    </div>
                </div>

                <!-- Results summary table -->
                <table class="params-table">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Downsampling</th>
                            <th>Gaussian Blur</th>
                            <th>Salt & Pepper</th>
                        </tr>
                    </thead>
                    <tbody>
                        {summary_rows_html}
                    </tbody>
                </table>
            </div>

            <!-- SECTION 2: Per-Model Robustness (3 graphs, 3 curves each) -->
            <div class="section">
                <h2 class="section-title">Model Robustness Analysis - Learning Curves by Degradation</h2>
                <p style="margin-bottom:20px; color:#666; font-size:14px;">
                    Each graph shows one model tested on three different degradation types.
                    Compare how each model handles different types of visual corruption.
                </p>
                <div class="charts-grid">
                    {model_chart_divs}
                </div>
            </div>

            <!-- SECTION 3: Train vs Validation Analysis -->
            <div class="section">
                <h2 class="section-title">Train vs Validation - Overfitting Analysis</h2>
                <p style="margin-bottom:20px; color:#666; font-size:14px;">
                    Compare training accuracy vs validation accuracy. Large gaps indicate overfitting.
                </p>
                <div class="charts-grid">
                    {trainval_chart_divs}
                </div>
            </div>

            <!-- SECTION 4: Performance Overview -->
            <div class="section">
                <h2 class="section-title">Performance Overview</h2>
                <div class="charts-grid">
                    <div class="chart-wrapper">
                        <h3 class="chart-title">Model Comparison (All Runs)</h3>
                        <div id="modelChart" style="height:350px;"></div>
                    </div>
                    <div class="chart-wrapper">
                        <h3 class="chart-title">Degradation Type Comparison</h3>
                        <div id="degradationBarChart" style="height:350px;"></div>
                    </div>
                </div>
            </div>

            <!-- SECTION 5: Interactive Learning Curve Viewer -->
            <div class="section">
                <h2 class="section-title">Interactive Learning Curve Viewer</h2>
                <p style="margin-bottom:15px; color:#666; font-size:14px;">
                    Click a run card to view its full learning curve.
                </p>
                <div class="top-runs-list" id="topRunsList"></div>
                <div class="chart-wrapper">
                    <div id="learningCurveChart" style="height:420px;"></div>
                </div>
            </div>

            <!-- SECTION 6: Detailed Results Table -->
            <div class="section">
                <h2 class="section-title">All Experiment Results</h2>
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
                    </div>
                </div>
                <div class="table-container">
                    <table id="resultsTable">
                        <thead>
                            <tr>
                                <th>Group</th>
                                <th>Run Name</th>
                                <th>Model</th>
                                <th>Degradation</th>
                                <th>Best Val Acc</th>
                                <th>Out Size</th>
                                <th>Batch</th>
                                <th>Epochs</th>
                            </tr>
                        </thead>
                        <tbody id="tableBody"></tbody>
                    </table>
                </div>
            </div>
        </div>

        <footer>
            <p>Advanced Results Dashboard | {len(completed_runs)}/{len(runs)} runs completed | Generated by generate_advanced_dashboard.py</p>
        </footer>
    </div>

    <script>
        const runsData = {json.dumps(runs)};
        const completedRuns = runsData.filter(r => r.best_val_acc !== null);
        const learningCurvesData = {json.dumps(learning_curves)};
        const topRunsData = {json.dumps(top_runs_data)};
        const perModelCurves = {json.dumps(per_model_curves)};
        const systematicTable = {json.dumps(systematic_table)};

        const DEG_COLORS = {json.dumps(DEG_COLORS)};
        const DEG_LABELS = {json.dumps(DEG_LABELS)};
        const MODEL_LABELS = {json.dumps(MODEL_LABELS)};
        const MODELS = {json.dumps(MODELS)};
        const DEGRADATIONS = {json.dumps(DEGRADATIONS)};

        // =====================================================
        // Per-Model Robustness Charts (3 graphs x 3 curves)
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
        // Train vs Validation Analysis (overfitting detection)
        // =====================================================
        function createTrainValCharts() {{
            MODELS.forEach(model => {{
                const traces = [];
                DEGRADATIONS.forEach(deg => {{
                    const curveData = perModelCurves[model]?.[deg];
                    if (curveData && curveData.length > 0) {{
                        // Train accuracy - dashed
                        traces.push({{
                            x: curveData.map(d => d.epoch),
                            y: curveData.map(d => d.train_acc),
                            name: DEG_LABELS[deg] + ' (Train)',
                            type: 'scatter',
                            mode: 'lines',
                            line: {{color: DEG_COLORS[deg], width: 1.5, dash: 'dash'}},
                            legendgroup: deg
                        }});
                        // Val accuracy - solid
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
        // Overview Charts
        // =====================================================
        function createOverviewCharts() {{
            const modelComparison = {json.dumps(model_comparison)};
            Plotly.newPlot('modelChart', [
                {{
                    x: modelComparison.map(m => m.model),
                    y: modelComparison.map(m => m.max_acc),
                    name: 'Max Accuracy',
                    type: 'bar',
                    marker: {{color: '#667eea'}}
                }},
                {{
                    x: modelComparison.map(m => m.model),
                    y: modelComparison.map(m => m.avg_acc),
                    name: 'Avg Accuracy',
                    type: 'bar',
                    marker: {{color: '#764ba2'}}
                }}
            ], {{
                xaxis: {{title: 'Model'}},
                yaxis: {{title: 'Accuracy'}},
                barmode: 'group',
                hovermode: 'closest'
            }}, {{responsive: true}});

            // Degradation comparison from systematic experiments
            const degData = {{}};
            systematicTable.forEach(row => {{
                if (row.best_val_acc !== null) {{
                    if (!degData[row.degradation]) degData[row.degradation] = [];
                    degData[row.degradation].push(row.best_val_acc);
                }}
            }});
            const degLabels = Object.keys(degData).map(d => DEG_LABELS[d] || d);
            const degAvg = Object.values(degData).map(accs => accs.reduce((a,b) => a+b, 0) / accs.length);
            const degMax = Object.values(degData).map(accs => Math.max(...accs));

            Plotly.newPlot('degradationBarChart', [
                {{ x: degLabels, y: degMax, name: 'Max Accuracy', type: 'bar', marker: {{color: '#2196F3'}} }},
                {{ x: degLabels, y: degAvg, name: 'Avg Accuracy', type: 'bar', marker: {{color: '#E91E63'}} }}
            ], {{
                xaxis: {{title: 'Degradation Type'}},
                yaxis: {{title: 'Accuracy'}},
                barmode: 'group',
                hovermode: 'closest'
            }}, {{responsive: true}});
        }}

        // =====================================================
        // Interactive Learning Curve Viewer
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
        // Results Table
        // =====================================================
        function populateTable(data) {{
            const tbody = document.getElementById('tableBody');
            tbody.innerHTML = '';
            data.forEach(run => {{
                const tr = document.createElement('tr');
                const accColor = run.best_val_acc === null ? '' :
                    (run.best_val_acc >= 0.6 ? 'acc-good' : run.best_val_acc >= 0.5 ? 'acc-warn' : 'acc-poor');
                tr.innerHTML = `
                    <td><span class="badge badge-${{run.group}}">${{run.group}}</span></td>
                    <td><small title="${{run.run_name}}">${{run.run_name.substring(0,45)}}</small></td>
                    <td><strong>${{run.model_name || '-'}}</strong></td>
                    <td><code>${{run.low_res ? 'lr='+run.low_res : '-'}}</code></td>
                    <td class="acc-cell ${{accColor}}">${{run.best_val_acc !== null ? (run.best_val_acc*100).toFixed(2)+'%' : '-'}}</td>
                    <td>${{run.out_size || '-'}}</td>
                    <td>${{run.batch_size || '-'}}</td>
                    <td>${{run.epochs || '-'}}</td>`;
                tbody.appendChild(tr);
            }});
        }}

        function filterTable() {{
            const search = document.getElementById('searchBox').value.toLowerCase();
            const group = document.getElementById('groupFilter').value;
            const model = document.getElementById('modelFilter').value;
            populateTable(runsData.filter(r =>
                (!search || r.run_name.toLowerCase().includes(search)) &&
                (!group || r.group === group) &&
                (!model || r.model_name === model)
            ));
        }}

        function resetFilters() {{
            document.getElementById('searchBox').value = '';
            document.getElementById('groupFilter').value = '';
            document.getElementById('modelFilter').value = '';
            populateTable(runsData);
        }}

        // =====================================================
        // Initialize
        // =====================================================
        document.addEventListener('DOMContentLoaded', function() {{
            createModelRobustnessCharts();
            createTrainValCharts();
            createOverviewCharts();
            populateTopRunsList();
            populateTable(runsData);
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


def main():
    csv_path = Path("artifacts/tables/run_summary.csv")
    if not csv_path.exists():
        print(f"[ERROR] Run summary not found: {csv_path}")
        return
    print("[GENERATE] Generating advanced dashboard...")
    runs = load_run_summary(csv_path)
    output = Path("artifacts/dashboard_advanced.html")
    generate_advanced_html(runs, output)


if __name__ == "__main__":
    main()
