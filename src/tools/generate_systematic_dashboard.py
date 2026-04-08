#!/usr/bin/env python3
"""
Generate Systematic Experiments Dashboard
==========================================

Creates an interactive HTML dashboard that:
1. Scans all run groups (official, pilot, systematic, etc.)
2. Filters OUT incomplete runs (no metrics) and low-accuracy runs (<30%)
3. Groups experiments by degradation configuration
4. Shows original vs degraded sample images with degradation parameters
5. Displays results tables with learning curves
"""

import csv
import json
import base64
import io
import sys
from pathlib import Path

import torch
from torchvision import datasets, transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.data.degrade import DegradeConfig, degrade_image


SAMPLE_IDX = 7  # horse from CIFAR-10


def get_cifar10_sample(idx: int = SAMPLE_IDX):
    ds = datasets.CIFAR10(root="./data", train=False, download=True,
                          transform=transforms.ToTensor())
    img, label = ds[idx]
    return img, ds.classes[label]


def tensor_to_base64(t: torch.Tensor, size: int = 128) -> str:
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


def scan_all_runs(runs_root: Path) -> list[dict]:
    """Scan all runs, filter out incomplete and <30% accuracy."""
    results = []
    for group_dir in sorted(runs_root.iterdir()):
        if not group_dir.is_dir():
            continue
        group = group_dir.name
        for run_dir in sorted(group_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            config = parse_config(run_dir)
            metrics = get_metrics(run_dir)

            # Filter: must have metrics
            if not metrics:
                continue

            best_val_acc = max(m["val_acc"] for m in metrics)

            # Filter: accuracy >= 30%
            if best_val_acc < 0.30:
                continue

            results.append({
                "group": group,
                "run_name": run_dir.name,
                "run_dir": str(run_dir),
                "model_name": config.get("model_name", "unknown"),
                "tag": config.get("tag", ""),
                "low_res": config.get("low_res", "16"),
                "out_size": config.get("out_size", "224"),
                "epochs": len(metrics),
                "best_val_acc": best_val_acc,
                "degradation_type": config.get("degradation_type", "all"),
                "lr": config.get("lr", ""),
                "backbone_lr": config.get("backbone_lr", "None"),
                "freeze_backbone": config.get("freeze_backbone", "False"),
                "weight_decay": config.get("weight_decay", ""),
                "label_smoothing": config.get("label_smoothing", ""),
                "blur_kernel": config.get("blur_kernel", "None"),
                "blur_sigma": config.get("blur_sigma", "None"),
                "gaussian_noise_std": config.get("gaussian_noise_std", "None"),
                "salt_pepper_amount": config.get("salt_pepper_amount", "None"),
                "p_grayscale": config.get("p_grayscale", "None"),
                "metrics": metrics,
            })

    return results


def make_deg_key(run: dict) -> str:
    """Create a unique key for the degradation configuration."""
    parts = [
        f"lr{run['low_res']}",
        f"deg_{run['degradation_type']}",
    ]
    if run.get("blur_kernel") not in (None, "None", ""):
        parts.append(f"bk{run['blur_kernel']}")
    if run.get("blur_sigma") not in (None, "None", ""):
        parts.append(f"bs{run['blur_sigma']}")
    if run.get("gaussian_noise_std") not in (None, "None", ""):
        parts.append(f"gn{run['gaussian_noise_std']}")
    if run.get("salt_pepper_amount") not in (None, "None", ""):
        parts.append(f"sp{run['salt_pepper_amount']}")
    return "_".join(parts)


def make_deg_label(run: dict) -> str:
    """Human-readable degradation description."""
    parts = []
    parts.append(f"Low Res: {run['low_res']}px")
    parts.append(f"Type: {run['degradation_type']}")
    if run.get("blur_kernel") not in (None, "None", ""):
        parts.append(f"Blur: kernel={run['blur_kernel']}, σ={run['blur_sigma']}")
    if run.get("gaussian_noise_std") not in (None, "None", ""):
        parts.append(f"Gaussian Noise: std={run['gaussian_noise_std']}")
    if run.get("salt_pepper_amount") not in (None, "None", ""):
        parts.append(f"Salt & Pepper: {run['salt_pepper_amount']}")
    if run.get("p_grayscale") not in (None, "None", ""):
        parts.append(f"Grayscale: p={run['p_grayscale']}")
    return " | ".join(parts)


def generate_sample_image(run: dict, original_img: torch.Tensor) -> str:
    """Generate base64 degraded image for a given run config."""
    low_res = int(run["low_res"])
    out_size = int(run.get("out_size", 224))
    deg_type = run.get("degradation_type", "all")

    kwargs = dict(
        low_res=low_res,
        out_size=out_size,
        degradation_type=deg_type,
        p_grayscale=0.0,  # deterministic for display
    )

    # Apply custom params if present
    for param in ["blur_kernel", "blur_sigma", "gaussian_noise_std", "salt_pepper_amount"]:
        val = run.get(param)
        if val not in (None, "None", ""):
            if param == "blur_kernel":
                kwargs[param] = int(val)
            else:
                kwargs[param] = float(val)

    deg_cfg = DegradeConfig(**kwargs)
    degraded = degrade_image(original_img.clone(), deg_cfg, seed=42)
    return tensor_to_base64(degraded, size=128)


def generate_html(runs: list[dict], original_b64: str, sample_images: dict,
                  deg_labels: dict) -> str:
    """Generate the full HTML dashboard."""

    # Group runs by degradation config
    groups = {}
    for run in runs:
        key = make_deg_key(run)
        if key not in groups:
            groups[key] = []
        groups[key].append(run)

    # Sort groups: systematic first, then by low_res
    sorted_keys = sorted(groups.keys(), key=lambda k: (
        0 if any(r["group"] == "systematic" for r in groups[k]) else 1,
        k,
    ))

    # Build metrics JSON for learning curves
    all_metrics = {}
    for run in runs:
        run_id = run["run_name"].replace(" ", "_")
        all_metrics[run_id] = {
            "epochs": [m["epoch"] for m in run["metrics"]],
            "train_acc": [m["train_acc"] for m in run["metrics"]],
            "val_acc": [m["val_acc"] for m in run["metrics"]],
            "train_loss": [m["train_loss"] for m in run["metrics"]],
            "val_loss": [m["val_loss"] for m in run["metrics"]],
        }

    MODEL_COLORS = {
        "resnet50": "#2196F3",
        "densenet121": "#4CAF50",
        "transnext_micro": "#FF9800",
    }
    MODEL_LABELS = {
        "resnet50": "ResNet-50",
        "densenet121": "DenseNet-121",
        "transnext_micro": "TransNeXt Micro",
    }

    # Build group sections HTML
    group_sections = []
    for gkey in sorted_keys:
        gruns = groups[gkey]
        label = deg_labels.get(gkey, gkey)
        degraded_b64 = sample_images.get(gkey, "")

        # Sort runs by model name, then best_val_acc desc
        gruns.sort(key=lambda r: (-r["best_val_acc"],))

        # Best result
        best_run = gruns[0]
        best_model = MODEL_LABELS.get(best_run["model_name"], best_run["model_name"])

        # Table rows
        table_rows = ""
        for run in gruns:
            model_label = MODEL_LABELS.get(run["model_name"], run["model_name"])
            color = MODEL_COLORS.get(run["model_name"], "#999")
            acc_pct = run["best_val_acc"] * 100
            acc_class = "acc-high" if acc_pct >= 70 else ("acc-mid" if acc_pct >= 50 else "acc-low")
            group_badge = f'<span class="badge badge-{run["group"]}">{run["group"]}</span>'
            run_id = run["run_name"].replace(" ", "_")

            table_rows += f"""
            <tr data-run-id="{run_id}" onclick="showCurve('{run_id}')" style="cursor:pointer;">
                <td><span class="model-dot" style="background:{color}"></span> {model_label}</td>
                <td class="{acc_class}">{acc_pct:.1f}%</td>
                <td>{run['epochs']}</td>
                <td>{run['lr']}</td>
                <td>{run['freeze_backbone']}</td>
                <td>{group_badge}</td>
                <td class="run-name-cell" title="{run['run_name']}">{run['tag'] or run['run_name'][:30]}</td>
            </tr>"""

        # Degradation params for display
        deg_params_html = label.replace(" | ", "<br>")

        section_html = f"""
        <div class="deg-group" id="group-{gkey}">
            <div class="deg-header">
                <div class="deg-images">
                    <div class="img-pair">
                        <div class="img-box">
                            <img src="data:image/png;base64,{original_b64}" alt="Original">
                            <div class="img-label">Original (CIFAR-10)</div>
                        </div>
                        <div class="arrow">→</div>
                        <div class="img-box">
                            <img src="data:image/png;base64,{degraded_b64}" alt="Degraded">
                            <div class="img-label">After Degradation</div>
                        </div>
                    </div>
                    <div class="deg-params">{deg_params_html}</div>
                </div>
                <div class="deg-summary">
                    <div class="best-result">
                        <span class="best-label">Best:</span>
                        <span class="best-value">{best_run['best_val_acc']*100:.1f}%</span>
                        <span class="best-model">({best_model})</span>
                    </div>
                    <div class="run-count">{len(gruns)} experiment(s)</div>
                </div>
            </div>
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Best Val Acc</th>
                        <th>Epochs</th>
                        <th>LR</th>
                        <th>Frozen</th>
                        <th>Group</th>
                        <th>Tag</th>
                    </tr>
                </thead>
                <tbody>{table_rows}
                </tbody>
            </table>
            <div class="chart-container" id="chart-{gkey}">
                <canvas id="canvas-{gkey}" width="600" height="250"></canvas>
            </div>
        </div>
        """
        group_sections.append(section_html)

    sections_html = "\n".join(group_sections)
    metrics_json = json.dumps(all_metrics)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Systematic Experiments Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
    :root {{
        --bg: #0d1117;
        --surface: #161b22;
        --surface2: #21262d;
        --border: #30363d;
        --text: #e6edf3;
        --text-dim: #8b949e;
        --accent: #58a6ff;
        --green: #3fb950;
        --orange: #d29922;
        --red: #f85149;
    }}
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
        background: var(--bg);
        color: var(--text);
        padding: 20px;
        line-height: 1.5;
    }}
    h1 {{
        font-size: 1.8em;
        margin-bottom: 4px;
        color: var(--accent);
    }}
    .subtitle {{
        color: var(--text-dim);
        font-size: 0.95em;
        margin-bottom: 24px;
    }}
    .filters {{
        display: flex;
        gap: 12px;
        margin-bottom: 20px;
        flex-wrap: wrap;
    }}
    .filters label {{
        color: var(--text-dim);
        font-size: 0.85em;
    }}
    .filters select {{
        background: var(--surface2);
        color: var(--text);
        border: 1px solid var(--border);
        padding: 6px 10px;
        border-radius: 6px;
        font-size: 0.9em;
    }}
    .stats-bar {{
        display: flex;
        gap: 20px;
        margin-bottom: 24px;
        flex-wrap: wrap;
    }}
    .stat-card {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 14px 20px;
        min-width: 140px;
    }}
    .stat-value {{
        font-size: 1.6em;
        font-weight: 700;
        color: var(--accent);
    }}
    .stat-label {{
        font-size: 0.8em;
        color: var(--text-dim);
    }}
    .deg-group {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        margin-bottom: 24px;
        overflow: hidden;
    }}
    .deg-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 20px;
        background: var(--surface2);
        border-bottom: 1px solid var(--border);
        flex-wrap: wrap;
        gap: 16px;
    }}
    .deg-images {{
        display: flex;
        align-items: center;
        gap: 16px;
        flex-wrap: wrap;
    }}
    .img-pair {{
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .img-box {{
        text-align: center;
    }}
    .img-box img {{
        width: 96px;
        height: 96px;
        border-radius: 6px;
        border: 2px solid var(--border);
        image-rendering: pixelated;
    }}
    .img-label {{
        font-size: 0.7em;
        color: var(--text-dim);
        margin-top: 4px;
    }}
    .arrow {{
        font-size: 1.8em;
        color: var(--text-dim);
    }}
    .deg-params {{
        font-size: 0.82em;
        color: var(--text);
        background: var(--bg);
        padding: 10px 14px;
        border-radius: 6px;
        border: 1px solid var(--border);
        line-height: 1.7;
        font-family: 'Consolas', 'Monaco', monospace;
    }}
    .deg-summary {{
        text-align: right;
    }}
    .best-result {{
        font-size: 1.1em;
    }}
    .best-label {{
        color: var(--text-dim);
    }}
    .best-value {{
        font-weight: 700;
        font-size: 1.4em;
        color: var(--green);
    }}
    .best-model {{
        color: var(--text-dim);
        font-size: 0.85em;
    }}
    .run-count {{
        color: var(--text-dim);
        font-size: 0.8em;
        margin-top: 4px;
    }}
    .results-table {{
        width: 100%;
        border-collapse: collapse;
    }}
    .results-table th {{
        background: var(--surface2);
        padding: 10px 14px;
        text-align: left;
        font-size: 0.8em;
        color: var(--text-dim);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border-bottom: 1px solid var(--border);
    }}
    .results-table td {{
        padding: 10px 14px;
        border-bottom: 1px solid var(--border);
        font-size: 0.9em;
    }}
    .results-table tr:hover {{
        background: var(--surface2);
    }}
    .results-table tr.selected {{
        background: rgba(88, 166, 255, 0.1);
        border-left: 3px solid var(--accent);
    }}
    .model-dot {{
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 6px;
    }}
    .acc-high {{ color: var(--green); font-weight: 600; }}
    .acc-mid {{ color: var(--orange); font-weight: 600; }}
    .acc-low {{ color: var(--red); font-weight: 600; }}
    .badge {{
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75em;
        font-weight: 600;
    }}
    .badge-systematic {{ background: rgba(88,166,255,0.15); color: var(--accent); }}
    .badge-official {{ background: rgba(63,185,80,0.15); color: var(--green); }}
    .badge-pilot {{ background: rgba(210,153,34,0.15); color: var(--orange); }}
    .run-name-cell {{
        max-width: 200px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        color: var(--text-dim);
        font-size: 0.82em;
    }}
    .chart-container {{
        padding: 16px 20px;
        display: none;
    }}
    .chart-container.visible {{
        display: block;
    }}
    footer {{
        text-align: center;
        color: var(--text-dim);
        font-size: 0.8em;
        margin-top: 30px;
        padding: 16px;
        border-top: 1px solid var(--border);
    }}
</style>
</head>
<body>

<h1>Systematic Experiments Dashboard</h1>
<div class="subtitle">
    Object Classification in Low-Resolution THz-like Imagery — Degradation Level Comparison<br>
    Filtered: incomplete runs and accuracy &lt; 30% removed
</div>

<div class="stats-bar">
    <div class="stat-card">
        <div class="stat-value">{len(runs)}</div>
        <div class="stat-label">Total Experiments</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{len(groups)}</div>
        <div class="stat-label">Degradation Configs</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{max(r['best_val_acc'] for r in runs)*100:.1f}%</div>
        <div class="stat-label">Best Accuracy</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{len(set(r['model_name'] for r in runs))}</div>
        <div class="stat-label">Models Tested</div>
    </div>
</div>

<div class="filters">
    <div>
        <label>Filter by Group:</label><br>
        <select id="filterGroup" onchange="applyFilters()">
            <option value="all">All Groups</option>
            <option value="systematic">Systematic</option>
            <option value="official">Official</option>
            <option value="pilot">Pilot</option>
        </select>
    </div>
    <div>
        <label>Filter by Model:</label><br>
        <select id="filterModel" onchange="applyFilters()">
            <option value="all">All Models</option>
            <option value="resnet50">ResNet-50</option>
            <option value="densenet121">DenseNet-121</option>
            <option value="transnext_micro">TransNeXt Micro</option>
        </select>
    </div>
</div>

{sections_html}

<footer>
    Generated by generate_systematic_dashboard.py | THz-like Image Classification Project
</footer>

<script>
const ALL_METRICS = {metrics_json};

const MODEL_COLORS = {{
    'resnet50': '#2196F3',
    'densenet121': '#4CAF50',
    'transnext_micro': '#FF9800',
}};

let activeCharts = {{}};

function showCurve(runId) {{
    const data = ALL_METRICS[runId];
    if (!data) return;

    // Find the parent group
    const row = document.querySelector(`tr[data-run-id="${{runId}}"]`);
    if (!row) return;
    const group = row.closest('.deg-group');
    const chartContainer = group.querySelector('.chart-container');
    const canvas = chartContainer.querySelector('canvas');
    const chartId = canvas.id;

    // Toggle selection
    group.querySelectorAll('tr').forEach(r => r.classList.remove('selected'));
    row.classList.add('selected');
    chartContainer.classList.add('visible');

    // Destroy old chart
    if (activeCharts[chartId]) {{
        activeCharts[chartId].destroy();
    }}

    activeCharts[chartId] = new Chart(canvas, {{
        type: 'line',
        data: {{
            labels: data.epochs,
            datasets: [
                {{
                    label: 'Train Acc',
                    data: data.train_acc.map(v => v * 100),
                    borderColor: '#58a6ff',
                    backgroundColor: 'rgba(88,166,255,0.1)',
                    tension: 0.3,
                    pointRadius: 2,
                }},
                {{
                    label: 'Val Acc',
                    data: data.val_acc.map(v => v * 100),
                    borderColor: '#3fb950',
                    backgroundColor: 'rgba(63,185,80,0.1)',
                    tension: 0.3,
                    pointRadius: 2,
                }},
            ]
        }},
        options: {{
            responsive: true,
            plugins: {{
                title: {{
                    display: true,
                    text: `Learning Curve: ${{runId.substring(0, 50)}}`,
                    color: '#e6edf3',
                    font: {{ size: 13 }},
                }},
                legend: {{
                    labels: {{ color: '#8b949e' }}
                }}
            }},
            scales: {{
                x: {{
                    title: {{ display: true, text: 'Epoch', color: '#8b949e' }},
                    ticks: {{ color: '#8b949e' }},
                    grid: {{ color: '#21262d' }},
                }},
                y: {{
                    title: {{ display: true, text: 'Accuracy (%)', color: '#8b949e' }},
                    ticks: {{ color: '#8b949e' }},
                    grid: {{ color: '#21262d' }},
                    min: 0,
                    max: 100,
                }},
            }},
        }},
    }});
}}

function applyFilters() {{
    const groupFilter = document.getElementById('filterGroup').value;
    const modelFilter = document.getElementById('filterModel').value;

    document.querySelectorAll('.results-table tbody tr').forEach(row => {{
        let show = true;
        if (modelFilter !== 'all') {{
            const modelCell = row.cells[0].textContent.toLowerCase();
            if (!modelCell.includes(modelFilter.replace('_', ' ').replace('50', '-50').replace('121', '-121'))) {{
                // Simple contains check
                const modelMap = {{
                    'resnet50': 'resnet-50',
                    'densenet121': 'densenet-121',
                    'transnext_micro': 'transnext micro',
                }};
                if (!modelCell.toLowerCase().includes(modelMap[modelFilter] || modelFilter)) {{
                    show = false;
                }}
            }}
        }}
        if (groupFilter !== 'all') {{
            const groupCell = row.cells[5].textContent.toLowerCase();
            if (!groupCell.includes(groupFilter)) {{
                show = false;
            }}
        }}
        row.style.display = show ? '' : 'none';
    }});
}}
</script>
</body>
</html>"""

    return html


def main():
    runs_root = Path("runs")
    if not runs_root.exists():
        print("[ERROR] No 'runs' directory found")
        return

    print("[SCAN] Scanning all experiment runs...")
    runs = scan_all_runs(runs_root)
    print(f"[OK] Found {len(runs)} valid experiments (filtered: incomplete & <30% accuracy)")

    if not runs:
        print("[WARN] No valid runs found. Dashboard will be empty.")
        return

    # Generate sample images
    print("[IMAGES] Generating sample degradation images...")
    original_img, label = get_cifar10_sample()
    original_b64 = tensor_to_base64(original_img, size=128)

    sample_images = {}
    deg_labels = {}
    seen_keys = set()
    for run in runs:
        key = make_deg_key(run)
        if key not in seen_keys:
            seen_keys.add(key)
            sample_images[key] = generate_sample_image(run, original_img)
            deg_labels[key] = make_deg_label(run)
            print(f"     {key}: {deg_labels[key]}")

    # Generate HTML
    print("[HTML] Generating dashboard...")
    html = generate_html(runs, original_b64, sample_images, deg_labels)

    out_path = Path("artifacts") / "dashboard_systematic.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[OK] Dashboard saved to: {out_path}")
    print(f"     {len(runs)} experiments across {len(seen_keys)} degradation configs")


if __name__ == "__main__":
    main()
