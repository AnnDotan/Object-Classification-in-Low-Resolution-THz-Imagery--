#!/usr/bin/env python3
"""
Generate an interactive HTML dashboard for experiment results.

The dashboard includes:
- Summary table of all runs
- Model comparison charts
- Learning curves
- Performance by degradation level
- Interactive filtering and sorting
"""

import csv
from pathlib import Path
from typing import Any

import json


def load_run_summary(csv_path: Path) -> list[dict]:
    """Load run summary from CSV."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            if row.get("best_val_acc"):
                try:
                    row["best_val_acc"] = float(row["best_val_acc"])
                except ValueError:
                    row["best_val_acc"] = None
            else:
                row["best_val_acc"] = None

            if row.get("total_time"):
                row["total_time_seconds"] = parse_time(row["total_time"])
            else:
                row["total_time_seconds"] = None

            rows.append(row)
    return rows


def parse_time(time_str: str) -> float:
    """Parse time string like '1890.9s' to float seconds."""
    try:
        return float(time_str.replace("s", "").strip())
    except (ValueError, AttributeError):
        return 0.0


def get_metrics_csv(run_dir: str) -> list[dict] | None:
    """Load metrics.csv from a run directory."""
    metrics_path = Path(run_dir) / "metrics.csv"
    if not metrics_path.exists():
        return None

    rows = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("epoch"):
                try:
                    row["epoch"] = int(row["epoch"])
                    row["train_loss"] = float(row["train_loss"])
                    row["train_acc"] = float(row["train_acc"])
                    row["val_loss"] = float(row["val_loss"])
                    row["val_acc"] = float(row["val_acc"])
                    rows.append(row)
                except (ValueError, KeyError):
                    continue
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


def load_sample_images() -> dict:
    path = Path("artifacts/tables/sample_images.json")
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def generate_html(runs: list[dict], output_path: Path) -> None:
    """Generate interactive HTML dashboard."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_images = load_sample_images()

    # Enrich runs with config key for sample images
    for r in runs:
        run_dir = r.get("run_dir", "")
        low_res = r.get("low_res", "16")
        out_size = r.get("out_size", "32")
        cfg_data = get_run_config(run_dir)
        deg_type = cfg_data.get("degradation_type", "all")
        r["_config_key"] = f"lr{low_res}_out{out_size}_deg{deg_type}" if low_res and out_size else ""

    # Group runs by model and degradation
    runs_by_model = {}
    runs_by_degradation = {}
    models = set()
    degradations = set()

    for run in runs:
        model = run.get("model_name", "unknown")
        degradation = run.get("low_res", "unknown")
        models.add(model)
        degradations.add(degradation)

        if model not in runs_by_model:
            runs_by_model[model] = []
        runs_by_model[model].append(run)

        if degradation not in runs_by_degradation:
            runs_by_degradation[degradation] = []
        runs_by_degradation[degradation].append(run)

    # Filter out runs with no best_val_acc
    completed_runs = [r for r in runs if r.get("best_val_acc") is not None]

    # Prepare chart data
    models_sorted = sorted([m for m in models if m])
    degradations_sorted = sorted([d for d in degradations if d], key=lambda x: int(x) if x.isdigit() else 999)

    # Model comparison data (best acc by model across all runs)
    model_comparison = []
    for model in models_sorted:
        model_runs = [r for r in completed_runs if r.get("model_name") == model]
        if model_runs:
            avg_acc = sum(r.get("best_val_acc", 0) for r in model_runs) / len(model_runs)
            max_acc = max(r.get("best_val_acc", 0) for r in model_runs)
            model_comparison.append({"model": model, "avg_acc": avg_acc, "max_acc": max_acc, "count": len(model_runs)})

    # Degradation comparison (best acc by degradation level)
    degradation_comparison = []
    for deg in degradations_sorted:
        deg_runs = [r for r in completed_runs if r.get("low_res") == deg]
        if deg_runs:
            avg_acc = sum(r.get("best_val_acc", 0) for r in deg_runs) / len(deg_runs)
            max_acc = max(r.get("best_val_acc", 0) for r in deg_runs)
            degradation_comparison.append(
                {"degradation": f"low_res={deg}", "avg_acc": avg_acc, "max_acc": max_acc, "count": len(deg_runs)}
            )

    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Experiment Results Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }}

        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 20px;
            text-align: center;
        }}

        h1 {{
            font-size: 32px;
            margin-bottom: 8px;
        }}

        .subtitle {{
            font-size: 14px;
            opacity: 0.9;
        }}

        .content {{
            padding: 40px;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}

        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}

        .stat-value {{
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 8px;
        }}

        .stat-label {{
            font-size: 12px;
            opacity: 0.9;
        }}

        .chart-container {{
            margin-bottom: 40px;
        }}

        .chart-title {{
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 15px;
            color: #333;
        }}

        .chart-wrapper {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            border: 1px solid #e0e0e0;
        }}

        .table-container {{
            overflow-x: auto;
            margin-bottom: 40px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}

        thead {{
            background: #f0f0f0;
            font-weight: 600;
        }}

        th {{
            padding: 12px;
            text-align: left;
            border-bottom: 2px solid #ddd;
            position: sticky;
            top: 0;
        }}

        td {{
            padding: 12px;
            border-bottom: 1px solid #eee;
        }}

        tbody tr:hover {{
            background: #f9f9f9;
        }}

        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 500;
        }}

        .badge-official {{
            background: #e3f2fd;
            color: #1976d2;
        }}

        .badge-pilot {{
            background: #f3e5f5;
            color: #7b1fa2;
        }}

        .badge-archive {{
            background: #f5f5f5;
            color: #666;
        }}

        .metric-good {{
            color: #2e7d32;
            font-weight: 500;
        }}

        .metric-warn {{
            color: #f57c00;
            font-weight: 500;
        }}

        .metric-poor {{
            color: #c62828;
            font-weight: 500;
        }}

        .filter-section {{
            margin-bottom: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }}

        .filter-title {{
            font-weight: 600;
            margin-bottom: 15px;
            color: #333;
        }}

        .filter-group {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            align-items: center;
        }}

        input[type="text"], select {{
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 13px;
        }}

        button {{
            padding: 8px 16px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
        }}

        button:hover {{
            background: #764ba2;
        }}

        footer {{
            background: #f0f0f0;
            padding: 20px;
            text-align: center;
            font-size: 12px;
            color: #666;
            border-top: 1px solid #ddd;
        }}
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
        .sample-box {{ text-align: center; }}
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
            <h1>📊 Experiment Results Dashboard</h1>
            <p class="subtitle">Low-Resolution Image Classification with Degradation</p>
        </header>

        <div class="content">
            <!-- Statistics -->
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{len(runs)}</div>
                    <div class="stat-label">Total Experiments</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(completed_runs)}</div>
                    <div class="stat-label">Completed Runs</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(models_sorted)}</div>
                    <div class="stat-label">Models Tested</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{max([r.get('best_val_acc', 0) for r in completed_runs], default=0):.2%}</div>
                    <div class="stat-label">Best Accuracy</div>
                </div>
            </div>

            <!-- Model Comparison -->
            <div class="chart-container">
                <h2 class="chart-title">🏆 Model Comparison</h2>
                <div class="chart-wrapper">
                    <div id="modelChart" style="height: 400px;"></div>
                </div>
            </div>

            <!-- Degradation Comparison -->
            <div class="chart-container">
                <h2 class="chart-title">📉 Performance by Degradation Level</h2>
                <div class="chart-wrapper">
                    <div id="degradationChart" style="height: 400px;"></div>
                </div>
            </div>

            <!-- Accuracy Distribution -->
            <div class="chart-container">
                <h2 class="chart-title">📈 Accuracy Distribution</h2>
                <div class="chart-wrapper">
                    <div id="accuracyChart" style="height: 400px;"></div>
                </div>
            </div>

            <!-- Runs Table -->
            <div class="chart-container">
                <h2 class="chart-title">📋 Detailed Results</h2>
                <div class="filter-section">
                    <div class="filter-title">Filter Results</div>
                    <div class="filter-group">
                        <input type="text" id="searchBox" placeholder="Search runs..." style="flex: 1; max-width: 300px;">
                        <select id="groupFilter">
                            <option value="">All Groups</option>
                            <option value="official">Official</option>
                            <option value="pilot">Pilot</option>
                            <option value="archive">Archive</option>
                        </select>
                        <select id="modelFilter">
                            <option value="">All Models</option>
                            {"".join(f'<option value="{m}">{m}</option>' for m in models_sorted)}
                        </select>
                        <button onclick="filterTable()">Apply Filters</button>
                        <button onclick="resetFilters()" style="background: #999;">Reset</button>
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
                                <th>Best Val Acc</th>
                                <th>Output Size</th>
                                <th>Batch Size</th>
                                <th>Epochs</th>
                                <th>Total Time</th>
                                <th>Sample</th>
                            </tr>
                        </thead>
                        <tbody id="tableBody">
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <footer>
            <p>Generated: {Path.cwd().name} | Total Runs: {len(runs)} | Completed: {len(completed_runs)}</p>
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
        const sampleImages = {json.dumps(sample_images)};
        const runsData = {json.dumps(runs)};
        const completedRuns = runsData.filter(r => r.best_val_acc !== null);

        // Color coding for degradation config groups
        const CONFIG_GROUP_COLORS = [
            'rgba(33,150,243,0.10)',
            'rgba(76,175,80,0.10)',
            'rgba(255,152,0,0.10)',
            'rgba(156,39,176,0.10)',
            'rgba(0,150,136,0.10)',
            'rgba(244,67,54,0.10)',
            'rgba(121,85,72,0.10)',
            'rgba(63,81,181,0.10)',
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
            if (!(configKey in keyMap)) keyMap[configKey] = Object.keys(keyMap).length;
            return CONFIG_GROUP_COLORS[keyMap[configKey] % CONFIG_GROUP_COLORS.length];
        }}
        function getConfigGroupBorder(configKey, keyMap) {{
            if (!configKey) return '';
            if (!(configKey in keyMap)) keyMap[configKey] = Object.keys(keyMap).length;
            return CONFIG_GROUP_BORDERS[keyMap[configKey] % CONFIG_GROUP_BORDERS.length];
        }}
        function configKeyToLabel(ck) {{
            if (!ck) return '-';
            const m = ck.match(/lr(\\d+)_out(\\d+)_deg(\\w+)/);
            if (!m) return ck;
            const deg = m[3] === 'all' ? 'Full Pipeline' : m[3];
            return `lr=${{m[1]}} out=${{m[2]}} ${{deg}}`;
        }}

        // Populate table
        function populateTable(dataToShow) {{
            const tbody = document.getElementById('tableBody');
            tbody.innerHTML = '';

            // Sort by config key so groups are together
            const sorted = [...dataToShow].sort((a, b) => {{
                const ka = a._config_key || 'zzz';
                const kb = b._config_key || 'zzz';
                if (ka !== kb) return ka.localeCompare(kb);
                const aa = a.best_val_acc || 0;
                const ab = b.best_val_acc || 0;
                return ab - aa;
            }});
            const keyMap = {{}};
            sorted.forEach(run => {{
                const tr = document.createElement('tr');
                const accColor = run.best_val_acc === null ? '' :
                    (run.best_val_acc >= 0.6 ? 'metric-good' :
                     run.best_val_acc >= 0.5 ? 'metric-warn' : 'metric-poor');

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
                tr.innerHTML = `
                    <td><span class="badge badge-${{run.group}}">${{run.group}}</span></td>
                    <td><small>${{run.run_name}}</small></td>
                    <td><strong>${{run.model_name || '-'}}</strong></td>
                    <td><code>${{configKeyToLabel(ck)}}</code></td>
                    <td class="${{accColor}}">${{run.best_val_acc !== null ? (run.best_val_acc * 100).toFixed(2) + '%' : '-'}}</td>
                    <td>${{run.out_size || '-'}}</td>
                    <td>${{run.batch_size || '-'}}</td>
                    <td>${{run.epochs || '-'}}</td>
                    <td><small>${{run.total_time || '-'}}</small></td>
                    <td>${{sampleBtn}}</td>
                `;
                tbody.appendChild(tr);
            }});
        }}

        // Filter table
        function filterTable() {{
            const search = document.getElementById('searchBox').value.toLowerCase();
            const group = document.getElementById('groupFilter').value;
            const model = document.getElementById('modelFilter').value;

            const filtered = runsData.filter(run => {{
                const matchSearch = !search || run.run_name.toLowerCase().includes(search);
                const matchGroup = !group || run.group === group;
                const matchModel = !model || run.model_name === model;
                return matchSearch && matchGroup && matchModel;
            }});

            populateTable(filtered);
        }}

        function resetFilters() {{
            document.getElementById('searchBox').value = '';
            document.getElementById('groupFilter').value = '';
            document.getElementById('modelFilter').value = '';
            populateTable(runsData);
        }}

        // Create charts
        function createCharts() {{
            // Model comparison
            const modelComparison = {json.dumps(model_comparison)};
            const modelTrace = {{
                x: modelComparison.map(m => m.model),
                y: modelComparison.map(m => m.max_acc),
                name: 'Max Accuracy',
                type: 'bar',
                marker: {{color: '#667eea'}}
            }};
            const modelTrace2 = {{
                x: modelComparison.map(m => m.model),
                y: modelComparison.map(m => m.avg_acc),
                name: 'Average Accuracy',
                type: 'bar',
                marker: {{color: '#764ba2'}}
            }};
            Plotly.newPlot('modelChart', [modelTrace, modelTrace2], {{
                xaxis: {{title: 'Model'}},
                yaxis: {{title: 'Accuracy'}},
                barmode: 'group',
                hovermode: 'closest'
            }}, {{responsive: true}});

            // Degradation comparison
            const degradationComparison = {json.dumps(degradation_comparison)};
            const degTrace = {{
                x: degradationComparison.map(d => d.degradation),
                y: degradationComparison.map(d => d.max_acc),
                name: 'Max Accuracy',
                type: 'bar',
                marker: {{color: '#667eea'}}
            }};
            const degTrace2 = {{
                x: degradationComparison.map(d => d.degradation),
                y: degradationComparison.map(d => d.avg_acc),
                name: 'Average Accuracy',
                type: 'bar',
                marker: {{color: '#764ba2'}}
            }};
            Plotly.newPlot('degradationChart', [degTrace, degTrace2], {{
                xaxis: {{title: 'Degradation Level'}},
                yaxis: {{title: 'Accuracy'}},
                barmode: 'group',
                hovermode: 'closest'
            }}, {{responsive: true}});

            // Accuracy distribution
            const accuracies = completedRuns.map(r => r.best_val_acc);
            const accuracyTrace = {{
                x: accuracies,
                type: 'histogram',
                nbinsx: 20,
                marker: {{color: '#667eea'}}
            }};
            Plotly.newPlot('accuracyChart', [accuracyTrace], {{
                xaxis: {{title: 'Accuracy'}},
                yaxis: {{title: 'Frequency'}},
                hovermode: 'closest'
            }}, {{responsive: true}});
        }}

        // Sample Image Modal
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

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {{
            populateTable(runsData);
            createCharts();

            // Search on enter
            document.getElementById('searchBox').addEventListener('keypress', function(e) {{
                if (e.key === 'Enter') filterTable();
            }});
        }});
    </script>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[OK] Dashboard generated: {output_path}")


def main():
    runs_root = Path("runs")
    csv_path = Path("artifacts/tables/run_summary.csv")

    if not csv_path.exists():
        print(f"[ERROR] Run summary not found: {csv_path}")
        print("Run 'python src/tools/summarize_runs.py' first.")
        return

    print("[GENERATE] Generating interactive dashboard...")
    runs = load_run_summary(csv_path)
    output = Path("artifacts/dashboard.html")
    generate_html(runs, output)


if __name__ == "__main__":
    main()
