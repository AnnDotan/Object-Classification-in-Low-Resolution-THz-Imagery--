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


def generate_html(runs: list[dict], output_path: Path) -> None:
    """Generate interactive HTML dashboard."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
                                <th>Degradation</th>
                                <th>Best Val Acc</th>
                                <th>Output Size</th>
                                <th>Batch Size</th>
                                <th>Epochs</th>
                                <th>Total Time</th>
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

    <script>
        const runsData = {json.dumps(runs)};
        const completedRuns = runsData.filter(r => r.best_val_acc !== null);

        // Populate table
        function populateTable(dataToShow) {{
            const tbody = document.getElementById('tableBody');
            tbody.innerHTML = '';

            dataToShow.forEach(run => {{
                const tr = document.createElement('tr');
                const accColor = run.best_val_acc === null ? '' :
                    (run.best_val_acc >= 0.6 ? 'metric-good' :
                     run.best_val_acc >= 0.5 ? 'metric-warn' : 'metric-poor');

                tr.innerHTML = `
                    <td><span class="badge badge-${{run.group}}">${{run.group}}</span></td>
                    <td><small>${{run.run_name}}</small></td>
                    <td><strong>${{run.model_name || '-'}}</strong></td>
                    <td><code>low_res=${{run.low_res || '-'}}</code></td>
                    <td class="${{accColor}}">${{run.best_val_acc !== null ? (run.best_val_acc * 100).toFixed(2) + '%' : '-'}}</td>
                    <td>${{run.out_size || '-'}}</td>
                    <td>${{run.batch_size || '-'}}</td>
                    <td>${{run.epochs || '-'}}</td>
                    <td><small>${{run.total_time || '-'}}</small></td>
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
