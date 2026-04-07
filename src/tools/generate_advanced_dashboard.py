#!/usr/bin/env python3
"""
Generate an advanced interactive HTML dashboard with learning curves.

Features:
- Model comparison charts
- Performance by degradation level
- Interactive learning curve viewer
- Detailed results table with filtering
- Run comparison capability
"""

import csv
from pathlib import Path
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

            rows.append(row)
    return rows


def get_metrics_csv(run_dir: str) -> list[dict] | None:
    """Load metrics.csv from a run directory."""
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
                        rows.append(
                            {
                                "epoch": int(row["epoch"]),
                                "train_loss": float(row["train_loss"]),
                                "train_acc": float(row["train_acc"]),
                                "val_loss": float(row["val_loss"]),
                                "val_acc": float(row["val_acc"]),
                            }
                        )
                    except (ValueError, KeyError):
                        continue
    except Exception:
        return None

    return rows if rows else None


def generate_advanced_html(runs: list[dict], output_path: Path) -> None:
    """Generate advanced interactive HTML dashboard."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Group runs by model and degradation
    models = sorted(set(r.get("model_name", "") for r in runs if r.get("model_name")))
    degradations = sorted(
        set(r.get("low_res", "") for r in runs if r.get("low_res")), key=lambda x: int(x) if x.isdigit() else 999
    )

    # Filter completed runs
    completed_runs = [r for r in runs if r.get("best_val_acc") is not None]

    # Load learning curves for top runs
    top_runs = sorted(completed_runs, key=lambda x: x.get("best_val_acc", 0), reverse=True)[:6]
    learning_curves = {}
    for run in top_runs:
        metrics = get_metrics_csv(run["run_dir"])
        if metrics:
            learning_curves[run["run_name"]] = metrics

    # Prepare top runs for display
    top_runs_data = [{"name": r["run_name"], "model": r["model_name"], "acc": r["best_val_acc"], "deg": r["low_res"]} for r in top_runs]

    # Prepare aggregated data
    model_comparison = []
    for model in models:
        model_runs = [r for r in completed_runs if r.get("model_name") == model]
        if model_runs:
            accs = [r.get("best_val_acc", 0) for r in model_runs]
            model_comparison.append({"model": model, "avg_acc": sum(accs) / len(accs), "max_acc": max(accs), "count": len(model_runs)})

    degradation_comparison = []
    for deg in degradations:
        deg_runs = [r for r in completed_runs if r.get("low_res") == deg]
        if deg_runs:
            accs = [r.get("best_val_acc", 0) for r in deg_runs]
            degradation_comparison.append({"degradation": f"low_res={deg}", "avg_acc": sum(accs) / len(accs), "max_acc": max(accs)})

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Results Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

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
            margin-bottom: 10px;
        }}

        .content {{
            padding: 40px;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
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

        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}

        @media (max-width: 1200px) {{
            .charts-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .chart-container {{
            margin-bottom: 40px;
        }}

        .chart-title {{
            font-size: 18px;
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

        .full-width {{
            grid-column: 1 / -1;
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

        .top-runs-list {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}

        .run-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            cursor: pointer;
            transition: all 0.3s ease;
        }}

        .run-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }}

        .run-card.active {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}

        .run-card-name {{
            font-weight: 600;
            font-size: 14px;
            margin-bottom: 8px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .run-card-stats {{
            font-size: 12px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }}

        .run-stat {{
            display: flex;
            justify-content: space-between;
        }}

        .section {{
            margin-bottom: 50px;
        }}

        .section-title {{
            font-size: 24px;
            font-weight: 700;
            margin-bottom: 20px;
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
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
            <h1>Advanced Results Dashboard</h1>
            <p class="subtitle">Low-Resolution Image Classification with Degradation</p>
            <p class="subtitle">Interactive Analysis & Comparison</p>
        </header>

        <div class="content">
            <!-- Statistics -->
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
                    <div class="stat-value">{len(models)}</div>
                    <div class="stat-label">Models</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{max([r.get('best_val_acc', 0) for r in completed_runs], default=0):.1%}</div>
                    <div class="stat-label">Best Accuracy</div>
                </div>
            </div>

            <!-- Section: Performance Overview -->
            <div class="section">
                <h2 class="section-title">Performance Overview</h2>
                <div class="charts-grid">
                    <div class="chart-wrapper">
                        <h3 class="chart-title">Model Comparison</h3>
                        <div id="modelChart" style="height: 350px;"></div>
                    </div>
                    <div class="chart-wrapper">
                        <h3 class="chart-title">Accuracy by Degradation</h3>
                        <div id="degradationChart" style="height: 350px;"></div>
                    </div>
                </div>
            </div>

            <!-- Section: Learning Curves -->
            <div class="section">
                <h2 class="section-title">Learning Curves</h2>
                <p style="margin-bottom: 20px; color: #666; font-size: 14px;">
                    Click on a run card below to display its learning curve.
                </p>
                <div class="top-runs-list" id="topRunsList"></div>
                <div class="chart-wrapper">
                    <div id="learningCurveChart" style="height: 450px;"></div>
                </div>
            </div>

            <!-- Section: Detailed Results -->
            <div class="section">
                <h2 class="section-title">Detailed Results</h2>
                <div class="filter-section">
                    <div class="filter-title">Filter & Search</div>
                    <div class="filter-group">
                        <input type="text" id="searchBox" placeholder="Search run names..." style="flex: 1; max-width: 300px;">
                        <select id="groupFilter">
                            <option value="">All Groups</option>
                            <option value="official">Official</option>
                            <option value="pilot">Pilot</option>
                            <option value="archive">Archive</option>
                        </select>
                        <select id="modelFilter">
                            <option value="">All Models</option>
                            {"".join(f'<option value="{m}">{m}</option>' for m in models)}
                        </select>
                        <button onclick="filterTable()">Apply</button>
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
                            </tr>
                        </thead>
                        <tbody id="tableBody">
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <footer>
            <p>Generated: Advanced Results Dashboard | {len(completed_runs)}/{len(runs)} runs completed</p>
        </footer>
    </div>

    <script>
        const runsData = {json.dumps(runs)};
        const completedRuns = runsData.filter(r => r.best_val_acc !== null);
        const learningCurvesData = {json.dumps(learning_curves)};
        const topRunsData = {json.dumps(top_runs_data)};

        let selectedRunName = null;

        // Populate top runs list
        function populateTopRunsList() {{
            const container = document.getElementById('topRunsList');
            container.innerHTML = '';

            topRunsData.forEach((run, idx) => {{
                const card = document.createElement('div');
                card.className = 'run-card';
                if (idx === 0) {{
                    card.classList.add('active');
                    selectedRunName = run.name;
                }}
                card.innerHTML = `
                    <div class="run-card-name" title="${{run.name}}">#${{idx + 1}} ${{run.model}}</div>
                    <div class="run-card-stats">
                        <div class="run-stat">
                            <span>Acc:</span>
                            <strong>${{(run.acc * 100).toFixed(1)}}%</strong>
                        </div>
                        <div class="run-stat">
                            <span>LowRes:</span>
                            <strong>${{run.deg}}</strong>
                        </div>
                    </div>
                `;
                card.onclick = () => selectRun(run.name, card);
                container.appendChild(card);
            }});

            // Display first run's learning curve
            if (topRunsData.length > 0) {{
                displayLearningCurve(topRunsData[0].name);
            }}
        }}

        function selectRun(runName, element) {{
            document.querySelectorAll('.run-card').forEach(e => e.classList.remove('active'));
            element.classList.add('active');
            selectedRunName = runName;
            displayLearningCurve(runName);
        }}

        function displayLearningCurve(runName) {{
            const metrics = learningCurvesData[runName];
            if (!metrics || metrics.length === 0) {{
                Plotly.purge('learningCurveChart');
                const div = document.getElementById('learningCurveChart');
                div.innerHTML = '<p style="padding: 20px; text-align: center; color: #999;">No metrics data available for this run</p>';
                return;
            }}

            const epochs = metrics.map(m => m.epoch);
            const trainLoss = metrics.map(m => m.train_loss);
            const valLoss = metrics.map(m => m.val_loss);
            const trainAcc = metrics.map(m => m.train_acc);
            const valAcc = metrics.map(m => m.val_acc);

            const trace1 = {{
                x: epochs, y: trainAcc, name: 'Train Accuracy',
                type: 'scatter', mode: 'lines+markers',
                line: {{color: '#667eea', width: 2}}, marker: {{size: 4}}
            }};
            const trace2 = {{
                x: epochs, y: valAcc, name: 'Val Accuracy',
                type: 'scatter', mode: 'lines+markers',
                line: {{color: '#764ba2', width: 2}}, marker: {{size: 4}}
            }};
            const trace3 = {{
                x: epochs, y: trainLoss, name: 'Train Loss',
                type: 'scatter', mode: 'lines',
                line: {{color: '#ff9999', width: 1, dash: 'dot'}},
                yaxis: 'y2', visible: 'legendonly'
            }};
            const trace4 = {{
                x: epochs, y: valLoss, name: 'Val Loss',
                type: 'scatter', mode: 'lines',
                line: {{color: '#ff6666', width: 1, dash: 'dot'}},
                yaxis: 'y2', visible: 'legendonly'
            }};

            const layout = {{
                title: 'Learning Curve - ' + runName.substring(0, 60),
                hovermode: 'x unified',
                xaxis: {{title: 'Epoch'}},
                yaxis: {{title: 'Accuracy', domain: [0, 0.7]}},
                yaxis2: {{title: 'Loss', overlaying: 'y', side: 'right', domain: [0, 0.7]}},
            }};

            Plotly.newPlot('learningCurveChart', [trace1, trace2, trace3, trace4], layout, {{responsive: true}});
        }}

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
                    <td><small title="${{run.run_name}}">${{run.run_name.substring(0, 40)}}</small></td>
                    <td><strong>${{run.model_name || '-'}}</strong></td>
                    <td><code>low_res=${{run.low_res || '-'}}</code></td>
                    <td class="${{accColor}}">${{run.best_val_acc !== null ? (run.best_val_acc * 100).toFixed(2) + '%' : '-'}}</td>
                    <td>${{run.out_size || '-'}}</td>
                    <td>${{run.batch_size || '-'}}</td>
                    <td>${{run.epochs || '-'}}</td>
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

        // Create overview charts
        function createOverviewCharts() {{
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
                name: 'Avg Accuracy',
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
                name: 'Avg Accuracy',
                type: 'bar',
                marker: {{color: '#764ba2'}}
            }};
            Plotly.newPlot('degradationChart', [degTrace, degTrace2], {{
                xaxis: {{title: 'Degradation Level'}},
                yaxis: {{title: 'Accuracy'}},
                barmode: 'group',
                hovermode: 'closest'
            }}, {{responsive: true}});
        }}

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {{
            populateTopRunsList();
            populateTable(runsData);
            createOverviewCharts();

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

    print(f"[OK] Advanced dashboard generated: {output_path}")


def main():
    csv_path = Path("artifacts/tables/run_summary.csv")

    if not csv_path.exists():
        print(f"[ERROR] Run summary not found: {csv_path}")
        print("Run 'python src/tools/summarize_runs.py' first.")
        return

    print("[GENERATE] Generating advanced dashboard with learning curves...")
    runs = load_run_summary(csv_path)
    output = Path("artifacts/dashboard_advanced.html")
    generate_advanced_html(runs, output)


if __name__ == "__main__":
    main()
