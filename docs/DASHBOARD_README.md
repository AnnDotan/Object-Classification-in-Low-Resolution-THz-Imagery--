# Interactive Results Dashboards

This project includes two interactive HTML dashboards for visualizing experiment results.

## Quick Start

### Generate Dashboards
```bash
python src/tools/generate_experiment_plan_dashboard.py
python src/tools/generate_systematic_dashboard.py
```

### View Dashboards
Open the generated HTML files in your web browser:
- **Experiment Plan Dashboard**: `artifacts/dashboard_experiment_plan.html`
- **Systematic Dashboard**: `artifacts/dashboard_systematic.html`

## Dashboard Features

### Experiment Plan Dashboard (`dashboard_experiment_plan.html`)

Tracks all 36 planned experiments across 4 phases:

- **Dark theme** with white text for readability
- **Phase Info Cards**: Shared hyperparameters + per-phase specific config
- **Original vs Degraded Images**: Side-by-side comparison for each dataset (CIFAR-10 truck, MNIST digit 7)
- **Experiment Table**: Status (completed/pending), date, duration, accuracy, degraded sample image
- **Comparative Bar Charts**: All 3 models side-by-side per degradation level
- **Cross-Level Robustness Chart**: Accuracy vs. severity line plot
- **Interim Conclusions**: Auto-generated insights per completed phase
- **Learning Curve Modal**: Click any row to see accuracy + loss curves
- **Filters**: Phase, model, dataset, status
- **Auto-refresh**: Dashboard regenerates automatically after each experiment

### Systematic Dashboard (`dashboard_systematic.html`)

Groups completed experiments by degradation configuration:

- **Sample Images**: Original vs degraded per configuration
- **Filterable Results Table**: Searchable, sortable
- **Learning Curve Viewer**: Click any row for detailed curves
- **Filters**: Removes incomplete runs and <30% accuracy

## Auto-Dashboard Refresh

The training runner (`src/runner.py`) automatically regenerates the experiment plan dashboard after each experiment completes. A background watcher (`watch_dashboard.py`) also monitors for new completed runs.

## Technical Details

- **Frontend**: Self-contained HTML + CSS + JavaScript (no server needed)
- **Charting**: Chart.js (embedded)
- **Data**: JSON embedded in HTML via Python `json.dumps()`
- **Images**: Base64-encoded inline (no external files)
- **Generation**: Python scripts scan `runs/systematic/` for metrics.csv, run_config.txt, and .png files
