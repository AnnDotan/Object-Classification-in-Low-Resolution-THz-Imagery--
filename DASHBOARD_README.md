# Interactive Results Dashboard

This project includes two interactive HTML dashboards for visualizing experiment results in real-time.

## Quick Start

### Generate Dashboards
```bash
# Refresh both dashboards (recommended after new experiments)
python src/tools/refresh_dashboards.py

# Or generate individually:
python src/tools/generate_dashboard.py
python src/tools/generate_advanced_dashboard.py
```

### View Dashboards
Open the generated HTML files in your web browser:
- **Basic Dashboard**: `artifacts/dashboard.html` - Overview and detailed results table
- **Advanced Dashboard**: `artifacts/dashboard_advanced.html` - Interactive learning curves and detailed analysis

## Dashboard Features

### Basic Dashboard (`dashboard.html`)
- **Statistics Cards**: Quick summary of total experiments, completed runs, models tested, and best accuracy
- **Model Comparison Chart**: Compare average and maximum accuracy across models
- **Degradation Comparison**: See how accuracy changes with different degradation levels (low_res=8, 16, 32)
- **Accuracy Distribution**: Histogram showing distribution of achieved accuracies
- **Results Table**: 
  - Searchable and filterable by group (official/pilot/archive), model, and run name
  - Shows: group, run name, model, degradation level, best accuracy, output size, batch size, epochs, and total time
  - Color-coded accuracy: green (≥60%), orange (50-60%), red (<50%)

### Advanced Dashboard (`dashboard_advanced.html`)
- **All features from basic dashboard** plus:
- **Top 6 Runs Section**: 
  - Display the 6 best-performing runs ranked by accuracy
  - Click any run card to view its learning curve
  - Shows rank, model, accuracy, and degradation level
- **Interactive Learning Curves**:
  - Train/Validation accuracy across epochs
  - Train/Validation loss (toggle visibility)
  - Hover for detailed values
  - Real-time display updates when selecting different runs
- **Enhanced Analytics**: Same detailed results table with better organization

## How to Use

### Searching and Filtering
1. Enter search term in the search box (matches run name)
2. Select a group (official, pilot, archive) from dropdown
3. Select a model (ResNet50, TransNeXt, DenseNet121, etc.) from dropdown
4. Click "Apply Filters" or press Enter
5. Click "Reset" to clear all filters

### Analyzing Learning Curves
1. Open the Advanced Dashboard
2. The top 6 best runs are displayed automatically
3. Click on any run card to view its learning curve
4. Hover over the chart to see epoch-by-epoch metrics
5. Click legend items to toggle visibility (loss curves are hidden by default)

### Interpreting Results
- **Best Val Acc**: The highest validation accuracy achieved during training
- **Degradation Level** (low_res): Image resolution after downsampling
  - low_res=32: Minimal degradation (original size)
  - low_res=16: Moderate degradation
  - low_res=8: Severe degradation (most challenging)
- **Color Coding**:
  - Green ≥60%: Excellent performance
  - Orange 50-60%: Good performance
  - Red <50%: Poor performance

## Data Sources

The dashboards aggregate data from:
- `runs/official/`: Production-quality experiments
- `runs/pilot/`: Quick validation experiments
- `runs/archive/`: Previous experimental runs

Each run directory contains:
- `run_config.txt`: Experiment parameters
- `log.txt`: Training logs with final metrics
- `metrics.csv`: Per-epoch metrics (used for learning curves)
- `best.pt`: Best model checkpoint
- `*_curves.png`: Static visualization files

## Updating the Dashboards

After running new experiments:
```bash
python refresh_dashboards.py
```

This will:
1. Scan all runs in the `runs/` directory
2. Extract best accuracy and other metrics
3. Load learning curves for top-performing runs
4. Regenerate both HTML dashboards

The dashboards are completely self-contained (single HTML file) and don't require any server or external resources (Plotly.js is loaded from CDN).

## Technical Details

### Dashboard Scripts

#### `src/tools/generate_dashboard.py`
- Loads run summary from `artifacts/tables/run_summary.csv`
- Creates bar charts for model and degradation comparison
- Generates histogram of accuracy distribution
- Builds filterable results table
- Output: `artifacts/dashboard.html` (~33KB)

#### `src/tools/generate_advanced_dashboard.py`
- Extends basic dashboard with learning curve support
- Loads metrics.csv from top 6 runs
- Creates interactive scatter plots with dual axes (accuracy + loss)
- Implements run selection cards
- Output: `artifacts/dashboard_advanced.html` (~46KB)

#### `refresh_dashboards.py`
- Convenience script to update all dashboards
- Runs summarization and generation steps in sequence
- Reports success/failure status

### Browser Compatibility
- Chrome/Edge: Fully supported
- Firefox: Fully supported
- Safari: Fully supported
- IE11: Not supported (uses ES6 features)

### Performance
- Dashboards load in <1 second (HTML + embedded JavaScript/CSS)
- Charts render in <2 seconds
- Filtering is instantaneous
- Learning curves render in <1 second

## Troubleshooting

**Dashboards don't show my new runs:**
```bash
python refresh_dashboards.py
```
Then reload the HTML in your browser.

**Learning curves not displaying:**
- Check that runs have completed and have `metrics.csv` files
- Advanced dashboard only loads curves for top 6 runs
- Incomplete runs are skipped automatically

**Chart not rendering:**
- Try refreshing the browser (Ctrl+F5)
- Check browser console (F12) for errors
- Ensure JavaScript is enabled

**Accuracy values showing as blank:**
- Run may not have completed successfully
- Check `log.txt` in the run directory for errors
- Missing or malformed `run_config.txt` may also cause this

## Examples

### Finding the best TransNeXt run
1. Open Advanced Dashboard
2. Filter: Model = "transnext_micro"
3. Check the top runs cards or sort the table by accuracy

### Comparing ResNet50 vs TransNeXt on same degradation
1. Open Basic Dashboard
2. Search for relevant runs or use Model filter
3. Look at Model Comparison chart
4. Check accuracy values in the results table

### Analyzing overfitting
1. Open Advanced Dashboard
2. Select a run by clicking its card
3. View the learning curve
4. If validation accuracy (purple line) diverges from training (blue line), overfitting is occurring

## Future Enhancements

Possible improvements:
- [ ] Export results to PDF report
- [ ] Compare specific runs side-by-side
- [ ] Statistical significance testing
- [ ] Per-epoch performance comparison
- [ ] Heatmaps of accuracy by model × degradation
- [ ] Automatic dashboard refresh when new runs complete
