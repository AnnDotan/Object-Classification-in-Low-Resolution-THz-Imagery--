#!/bin/bash
# Final analysis script - runs after experiments complete

echo "==================================="
echo "🎯 Final Analysis - TransNeXt Results"
echo "==================================="

cd "i:/Object-Classification-in-Low-Resolution-THz-Imagery--"

# 1. Summarize all runs
echo ""
echo "📊 Aggregating all experiment results..."
python src/tools/summarize_runs.py
echo "✓ Results saved to artifacts/tables/run_summary.csv"

# 2. Generate plots
echo ""
echo "📈 Generating publication-ready plots..."
python src/tools/plot_experiments.py
echo "✓ Plots saved to artifacts/figures/"

# 3. Display summary
echo ""
echo "=================================="
echo "✅ Analysis Complete!"
echo "=================================="
echo ""
echo "Key Results:"
echo "- Linear Probe Summary: artifacts/tables/run_summary.csv"
echo "- Accuracy vs Epoch: artifacts/figures/accuracy_vs_epoch.png"
echo "- Model Comparison: artifacts/figures/model_comparison_bar.png"
echo "- Robustness Curve: artifacts/figures/accuracy_vs_degradation.png"
