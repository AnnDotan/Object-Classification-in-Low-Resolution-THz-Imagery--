#!/usr/bin/env bash
# US-045 close pipeline — runs all post-inference diagnostics in sequence.
# Started right after run_test_set_inference.py finishes (test_set_results.json exists).
# Each step is idempotent; rerun-safe.
set -euo pipefail

echo "[us045-close] Waiting for test_set_results.json..."
while [ ! -f artifacts/validation/test_set_results.json ]; do sleep 10; done
echo "[us045-close] test_set_results.json present. Starting diagnostics."

echo "[us045-close] 1/5 confusion matrices (18 L5 cells)..."
.venv-gpu/Scripts/python.exe scripts/generate_confusion_matrices.py 2>&1 | tail -20

echo "[us045-close] 2/5 calibration diagrams (42 cells)..."
.venv-gpu/Scripts/python.exe scripts/generate_calibration_diagrams.py 2>&1 | tail -20

echo "[us045-close] 3/5 inference throughput..."
.venv-gpu/Scripts/python.exe scripts/measure_inference_throughput.py 2>&1 | tail -10

echo "[us045-close] 4/5 build diagnostics LaTeX tables..."
.venv-gpu/Scripts/python.exe scripts/build_diagnostics_latex.py 2>&1 | tail -10

echo "[us045-close] 5/5 copy throughput PNG to docs figs..."
cp artifacts/figures/inference_throughput.png docs/_autogen/figs/ || true

echo "[us045-close] Done. Now rerun pdflatex+bibtex+pdflatex+pdflatex."
cd docs
pdflatex -interaction=nonstopmode Final_Report.tex 2>&1 | grep -E "Output written|error" | head -5
bibtex Final_Report 2>&1 | tail -3
pdflatex -interaction=nonstopmode Final_Report.tex 2>&1 | grep -E "Output written|error" | head -3
pdflatex -interaction=nonstopmode Final_Report.tex 2>&1 | grep -E "Output written|error" | head -3
echo "[us045-close] PDF rev3 build complete."
