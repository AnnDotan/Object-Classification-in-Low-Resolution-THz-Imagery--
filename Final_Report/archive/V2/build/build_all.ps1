# PowerShell entrypoint for the full Final_Report build.
# Run from the repository root:
#   pwsh Final_Report\build\build_all.ps1
#
# Order matters: data -> reference docx -> figures/tables -> docx -> pdf -> validate.
# All steps are deterministic and re-runnable.

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Get-Location).Path

Write-Output "=== [1/8] aggregate per-pipeline PSNR/SSIM ==="
python Final_Report/scripts/aggregate_pipelines.py

Write-Output "=== [2/8] build IEEE tables ==="
python Final_Report/scripts/build_tables.py

Write-Output "=== [3/8] build figures (curves, scatters, attribution, multiseed) ==="
python Final_Report/scripts/build_figures.py

Write-Output "=== [4/8] build sample-image grids ==="
python Final_Report/scripts/build_samples.py

Write-Output "=== [5/8] copy reused figures from artifacts/figures ==="
python Final_Report/scripts/copy_reused_figures.py

Write-Output "=== [6/8] build BGU-styled reference.docx (one-time) ==="
python Final_Report/scripts/make_reference_docx.py

Write-Output "=== [7/8] build the DOCX deliverable + reference PDF ==="
python Final_Report/scripts/build_report.py pdf docx

Write-Output "=== [8/8] validate DOCX against BGU + PRD rules ==="
python Final_Report/scripts/validate_report.py

Write-Output ""
Write-Output "Done. Deliverables:"
Get-ChildItem Final_Report\fin-2026-061.* | Format-Table Name, Length, LastWriteTime
