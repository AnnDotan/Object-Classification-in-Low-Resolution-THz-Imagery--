

# Object Classification in Low-Resolution THz Imagery

A deep learning project for evaluating image classification robustness under simulated low-resolution, high-noise conditions inspired by THz-like imaging constraints.

## Overview

This repository presents a framework for analyzing how image classification models perform under challenging visual conditions, including reduced resolution and added noise. The project focuses on benchmarking model robustness, comparing architectures, and organizing reproducible experiments for future development.

The repository includes:
- dataset preparation and preprocessing
- model training and evaluation
- robustness analysis under degraded image conditions
- experiment outputs, logs, and saved artifacts

## Motivation

In practical sensing systems, especially under constrained or noisy imaging conditions, model performance may degrade significantly. This project studies that behavior systematically by simulating difficult visual conditions and measuring how different architectures respond in terms of:
- classification accuracy
- robustness trends across degradation levels
- inference behavior
- experiment reproducibility

## Repository Structure

```text
.
├── artifacts/                  # Saved outputs, figures, summaries, tables, and model-related artifacts
├── data/                       # Dataset files and local data resources
├── docs/                       # Project documentation and supporting material
├── runs/                       # Training and evaluation runs, checkpoints, and logs
├── src/                        # Main source code
├── main.py                     # Main entry point
├── requirements.txt            # Python dependencies
└── README.md
```

## Main Features

- Training and evaluating image classification models in Python
- Testing model robustness under simulated degradation
- Comparing multiple architectures under identical conditions
- Organizing experiment outputs for later analysis
- Reproducible environment setup with `requirements.txt`

## Tech Stack

- Python
- PyTorch
- Torchvision
- timm
- NumPy
- Matplotlib

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/AnnDotan/Object-Classification-in-Low-Resolution-THz-Imagery.git
cd Object-Classification-in-Low-Resolution-THz-Imagery
```

### 2. Create a virtual environment

**Windows**

```bash
py -3.12 -m venv .venv
.\.venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

Run the main project entry point:

```bash
python main.py
```

If your workflow uses additional scripts inside `src/`, run them from the repository root after activating the virtual environment.

## Expected Workflow

A typical workflow in this project is:

1. Prepare or load the dataset
2. Apply the desired degradation and preprocessing pipeline
3. Train or evaluate the selected model
4. Save logs, outputs, and artifacts
5. Compare robustness across runs and architectures

## Notes

- `data/` may contain local dataset files that are not always necessary to version-control
- `runs/` and `artifacts/` may contain large outputs and checkpoints
- `.venv/` should remain local and should not be committed
- Large model files such as `.pt` and `.pth` may require Git LFS

## Future Improvements

- Add configuration files for experiments
- Add pretrained model loading options
- Add automated robustness report generation
- Add visual summaries for degradation-performance curves
- Add clear CLI arguments and reproducible experiment presets

## Contributors

- Ann Dotan
- Itamar Bahat

## License

This project is intended for academic and research use unless stated otherwise.
