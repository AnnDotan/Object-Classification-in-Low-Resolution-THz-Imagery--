

# Object Classification in Low-Resolution THz Imagery

A deep learning project for evaluating image classification robustness under simulated low-resolution, high-noise conditions inspired by THz-like imaging constraints.

## Overview

This repository presents a framework for analyzing how image classification models perform under challenging visual conditions, including reduced resolution and added noise. The project focuses on benchmarking model robustness, comparing architectures, and organizing reproducible experiments for future development.

The repository includes:
- dataset preparation and preprocessing
- model training and evaluation
- robustness analysis under degraded image conditions
- experiment outputs, logs, and saved artifacts

## Key Finding (STAGE 1 Complete)

**TransNeXt with Linear Probe significantly outperforms CNN baselines:**

| Model | Low-Res=16 | Improvement |
|-------|-----------|-------------|
| **TransNeXt (Linear Probe)** | **68.75%** | **+14.25%** |
| ResNet50 | 54.50% | baseline |
| DenseNet121 | 51.15% | -3.35% |

**Conclusion**: Pretrained TransNeXt features are highly effective for degraded images. Frozen backbone with head-only training outperforms full fine-tuning (10-11%).

## Degradation Type Isolation (STAGE 2)

**New Feature**: Separate analysis of robustness to specific degradation types:

- **Downsampling**: Low-resolution information loss
- **Blur**: Edge and detail loss
- **Noise**: Random signal corruption

This enables understanding which degradation types each model handles best, informing design decisions for robust systems.

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

### Basic Training

Run the main project entry point:

```bash
python main.py
```

### Advanced Usage

**Specify model and degradation:**

```bash
# Train ResNet50 with all degradation types (default)
python main.py --model resnet50 --pretrained --epochs 20

# Train TransNeXt with only blur degradation
python main.py --model transnext_micro --pretrained --degradation_type blur --epochs 20

# Train DenseNet with only noise degradation
python main.py --model densenet121 --pretrained --degradation_type noise --epochs 20

# Train with only downsampling (low resolution)
python main.py --model resnet50 --degradation_type downsampling --epochs 20
```

**Degradation types:**
- `all` (default): Downsampling + Blur + Noise + optional Grayscale
- `downsampling`: Low-resolution information loss only
- `blur`: Gaussian blur only
- `noise`: Gaussian noise only

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

## Improvements Implemented

- ✅ Add pretrained model loading options (already exists)
- ✅ Add automated robustness report generation (visualize_run.py)
- ✅ Add visual summaries for degradation-performance curves (top_runs_comparison.py)
- ✅ Add clear CLI arguments (--degradation_type, --backbone_lr, etc.)
- ✅ Add degradation type isolation for robustness analysis

## Future Improvements

- Add configuration files (YAML/JSON) for experiments
- Add experiment presets for common scenarios
- Add interactive HTML dashboard for results
- Add additional degradation types (motion blur, color noise)
- Add automated report generation (PDF)

## Contributors

- Itamar Bahat

## License

This project is intended for academic and research use.
