# Low-Resolution Image Classification Under THz-Like Degradation

A deep learning project for evaluating image classification robustness under simulated low-resolution, high-noise conditions inspired by THz-like imaging constraints.

## Overview

This repository focuses on benchmarking and analyzing image classification models when image quality is degraded by resolution loss and noise.  
The goal is to evaluate how well modern neural networks maintain performance under challenging sensing conditions, and to provide a reproducible framework for robustness testing, comparison, and future experimentation.

The project includes:
- dataset preparation and preprocessing
- model training and evaluation
- robustness analysis under degraded image conditions
- experiment outputs, logs, and saved artifacts

## Motivation

In practical sensing systems, especially under constrained or noisy imaging settings, model accuracy can degrade significantly.  
This project studies that behavior systematically by simulating difficult visual conditions and measuring how different architectures respond in terms of:
- classification accuracy
- robustness trends
- inference behavior
- experiment reproducibility

## Repository Structure

```text
.
├── artifacts/                  # Saved outputs, checkpoints, or generated artifacts
├── data/                       # Dataset files and local data resources
│   └── cifar-10-batches-py/
├── docs/                       # Project documentation and supporting material
├── runs/                       # Training / evaluation logs and run outputs
├── src/                        # Main source code
├── main.py                     # Main entry point
├── requirements.txt            # Python dependencies
├── test_transnext_pretrained.py
└── README.md
