#!/bin/bash
cd "i:/Object-Classification-in-Low-Resolution-THz-Imagery--"

echo "Starting GPU Experiments..."

# Experiment 1: TransNeXt Linear Probe (low_res=16)
echo "Exp 1: TransNeXt LP low_res=16"
python main.py --model transnext_micro --pretrained --freeze_backbone \
  --low_res 16 --out_size 224 --epochs 20 \
  --train_subset 5000 --val_subset 2000 \
  --batch_size 32 --lr 1e-3 \
  --group official --tag transnext_linear_probe_gpu

# Experiment 2: TransNeXt Linear Probe (low_res=8)  
echo "Exp 2: TransNeXt LP low_res=8"
python main.py --model transnext_micro --pretrained --freeze_backbone \
  --low_res 8 --out_size 224 --epochs 20 \
  --train_subset 5000 --val_subset 2000 \
  --batch_size 32 --lr 1e-3 \
  --group official --tag transnext_linear_probe_gpu_low8

echo "Experiments completed!"
