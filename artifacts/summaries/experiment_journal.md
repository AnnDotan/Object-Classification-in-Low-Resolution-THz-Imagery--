# Experiment Journal

## Systematic Experiment Plan (36 experiments)

### Phase A: CIFAR-10 Systematic — COMPLETE (9/9)

#### Level 1 Mild (low_res=16, blur k=3 s=0.5, noise 0.04, S&P 2%)
- ResNet50: 81.2% (26 epochs, early stopped)
- DenseNet121: 81.9% (16 epochs, early stopped)
- TransNeXt Micro: 68.2% (19 epochs, early stopped)

#### Level 2 Moderate (low_res=16, blur k=5 s=1.0, noise 0.08, S&P 5%)
- ResNet50: 78.7% (26 epochs, early stopped)
- DenseNet121: 80.4% (26 epochs, early stopped)
- TransNeXt Micro: 63.6% (15 epochs, early stopped)

#### Level 3 Severe (low_res=8, blur k=7 s=1.5, noise 0.12, S&P 8%)
- ResNet50: 61.5% (27 epochs, early stopped)
- DenseNet121: 63.4% (15 epochs, early stopped)
- TransNeXt Micro: 44.9% (18 epochs, early stopped)

**Conclusion**: DenseNet121 consistently leads on CIFAR-10. Dense connections help preserve low-level features under degradation. Severe degradation (8px) causes ~20% accuracy drop. TransNeXt frozen backbone benefits from ImageNet features but trails CNNs by 15-18%.

### Phase B: MNIST Systematic — IN PROGRESS (6/9)

#### Level 1 Mild
- ResNet50: 99.1% (14 epochs)
- DenseNet121: 98.7% (7 epochs)
- TransNeXt Micro: 92.5% (30 epochs, full training)

#### Level 2 Moderate
- ResNet50: 99.1% (22 epochs)
- DenseNet121: 99.0% (15 epochs)
- TransNeXt Micro: 92.8% (26 epochs)

#### Level 3 Severe — Pending (3 experiments)

**Conclusion**: MNIST is much easier — even TransNeXt achieves 92%+ under degradation. ResNet50 slightly leads DenseNet121 on MNIST (99.1% vs 98.7-99.0%), reversing the CIFAR-10 ranking. This suggests task complexity matters more than architecture choice for simple visual domains.

### Phase C: Single-Degradation Isolation — TODO (12 experiments)
### Phase D: Clean Baselines — TODO (6 experiments)

---

## Pre-Systematic Reference Results (CIFAR-10, combined degradation)

Best runs preserved in `runs/official/`:
- DenseNet121: 80.7% (combined_densenet121_tuned, 30 epochs)
- ResNet50: 78.8% (combined_resnet50_tuned, 30 epochs)
- TransNeXt Micro: 64.2% (combined_transnext_lp_tuned, 30 epochs)

These confirm the systematic L2 results are consistent and reproducible.
