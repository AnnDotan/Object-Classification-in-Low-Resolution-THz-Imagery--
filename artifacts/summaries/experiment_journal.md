# Experiment Journal

מטרת הקובץ:
לתעד כל ניסוי בצורה ידנית-מחקרית, בנוסף ללוגים האוטומטיים.

## איך לתעד כל ניסוי
לכל ניסוי להוסיף:
- תאריך
- מטרה
- מודל
- האם pretrained
- out_size
- low_res
- epochs
- batch_size
- subset
- best_val_acc
- זמן ריצה
- מסקנה קצרה
- האם זה pilot או official

---

## Suggested Sections

### Pilot Runs
ניסויי sanity check, בדיקות מהירות, בדיקות pipeline.

### Official Runs
ניסויי baseline, DenseNet, TransNeXt, וניסויי השוואה שיופיעו בדוח.

# Experiment Journal

## Archived Pilot Runs

### Resolution comparison with ResNet50 (pretrained)
Goal: verify that stronger degradation reduces validation accuracy.

Runs:
- low_res=32 -> best_val_acc=0.5055
- low_res=16 -> best_val_acc=0.4555
- low_res=8  -> best_val_acc=0.4170

Conclusion:
Validation accuracy decreases as low_res decreases, which confirms that the degradation pipeline affects model performance in a meaningful way.

### Early model comparison at low_res=16
Goal: perform a quick pilot comparison under identical settings.

Runs:
- ResNet18 pretrained -> best_val_acc=0.4850
- ResNet50 pretrained -> best_val_acc=0.4555
- DenseNet121 pretrained -> best_val_acc=0.4550

Conclusion:
These are pilot-only results on small subsets and short training, so they are not used for final architectural conclusions.


## Official Comparison 1: ResNet50 vs DenseNet121

Setup:
- out_size=32
- low_res=16
- epochs=20
- batch_size=32
- train_subset=5000
- val_subset=2000
- lr=0.001
- pretrained=True

Results:
- ResNet50: best_val_acc=0.5450
- DenseNet121: best_val_acc=0.5115

Initial conclusion:
In the first official comparison, DenseNet121 did not outperform the ResNet50 baseline under the selected training setup.
This does not yet invalidate the project hypothesis, but suggests that further controlled experiments are needed.

## Official Comparison 1: ResNet50 vs DenseNet121 at low_res=16

Setup:
- pretrained=True
- out_size=32
- low_res=16
- epochs=20
- batch_size=32
- train_subset=5000
- val_subset=2000
- lr=0.001

Results:
- ResNet50: best_val_acc=0.5450
- DenseNet121: best_val_acc=0.5115

Conclusion:
Under this setup, ResNet50 outperformed DenseNet121.

## Official Comparison 2: ResNet50 vs DenseNet121 at low_res=8

Setup:
- pretrained=True
- out_size=32
- low_res=8
- epochs=20
- batch_size=32
- train_subset=5000
- val_subset=2000
- lr=0.001

Results:
- ResNet50: best_val_acc=0.4795
- DenseNet121: best_val_acc=0.4505

Conclusion:
Under stronger degradation, both models degraded, and ResNet50 still outperformed DenseNet121.

## TransNeXt Micro - pretrained pilot (small)

- Model: transnext_micro
- Pretrained: yes
- Input size: 224
- low_res: 16
- Train subset: 256
- Val subset: 128
- Epochs: 3
- Batch size: 2
- Best val acc: 0.1484
- Runtime: ~353s on CPU

Notes:
- Pretrained weights loaded successfully.
- Missing keys were only head.weight and head.bias, as expected after replacing the classifier for 10 classes.
- Performance improved compared to scratch (0.1484 vs 0.0859), but runtime is still high and accuracy is still low relative to existing baselines.

## TransNeXt Micro - final pilot

- Model: transnext_micro
- Pretrained: yes
- Input size: 224
- low_res: 16
- Train subset: 2000
- Val subset: 500
- Epochs: 10
- Batch size: 2
- Best val acc: 0.1140
- Runtime: 8742.6s (~2.43 hours) on CPU

Conclusion:
- Full integration of TransNeXt was completed successfully.
- Official pretrained weights were loaded correctly.
- Despite successful integration, performance remained very low relative to the CNN baselines.
- Under the current project constraints (CPU-only, degraded CIFAR-10, small-to-medium subsets), TransNeXt is not competitive with ResNet50 or DenseNet121.

## TransNeXt Micro - pilot with lower learning rate

- Model: transnext_micro
- Pretrained: yes
- Input size: 224
- low_res: 16
- Train subset: 2000
- Val subset: 500
- Epochs: 10
- Batch size: 2
- Learning rate: 0.0001
- Best val acc: 0.1140
- Runtime: 9314.5s (~2.59 hours) on CPU

Epoch summary:
- Epoch 1: 0.1120
- Epoch 2: 0.1020
- Epoch 3: 0.1080
- Epoch 4: 0.0840
- Epoch 5: 0.0940
- Epoch 6: 0.0940
- Epoch 7: 0.0800
- Epoch 8: 0.1140
- Epoch 9: 0.1140
- Epoch 10: 0.1080

Conclusion:
- Lowering the learning rate from 1e-3 to 1e-4 did not improve the best validation accuracy.
- Best validation accuracy remained 0.1140, similar to the previous pretrained pilot.
- Runtime increased slightly, so LR reduction alone does not solve the optimization issue.

## TransNeXt Micro - pilot with low_res=32

- Model: transnext_micro
- Pretrained: yes
- Input size: 224
- low_res: 32
- Train subset: 2000
- Val subset: 500
- Epochs: 10
- Batch size: 2
- Learning rate: 0.001
- Best val acc: 0.1140
- Runtime: 9223.7s (~2.56 hours) on CPU

Epoch summary:
- Epoch 1: 0.1020
- Epoch 2: 0.1120
- Epoch 3: 0.0960
- Epoch 4: 0.0980
- Epoch 5: 0.0820
- Epoch 6: 0.0940
- Epoch 7: 0.0800
- Epoch 8: 0.1140
- Epoch 9: 0.0980
- Epoch 10: 0.0800

Conclusion:
Changing the degradation level from low_res=16 to low_res=32 did not improve the performance.
The best validation accuracy remained 0.1140, suggesting that the issue is not primarily caused by the degradation level.



## TransNeXt Micro - larger dataset experiment

- Model: transnext_micro
- Pretrained: yes
- Input size: 224
- low_res: 16
- Train subset: 5000
- Val subset: 2000
- Epochs: 10
- Batch size: 2
- Learning rate: 0.001
- Best val acc: 0.1080
- Runtime: 5666.7s (~1.57 hours) on GPU

Epoch summary:
- Epoch 1: 0.0990
- Epoch 2: 0.0995
- Epoch 3: 0.1015
- Epoch 4: 0.0975
- Epoch 5: 0.1080
- Epoch 6: 0.0980
- Epoch 7: 0.0990
- Epoch 8: 0.0965
- Epoch 9: 0.1080
- Epoch 10: 0.1080

Conclusion:
Increasing the dataset size from 2000 to 5000 training samples did not significantly improve performance.
The model remains close to random guess (~10% accuracy).


## TransNeXt Micro - batch size 8 experiment

- Model: transnext_micro
- Pretrained: yes
- Input size: 224
- low_res: 16
- Train subset: 5000
- Val subset: 2000
- Epochs: 10
- Batch size: 8
- Learning rate: 0.001
- Best val acc: 0.1085
- Runtime: 1710.8s (~28.5 minutes) on GPU

Epoch summary:
- Epoch 1: 0.0995
- Epoch 2: 0.0995
- Epoch 3: 0.0990
- Epoch 4: 0.0975
- Epoch 5: 0.1085
- Epoch 6: 0.0965
- Epoch 7: 0.0965
- Epoch 8: 0.1015
- Epoch 9: 0.0980
- Epoch 10: 0.1080

Conclusion:
Increasing the batch size from 2 to 8 significantly reduced training time,
but did not improve validation accuracy.
The model still remains near random guess (~10% accuracy).


## TransNeXt Micro - batch size 8 experiment

- Model: transnext_micro
- Pretrained: yes
- Input size: 224
- low_res: 16
- Train subset: 5000
- Val subset: 2000
- Epochs: 10
- Batch size: 8
- Learning rate: 0.001
- Best val acc: 0.1085
- Runtime: 1710.8s (~28.5 minutes) on GPU

Epoch summary:
- Epoch 1: 0.0995
- Epoch 2: 0.0995
- Epoch 3: 0.0990
- Epoch 4: 0.0975
- Epoch 5: 0.1085
- Epoch 6: 0.0965
- Epoch 7: 0.0965
- Epoch 8: 0.1015
- Epoch 9: 0.0980
- Epoch 10: 0.1080

Conclusion:
Increasing the batch size from 2 to 8 significantly reduced runtime,
but did not improve validation accuracy.
The model still remains close to random guess (~10% accuracy).



## TransNeXt Micro - long training with normalization

- Model: transnext_micro
- Pretrained: yes
- Input size: 224
- low_res: 16
- Train subset: 5000
- Val subset: 2000
- Epochs: 40
- Batch size: 8
- Learning rate: 0.0005
- Best val acc: 0.1085
- Runtime: 6491.4s (~1.80 hours) on GPU

Conclusion:
Even with longer training (40 epochs), normalization, GPU acceleration, and a larger batch size, the model did not improve beyond random-guess level.
This suggests the issue is not insufficient training time, but a deeper mismatch in the fine-tuning setup.


