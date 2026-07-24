# Cats vs Dogs Classifier — CNN from Scratch

A binary image classifier that tells cats and dogs apart, built with a custom convolutional neural network in TensorFlow/Keras (no pretrained backbone). The project started as an extension of an MNIST lab — Dropout, BatchNorm, Early Stopping, LR scheduling, optimizer comparison — adapted to a real photo dataset, and grew into a small set of controlled experiments around architecture depth, dropout scheduling, and optimizer choice.

The best configuration (a 5-block CNN trained with AdamW) reaches **95.04% validation accuracy**.

## Dataset

[Cats vs Dogs (10k cat / 10k dog images)](https://www.kaggle.com/datasets/haroon669/cats-vs-dogs-dataset-10k-cat-10k-dog-images) from Kaggle, downloaded directly in-notebook via `kagglehub`.

- **20,000** training images, **5,000** test images
- 2 classes: `cats`, `dogs`
- Already split into `train/` and `test/` folders, so no manual splitting was needed
- Images resized to 128×128 and rescaled to `[0, 1]`

## Technologies Used

- Python
- TensorFlow / Keras (2.20)
- KaggleHub (dataset download)
- NumPy
- Matplotlib
- Google Colab (GPU runtime)

## Project Structure

The notebook is organized as a series of experiments rather than a single linear pipeline:

| Section | What it does |
|---|---|
| Baseline CNN | 4 conv blocks, Dropout 0.3, BatchNorm, Adam — first working model |
| Dropout/epoch variant | Same idea with Dropout 0.5 and 3 conv blocks, 30 max epochs |
| Hyperparameter experiments (A–D) | Config-driven architecture toggle (filters, dropout schedule, pooling type, LR schedule) |
| Optimizer comparison | Adam vs AdamW vs SGD vs RMSprop on a fixed architecture |
| Final reproducible run (E → F → G) | Seeded, deterministic experiments comparing 3 architecture variants back-to-back |

## Model Architecture

Every model in the notebook follows the same building block, repeated a different number of times per experiment:

```
Conv2D → BatchNormalization → MaxPooling2D → Dropout   (× N blocks)
GlobalAveragePooling2D
Dense → BatchNormalization → Dropout
Dense(1, sigmoid)
```

- **Conv blocks:** 3–5 depending on the experiment, with filter sizes scaling up (32 → 64 → 128 → 256 → 512 in the deepest version)
- **Dropout:** graduated per block (lower in early layers, higher in deeper layers) rather than a single fixed rate
- **Global Average Pooling** used instead of `Flatten()` to keep the classifier head lighter
- **Output:** single sigmoid unit for binary classification (`cat` = 0, `dog` = 1)

The best-performing setup (Experiment F) uses **5 conv blocks** (32→64→128→256→512 filters), a dropout schedule from 0.15 to 0.35, a 512-unit dense layer, and the **AdamW** optimizer.

## Training Configuration

- **Input size:** 128×128×3
- **Batch size:** 32
- **Loss:** Binary cross-entropy
- **Augmentation:** Random horizontal flip, rotation (±10%), zoom (±10%) — training set only
- **Optimizers tested:** Adam, AdamW, SGD (with Nesterov momentum), RMSprop
- **LR scheduling:** `ReduceLROnPlateau` in most runs; a custom cosine decay `LearningRateScheduler` in others
- **Callbacks:** `EarlyStopping` (patience 5–8, restores best weights), `ModelCheckpoint` (saves best model by validation accuracy), `ReduceLROnPlateau` / cosine decay
- **Reproducibility:** the final experiment run sets seeds for Python, NumPy, and TensorFlow, enables `tf.config.experimental.enable_op_determinism()`, and seeds the augmentation layers individually so results are consistent across re-runs

## Features

- Config-driven experiments — architecture and hyperparameters are swapped through a single `CONFIGS` dictionary rather than duplicated code
- Side-by-side optimizer benchmarking (accuracy, loss, training time, seconds/epoch)
- Overfitting check after every run (train/val accuracy gap, labeled Good / Moderate / Warning)
- Prediction visualization on 25 sample test images with model confidence, color-coded green (correct) / red (incorrect)
- Deterministic re-runs via full RNG seeding and forced deterministic GPU ops

## Results

| Experiment | Architecture | Optimizer | LR Schedule | Val Accuracy | Val Loss | Epochs Run |
|---|---|---|---|---|---|---|
| Baseline | 4 conv blocks, Dropout 0.3 | Adam | ReduceLROnPlateau | 0.9264 | 0.1925 | 25 |
| Dropout 0.5 variant | 3 conv blocks, Dropout 0.5 | Adam | ReduceLROnPlateau | 0.5876 | 0.7011 | 6 (stopped early) |
| Experiment A | 4 conv blocks, graduated dropout | Adam | ReduceLROnPlateau | 0.9366 | 0.1560 | 32 |
| Experiment E | 4 conv blocks, graduated dropout | Adam | Cosine decay | 0.9468 | 0.1460 | 47 |
| **Experiment F (best)** | **5 conv blocks, graduated dropout** | **AdamW** | **ReduceLROnPlateau** | **0.9504** | **0.1236** | 46 |
| Experiment G | 4 conv blocks, dense=512 | Adam | Cosine decay | 0.9162 | 0.2070 | 26 |

**Best model — training curves (Experiment F):**

![Training curves — best model](images/training_curves_best_model.png)

**Best model — sample predictions on test images:**

![Sample predictions](images/sample_predictions.png)

**Architecture experiment comparison (E vs F vs G):**

![Experiment comparison — accuracy and loss](images/experiment_comparison_accuracy_loss.png)

![Experiment comparison — speed tradeoff](images/experiment_comparison_speed_tradeoff.png)

**Optimizer comparison** (fixed 4-conv-block architecture, cosine LR decay, 25 max epochs):

| Optimizer | Val Accuracy | Train Accuracy | Gap | Epochs | Time/Epoch |
|---|---|---|---|---|---|
| Adam | 0.9212 | 0.9115 | -0.0057 | 22 | 97.5s |
| AdamW | 0.9200 | 0.9071 | 0.0028 | 20 | 99.1s |
| RMSprop | 0.8928 | 0.8995 | 0.0019 | 17 | 104.1s |
| SGD | 0.6748 | 0.7552 | 0.0596 | 9 | 101.0s |

SGD needed a much higher learning rate than the 0.001 used here to be competitive — it underperformed across every run in the notebook.

![Optimizer comparison — accuracy and loss](images/optimizer_comparison_accuracy_loss.png)

![Optimizer comparison — speed tradeoff](images/optimizer_speed_tradeoff.png)

## Performance Analysis

- **Best model:** Experiment F — 5 conv blocks, AdamW, graduated dropout — 95.04% validation accuracy with a small train/val gap (0.0145), indicating good generalization rather than overfitting.
- **Adam vs AdamW vs RMSprop** land within a couple of points of each other; **SGD** consistently trails by 20+ points at the same learning rate, since it doesn't adapt per-parameter and would need a separate LR tune to be fair.
- The **Dropout 0.5 / 3-block run** is a clear outlier — it stopped after only 6 epochs at 58.76% validation accuracy. This looks like too much dropout combined with a shallower network caused the model to underfit before `EarlyStopping` could give it a chance to recover.
- Deeper isn't always better in isolation — Experiment G (dense=512, 4 blocks) underperforms Experiment E (dense=256, 4 blocks) despite more capacity, suggesting the extra dense width didn't help without also increasing conv depth (as in Experiment F).
- Cosine decay and `ReduceLROnPlateau` both perform well; the notebook doesn't isolate which schedule contributed more since architecture also changed between those runs.

## Visualizations

All six images embedded above come straight from the notebook's actual output cells:

| Image | Source in notebook |
|---|---|
| `training_curves_best_model.png` | Cell 6 ("AUTO LOOP: E → F → G"), Experiment F block — accuracy/loss plot |
| `sample_predictions.png` | Cell 6, Experiment F block — 5×5 grid of test predictions with confidence |
| `experiment_comparison_accuracy_loss.png` | Cell 6, final comparison section — val accuracy/loss for E, F, G overlaid |
| `experiment_comparison_speed_tradeoff.png` | Cell 6, final comparison section — accuracy vs seconds/epoch bar chart |
| `optimizer_comparison_accuracy_loss.png` | Cell 5 ("OPTIMIZER COMPARISON") — val accuracy/loss for Adam/AdamW/SGD/RMSprop |
| `optimizer_speed_tradeoff.png` | Cell 5, same section — accuracy vs seconds/epoch bar chart |

**Important:** these `![...]` image links only render on GitHub once you push an `images/` folder containing these six PNGs to your repo, sitting next to this README. Keep the folder name and filenames exactly as above, or update the paths in this file to match.

### Missing but useful — recommended additions

Your notebook tracks accuracy and loss well, but never computes a confusion matrix or per-class metrics. Add this cell after evaluating your best model, and save the output for the README:

```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Collect predictions across the full validation set (not just one batch)
y_true, y_pred = [], []
for images, labels in val_ds_prepped:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy().flatten().astype(int))
    y_pred.extend((preds.flatten() > 0.5).astype(int))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted'); plt.ylabel('True')
plt.title('Confusion Matrix — Best Model (Experiment F)')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

print(classification_report(y_true, y_pred, target_names=class_names))
```

Save the figure as `confusion_matrix.png` and place it under **Results**, right after `sample_predictions.png`.

## Repository Structure (Recommended)

```
cats-vs-dogs-cnn/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── catdog.ipynb
├── models/
│   └── best_model_expF.h5
└── images/
    ├── training_curves_best_model.png
    ├── sample_predictions.png
    ├── experiment_comparison_accuracy_loss.png
    ├── experiment_comparison_speed_tradeoff.png
    ├── optimizer_comparison_accuracy_loss.png
    └── optimizer_speed_tradeoff.png
```

## Future Improvements

- Try transfer learning (MobileNetV2 / EfficientNet) as a comparison point against the from-scratch CNN
- Add a confusion matrix and precision/recall/F1 instead of relying on accuracy alone
- Save models in the native `.keras` format instead of legacy `.h5` (Keras currently warns about this on every checkpoint save)
- Automate the hyperparameter search (Keras Tuner / Optuna) instead of manually toggling config dictionaries
- Isolate LR schedule from architecture changes to know which one is actually driving the accuracy gains
- Add Grad-CAM visualizations to see what the CNN is actually looking at

## Author

**Nimra Maqbool**
University of Management and Technology

---

*This README was written based on an actual training run of this notebook — all numbers above come directly from the notebook's output cells.*
