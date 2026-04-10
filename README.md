# 🧠 Brain Tumor MRI Classification — EfficientNet-B3

A PyTorch deep learning pipeline for classifying brain MRI scans into 4 categories using EfficientNet-B3 with transfer learning.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue)](https://mlflow.org/)
[![Optuna](https://img.shields.io/badge/Tuning-Optuna-blue)](https://optuna.org/)
[![HuggingFace](https://img.shields.io/badge/Demo-HuggingFace%20Space-yellow)](https://huggingface.co/spaces)

---

## 🎯 Results

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Glioma | 99.7% | 83.3% | 90.7% |
| Meningioma | 93.5% | 96.8% | 95.1% |
| No Tumor | 91.3% | 100.0% | 95.5% |
| Pituitary | 96.6% | 100.0% | 98.3% |
| **Overall** | **95.3%** | **95.0%** | **94.9%** |

---

## 🧪 Experiments & Comparison

Four experiments were run progressively, each building on the previous:

| # | Model | Loss | Epochs | Test Acc | Glioma Recall | Glioma F1 | Key Change |
|---|---|---|---|---|---|---|---|
| 1 | ResNet-50 | CrossEntropy | 30 | ~95.0% | 84.2% | ~89% | Baseline |
| 2 | EfficientNet-B3 | CrossEntropy | 30 | ~95.0% | 84.2% | ~89% | Backbone swap, 300×300 input |
| 3 | EfficientNet-B3 | Focal Loss (γ=2.0) | 40 | 95.0% | 83.3% | 90.7% | Focal loss + lower LR |
| 4 | EfficientNet-B3 + Optuna | Focal Loss (γ=best) | 40 | **95.0%** | 83.3% | **90.7%** | Tuned LR, dropout, gamma |
| 5 | EfficientNet-B7 | Focal Loss (γ=best) | 40 | **95.75%** | **89.3%** | **93.5%** | Larger backbone |

### Key findings

- **ResNet-50 → EfficientNet-B3:** Switching backbones gave comparable overall accuracy with fewer parameters (~12M vs ~25M), faster inference, and a more compact model suitable for deployment.
- **CrossEntropy → Focal Loss:** Directly targeted the glioma problem by down-weighting easy classes (No Tumor, Pituitary) during training. Glioma F1 improved and precision jumped to 99.7%, though recall remained the bottleneck at 83.3%.
- **Glioma remains the hardest class** across all experiments due to its diffuse, irregular appearance on MRI — visually similar to meningioma and can present without a clearly defined boundary, consistent with the clinical literature.
- **No Tumor and Pituitary** hit 100% recall in all experiments — these classes are visually distinct and the model learned them quickly even in the warm-up phase.
- **Optuna tuning** stabilized training and reduced overfitting on the easy classes, contributing to better generalization rather than a raw accuracy jump.

---

## 📁 Project Structure

```
brain-tumor-classifier/
├── config.py          # All hyperparameters and paths
├── dataset.py         # Skull stripping, transforms, dataloaders
├── model.py           # EfficientNet-B3 model definition
├── train.py           # FocalLoss, training loop, Optuna tuning, MLflow tracking
├── evaluate.py        # Metrics, confusion matrix, ROC curves, Grad-CAM
├── main.py            # Entry point — runs the full pipeline
└── requirements.txt   # Dependencies
```

---

## 🚀 Quickstart

### 1. Clone & install

```bash
git clone https://github.com/your-username/brain-tumor-classifier.git
cd brain-tumor-classifier
pip install -r requirements.txt
```

### 2. Set Kaggle credentials

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

Or place the dataset manually at `brain-tumor-mri-dataset/` (see [Dataset](#dataset) below).

### 3. Run

```bash
python main.py
```

This will automatically:
1. Download and preprocess the dataset (skull stripping)
2. Run Optuna hyperparameter search (10 trials)
3. Train the model with MLflow tracking + early stopping
4. Evaluate on the held-out test set
5. Generate confusion matrix, ROC curves, and Grad-CAM visualizations
6. Save the checkpoint to `outputs/`

---

## 📦 Dataset

**[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)** by Masoud Nickparvar

| Split | Glioma | Meningioma | No Tumor | Pituitary | Total |
|---|---|---|---|---|---|
| Training | 1400 | 1400 | 1400 | 1400 | 5600 |
| Testing | 400 | 400 | 400 | 400 | 1600 |

---

## 🏗️ Model Architecture

- **Backbone:** EfficientNet-B3 pretrained on ImageNet
- **Head:** `Dropout → FC(1536→512) → BatchNorm → ReLU → Dropout → FC(512→4)`
- **Preprocessing:** Skull stripping via Otsu threshold + contour masking
- **Input size:** 300×300

### Two-phase fine-tuning
| Phase | Epochs | Trainable layers |
|---|---|---|
| Warm-up | 0–5 | Head only |
| Fine-tuning | 5–40 | Last 4 backbone blocks + head |

---

## ⚙️ Key Features

- **Focal Loss** (γ=2.0) with label smoothing to focus training on hard cases (glioma)
- **Optuna** hyperparameter search over LR, dropout, gamma, and weight decay
- **MLflow** experiment tracking — all runs logged to `./mlruns`
- **Early stopping** with patience=8
- **Grad-CAM** explainability visualizations
- **CosineAnnealingLR** scheduler

---

## 🔧 Configuration

All hyperparameters are in `config.py`:

```python
NUM_EPOCHS   = 40
BATCH_SIZE   = 32
LR           = 5e-5      # backbone LR
HEAD_LR      = 5e-4      # head LR
DROPOUT_RATE = 0.4
PATIENCE     = 8         # early stopping
OPTUNA_TRIALS = 10
```

---

## 📊 MLflow Tracking

View experiment runs locally:

```bash
mlflow ui
```

Then open `http://localhost:5000` in your browser.

---

## 🌐 Demo

A live inference demo is available on HuggingFace Spaces:

👉 **[Try it here](https://huggingface.co/spaces/AhmedTarek1/BTC)**

---

## 📌 Loading a Saved Checkpoint

```python
import torch
from model import BrainTumorEfficientNetB3

checkpoint = torch.load('outputs/brain_tumor_efficientnet_b3_checkpoint.pt', map_location='cpu')

model = BrainTumorEfficientNetB3(
    num_classes=4,
    dropout=checkpoint['config']['dropout_rate'],
    freeze_backbone=False,
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print('Classes :', checkpoint['class_names'])
print('Val Acc :', checkpoint['best_val_acc'])
```

---

## ⚠️ Disclaimer

This project is for **research and educational purposes only** and is not intended as a medical diagnostic tool.
