"""
Brain Tumor MRI Classification — EfficientNet-B3
=================================================
Run:
    python main.py

Requires:
    - Kaggle credentials set as environment variables KAGGLE_USERNAME / KAGGLE_KEY
      OR a pre-downloaded dataset at brain-tumor-mri-dataset/
    - GPU recommended (CUDA)

Pipeline:
    1. Download & preprocess dataset (skull stripping)
    2. Build dataloaders
    3. EDA plots
    4. Optuna hyperparameter search
    5. Full training with MLflow tracking + early stopping
    6. Evaluation: metrics, confusion matrix, ROC, Grad-CAM
    7. Single-image inference demo
    8. Save checkpoint
"""

import json
import os
import subprocess

import mlflow
import torch

from config import DEVICE, cfg
from dataset import get_dataloaders, preprocess_dataset
from evaluate import (
    plot_confusion_matrix,
    plot_gradcam,
    plot_roc_curves,
    plot_training_curves,
    predict_single_image,
    run_test_evaluation,
)
from model import BrainTumorEfficientNetB3
from train import run_optuna, train


# ── 1. Download dataset (skip if already present) ────────────────────────────

def download_dataset():
    if cfg.TRAIN_DIR.exists():
        print('Dataset already present, skipping download.')
        return

    kaggle_user = os.environ.get('KAGGLE_USERNAME')
    kaggle_key  = os.environ.get('KAGGLE_KEY')
    if not kaggle_user or not kaggle_key:
        raise EnvironmentError(
            'Set KAGGLE_USERNAME and KAGGLE_KEY environment variables '
            'before running, or place the dataset at brain-tumor-mri-dataset/.'
        )

    # Write kaggle.json
    kaggle_dir = os.path.expanduser('~/.kaggle')
    os.makedirs(kaggle_dir, exist_ok=True)
    with open(os.path.join(kaggle_dir, 'kaggle.json'), 'w') as f:
        json.dump({'username': kaggle_user, 'key': kaggle_key}, f)
    os.chmod(os.path.join(kaggle_dir, 'kaggle.json'), 0o600)

    subprocess.run(
        ['kaggle', 'datasets', 'download', '-d',
         'masoudnickparvar/brain-tumor-mri-dataset'],
        check=True,
    )
    subprocess.run(
        ['unzip', '-q', 'brain-tumor-mri-dataset.zip', '-d', 'brain-tumor-mri-dataset'],
        check=True,
    )
    print('Dataset downloaded ✅')


# ── 2. EDA plots ─────────────────────────────────────────────────────────────

def run_eda(class_names):
    import matplotlib.pyplot as plt
    import numpy as np
    from torchvision.datasets import ImageFolder

    train_ds_raw = ImageFolder(root=cfg.TRAIN_DIR)
    test_ds_raw  = ImageFolder(root=cfg.TEST_DIR)
    train_counts = np.bincount([l for _, l in train_ds_raw.imgs])
    test_counts  = np.bincount([l for _, l in test_ds_raw.imgs])

    x = np.arange(len(class_names))
    fig, ax = plt.subplots(figsize=(9, 4))
    bars1 = ax.bar(x - 0.2, train_counts, width=0.4, label='Train', color='#4361ee', alpha=0.85)
    bars2 = ax.bar(x + 0.2, test_counts,  width=0.4, label='Test',  color='#f72585', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(class_names, fontsize=12)
    ax.set_ylabel('Number of Images')
    ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(int(bar.get_height())), ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(int(bar.get_height())), ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(cfg.OUTPUT_DIR / 'class_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print(f'Device : {DEVICE}')
    if DEVICE.type == 'cuda':
        print(f'GPU    : {torch.cuda.get_device_name(0)}')
        print(f'Memory : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    # ── Dataset ──────────────────────────────────────────────────────────
    download_dataset()

    if not cfg.TRAIN_STRIPPED_DIR.exists():
        print('\nPreprocessing training images...')
        preprocess_dataset(cfg.TRAIN_DIR, cfg.TRAIN_STRIPPED_DIR)
        print('Preprocessing test images...')
        preprocess_dataset(cfg.TEST_DIR, cfg.TEST_STRIPPED_DIR)

    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders()
    class_names = list(class_to_idx.keys())
    print(f'\nClass → index mapping: {class_to_idx}')

    # ── EDA ──────────────────────────────────────────────────────────────
    run_eda(class_names)

    # ── MLflow setup ─────────────────────────────────────────────────────
    mlflow.set_tracking_uri('file:./mlruns')
    mlflow.set_experiment(cfg.MLFLOW_EXPERIMENT)

    # ── Optuna hyperparameter search ──────────────────────────────────────
    best_params = run_optuna(train_loader, val_loader)
    best_gamma  = best_params['gamma']

    # ── Re-instantiate model with best dropout ────────────────────────────
    model = BrainTumorEfficientNetB3(
        num_classes=cfg.NUM_CLASSES,
        dropout=cfg.DROPOUT_RATE,   # patched by run_optuna
        freeze_backbone=True,
    ).to(DEVICE)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nTotal params     : {total_params:,}')
    print(f'Trainable params : {trainable_params:,}  ({100*trainable_params/total_params:.1f}%)')

    # ── Training ──────────────────────────────────────────────────────────
    model, history, best_acc = train(model, train_loader, val_loader, best_gamma)

    # ── Training curves ───────────────────────────────────────────────────
    plot_training_curves(history)

    # ── Evaluation ────────────────────────────────────────────────────────
    preds, labels, probs, y_bin = run_test_evaluation(model, test_loader, class_names)
    plot_confusion_matrix(labels, preds, class_names)
    plot_roc_curves(y_bin, probs, class_names)
    plot_gradcam(model, test_loader, class_names)

    # ── Single-image demo ─────────────────────────────────────────────────
    from torchvision.datasets import ImageFolder
    test_ds_raw = ImageFolder(root=cfg.TEST_DIR)
    sample_path, sample_label = test_ds_raw.imgs[10]
    print(f'\nRunning inference on: {sample_path}')
    print(f'True class          : {class_names[sample_label]}')
    predict_single_image(model, sample_path, class_names)

    # ── Save checkpoint ───────────────────────────────────────────────────
    checkpoint = {
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': None,   # optimizer not in scope here; save separately if needed
        'class_to_idx':         class_to_idx,
        'class_names':          class_names,
        'best_val_acc':         best_acc,
        'config': {
            'num_classes':  cfg.NUM_CLASSES,
            'image_size':   cfg.IMAGE_SIZE,
            'dropout_rate': cfg.DROPOUT_RATE,
            'num_epochs':   cfg.NUM_EPOCHS,
        },
    }
    ckpt_path = cfg.OUTPUT_DIR / 'brain_tumor_efficientnet_b3_checkpoint.pt'
    torch.save(checkpoint, ckpt_path)
    print(f'\nCheckpoint saved: {ckpt_path}')

    hist_path = cfg.OUTPUT_DIR / 'training_history.json'
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f'History saved   : {hist_path}')

    print('\nOutputs directory:')
    for p in sorted(cfg.OUTPUT_DIR.iterdir()):
        print(f'  {p.name:40s}  {p.stat().st_size / 1e6:.2f} MB')


if __name__ == '__main__':
    main()
