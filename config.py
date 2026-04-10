import random
from pathlib import Path

import numpy as np
import torch


# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Config:
    # ── Paths ─────────────────────────────────────────────────────────────
    DATA_ROOT          = Path('brain-tumor-mri-dataset')
    TRAIN_DIR          = DATA_ROOT / 'Training'
    TEST_DIR           = DATA_ROOT / 'Testing'
    TRAIN_STRIPPED_DIR = DATA_ROOT / 'Training_stripped'
    TEST_STRIPPED_DIR  = DATA_ROOT / 'Testing_stripped'
    OUTPUT_DIR         = Path('outputs')

    # ── Model ─────────────────────────────────────────────────────────────
    NUM_CLASSES  = 4
    IMAGE_SIZE   = 300      # EfficientNet-B3 native input size
    DROPOUT_RATE = 0.4

    # ── Training ──────────────────────────────────────────────────────────
    BATCH_SIZE   = 32
    NUM_EPOCHS   = 40
    LR           = 5e-5     # backbone (fine-tuning phase)
    HEAD_LR      = 5e-4     # classification head
    WEIGHT_DECAY = 1e-4
    T_MAX        = 40       # CosineAnnealingLR period

    # ── Validation split ──────────────────────────────────────────────────
    VAL_SPLIT = 0.15

    # ── Two-phase fine-tuning ─────────────────────────────────────────────
    # Phase 1 [0 .. FREEZE_EPOCHS):   head only
    # Phase 2 [FREEZE_EPOCHS .. end): last N backbone blocks + head
    FREEZE_EPOCHS        = 5
    UNFREEZE_LAST_LAYERS = 4

    # ── Early stopping ────────────────────────────────────────────────────
    PATIENCE = 8   # epochs without val_acc improvement before stopping

    CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

    # ── MLflow / Optuna ───────────────────────────────────────────────────
    MLFLOW_EXPERIMENT = 'brain_tumor_efficientnet_b3'
    OPTUNA_TRIALS     = 10
    OPTUNA_EPOCHS     = 5


cfg = Config()
cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
