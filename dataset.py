import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from config import DEVICE, SEED, cfg


# ── Skull stripping ──────────────────────────────────────────────────────────

def skull_strip(image: Image.Image) -> Image.Image:
    """
    Simple skull stripping via Otsu threshold + largest contour masking.
    Returns original image as fallback if stripping fails.
    """
    img = np.array(image.convert('L'))
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 500:
        return image

    mask = np.zeros_like(img)
    cv2.drawContours(mask, [c], -1, 255, thickness=-1)
    stripped = cv2.bitwise_and(img, img, mask=mask)
    stripped = cv2.cvtColor(stripped, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(stripped)


# ── Offline preprocessing ────────────────────────────────────────────────────

def preprocess_dataset(input_dir, output_dir) -> None:
    """Skull-strip every image in input_dir and save to output_dir."""
    from pathlib import Path
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    error_count = 0

    for class_name in os.listdir(input_dir):
        class_in  = input_dir  / class_name
        class_out = output_dir / class_name
        if not class_in.is_dir():
            continue
        os.makedirs(class_out, exist_ok=True)

        for img_name in tqdm(os.listdir(class_in), desc=f'Processing {class_name}'):
            in_path  = class_in  / img_name
            out_path = class_out / img_name
            if out_path.exists():
                continue
            try:
                img      = Image.open(in_path).convert('RGB')
                stripped = skull_strip(img)
                stripped.save(out_path)
            except Exception as e:
                error_count += 1
                print(f'  ⚠️  Error processing {in_path} — {e}')

    if error_count:
        print(f'\n⚠️  Total preprocessing errors: {error_count}')
    else:
        print('Preprocessing complete — no errors ✅')


# ── Transforms ───────────────────────────────────────────────────────────────

def get_transforms():
    """
    Returns (train_tf, eval_tf).
    Train: heavy augmentation. Eval: deterministic resize + normalise.
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((320, 320)),   # larger than IMAGE_SIZE=300 to allow RandomCrop
        transforms.RandomCrop(cfg.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        transforms.Normalize(mean, std),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


# ── DataLoaders ──────────────────────────────────────────────────────────────

def get_dataloaders():
    """
    Returns (train_loader, val_loader, test_loader, class_to_idx).
    15% of training data is held out as validation.
    Testing/ is kept exclusively for final evaluation.
    """
    train_tf, eval_tf = get_transforms()

    full_train_ds = ImageFolder(root=cfg.TRAIN_STRIPPED_DIR, transform=train_tf)
    n_total = len(full_train_ds)
    n_val   = int(n_total * cfg.VAL_SPLIT)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(SEED)
    train_indices, val_indices = [
        s.indices
        for s in random_split(full_train_ds, [n_train, n_val], generator=generator)
    ]

    train_subset = Subset(full_train_ds, train_indices)

    # Validation must use eval transforms (no augmentation)
    val_ds_eval = ImageFolder(root=cfg.TRAIN_STRIPPED_DIR, transform=eval_tf)
    val_subset  = Subset(val_ds_eval, val_indices)

    test_ds = ImageFolder(root=cfg.TEST_STRIPPED_DIR, transform=eval_tf)

    num_workers = min(4, os.cpu_count() or 1)

    train_loader = DataLoader(
        train_subset, batch_size=cfg.BATCH_SIZE,
        shuffle=True, num_workers=num_workers,
        pin_memory=True, persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_subset, batch_size=cfg.BATCH_SIZE,
        shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.BATCH_SIZE,
        shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=(num_workers > 0),
    )

    print(f'\nTrain samples : {len(train_subset)}')
    print(f'Val   samples : {len(val_subset)}')
    print(f'Test  samples : {len(test_ds)}  (held out)')
    return train_loader, val_loader, test_loader, full_train_ds.class_to_idx
