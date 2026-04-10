import copy
import time

import mlflow
import mlflow.pytorch
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from optuna.samplers import TPESampler
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import DEVICE, SEED, cfg
from model import BrainTumorEfficientNetB3

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Focal Loss ───────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss with optional label smoothing.
    Down-weights well-classified examples so training focuses on hard cases
    like glioma.

    Args:
        gamma (float): Focusing parameter. 0 = standard CE. 2.0 recommended.
        label_smoothing (float): Smoothing factor applied before focal weighting.
    """

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.1):
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce   = F.cross_entropy(logits, targets,
                               label_smoothing=self.label_smoothing,
                               reduction='none')    # [B]
        pt   = torch.exp(-ce)                        # p(correct class)
        loss = ((1 - pt) ** self.gamma * ce).mean()
        return loss


# ── Training utilities ───────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0.0
    def update(self, val, n=1):
        self.sum += val * n; self.count += n; self.avg = self.sum / self.count


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    loss_m = AverageMeter()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        preds    = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        loss_m.update(loss.item(), labels.size(0))
    return loss_m.avg, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    loss_m = AverageMeter()
    correct = total = 0
    all_preds, all_labels, all_probs = [], [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        loss_m.update(criterion(logits, labels).item(), labels.size(0))
        probs  = torch.softmax(logits, 1)
        preds  = probs.argmax(1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    return (loss_m.avg, correct / total,
            np.array(all_preds), np.array(all_labels), np.array(all_probs))


def build_optimizer(model):
    head_params     = list(model.classifier.parameters())
    backbone_params = [p for p in model.features.parameters() if p.requires_grad]
    return optim.AdamW(
        [{'params': head_params,     'lr': cfg.HEAD_LR},
         {'params': backbone_params, 'lr': cfg.LR}],
        weight_decay=cfg.WEIGHT_DECAY,
    )


# ── Optuna hyperparameter tuning ─────────────────────────────────────────────

def run_optuna(train_loader, val_loader) -> dict:
    """
    Run an Optuna study to find the best hyperparameters.
    Each trial trains a fresh model for OPTUNA_EPOCHS epochs.
    All trials are logged as nested MLflow runs under 'optuna_study'.

    Returns the best_params dict (also patches cfg in-place).
    """
    mlflow.set_tracking_uri('file:./mlruns')
    mlflow.set_experiment(cfg.MLFLOW_EXPERIMENT)

    def objective(trial, parent_run_id=None):
        lr           = trial.suggest_float('lr',           1e-5, 5e-4, log=True)
        head_lr      = trial.suggest_float('head_lr',      1e-4, 1e-2, log=True)
        dropout      = trial.suggest_float('dropout',      0.2,  0.5)
        gamma        = trial.suggest_float('gamma',        1.0,  3.0)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

        trial_model = BrainTumorEfficientNetB3(
            num_classes=cfg.NUM_CLASSES,
            dropout=dropout,
            freeze_backbone=False,   # fully unfrozen for fast proxy evaluation
        ).to(DEVICE)

        criterion_t = FocalLoss(gamma=gamma, label_smoothing=0.1)
        optimizer_t = optim.AdamW(
            [{'params': list(trial_model.classifier.parameters()), 'lr': head_lr},
             {'params': list(trial_model.features.parameters()),   'lr': lr}],
            weight_decay=weight_decay,
        )
        scheduler_t = CosineAnnealingLR(optimizer_t, T_max=cfg.OPTUNA_EPOCHS, eta_min=1e-6)

        for _ in range(cfg.OPTUNA_EPOCHS):
            train_one_epoch(trial_model, train_loader, criterion_t, optimizer_t)
            scheduler_t.step()

        _, val_acc, _, _, _ = evaluate(trial_model, val_loader, criterion_t)

        with mlflow.start_run(run_name=f'optuna_trial_{trial.number}',
                              nested=True, parent_run_id=parent_run_id):
            mlflow.log_params({
                'lr': lr, 'head_lr': head_lr, 'dropout': dropout,
                'gamma': gamma, 'weight_decay': weight_decay,
            })
            mlflow.log_metric('val_acc', val_acc)

        del trial_model
        torch.cuda.empty_cache()
        return val_acc

    print(f'Running Optuna study — {cfg.OPTUNA_TRIALS} trials × {cfg.OPTUNA_EPOCHS} epochs each...')
    with mlflow.start_run(run_name='optuna_study') as run:
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=SEED),
            study_name=cfg.MLFLOW_EXPERIMENT,
        )
        study.optimize(
            lambda t: objective(t, parent_run_id=run.info.run_id),
            n_trials=cfg.OPTUNA_TRIALS,
            show_progress_bar=True,
        )

    best = study.best_params
    print('\n✅ Best trial:')
    for k, v in best.items():
        print(f'  {k:<15}: {v:.5g}')
    print(f'  val_acc        : {study.best_value:.4f}')

    # Patch cfg with best values
    cfg.LR           = best['lr']
    cfg.HEAD_LR      = best['head_lr']
    cfg.DROPOUT_RATE = best['dropout']
    cfg.WEIGHT_DECAY = best['weight_decay']
    print('\ncfg updated with best hyperparameters ✅')
    return best


# ── Main training loop ───────────────────────────────────────────────────────

def train(model, train_loader, val_loader, best_gamma: float):
    """
    Full training loop with two-phase fine-tuning, early stopping,
    and MLflow metric logging.

    Returns (model, history, best_acc).
    """
    criterion = FocalLoss(gamma=best_gamma, label_smoothing=0.1)
    optimizer = build_optimizer(model)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.T_MAX, eta_min=1e-6)

    history   = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc  = 0.0
    best_wts  = None
    es_counter  = 0
    es_best_acc = 0.0

    print('=' * 70)
    print('                         TRAINING')
    print('=' * 70)

    with mlflow.start_run(run_name='final_training') as run:
        mlflow.log_params({
            'model':         'efficientnet_b3',
            'num_epochs':    cfg.NUM_EPOCHS,
            'batch_size':    cfg.BATCH_SIZE,
            'lr':            cfg.LR,
            'head_lr':       cfg.HEAD_LR,
            'weight_decay':  cfg.WEIGHT_DECAY,
            'dropout':       cfg.DROPOUT_RATE,
            'gamma':         best_gamma,
            'patience':      cfg.PATIENCE,
            'image_size':    cfg.IMAGE_SIZE,
            'freeze_epochs': cfg.FREEZE_EPOCHS,
            'unfreeze_last': cfg.UNFREEZE_LAST_LAYERS,
        })

        for epoch in range(cfg.NUM_EPOCHS):
            t0 = time.time()

            if epoch == cfg.FREEZE_EPOCHS:
                model.unfreeze_last_n_blocks(cfg.UNFREEZE_LAST_LAYERS)
                optimizer = build_optimizer(model)
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=cfg.T_MAX - cfg.FREEZE_EPOCHS,
                    eta_min=1e-6,
                )
                print(f'[Epoch {epoch}] Fine-tuning phase activated.')

            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            vl_loss, vl_acc, _, _, _ = evaluate(model, val_loader, criterion)
            scheduler.step()

            for k, v in zip(history, [tr_loss, tr_acc, vl_loss, vl_acc]):
                history[k].append(v)

            mlflow.log_metrics({
                'train_loss': tr_loss,
                'train_acc':  tr_acc,
                'val_loss':   vl_loss,
                'val_acc':    vl_acc,
            }, step=epoch)

            if vl_acc > best_acc:
                best_acc = vl_acc
                best_wts = copy.deepcopy(model.state_dict())
                torch.save(best_wts, cfg.OUTPUT_DIR / 'best_model.pt')
                mlflow.log_metric('best_val_acc', best_acc, step=epoch)

            # Early stopping
            if vl_acc > es_best_acc:
                es_best_acc = vl_acc
                es_counter  = 0
            else:
                es_counter += 1

            lrs    = scheduler.get_last_lr()
            lr_str = f'LR head={lrs[0]:.1e}' + (f' bb={lrs[1]:.1e}' if len(lrs) > 1 else '')

            print(
                f'Ep [{epoch+1:02d}/{cfg.NUM_EPOCHS}]  '
                f'TrLoss {tr_loss:.4f}  TrAcc {tr_acc:.4f}  '
                f'VlLoss {vl_loss:.4f}  VlAcc {vl_acc:.4f}  '
                f'{lr_str}  ES {es_counter}/{cfg.PATIENCE}  '
                f'[{time.time()-t0:.0f}s]'
            )

            if es_counter >= cfg.PATIENCE:
                print(f'\n⏹  Early stopping triggered — no improvement for {cfg.PATIENCE} epochs.')
                mlflow.log_param('stopped_at_epoch', epoch + 1)
                break

        mlflow.log_metric('final_best_val_acc', best_acc)
        mlflow.pytorch.log_model(model, artifact_path='model')
        mlflow.log_artifact(str(cfg.OUTPUT_DIR / 'best_model.pt'))
        print(f'  Run ID: {run.info.run_id}')

    print(f'\n🏆 Best Validation Accuracy: {best_acc:.4f}')
    model.load_state_dict(best_wts)
    return model, history, best_acc
