import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    auc, classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
)
from sklearn.preprocessing import label_binarize

from config import DEVICE, cfg
from dataset import get_transforms, skull_strip
from train import AverageMeter, evaluate


# ── Test-set metrics ─────────────────────────────────────────────────────────

def run_test_evaluation(model, test_loader, class_names):
    """Print classification report + macro AUC on the held-out test set."""
    criterion_eval = nn.CrossEntropyLoss()
    vl_loss, vl_acc, preds, labels, probs = evaluate(model, test_loader, criterion_eval)

    print('=' * 65)
    print('                    TEST SET RESULTS')
    print('=' * 65)
    print(f'Loss     : {vl_loss:.4f}')
    print(f'Accuracy : {vl_acc:.4f}  ({vl_acc*100:.2f}%)')
    print('\nPer-class Report:')
    print(classification_report(labels, preds, target_names=class_names, digits=4))

    y_bin      = label_binarize(labels, classes=list(range(cfg.NUM_CLASSES)))
    macro_auc  = roc_auc_score(y_bin, probs, multi_class='ovr', average='macro')
    print(f'Macro-average AUC : {macro_auc:.4f}')
    return preds, labels, probs, y_bin


# ── Training curves ──────────────────────────────────────────────────────────

def plot_training_curves(history: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    for ax, metric in zip(axes, ['loss', 'acc']):
        ax.plot(epochs, history[f'train_{metric}'], label='Train',      linewidth=2, color='#4361ee')
        ax.plot(epochs, history[f'val_{metric}'],   label='Validation', linewidth=2, color='#f72585')
        ax.axvline(cfg.FREEZE_EPOCHS, linestyle='--', color='gray', alpha=0.6, label='Fine-tune start')
        ax.set_title('Loss' if metric == 'loss' else 'Accuracy', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Training Curves', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(cfg.OUTPUT_DIR / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()


# ── Confusion matrix ─────────────────────────────────────────────────────────

def plot_confusion_matrix(labels, preds, class_names) -> None:
    cm     = confusion_matrix(labels, preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=axes[0])
    axes[0].set_title('Confusion Matrix (counts)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=axes[1], vmin=0, vmax=100)
    axes[1].set_title('Confusion Matrix (%)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(cfg.OUTPUT_DIR / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()


# ── ROC curves ───────────────────────────────────────────────────────────────

def plot_roc_curves(y_bin, probs, class_names) -> None:
    colors = ['#e63946', '#2a9d8f', '#e9c46a', '#264653']
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2.5,
                label=f'{name}  (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_title('ROC Curves — One-vs-Rest', fontsize=14, fontweight='bold')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(cfg.OUTPUT_DIR / 'roc_curves.png', dpi=150, bbox_inches='tight')
    plt.show()


# ── Grad-CAM ─────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Hooks into EfficientNet-B3's last MBConv stage (features[-2]).
    Uses tensor-level gradient hooks to capture d(loss)/d(activation) correctly.
    """

    def __init__(self, model):
        self.model        = model
        self.gradients    = None
        self.activations  = None
        self._hook_handle = None

        target = list(model.features.children())[-2]  # MBConv stage 7
        target.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
        if self._hook_handle is not None:
            self._hook_handle.remove()
        self._hook_handle = output.register_hook(self._gradient_hook)

    def _gradient_hook(self, grad):
        self.gradients = grad.detach()

    def generate(self, img_tensor, class_idx=None):
        self.model.eval()
        t = img_tensor.unsqueeze(0).to(DEVICE)
        with torch.enable_grad():
            logits = self.model(t)
            if class_idx is None:
                class_idx = logits.argmax(1).item()
            self.model.zero_grad()
            logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = torch.relu((weights * self.activations).sum(1)).squeeze()
        cam    -= cam.min()
        cam    /= cam.max() + 1e-8
        return cam.cpu().numpy(), class_idx


def plot_gradcam(model, test_loader, class_names, n: int = 4) -> None:
    gcam     = GradCAM(model)
    inv_mean = np.array([0.485, 0.456, 0.406])
    inv_std  = np.array([0.229, 0.224, 0.225])

    imgs_b, lbls_b = next(iter(test_loader))
    imgs_b, lbls_b = imgs_b[:n], lbls_b[:n]

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))

    for i in range(n):
        img_t    = imgs_b[i]
        true_lbl = class_names[lbls_b[i].item()]
        cam, p_idx = gcam.generate(img_t)
        pred_lbl = class_names[p_idx]

        img_np = (img_t.permute(1, 2, 0).numpy() * inv_std + inv_mean).clip(0, 1)
        cam_r  = np.array(
            Image.fromarray((cam * 255).astype(np.uint8))
                 .resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), Image.BILINEAR)
        ) / 255.0

        axes[0, i].imshow(img_np)
        axes[0, i].set_title(f'True: {true_lbl}', fontsize=11)
        axes[0, i].axis('off')

        axes[1, i].imshow(img_np)
        axes[1, i].imshow(cam_r, cmap='jet', alpha=0.45)
        axes[1, i].set_title(f'Pred: {pred_lbl}', fontsize=11,
                              color='green' if pred_lbl == true_lbl else 'red')
        axes[1, i].axis('off')

    plt.suptitle('Grad-CAM Heatmaps', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(cfg.OUTPUT_DIR / 'gradcam.png', dpi=150, bbox_inches='tight')
    plt.show()


# ── Single-image inference ───────────────────────────────────────────────────

def predict_single_image(model, image_path, class_names):
    """
    Run inference on one MRI image.
    Applies skull_strip first to match the training pipeline exactly.
    """
    _, eval_tf = get_transforms()
    raw_img  = Image.open(image_path).convert('RGB')
    stripped = skull_strip(raw_img)
    tensor   = eval_tf(stripped).unsqueeze(0).to(DEVICE)

    model.eval()
    probs = torch.softmax(model(tensor), 1).squeeze().detach().cpu().numpy()
    pred  = class_names[np.argmax(probs)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(raw_img)
    axes[0].axis('off')
    axes[0].set_title(f'Prediction: {pred.upper()}', fontsize=13,
                      fontweight='bold', color='#2a9d8f')

    colors = ['#e63946' if n == pred else '#adb5bd' for n in class_names]
    bars   = axes[1].barh(class_names, probs * 100, color=colors, height=0.5)
    axes[1].set_xlabel('Confidence (%)')
    axes[1].set_title('Class Probabilities', fontsize=12)
    axes[1].set_xlim(0, 100)
    for bar, p in zip(bars, probs):
        axes[1].text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                     f'{p*100:.1f}%', va='center', fontsize=10)

    plt.tight_layout()
    plt.show()
    return pred, probs
