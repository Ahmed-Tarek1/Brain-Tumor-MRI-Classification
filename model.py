import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3

from config import cfg


class BrainTumorEfficientNetB3(nn.Module):
    """
    EfficientNet-B3 backbone + custom 2-layer classification head.

    Head architecture:
        features (MBConv stages) → AdaptiveAvgPool → Flatten
        → Dropout → FC(1536→512) → BatchNorm → ReLU → Dropout → FC(512→4)

    EfficientNet-B3 features layout:
      Index  Layer
      ─────  ─────────────────────────────
        0    stem conv
        1-7  MBConv block stages 1-7
        8    head conv
    With n=2: unfreezes stage 7 + head conv (last 2).
    """

    def __init__(self, num_classes: int = 4, dropout: float = 0.4,
                 freeze_backbone: bool = True):
        super().__init__()
        backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)

        self.features = backbone.features
        self.avgpool  = backbone.avgpool   # AdaptiveAvgPool2d(1)

        # EfficientNet-B3 outputs 1536 feature channels
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1536, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

        # Kaiming init for the head
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int) -> None:
        """Unfreeze the last n stages of backbone features (from the end).
        With n=2: unfreezes MBConv stage 7 + head conv.
        """
        for layer in list(self.features.children())[-n:]:
            for p in layer.parameters():
                p.requires_grad = True
        print(f'  Unfroze last {n} backbone block(s) ✅')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)       # [B, 1536, H, W]
        x = self.avgpool(x)        # [B, 1536, 1, 1]
        x = torch.flatten(x, 1)   # [B, 1536]
        return self.classifier(x)  # [B, num_classes]
