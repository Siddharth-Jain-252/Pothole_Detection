"""
=============================================================
  Pothole Detection — YOLOv3-Tiny Model Architecture
  Custom lightweight YOLO implementation for pothole detection
=============================================================

Architecture Overview:
──────────────────────
  Input (3×416×416)
    │
    ├── Backbone (7 Conv-BN-LeakyReLU blocks + MaxPool)
    │     Conv2d(3→16) → Conv2d(16→32) → Conv2d(32→64)
    │     Conv2d(64→128) → Conv2d(128→256) → Conv2d(256→512)
    │     Conv2d(512→1024)
    │
    └── Detection Head
          Conv2d(1024 → num_anchors × (5 + num_classes))
          Reshape → [B, A, G, G, (5+C)]

Key Concepts (Interview):
─────────────────────────
  • Single-shot detector — predicts boxes & classes in one pass
  • Grid-based detection — image divided into G×G cells
  • Anchor boxes — predefined box shapes for each cell
  • Prediction per anchor: [tx, ty, tw, th, objectness, class_probs]
"""

import torch
import torch.nn as nn
from config import DEVICE, NUM_CLASSES, NUM_ANCHORS, DEFAULT_WEIGHTS


class YOLOv3Tiny(nn.Module):
    """
    YOLOv3-Tiny: A lightweight single-shot object detector.

    The backbone uses 7 convolutional blocks with batch normalization
    and LeakyReLU activation, followed by max-pooling for downsampling.
    The detection head produces per-anchor predictions on the final
    feature map grid.

    Args:
        num_classes (int): Number of object classes (default: 10).
        num_anchors (int): Number of anchor boxes per grid cell (default: 3).
    """

    def __init__(self, num_classes=NUM_CLASSES, num_anchors=NUM_ANCHORS):
        super(YOLOv3Tiny, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # ── Backbone (Feature Extractor) ──────────────────
        # Progressive channel expansion: 3 → 16 → 32 → 64 → 128 → 256 → 512 → 1024
        # Each block: Conv2d → BatchNorm → LeakyReLU(0.1) → MaxPool (except last)
        self.backbone = nn.Sequential(
            # Block 1: 3 → 16, spatial: 416 → 208
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            # Block 2: 16 → 32, spatial: 208 → 104
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            # Block 3: 32 → 64, spatial: 104 → 52
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            # Block 4: 64 → 128, spatial: 52 → 26
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            # Block 5: 128 → 256, spatial: 26 → 13
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            # Block 6: 256 → 512, spatial: 13 → 6
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            # Block 7: 512 → 1024, spatial stays 6×6 (no pool)
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        )

        # ── Detection Head ────────────────────────────────
        # 1×1 convolution to produce predictions
        # Output channels: num_anchors × (5 + num_classes)
        #   5 = [tx, ty, tw, th, objectness]
        #   num_classes = class probabilities
        self.detector = nn.Conv2d(
            1024, self.num_anchors * (5 + num_classes), kernel_size=1
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, 3, H, W]

        Returns:
            Tensor of shape [B, num_anchors, grid_h, grid_w, (5 + num_classes)]
        """
        x = self.backbone(x)
        x = self.detector(x)

        # Reshape: [B, A*(5+C), G, G] → [B, A, (5+C), G, G] → [B, A, G, G, (5+C)]
        batch_size = x.size(0)
        grid_h, grid_w = x.size(2), x.size(3)
        x = x.view(batch_size, self.num_anchors, 5 + self.num_classes, grid_h, grid_w)
        x = x.permute(0, 1, 3, 4, 2)  # [B, A, G, G, 5+C]

        return x


def load_model(weights_path=DEFAULT_WEIGHTS, device=DEVICE):
    """
    Load YOLOv3-Tiny model with pre-trained weights.

    Args:
        weights_path (str): Path to .pth weights file.
        device: torch.device to load model onto.

    Returns:
        model (YOLOv3Tiny): Model in eval mode with loaded weights.
    """
    model = YOLOv3Tiny(num_classes=NUM_CLASSES, num_anchors=NUM_ANCHORS)

    # Load state dict
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    result = model.load_state_dict(state_dict)
    print(f"✅ Model loaded from: {weights_path}")
    print(f"   Keys matched: {result}")

    model = model.to(device)
    model.eval()
    return model


def model_summary(model):
    """
    Print a summary of the model architecture.

    Shows total parameters, trainable parameters, and layer details.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 60)
    print("  YOLOv3-Tiny Model Summary")
    print("=" * 60)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size:           {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    print(f"  Num classes:          {model.num_classes}")
    print(f"  Num anchors:          {model.num_anchors}")
    print("=" * 60)

    print("\n  Layer Details:")
    print("  " + "-" * 56)
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d)):
            params = sum(p.numel() for p in module.parameters())
            print(f"  {name:30s} | {str(module.__class__.__name__):15s} | {params:>8,} params")
    print("  " + "-" * 56)
    print()


if __name__ == "__main__":
    """Quick test: load model and print summary."""
    model = load_model()
    model_summary(model)

    # Test forward pass
    dummy_input = torch.randn(1, 3, 416, 416).to(DEVICE)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"  Input shape:  {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Grid size:    {output.shape[2]}×{output.shape[3]}")
    print(f"  Predictions per cell: {output.shape[-1]} "
          f"(5 box params + {model.num_classes} classes)")
