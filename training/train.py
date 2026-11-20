"""
=============================================================
  Pothole Detection — Training Script
  Complete training loop with logging, checkpointing, and LR scheduling
=============================================================

NOTE: Pre-trained weights (yolo_rdd_1.pth) already exist.
      This script is provided for REFERENCE and for retraining
      if needed. You do NOT need to run this to use the project.

Training Pipeline:
──────────────────
  1. Load dataset with augmentation
  2. Initialize model (or load from checkpoint)
  3. Define loss function (YOLO loss: class + bbox + confidence)
  4. Train with Adam optimizer + learning rate scheduling
  5. Log metrics per epoch
  6. Save best model checkpoint
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

from config import (
    DEVICE, NUM_CLASSES, NUM_ANCHORS,
    LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, WEIGHT_DECAY,
    RESULTS_DIR, DEFAULT_WEIGHTS,
)
from model import YOLOv3Tiny, model_summary
from dataset import get_train_loader, get_test_loader


# ─── YOLO Loss Function ──────────────────────────────────

def yolo_loss(predictions, targets, num_classes=NUM_CLASSES):
    """
    Compute YOLO detection loss.

    YOLO Loss = λ_coord × Localization Loss
              + Confidence Loss
              + λ_class × Classification Loss

    Interview: "Explain the YOLO loss function."
    The loss has 3 components:
    1. Bounding box regression (MSE on x, y, w, h)
    2. Objectness/confidence (MSE: 1 if object, 0 if not)
    3. Classification (Cross-entropy for class probabilities)

    Args:
        predictions: Model output [B, A, G, G, 5+C]
        targets: List of label tensors per image

    Returns:
        Total loss (scalar)
    """
    total_class_loss = 0
    total_bbox_loss = 0
    total_confidence_loss = 0
    num_images = len(predictions)

    for i in range(num_images):
        # Reshape predictions per grid cell
        pred = predictions[i].reshape(-1, num_classes + 5)
        pred_classes = pred[:, :num_classes]
        pred_confidence = pred[:, num_classes:num_classes + 1]
        pred_bboxes = pred[:, num_classes + 1:]

        # Targets for this image
        target = targets[i]
        num_target_boxes = target.size(0)
        if num_target_boxes == 0:
            continue

        target_classes = target[:, 0].long()
        target_confidence = torch.ones_like(pred_confidence[:num_target_boxes])
        target_bboxes = target[:, 1:]

        # Compute each component of the loss
        class_loss = F.cross_entropy(pred_classes[:num_target_boxes], target_classes)
        confidence_loss = F.mse_loss(pred_confidence[:num_target_boxes], target_confidence)
        bbox_loss = F.mse_loss(pred_bboxes[:num_target_boxes], target_bboxes)

        total_class_loss += class_loss
        total_confidence_loss += confidence_loss
        total_bbox_loss += bbox_loss

    avg_class_loss = total_class_loss / num_images
    avg_confidence_loss = total_confidence_loss / num_images
    avg_bbox_loss = total_bbox_loss / num_images

    return avg_class_loss + avg_confidence_loss + avg_bbox_loss


# ─── Accuracy Calculation ─────────────────────────────────

def calculate_accuracy(predictions, targets, num_classes=NUM_CLASSES):
    """
    Calculate classification accuracy for detected objects.

    This is a simplified metric — for true detection accuracy,
    use mAP from evaluate.py instead.
    """
    total_correct = 0
    total_boxes = 0

    for i in range(len(predictions)):
        pred = predictions[i].reshape(-1, num_classes + 5)
        pred_classes = pred[:, :num_classes].argmax(dim=1)

        target = targets[i]
        num_target_boxes = target.size(0)
        if num_target_boxes == 0:
            continue

        target_classes = target[:, 0].long()
        total_boxes += num_target_boxes
        total_correct += (pred_classes[:num_target_boxes] == target_classes).sum().item()

    accuracy = total_correct / total_boxes if total_boxes > 0 else 0
    return accuracy


# ─── Training Loop ────────────────────────────────────────

def train(num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
          learning_rate=LEARNING_RATE, resume_from=None):
    """
    Full training loop with logging and checkpointing.

    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        resume_from: Path to checkpoint to resume from (optional)
    """
    print("\n" + "=" * 60)
    print("  🚀 Pothole Detection — Training")
    print("=" * 60)
    print(f"  Device:        {DEVICE}")
    print(f"  Epochs:        {num_epochs}")
    print(f"  Batch size:    {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay:  {WEIGHT_DECAY}")
    print("=" * 60)

    # Model
    model = YOLOv3Tiny(num_classes=NUM_CLASSES, num_anchors=NUM_ANCHORS).to(DEVICE)

    if resume_from and os.path.exists(resume_from):
        state_dict = torch.load(resume_from, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"  📥 Resumed from: {resume_from}")

    model_summary(model)

    # Optimizer & Scheduler
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # DataLoader
    train_loader = get_train_loader(batch_size=batch_size, augment=True)

    # Training history
    history = {
        'loss': [], 'accuracy': [], 'lr': []
    }

    best_loss = float('inf')
    save_dir = os.path.join(RESULTS_DIR, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0

        for batch_idx, (images, boxes) in enumerate(train_loader):
            images = images.to(DEVICE)
            boxes = [box.to(DEVICE) for box in boxes]

            # Forward pass
            outputs = model(images)
            loss = yolo_loss(outputs, boxes, NUM_CLASSES)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            epoch_loss += loss.item()
            accuracy = calculate_accuracy(outputs, boxes, NUM_CLASSES)
            epoch_accuracy += accuracy

            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch [{epoch+1}/{num_epochs}] "
                      f"Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} Acc: {accuracy:.4f}")

        # Epoch summary
        avg_loss = epoch_loss / len(train_loader)
        avg_accuracy = epoch_accuracy / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']

        history['loss'].append(avg_loss)
        history['accuracy'].append(avg_accuracy)
        history['lr'].append(current_lr)

        print(f"\n  ✅ Epoch [{epoch+1}/{num_epochs}] "
              f"Loss: {avg_loss:.4f} Acc: {avg_accuracy:.4f} "
              f"LR: {current_lr:.6f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(save_dir, "best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  💾 Best model saved (loss: {best_loss:.4f})")

        # Update learning rate
        scheduler.step()

    # Save final model
    final_path = os.path.join(save_dir, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\n  💾 Final model saved to: {final_path}")

    # Plot training curves
    plot_training_history(history, save_dir)

    return model, history


def plot_training_history(history, save_dir):
    """Plot and save training loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#1a1a2e')

    epochs = range(1, len(history['loss']) + 1)

    # Loss curve
    ax1.set_facecolor('#16213e')
    ax1.plot(epochs, history['loss'], color='#ff6b6b', linewidth=2, marker='o', markersize=4)
    ax1.set_title('Training Loss', fontsize=14, color='white', fontweight='bold')
    ax1.set_xlabel('Epoch', color='white')
    ax1.set_ylabel('Loss', color='white')
    ax1.tick_params(colors='white')
    ax1.grid(True, alpha=0.2, color='white')
    for spine in ax1.spines.values():
        spine.set_color('#ffffff40')

    # Accuracy curve
    ax2.set_facecolor('#16213e')
    ax2.plot(epochs, history['accuracy'], color='#51cf66', linewidth=2, marker='o', markersize=4)
    ax2.set_title('Training Accuracy', fontsize=14, color='white', fontweight='bold')
    ax2.set_xlabel('Epoch', color='white')
    ax2.set_ylabel('Accuracy', color='white')
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.2, color='white')
    for spine in ax2.spines.values():
        spine.set_color('#ffffff40')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  📈 Training curves saved to: {save_path}")


if __name__ == "__main__":
    """
    Usage:
        python train.py                  # Train from scratch
        python train.py --resume         # Resume from existing weights
        python train.py --epochs 5       # Quick training run
    """
    import argparse
    parser = argparse.ArgumentParser(description='Train pothole detector')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing weights')
    args = parser.parse_args()

    resume_path = DEFAULT_WEIGHTS if args.resume else None
    train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        resume_from=resume_path,
    )
