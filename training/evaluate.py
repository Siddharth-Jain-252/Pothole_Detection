"""
=============================================================
  Pothole Detection — Evaluation Pipeline
  mAP, Precision, Recall, F1, PR Curves, Confusion Matrix
=============================================================

Evaluation Metrics (Interview Must-Know):
──────────────────────────────────────────
  • Precision: TP / (TP + FP) — "Of all detections, how many are correct?"
  • Recall:    TP / (TP + FN) — "Of all real objects, how many did we find?"
  • F1 Score:  2 × (P × R) / (P + R) — harmonic mean
  • AP:        Area under Precision-Recall curve
  • mAP@0.5:  Mean AP at IoU threshold 0.5 (PASCAL VOC standard)
"""

import os
import sys
import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')       # Non-interactive backend
import matplotlib.pyplot as plt
from collections import defaultdict

from config import (
    DEVICE, TEST_IMAGES, METRICS_OUTPUT_DIR,
    CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD,
    IMAGE_SIZE, CLASS_NAMES,
)
from model import load_model
from dataset import PotholeDataset
from inference import preprocess_image, decode_predictions
from utils import compute_iou, non_max_suppression, xywh_to_xyxy
import torchvision.transforms as transforms


def evaluate_model(model, test_folder=TEST_IMAGES, iou_threshold=0.5,
                   conf_thresh=CONFIDENCE_THRESHOLD, max_images=None):
    """
    Evaluate model on test set and compute detection metrics.

    Algorithm:
    1. For each image: get predictions and ground truth
    2. Match predictions to ground truth using IoU
    3. Classify each prediction as TP or FP
    4. Compute Precision, Recall, F1, and AP

    Args:
        model: YOLOv3Tiny model in eval mode
        test_folder: Path to test images
        iou_threshold: IoU threshold for matching (default 0.5)
        conf_thresh: Confidence threshold
        max_images: Limit number of images (None = all)

    Returns:
        dict with metrics: precision, recall, f1, ap, all_tp, all_fp, etc.
    """
    dataset = PotholeDataset(test_folder, transform=transforms.ToTensor())

    all_predictions = []   # [(confidence, is_tp)]
    total_gt = 0
    total_detected = 0
    total_images = 0

    n_images = len(dataset) if max_images is None else min(max_images, len(dataset))

    print(f"\n📊 Evaluating on {n_images} test images...")
    print(f"   IoU threshold: {iou_threshold}")
    print(f"   Confidence threshold: {conf_thresh}")
    print("=" * 50)

    for idx in range(n_images):
        img_path = dataset.get_image_path(idx)
        image_tensor, original = preprocess_image(img_path)
        h, w = original.shape[:2]

        # Get ground truth
        _, gt_boxes = dataset[idx]
        gt_used = [False] * len(gt_boxes)

        # Convert ground truth to pixel coordinates
        gt_pixel = []
        for box in gt_boxes:
            cls, cx, cy, bw, bh = box.tolist()
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            gt_pixel.append([x1, y1, x2, y2, int(cls)])
        total_gt += len(gt_pixel)

        # Get predictions
        with torch.no_grad():
            output = model(image_tensor)

        raw_dets = decode_predictions(output, (h, w), conf_thresh)
        detections = non_max_suppression(raw_dets, conf_thresh, NMS_IOU_THRESHOLD)
        total_detected += len(detections)

        # Match predictions to ground truth
        detections_sorted = sorted(detections, key=lambda x: -x[4])

        for det in detections_sorted:
            det_box = det[:4]
            det_conf = det[4]

            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gt_pixel):
                if gt_used[gt_idx]:
                    continue
                iou = compute_iou(det_box, gt[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                all_predictions.append((det_conf, True))   # True Positive
                gt_used[best_gt_idx] = True
            else:
                all_predictions.append((det_conf, False))  # False Positive

        total_images += 1
        if (total_images) % 50 == 0:
            print(f"  [{total_images}/{n_images}] images evaluated...")

    # Sort by confidence (descending)
    all_predictions.sort(key=lambda x: -x[0])

    # Compute precision-recall curve
    tp_cumsum = 0
    fp_cumsum = 0
    precisions = []
    recalls = []

    for conf, is_tp in all_predictions:
        if is_tp:
            tp_cumsum += 1
        else:
            fp_cumsum += 1

        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / total_gt if total_gt > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    # Compute AP (Area under PR curve) using 11-point interpolation
    ap = compute_ap(precisions, recalls)

    # Final metrics
    final_precision = tp_cumsum / (tp_cumsum + fp_cumsum) if (tp_cumsum + fp_cumsum) > 0 else 0
    final_recall = tp_cumsum / total_gt if total_gt > 0 else 0
    f1 = 2 * final_precision * final_recall / (final_precision + final_recall + 1e-6)

    metrics = {
        "precision": final_precision,
        "recall": final_recall,
        "f1": f1,
        "ap": ap,
        "mAP@0.5": ap,  # Single class, so AP = mAP
        "total_gt": total_gt,
        "total_detections": total_detected,
        "true_positives": tp_cumsum,
        "false_positives": fp_cumsum,
        "false_negatives": total_gt - tp_cumsum,
        "images_evaluated": total_images,
        "precisions": precisions,
        "recalls": recalls,
    }

    return metrics


def compute_ap(precisions, recalls):
    """
    Compute Average Precision using 11-point interpolation.

    This is the PASCAL VOC standard method:
    AP = (1/11) × Σ max(precision) at recall ∈ {0, 0.1, 0.2, ..., 1.0}

    Interview: "How is mAP calculated?"
    For each class, compute AP from the PR curve.
    mAP = mean of all per-class APs.
    """
    if len(precisions) == 0:
        return 0.0

    recall_points = np.linspace(0, 1, 11)
    ap = 0.0

    for r in recall_points:
        # Find max precision at recall >= r
        precs_at_recall = [p for p, rec in zip(precisions, recalls) if rec >= r]
        if precs_at_recall:
            ap += max(precs_at_recall)

    ap /= 11.0
    return ap


def plot_pr_curve(precisions, recalls, ap, save_path=None):
    """
    Plot and save the Precision-Recall curve.

    The PR curve is a key visualization for evaluating detectors:
    - A curve close to (1,1) = perfect detector
    - Area under curve = AP
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Background styling
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    ax.plot(recalls, precisions, color='#00d2ff', linewidth=2.5, label=f'Pothole AP={ap:.3f}')
    ax.fill_between(recalls, precisions, alpha=0.2, color='#00d2ff')

    ax.set_xlabel('Recall', fontsize=14, color='white', fontweight='bold')
    ax.set_ylabel('Precision', fontsize=14, color='white', fontweight='bold')
    ax.set_title(f'Precision-Recall Curve (mAP@0.5 = {ap:.3f})',
                 fontsize=16, color='white', fontweight='bold', pad=15)

    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.2, color='white')
    ax.legend(fontsize=12, loc='lower left',
              facecolor='#0f3460', edgecolor='white', labelcolor='white')
    ax.tick_params(colors='white')

    for spine in ax.spines.values():
        spine.set_color('#ffffff40')

    plt.tight_layout()

    save_path = save_path or os.path.join(METRICS_OUTPUT_DIR, 'pr_curve.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  📈 PR curve saved to: {save_path}")


def plot_metrics_summary(metrics, save_path=None):
    """
    Create a visual summary of evaluation metrics.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#1a1a2e')

    metric_names = ['Precision', 'Recall', 'F1 Score']
    metric_values = [metrics['precision'], metrics['recall'], metrics['f1']]
    colors = ['#00d2ff', '#ff6b6b', '#51cf66']

    for ax, name, value, color in zip(axes, metric_names, metric_values, colors):
        ax.set_facecolor('#16213e')

        # Circular gauge-style bar
        ax.barh([0], [value], height=0.5, color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
        ax.barh([0], [1.0], height=0.5, color='white', alpha=0.1)

        ax.set_xlim([0, 1.1])
        ax.set_ylim([-0.5, 0.5])
        ax.set_title(f'{name}: {value:.3f}', fontsize=14,
                     color='white', fontweight='bold', pad=10)
        ax.set_yticks([])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(colors='white')

        for spine in ax.spines.values():
            spine.set_color('#ffffff40')

    plt.suptitle(f'Pothole Detection — Evaluation Results (mAP@0.5 = {metrics["mAP@0.5"]:.3f})',
                 fontsize=16, color='white', fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = save_path or os.path.join(METRICS_OUTPUT_DIR, 'metrics_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  📊 Metrics summary saved to: {save_path}")


def print_report(metrics):
    """Print a formatted evaluation report."""
    print("\n" + "=" * 60)
    print("  📋 EVALUATION REPORT — Pothole Detection")
    print("=" * 60)
    print(f"  Images evaluated:    {metrics['images_evaluated']}")
    print(f"  Ground truth boxes:  {metrics['total_gt']}")
    print(f"  Total detections:    {metrics['total_detections']}")
    print("-" * 60)
    print(f"  True Positives:      {metrics['true_positives']}")
    print(f"  False Positives:     {metrics['false_positives']}")
    print(f"  False Negatives:     {metrics['false_negatives']}")
    print("-" * 60)
    print(f"  Precision:           {metrics['precision']:.4f}")
    print(f"  Recall:              {metrics['recall']:.4f}")
    print(f"  F1 Score:            {metrics['f1']:.4f}")
    print(f"  mAP@0.5:            {metrics['mAP@0.5']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    """
    Run full evaluation pipeline.

    Usage:
        python evaluate.py           # Evaluate on all test images
        python evaluate.py --quick   # Quick eval on 50 images
    """
    print("\n" + "=" * 60)
    print("  📊 Pothole Detection — Evaluation Pipeline")
    print("=" * 60)

    model = load_model()

    max_imgs = 50 if '--quick' in sys.argv else None

    metrics = evaluate_model(model, max_images=max_imgs)
    print_report(metrics)

    # Generate plots
    if len(metrics['precisions']) > 0:
        plot_pr_curve(metrics['precisions'], metrics['recalls'], metrics['ap'])
        plot_metrics_summary(metrics)

    print("\n✅ Evaluation complete!")
