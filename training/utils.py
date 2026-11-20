"""
=============================================================
  Pothole Detection — Utility Functions
  IoU, NMS, box drawing, coordinate conversions, severity
=============================================================

Interview-Critical Functions:
─────────────────────────────
  • compute_iou()          → Core metric for object detection
  • non_max_suppression()  → Removes duplicate detections
  • xywh_to_xyxy()         → YOLO ↔ standard coordinate conversion
  • classify_severity()    → Real-world application logic
"""

import numpy as np
import cv2
import torch
from config import (
    CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD,
    SEVERITY_THRESHOLDS, SEVERITY_COLORS,
    BOX_THICKNESS, FONT_SCALE, CLASS_NAMES,
)


# ─── Coordinate Conversions ──────────────────────────────

def xywh_to_xyxy(boxes):
    """
    Convert YOLO format (center_x, center_y, width, height) to
    corner format (x1, y1, x2, y2).

    This is one of the most common interview questions:
    "How does YOLO encode bounding boxes?"

    Args:
        boxes: numpy array or tensor of shape [..., 4]

    Returns:
        Converted boxes in [x1, y1, x2, y2] format
    """
    if isinstance(boxes, torch.Tensor):
        converted = boxes.clone()
    else:
        converted = boxes.copy()

    converted[..., 0] = boxes[..., 0] - boxes[..., 2] / 2  # x1 = cx - w/2
    converted[..., 1] = boxes[..., 1] - boxes[..., 3] / 2  # y1 = cy - h/2
    converted[..., 2] = boxes[..., 0] + boxes[..., 2] / 2  # x2 = cx + w/2
    converted[..., 3] = boxes[..., 1] + boxes[..., 3] / 2  # y2 = cy + h/2
    return converted


def xyxy_to_xywh(boxes):
    """
    Convert corner format (x1, y1, x2, y2) to
    YOLO format (center_x, center_y, width, height).

    Args:
        boxes: numpy array or tensor of shape [..., 4]

    Returns:
        Converted boxes in [cx, cy, w, h] format
    """
    if isinstance(boxes, torch.Tensor):
        converted = boxes.clone()
    else:
        converted = boxes.copy()

    converted[..., 0] = (boxes[..., 0] + boxes[..., 2]) / 2  # cx = (x1+x2)/2
    converted[..., 1] = (boxes[..., 1] + boxes[..., 3]) / 2  # cy = (y1+y2)/2
    converted[..., 2] = boxes[..., 2] - boxes[..., 0]          # w = x2 - x1
    converted[..., 3] = boxes[..., 3] - boxes[..., 1]          # h = y2 - y1
    return converted


# ─── Intersection over Union (IoU) ───────────────────────

def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    IoU = Area of Intersection / Area of Union

    This is THE most important metric in object detection and a
    guaranteed interview question.

    Args:
        box1: [x1, y1, x2, y2] format (can be numpy or tensor)
        box2: [x1, y1, x2, y2] format

    Returns:
        float: IoU value in [0, 1]

    Interview Tip:
        IoU > 0.5 → "correct detection" (PASCAL VOC standard)
        IoU > 0.75 → "strict match" (COCO standard)
    """
    # Intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Intersection area (clamp to 0 if no overlap)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def compute_iou_batch(boxes1, boxes2):
    """
    Compute pairwise IoU between two sets of boxes.

    Args:
        boxes1: [N, 4] tensor in xyxy format
        boxes2: [M, 4] tensor in xyxy format

    Returns:
        [N, M] tensor of IoU values
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Intersection
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    union = area1[:, None] + area2[None, :] - inter_area
    return inter_area / (union + 1e-6)


# ─── Non-Maximum Suppression (NMS) ───────────────────────

def non_max_suppression(detections, conf_thresh=CONFIDENCE_THRESHOLD,
                         iou_thresh=NMS_IOU_THRESHOLD):
    """
    Apply Non-Maximum Suppression to filter overlapping detections.

    NMS Algorithm:
    1. Sort detections by confidence (descending)
    2. Pick the highest-confidence box → add to results
    3. Remove all boxes that overlap with it above iou_thresh
    4. Repeat until no boxes remain

    Interview Question: "Why do we need NMS in object detection?"
    Answer: Multiple grid cells / anchors can detect the same object,
    producing duplicate boxes. NMS keeps only the best one.

    Args:
        detections: list of [x1, y1, x2, y2, confidence, class_id]
        conf_thresh: Minimum confidence score
        iou_thresh: IoU threshold for suppression

    Returns:
        List of surviving detections after NMS
    """
    if len(detections) == 0:
        return []

    # Convert to numpy for easier manipulation
    detections = np.array(detections)

    # Filter by confidence threshold
    mask = detections[:, 4] >= conf_thresh
    detections = detections[mask]

    if len(detections) == 0:
        return []

    # Sort by confidence (descending)
    sorted_indices = np.argsort(-detections[:, 4])
    detections = detections[sorted_indices]

    keep = []

    while len(detections) > 0:
        # Keep the highest confidence detection
        best = detections[0]
        keep.append(best)

        if len(detections) == 1:
            break

        # Compute IoU with remaining boxes
        remaining = detections[1:]
        ious = np.array([compute_iou(best[:4], det[:4]) for det in remaining])

        # Keep boxes with IoU below threshold (different objects)
        mask = ious < iou_thresh
        detections = remaining[mask]

    return keep


# ─── Severity Classification ─────────────────────────────

def classify_severity(box_wh, image_wh):
    """
    Classify pothole severity based on its size relative to the image.

    Real-world application: Helps road maintenance teams prioritize
    repairs based on pothole damage level.

    Args:
        box_wh: (width, height) of the bounding box in pixels
        image_wh: (width, height) of the image in pixels

    Returns:
        str: Severity level ("Small", "Medium", "Large", "Critical")
    """
    box_area = box_wh[0] * box_wh[1]
    image_area = image_wh[0] * image_wh[1]
    ratio = box_area / (image_area + 1e-6)

    if ratio < SEVERITY_THRESHOLDS["small"]:
        return "Small"
    elif ratio < SEVERITY_THRESHOLDS["medium"]:
        return "Medium"
    elif ratio < SEVERITY_THRESHOLDS["large"]:
        return "Large"
    else:
        return "Critical"


# ─── Visualization ───────────────────────────────────────

def draw_detections(image, detections, show_severity=True):
    """
    Draw bounding boxes, labels, and severity on an image.

    Args:
        image: numpy array (BGR format from OpenCV)
        detections: list of [x1, y1, x2, y2, confidence, class_id]
        show_severity: whether to show severity labels

    Returns:
        Annotated image (numpy array)
    """
    img = image.copy()
    h, w = img.shape[:2]

    for det in detections:
        x1, y1, x2, y2, conf, class_id = det[:6]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_id = int(class_id)

        # Get class name
        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"cls_{class_id}"

        # Determine severity and color
        if show_severity:
            box_w, box_h = x2 - x1, y2 - y1
            severity = classify_severity((box_w, box_h), (w, h))
            color = SEVERITY_COLORS.get(severity, (0, 255, 0))
            label = f"{class_name} {conf:.2f} [{severity}]"
        else:
            color = (0, 255, 0)
            label = f"{class_name} {conf:.2f}"

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        # Draw label background
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1
        )
        cv2.rectangle(
            img, (x1, y1 - label_h - baseline - 5),
            (x1 + label_w, y1), color, -1
        )

        # Draw label text
        cv2.putText(
            img, label, (x1, y1 - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), 1, cv2.LINE_AA
        )

    # Draw detection summary
    num_potholes = len(detections)
    summary = f"Detected: {num_potholes} pothole{'s' if num_potholes != 1 else ''}"
    cv2.putText(
        img, summary, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA
    )

    return img


def create_comparison(original, detected, max_width=1200):
    """
    Create a side-by-side comparison image.

    Args:
        original: Original image (numpy array)
        detected: Image with detections drawn (numpy array)
        max_width: Maximum width of the combined image

    Returns:
        Combined comparison image
    """
    h1, w1 = original.shape[:2]
    h2, w2 = detected.shape[:2]

    # Resize to same height
    target_h = min(h1, h2, 600)
    scale1 = target_h / h1
    scale2 = target_h / h2

    img1 = cv2.resize(original, (int(w1 * scale1), target_h))
    img2 = cv2.resize(detected, (int(w2 * scale2), target_h))

    # Add labels
    cv2.putText(img1, "Original", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img2, "Detected", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Separator
    sep = np.full((target_h, 3, 3), 128, dtype=np.uint8)

    # Combine
    combined = np.hstack([img1, sep, img2])
    return combined
