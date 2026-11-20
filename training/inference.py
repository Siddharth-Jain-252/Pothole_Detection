"""
=============================================================
  Pothole Detection — Inference Pipeline
  End-to-end detection: load image → model → post-process → visualize
=============================================================

Pipeline Steps:
───────────────
  1. Preprocess image (resize, normalize, to tensor)
  2. Forward pass through YOLOv3-Tiny
  3. Decode predictions (sigmoid for confidence/class, decode boxes)
  4. Filter by confidence threshold
  5. Apply Non-Maximum Suppression (NMS)
  6. Draw bounding boxes with severity labels
  7. Save / display results
"""

import os
import sys
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from glob import glob

from config import (
    DEVICE, IMAGE_SIZE, NUM_CLASSES, NUM_ANCHORS,
    CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD,
    DETECTION_OUTPUT_DIR, DEFAULT_WEIGHTS, CLASS_NAMES,
)
from model import load_model
from utils import (
    non_max_suppression, draw_detections,
    create_comparison, classify_severity,
)


def preprocess_image(image_path, image_size=IMAGE_SIZE):
    """
    Load and preprocess an image for model inference.

    Steps:
        1. Read image with OpenCV (BGR)
        2. Convert to RGB
        3. Resize to model input size
        4. Normalize to [0, 1]
        5. Convert to tensor with batch dimension

    Args:
        image_path: Path to input image
        image_size: Target size for the model

    Returns:
        tensor: Preprocessed image tensor [1, 3, H, W]
        original: Original image (BGR numpy array)
    """
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # Convert and resize
    rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (image_size, image_size))

    # To tensor and add batch dimension
    tensor = transforms.ToTensor()(resized).unsqueeze(0)

    return tensor.to(DEVICE), original


def decode_predictions(output, image_shape, conf_thresh=CONFIDENCE_THRESHOLD):
    """
    Decode raw model output into bounding box detections.

    The model outputs a tensor of shape [B, A, G, G, 5+C] where:
        - B = batch size
        - A = number of anchors (3)
        - G = grid size (6)
        - Pred channels: [class_0..class_9, objectness, tx, ty, tw, th]

    During training, the loss function uses:
        pred[:, :num_classes]  → class scores
        pred[:, num_classes]   → confidence/objectness
        pred[:, num_classes+1:] → bbox (x, y, w, h)

    Interview: "Walk me through YOLO's output decoding."
    1. Apply sigmoid to objectness → confidence in [0,1]
    2. Apply softmax/sigmoid to class scores
    3. Convert grid-relative coords to absolute image coords
    4. Filter low-confidence detections

    Args:
        output: Model output tensor [1, A, G, G, 5+C]
        image_shape: (height, width) of the original image
        conf_thresh: Minimum confidence to keep

    Returns:
        List of [x1, y1, x2, y2, confidence, class_id]
    """
    output = output[0]  # Remove batch dimension → [A, G, G, 5+C]
    num_anchors, grid_h, grid_w, pred_size = output.shape

    detections = []
    orig_h, orig_w = image_shape

    for a in range(num_anchors):
        for gy in range(grid_h):
            for gx in range(grid_w):
                pred = output[a, gy, gx]

                # Extract components matching training format
                class_scores = pred[:NUM_CLASSES]
                objectness = torch.sigmoid(pred[NUM_CLASSES])
                bbox = pred[NUM_CLASSES + 1:]

                # Class prediction
                class_probs = torch.softmax(class_scores, dim=0)
                class_conf, class_id = class_probs.max(dim=0)

                # Combined confidence = objectness × class_probability
                confidence = (objectness * class_conf).item()

                if confidence < conf_thresh:
                    continue

                # Force class_id to 0 (pothole) since it's a single-class dataset
                # The model may predict other class indices due to limited training
                class_id_val = 0

                # Decode bounding box from grid-relative to absolute
                # The bbox values are raw regression targets (not normalized)
                # Decode center: sigmoid maps to [0,1] within the cell
                cx = (gx + torch.sigmoid(bbox[0]).item()) / grid_w
                cy = (gy + torch.sigmoid(bbox[1]).item()) / grid_h

                # Decode width/height: use sigmoid to bound within reasonable range
                # Scale by grid cell size for proportional boxes
                bw = torch.sigmoid(bbox[2]).item() * (2.0 / grid_w)
                bh = torch.sigmoid(bbox[3]).item() * (2.0 / grid_h)

                # Ensure minimum box size (at least 1 grid cell worth)
                bw = max(bw, 0.5 / grid_w)
                bh = max(bh, 0.5 / grid_h)

                # Convert to pixel coordinates (on original image)
                x1 = int((cx - bw / 2) * orig_w)
                y1 = int((cy - bh / 2) * orig_h)
                x2 = int((cx + bw / 2) * orig_w)
                y2 = int((cy + bh / 2) * orig_h)

                # Clip to image boundaries
                x1 = max(0, min(x1, orig_w))
                y1 = max(0, min(y1, orig_h))
                x2 = max(0, min(x2, orig_w))
                y2 = max(0, min(y2, orig_h))

                # Skip tiny boxes (less than 10px)
                if (x2 - x1) < 10 or (y2 - y1) < 10:
                    continue

                detections.append([x1, y1, x2, y2, confidence, class_id_val])

    return detections


def detect_image(model, image_path, conf_thresh=CONFIDENCE_THRESHOLD,
                 nms_thresh=NMS_IOU_THRESHOLD, save_path=None):
    """
    Run full detection pipeline on a single image.

    Args:
        model: YOLOv3Tiny model (in eval mode)
        image_path: Path to input image
        conf_thresh: Confidence threshold
        nms_thresh: NMS IoU threshold
        save_path: Optional path to save the annotated image

    Returns:
        detections: List of [x1, y1, x2, y2, confidence, class_id]
        annotated_img: Image with drawn detections (numpy array)
    """
    # 1. Preprocess
    tensor, original = preprocess_image(image_path)

    # 2. Forward pass
    with torch.no_grad():
        output = model(tensor)

    # 3. Decode predictions
    h, w = original.shape[:2]
    raw_detections = decode_predictions(output, (h, w), conf_thresh)

    # 4. Apply NMS
    detections = non_max_suppression(raw_detections, conf_thresh, nms_thresh)

    # 5. Draw results
    annotated = draw_detections(original, detections)

    # 6. Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, annotated)

    return detections, annotated


def detect_batch(model, image_folder, output_folder=DETECTION_OUTPUT_DIR,
                 max_images=None, conf_thresh=CONFIDENCE_THRESHOLD,
                 nms_thresh=NMS_IOU_THRESHOLD):
    """
    Run detection on all images in a folder.

    Args:
        model: YOLOv3Tiny model
        image_folder: Folder containing images
        output_folder: Folder to save annotated images
        max_images: Maximum number of images to process (None = all)
        conf_thresh: Confidence threshold
        nms_thresh: NMS IoU threshold

    Returns:
        results: dict mapping filename → list of detections
    """
    os.makedirs(output_folder, exist_ok=True)

    # Find all images
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob(os.path.join(image_folder, ext)))

    if max_images:
        image_paths = image_paths[:max_images]

    results = {}
    total_detections = 0

    print(f"\n🔍 Running detection on {len(image_paths)} images...")
    print("=" * 50)

    for i, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        save_path = os.path.join(output_folder, f"det_{filename}")

        try:
            detections, _ = detect_image(
                model, img_path, conf_thresh, nms_thresh, save_path
            )
            results[filename] = detections
            total_detections += len(detections)

            if (i + 1) % 50 == 0 or (i + 1) == len(image_paths):
                print(f"  [{i+1}/{len(image_paths)}] Processed, "
                      f"total detections so far: {total_detections}")

        except Exception as e:
            print(f"  ⚠️ Error processing {filename}: {e}")
            results[filename] = []

    print(f"\n✅ Detection complete!")
    print(f"   Images processed: {len(image_paths)}")
    print(f"   Total detections: {total_detections}")
    print(f"   Results saved to: {output_folder}")

    return results


if __name__ == "__main__":
    """
    Quick demo: Run inference on test images.

    Usage:
        python inference.py                    # Run on 10 test images
        python inference.py path/to/image.jpg  # Run on specific image
    """
    print("\n" + "=" * 60)
    print("  🔍 Pothole Detection — Inference Demo")
    print("=" * 60)

    # Load model
    model = load_model()

    if len(sys.argv) > 1:
        # Single image
        img_path = sys.argv[1]
        save_path = os.path.join(DETECTION_OUTPUT_DIR, f"det_{os.path.basename(img_path)}")
        detections, annotated = detect_image(model, img_path, save_path=save_path)
        print(f"\n  Found {len(detections)} potholes in {os.path.basename(img_path)}")
        for det in detections:
            x1, y1, x2, y2, conf, cls = det[:6]
            severity = classify_severity((x2-x1, y2-y1), (annotated.shape[1], annotated.shape[0]))
            print(f"    • [{severity}] conf={conf:.3f} box=({x1},{y1},{x2},{y2})")
    else:
        # Batch on first 10 test images
        from config import TEST_IMAGES
        results = detect_batch(model, TEST_IMAGES, max_images=10)
