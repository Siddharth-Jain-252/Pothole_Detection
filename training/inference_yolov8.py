"""
=============================================================
  Pothole Detection — Optimized YOLOv8 Inference Pipeline
  Strict Inference-Time Optimizations WITHOUT Retraining
=============================================================

Features:
  - Optimized model.predict() parameters (conf, iou, agnostic_nms, max_det, TTA)
  - Box aspect ratio & area filtering
  - Anti-overlap label drawing
  - Video stability with simple tracking
"""

import os
import sys
import cv2
import numpy as np
from glob import glob
from collections import deque

# Project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import (
    DETECTION_OUTPUT_DIR,
    YOLOV8_CONFIDENCE_THRESHOLD,
    YOLOV8_POST_FILTER_CONFIDENCE,
    YOLOV8_NMS_IOU_THRESHOLD,
    YOLOV8_MAX_PREDICTIONS,
    YOLOV8_MAX_DISPLAY_DETECTIONS,
    YOLOV8_MIN_BOX_AREA,
    YOLOV8_MIN_ASPECT_RATIO,
    YOLOV8_MAX_ASPECT_RATIO,
    YOLOV8_AGNOSTIC_NMS,
    YOLOV8_TTA,
    BOX_THICKNESS,
    FONT_SCALE,
)

def load_yolov8_model(weights_path=None):
    """Load the fine-tuned YOLOv8 model."""
    from ultralytics import YOLO

    if weights_path is None:
        candidates = [
            os.path.join(PROJECT_ROOT, "yolov8_pothole_best.pt"),
            os.path.join(PROJECT_ROOT, "runs", "pothole_yolov8", "weights", "best.pt"),
            os.path.join(PROJECT_ROOT, "runs", "pothole_yolov8", "weights", "last.pt"),
        ]
        for c in candidates:
            if os.path.exists(c):
                weights_path = c
                break

    if weights_path is None or not os.path.exists(weights_path):
        weights_path = "yolov8n.pt"

    model = YOLO(weights_path)
    print(f"  Model loaded: {weights_path}")
    return model


def filter_detections(boxes, img_w, img_h):
    """
    Apply false positive filtering:
    - Final confidence cleanup
    - Min area threshold
    - Aspect ratio between 0.3 and 3
    - Limit the final UI to the strongest boxes
    """
    filtered = []

    for box in boxes:
        x1, y1, x2, y2, conf, cls_id = box
        w = x2 - x1
        h = y2 - y1

        if w <= 0 or h <= 0:
            continue

        if conf < YOLOV8_POST_FILTER_CONFIDENCE:
            continue

        area = w * h
        aspect_ratio = w / h

        if area < YOLOV8_MIN_BOX_AREA:
            continue
            
        if not (YOLOV8_MIN_ASPECT_RATIO <= aspect_ratio <= YOLOV8_MAX_ASPECT_RATIO):
            continue

        filtered.append(box)

    filtered.sort(key=lambda det: det[4], reverse=True)
    return filtered[:YOLOV8_MAX_DISPLAY_DETECTIONS]


def detect_image_v8(model, image_path, conf_thresh=YOLOV8_CONFIDENCE_THRESHOLD,
                    iou_thresh=YOLOV8_NMS_IOU_THRESHOLD, save_path=None):
    """Run optimized YOLOv8 detection on a single image."""
    
    # 1. inference-time optimizations
    results = model.predict(
        image_path,
        conf=conf_thresh,
        iou=iou_thresh,
        agnostic_nms=YOLOV8_AGNOSTIC_NMS,
        max_det=YOLOV8_MAX_PREDICTIONS,
        augment=YOLOV8_TTA,
        verbose=False
    )

    original = cv2.imread(image_path)
    if original is None:
        return [], None
        
    h, w = original.shape[:2]
    raw_detections = []

    for result in results:
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                raw_detections.append([x1, y1, x2, y2, conf, cls_id])

    # 2. Filter False Positives (Aspect ratio & Area)
    final_detections = filter_detections(raw_detections, w, h)

    # 3. Clean Output Visualization
    annotated = draw_detections_v8(original, final_detections)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, annotated)

    return final_detections, annotated


def draw_detections_v8(image, detections):
    """Draw bounding boxes with anti-overlap labels."""
    img = image.copy()
    
    # Track used label areas to prevent overlap
    used_label_regions = []

    for det in detections:
        x1, y1, x2, y2, conf = det[:5]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        color = (0, 165, 255) # Orange for pothole
        label = f"Pothole {conf:.2f}"

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        # Draw label background
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1
        )
        
        # Calculate label position, avoid overlap
        label_x = x1
        label_y = y1 - 5
        
        # Simple anti-overlap pushdown
        for (ux1, uy1, ux2, uy2) in used_label_regions:
            if not (label_x + tw < ux1 or label_x > ux2 or label_y - th - baseline > uy2 or label_y < uy1):
                # Overlap detected! Push label down inside the box
                label_y = y1 + th + baseline + 5
                
        # Register new occupied space
        used_label_regions.append((label_x, label_y - th - baseline, label_x + tw, label_y))

        cv2.rectangle(img, (label_x, label_y - th - baseline),
                      (label_x + tw, label_y), color, -1)
        cv2.putText(img, label, (label_x, label_y - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
                    (0, 0, 0), 1, cv2.LINE_AA)

    return img


def detect_batch_v8(model, image_folder, output_folder=None,
                     max_images=None, conf_thresh=YOLOV8_CONFIDENCE_THRESHOLD,
                     iou_thresh=YOLOV8_NMS_IOU_THRESHOLD):
    """Run YOLOv8 batch detection."""
    if output_folder is None:
        output_folder = os.path.join(DETECTION_OUTPUT_DIR, "yolov8_optimized")
    os.makedirs(output_folder, exist_ok=True)

    extensions = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob(os.path.join(image_folder, ext)))

    if max_images:
        image_paths = image_paths[:max_images]

    results = {}
    total_dets = 0

    print(f"\n  Running Optimized YOLOv8 detection on {len(image_paths)} images...")
    print("=" * 50)

    for i, path in enumerate(image_paths):
        fname = os.path.basename(path)
        save_path = os.path.join(output_folder, f"v8_opt_{fname}")

        dets, _ = detect_image_v8(model, path, conf_thresh, iou_thresh, save_path)
        results[fname] = dets
        total_dets += len(dets)

        if (i + 1) % 20 == 0 or (i + 1) == len(image_paths):
            print(f"  [{i+1}/{len(image_paths)}] total detections: {total_dets}")

    print(f"\n  Done! {len(image_paths)} images, {total_dets} detections")
    print(f"  Saved to: {output_folder}")
    return results


def bb_intersection_over_union(boxA, boxB):
    """Compute IOU between two boxes [x1, y1, x2, y2]"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
    return iou


def detect_video_v8(model, video_path, output_path=None,
                     conf_thresh=YOLOV8_CONFIDENCE_THRESHOLD,
                     iou_thresh=YOLOV8_NMS_IOU_THRESHOLD, max_frames=None):
    """
    Run prioritized YOLOv8 detection on a video with STABILITY tracking.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return 0

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(DETECTION_OUTPUT_DIR, f"v8_opt_{base}.mp4")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_count = 0
    total_dets = 0

    # History tracker for detections in previous 3 frames
    history = deque(maxlen=3)

    print(f"\n  Processing video: {video_path}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames and frame_count >= max_frames:
            break
            
        # Write to temp image format because model.predict is optimized for paths/numpy list
        # Using native numpy inference
        results = model.predict(
            frame,
            conf=conf_thresh,
            iou=iou_thresh,
            agnostic_nms=YOLOV8_AGNOSTIC_NMS,
            max_det=YOLOV8_MAX_PREDICTIONS,
            augment=YOLOV8_TTA,
            verbose=False
        )
        
        raw_dets = []
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    raw_dets.append([x1, y1, x2, y2, float(box.conf[0]), int(box.cls[0])])

        # Filter areas and aspect ratio
        filtered = filter_detections(raw_dets, w, h)
        
        # Stable tracking logic
        stable_dets = []
        if len(history) < 2:
            # Not enough history, show everything filtered
            stable_dets = filtered
        else:
            # Check if each detection roughly existed in previous 2 frames
            for det in filtered:
                match_count = 0
                for past_frame_dets in history:
                    for past_det in past_frame_dets:
                        if bb_intersection_over_union(det[:4], past_det[:4]) > 0.4:
                            match_count += 1
                            break
                # Only keep if it appeared in at least 1 recent frame (allowing slight flicker)
                if match_count >= 1:
                    stable_dets.append(det)

        history.append(filtered)
        total_dets += len(stable_dets)
        
        annotated = draw_detections_v8(frame, stable_dets)
        out.write(annotated)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"  Frame {frame_count}/{total_frames} | Stable Detections: {total_dets}")

    cap.release()
    out.release()
    print(f"\n  Video complete: {frame_count} frames, {total_dets} stable detections")
    return total_dets


if __name__ == "__main__":
    from config import TEST_IMAGES

    print("\n" + "=" * 60)
    print("  Optimized YOLOv8 Pothole Detection Demo")
    print("=" * 60)

    model = load_yolov8_model()

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        save_path = os.path.join(DETECTION_OUTPUT_DIR, "yolov8_optimized",
                                 f"v8_opt_{os.path.basename(img_path)}")
        dets, _ = detect_image_v8(model, img_path, save_path=save_path)
        print(f"\n  Found {len(dets)} potholes:")
    else:
        detect_batch_v8(model, TEST_IMAGES, max_images=10)
