"""
=============================================================
  Pothole Detection — YOLOv8 Fine-Tuning Script
  Transfer Learning with Pre-trained COCO Weights
=============================================================

Why YOLOv8?
───────────
  • Pre-trained on COCO (80 classes, millions of images)
  • State-of-the-art accuracy and speed
  • Built-in data augmentation (mosaic, mixup, etc.)
  • Automatic anchor optimization
  • Modern CSPDarknet backbone with C2f modules
  • Native support for training, validation, export

Transfer Learning Strategy:
───────────────────────────
  1. Load YOLOv8n pre-trained on COCO (80 classes)
  2. Replace detection head for 1 class (pothole)
  3. Fine-tune ALL layers on pothole dataset
  4. Backbone already knows edges, textures, shapes → fast convergence
"""

import os
import sys

# Set project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)


def train_yolov8(epochs=25, imgsz=640, batch=8, model_size='n', resume=False):
    """
    Fine-tune YOLOv8 on the pothole dataset.

    Args:
        epochs: Number of training epochs (default: 25)
        imgsz: Input image size (default: 640)
        batch: Batch size (default: 8 for CPU)
        model_size: Model variant: 'n' (nano), 's' (small), 'm' (medium)
        resume: Resume from last checkpoint
    """
    from ultralytics import YOLO

    # Model selection
    model_name = f"yolov8{model_size}.pt"
    print(f"\n{'='*60}")
    print(f"  Pothole Detection — YOLOv8{model_size.upper()} Fine-Tuning")
    print(f"{'='*60}")
    print(f"  Base model:  {model_name} (COCO pre-trained)")
    print(f"  Epochs:      {epochs}")
    print(f"  Image size:  {imgsz}")
    print(f"  Batch size:  {batch}")
    print(f"  Device:      CPU")
    print(f"{'='*60}\n")

    # Load pre-trained YOLOv8
    model = YOLO(model_name)

    # Fine-tune on pothole dataset
    results = model.train(
        data=os.path.join(PROJECT_ROOT, "pothole_dataset.yaml"),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device="cpu",
        workers=0,
        project=os.path.join(PROJECT_ROOT, "runs"),
        name="pothole_yolov8",
        exist_ok=True,
        # Optimization
        optimizer="Adam",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        # Augmentation (built-in)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        # Regularization
        dropout=0.1,
        # Display
        verbose=True,
        save=True,
        save_period=5,
        plots=True,
    )

    # Copy best weights to project root
    best_weights = os.path.join(PROJECT_ROOT, "runs", "pothole_yolov8", "weights", "best.pt")
    if os.path.exists(best_weights):
        import shutil
        dest = os.path.join(PROJECT_ROOT, "yolov8_pothole_best.pt")
        shutil.copy2(best_weights, dest)
        print(f"\n  Best weights copied to: {dest}")

    print(f"\n  Training complete!")
    print(f"  Results saved to: runs/pothole_yolov8/")

    return results


def validate_yolov8(weights_path=None):
    """Run validation on the test set."""
    from ultralytics import YOLO

    if weights_path is None:
        weights_path = os.path.join(PROJECT_ROOT, "yolov8_pothole_best.pt")

    if not os.path.exists(weights_path):
        print(f"Weights not found: {weights_path}")
        print("Run training first: python train_yolov8.py")
        return None

    model = YOLO(weights_path)
    results = model.val(
        data=os.path.join(PROJECT_ROOT, "pothole_dataset.yaml"),
        imgsz=640,
        batch=8,
        device="cpu",
        workers=0,
        verbose=True,
        plots=True,
    )
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 Pothole Detection Training")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--model", type=str, default="n", choices=["n", "s", "m"],
                        help="Model size: n(ano), s(mall), m(edium)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only run validation, no training")
    args = parser.parse_args()

    if args.validate_only:
        validate_yolov8()
    else:
        train_yolov8(
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            model_size=args.model,
        )
