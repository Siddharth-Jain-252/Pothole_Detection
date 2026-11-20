# Model Training & Evaluation

This document describes the machine learning workflow for the Pothole Detection project, including dataset organization, YOLOv8 training, validation, evaluation, and model deployment.

The project uses Ultralytics YOLOv8 for pothole object detection.

## 1. Training Overview

The training pipeline consists of the following stages:

```text
        Dataset Collection
                |
                v
        Dataset Preparation
                |
                v
       Annotation Validation
                |
                v
        Train / Val / Test Split
                |
                v
          YOLOv8 Training
                |
                v
        Model Validation
                |
                v
         Performance Analysis
                |
                v
       Best Model Checkpoint
                |
                v
        Backend Integration
```

The objective is to train a model that can identify potholes in road images and localize them using bounding boxes.

## 2. Technology Requirements

| Component | Technology |
|---|---|
| Programming Language | Python 3.11 |
| Deep Learning Framework | PyTorch |
| Object Detection | Ultralytics YOLOv8 |
| Image Processing | OpenCV |
| Numerical Processing | NumPy |
| Visualization | Matplotlib |
| Training Hardware | NVIDIA GPU recommended |
| Dataset Format | YOLO detection format |

## 3. Dataset Organization

The project contains the following dataset directories:

- `dataset_yolo/`
- `dataset-pothole-main/`

Before training, identify which dataset is referenced by `pothole_dataset.yaml` and which dataset was used for the existing trained checkpoint.

A standard YOLO detection dataset follows this structure:

```text
dataset/
│
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
│
└── data.yaml
```

Each image should have a corresponding annotation file containing its object labels and bounding-box coordinates.

### YOLO Annotation Format

Each annotation line follows:

```text
class_id x_center y_center width height
```

The bounding-box coordinates are normalized between 0 and 1 relative to image dimensions.

For example:

```text
0 0.512 0.438 0.215 0.172
```

The class ID must correspond to the class mapping defined in the dataset configuration.

## 4. Dataset Configuration

The dataset configuration is maintained in:

```text
pothole_dataset.yaml
```

A typical YOLO dataset configuration looks like:

```yaml
path: ./dataset_yolo

train: images/train
val: images/val
test: images/test

names:
  0: pothole
```

Important: This is an example configuration. Keep the paths and class names consistent with your actual dataset. Do not change the class mapping when using an existing trained checkpoint.

## 5. Environment Setup

Create and activate a Python virtual environment.

Windows:

```powershell
py -3.11 -m venv .venv

.venv\Scripts\activate
```

Linux / macOS:

```bash
python3.11 -m venv .venv

source .venv/bin/activate
```

Install the training dependencies:

```bash
pip install --upgrade pip

pip install torch torchvision ultralytics numpy opencv-python matplotlib Pillow
```

Verify the Ultralytics installation:

```bash
yolo checks
```

Verify PyTorch and CUDA availability:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

A CUDA-enabled PyTorch installation and compatible NVIDIA drivers are required for GPU training.

## 6. Training

The repository includes:

```text
train_yolov8.py
```

Run the training script:

```bash
python train_yolov8.py
```

The script should use the project's dataset configuration and selected YOLOv8 architecture.

Before training, verify the following in the script:

- Dataset YAML path.
- Initial model checkpoint.
- Number of epochs.
- Image size.
- Batch size.
- Optimizer.
- Learning rate.
- Device selection.
- Output directory.

Do not assume these settings from this document; the actual values are defined by the training script.

### Example Ultralytics Training

For a standalone training experiment, the following is a generic YOLOv8 example:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="pothole_dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0
)
```

These are example hyperparameters, not a claim about the settings used to produce the project's existing checkpoint.

For CPU training, use:

```python
device="cpu"
```

GPU training is recommended for faster experimentation.

## 7. Training Outputs

Ultralytics typically creates a run directory containing training artifacts.

Example:

```text
runs/
└── detect/
    └── train/
        ├── weights/
        │   ├── best.pt
        │   └── last.pt
        │
        ├── results.csv
        ├── results.png
        ├── confusion_matrix.png
        ├── PR_curve.png
        ├── F1_curve.png
        ├── P_curve.png
        └── R_curve.png
```

The exact output directory depends on the training configuration.

### Checkpoint Information

| File | Purpose |
|---|---|
| `best.pt` | Checkpoint selected as best during validation |
| `last.pt` | Checkpoint from the final training epoch |
| `results.csv` | Epoch-wise training and validation metrics |
| `results.png` | Training metric visualizations |
| `confusion_matrix.png` | Classification error analysis |
| `PR_curve.png` | Precision-recall analysis |

## 8. Model Validation

After training, validate the checkpoint on the validation or test dataset.

Example:

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

metrics = model.val(
    data="pothole_dataset.yaml"
)

print(metrics)
```

Validation helps assess how well the model performs on data that was not used to update its weights.

## 9. Evaluation Metrics

The following metrics are useful when evaluating an object detection model.

| Metric | Description |
|---|---|
| Precision | Fraction of predicted potholes that are correct |
| Recall | Fraction of actual potholes detected |
| mAP@50 | Mean average precision at IoU 0.50 |
| mAP@50-95 | Mean average precision averaged over IoU thresholds |
| F1-score | Harmonic mean of precision and recall |

### Precision

Precision measures how many predicted potholes are correct.

```text
Precision = TP / (TP + FP)
```

### Recall

Recall measures how many actual potholes were detected.

```text
Recall = TP / (TP + FN)
```

### F1-score

```text
F1 = 2 * Precision * Recall / (Precision + Recall)
```

Where:

- TP = True Positives
- FP = False Positives
- FN = False Negatives

Report actual measured values from the validation output. Do not use training-set performance as a substitute for test-set performance.

## 10. Inference Testing

The repository includes:

```text
inference_yolov8.py
```

Use this script to test the trained model on sample images.

```bash
python inference_yolov8.py
```

The exact arguments and input handling depend on the implementation in that script.

You should test images containing:

- Clear potholes.
- Multiple potholes.
- Small potholes.
- Different road surfaces.
- Shadows and reflections.
- Normal roads without potholes.

This helps identify false positives and false negatives.
