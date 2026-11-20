"""
=============================================================
  Pothole Detection — Dataset & DataLoader
  PyTorch Dataset class for YOLO-format pothole annotations
=============================================================

Interview Key Points:
─────────────────────
  • Custom Dataset class extends torch.utils.data.Dataset
  • __getitem__ returns (image_tensor, label_tensor) pairs
  • Data augmentation improves generalization
  • collate_fn handles variable-length labels per image
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from config import (
    TRAIN_IMAGES, TEST_IMAGES, IMAGE_SIZE,
    BATCH_SIZE, AUGMENTATION,
)


class PotholeDataset(Dataset):
    """
    PyTorch Dataset for pothole detection in YOLO format.

    Each image has a corresponding .txt label file with one line per object:
        class_id center_x center_y width height
    (all values normalized to [0, 1])

    Args:
        images_folder (str): Path to folder containing images and labels
        image_size (int): Target image size (square)
        transform: torchvision transforms to apply
        augment (bool): Whether to apply data augmentation
    """

    def __init__(self, images_folder, image_size=IMAGE_SIZE,
                 transform=None, augment=False):
        self.images_folder = images_folder
        self.image_size = image_size
        self.transform = transform
        self.augment = augment

        # Collect all image files
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        all_files = sorted(os.listdir(images_folder))
        self.image_files = [
            f for f in all_files
            if os.path.splitext(f)[1].lower() in valid_extensions
        ]

        # Verify label files exist
        self.valid_pairs = []
        for img_file in self.image_files:
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(images_folder, label_file)
            if os.path.exists(label_path):
                self.valid_pairs.append((img_file, label_file))

        print(f"  📂 Loaded {len(self.valid_pairs)} image-label pairs "
              f"from {os.path.basename(images_folder)}/")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        img_file, label_file = self.valid_pairs[idx]

        # ── Load Image ────────────────────────────────────
        img_path = os.path.join(self.images_folder, img_file)
        image = cv2.imread(img_path)

        if image is None:
            # Return a blank image and empty labels as fallback
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            return transforms.ToTensor()(image), torch.zeros((0, 5), dtype=torch.float32)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ── Data Augmentation ─────────────────────────────
        if self.augment:
            image = self._augment(image)

        # ── Resize to target size ─────────────────────────
        image = cv2.resize(image, (self.image_size, self.image_size))

        # ── Apply transforms ──────────────────────────────
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        # ── Load Labels ───────────────────────────────────
        label_path = os.path.join(self.images_folder, label_file)
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x, y, w, h = map(float, parts)
                        boxes.append([class_id, x, y, w, h])

        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
        else:
            boxes = torch.zeros((0, 5), dtype=torch.float32)

        return image, boxes

    def _augment(self, image):
        """
        Apply data augmentation to training images.

        Augmentation Strategy (Interview Topic):
        - Horizontal flip: Potholes can appear on either side
        - Color jitter: Handles varying lighting conditions
        - Small rotation: Real-world camera angles vary slightly
        - Gaussian blur: Simulates motion blur / camera quality
        """
        # Random horizontal flip
        if np.random.random() < AUGMENTATION["horizontal_flip_prob"]:
            image = cv2.flip(image, 1)

        # Random brightness/contrast adjustment
        if np.random.random() < 0.5:
            alpha = 1.0 + np.random.uniform(
                -AUGMENTATION["color_jitter"]["contrast"],
                AUGMENTATION["color_jitter"]["contrast"]
            )
            beta = np.random.uniform(
                -AUGMENTATION["color_jitter"]["brightness"] * 50,
                AUGMENTATION["color_jitter"]["brightness"] * 50
            )
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        # Random Gaussian blur
        if np.random.random() < 0.1:
            ksize = np.random.choice([3, 5])
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)

        return image

    def get_image_path(self, idx):
        """Get the file path for an image at a given index."""
        img_file, _ = self.valid_pairs[idx]
        return os.path.join(self.images_folder, img_file)


def collate_fn(batch):
    """
    Custom collate function for variable-length label tensors.

    Standard DataLoader requires all items in a batch to have the
    same shape. But different images have different numbers of objects,
    so we keep labels as a list of tensors.

    Interview Tip: This is a common gotcha in object detection code!
    """
    images, targets = zip(*batch)
    images = torch.stack(images)
    return images, list(targets)


def get_train_loader(batch_size=BATCH_SIZE, augment=True):
    """Create a DataLoader for the training set."""
    dataset = PotholeDataset(
        TRAIN_IMAGES,
        augment=augment,
        transform=transforms.ToTensor(),
    )
    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, collate_fn=collate_fn,
        num_workers=0, pin_memory=False,
    )
    return loader


def get_test_loader(batch_size=BATCH_SIZE):
    """Create a DataLoader for the test set."""
    dataset = PotholeDataset(
        TEST_IMAGES,
        augment=False,
        transform=transforms.ToTensor(),
    )
    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn,
        num_workers=0, pin_memory=False,
    )
    return loader


if __name__ == "__main__":
    """Quick test: load dataset and print statistics."""
    print("\n📊 Dataset Statistics:")
    print("=" * 50)

    train_ds = PotholeDataset(TRAIN_IMAGES, transform=transforms.ToTensor())
    test_ds = PotholeDataset(TEST_IMAGES, transform=transforms.ToTensor())

    print(f"\n  Training images: {len(train_ds)}")
    print(f"  Test images:     {len(test_ds)}")

    # Sample a training image
    img, boxes = train_ds[0]
    print(f"\n  Sample image shape: {img.shape}")
    print(f"  Sample labels:      {boxes.shape[0]} objects")
    if boxes.shape[0] > 0:
        print(f"  First label:        class={int(boxes[0, 0])}, "
              f"cx={boxes[0, 1]:.3f}, cy={boxes[0, 2]:.3f}, "
              f"w={boxes[0, 3]:.3f}, h={boxes[0, 4]:.3f}")
