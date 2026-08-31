"""
Data augmentation pipeline using Albumentations for YOLOv5-CASP lung nodule detection.
"""

import albumentations as A
import cv2
import numpy as np


def get_casp_transform(img_size=640):
    """
    Returns Albumentations Compose pipeline matching thesis specifications:
    - Horizontal Flipping (p=0.5)
    - Rotation (±5°) & Scaling (±20%) via ShiftScaleRotate
    - Brightness & Contrast adjustment (±0.2)
    - Target resizing to 640x640
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0,
                scale_limit=0.20,
                rotate_limit=5,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Resize(height=img_size, width=img_size),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )


def apply_augmentations(image, bboxes=None, class_labels=None, transform=None):
    """
    Applies data augmentations to an image and its corresponding bounding boxes.
    """
    if transform is None:
        transform = get_casp_transform()

    if bboxes is None:
        bboxes = []
    if class_labels is None:
        class_labels = []

    augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    return augmented["image"], augmented["bboxes"], augmented["class_labels"]
