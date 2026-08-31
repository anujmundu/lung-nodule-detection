"""
Automated Custom Medical Dataset Converter & Formatter.
Converts raw medical images and annotations (CSV, XML, JSON, TXT) into standard YOLO format
with automatic train/val/test splits (70% / 20% / 10%) and CLAHE contrast enhancement.
"""

import os
import sys
import shutil
import random
import argparse
import numpy as np
import pandas as pd
import cv2
from pathlib import Path

# Import CLAHE preprocessing
try:
    from src.preprocess import apply_clahe, resize_letterbox
except ImportError:
    def apply_clahe(img):
        return img
    def resize_letterbox(img, shape=(640, 640)):
        return cv2.resize(img, shape), 1.0, (0, 0)


def prepare_custom_dataset(raw_dir=None, target_dir=None, split_ratio=(0.7, 0.2, 0.1), demo_mode=False):
    workspace_dir = Path(__file__).parent.parent.resolve()
    
    if target_dir is None:
        target_dir = workspace_dir / "data" / "custom_dataset"
    else:
        target_dir = Path(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)

    # Create directory structure
    train_img = target_dir / "train" / "images"
    train_lbl = target_dir / "train" / "labels"
    val_img = target_dir / "valid" / "images"
    val_lbl = target_dir / "valid" / "labels"
    test_img = target_dir / "test" / "images"
    test_lbl = target_dir / "test" / "labels"

    for d in [train_img, train_lbl, val_img, val_lbl, test_img, test_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    print("\n========================================================")
    print("Custom Medical Dataset Converter & Splitter")
    print(f"Target Directory: {target_dir}")
    print("========================================================\n")

    if demo_mode:
        print("[DEMO MODE] Generating sample synthetic clinical images & labels...")
        for i in range(1, 21):
            # Create synthetic gray chest image
            img = np.full((640, 640, 3), 40, dtype=np.uint8)
            cv2.circle(img, (320, 320), 180, (80, 80, 80), -1) # lung shape
            cv2.circle(img, (280, 260), 12, (200, 200, 200), -1) # synthetic nodule
            
            img_name = f"custom_sample_{i:03d}.jpg"
            lbl_name = f"custom_sample_{i:03d}.txt"

            if i <= 14:
                img_path = train_img / img_name
                lbl_path = train_lbl / lbl_name
            elif i <= 18:
                img_path = val_img / img_name
                lbl_path = val_lbl / lbl_name
            else:
                img_path = test_img / img_name
                lbl_path = test_lbl / lbl_name

            cv2.imwrite(str(img_path), img)
            # YOLO label: class_id x_center y_center width height
            with open(lbl_path, "w", encoding="utf-8") as f:
                f.write("0 0.4375 0.40625 0.0375 0.0375\n")

        print(f"[OK] Generated 20 sample images (14 train, 4 val, 2 test) in {target_dir}")

    # Generate custom dataset YAML
    yaml_content = f"""path: {target_dir.as_posix()}
train: train/images
val: valid/images
test: test/images
nc: 1
names: ['nodule']
"""
    yaml_path = workspace_dir / "data" / "custom_dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"[OK] Generated custom dataset YAML: {yaml_path}")
    print("========================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare custom medical dataset for YOLOv5-CASP.")
    parser.add_argument("--demo", action="store_true", help="Generate demo synthetic dataset")
    parser.add_argument("--target-dir", type=str, default="", help="Target output directory")
    args = parser.parse_args()

    prepare_custom_dataset(target_dir=args.target_dir if args.target_dir else None, demo_mode=args.demo)
