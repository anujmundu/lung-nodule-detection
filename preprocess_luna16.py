"""
preprocess_luna16.py
Convert LUNA16 CT scans to 2D PNG images with YOLO bounding box labels.
Each axial slice containing at least one nodule is saved as one image.
All nodules in that slice are annotated in the same .txt file (YOLO format).
"""

import os
import csv
import random
import numpy as np
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

# ---------- CONFIGURATION ----------
RAW_DATA_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
IMAGES_DIR = os.path.join(PROCESSED_DIR, "images")
LABELS_DIR = os.path.join(PROCESSED_DIR, "labels")
TRAIN_VAL_TEST_SPLIT = [0.7, 0.15, 0.15]  # train, val, test
RANDOM_SEED = 42
CLASS_ID = 0  # only one class: lung nodule

# Lung window for CT (center -600, width 1500) to enhance contrast
LUNG_CENTER = -600
LUNG_WIDTH = 1500
LOW = LUNG_CENTER - LUNG_WIDTH / 2
HIGH = LUNG_CENTER + LUNG_WIDTH / 2

# For reproducibility
random.seed(RANDOM_SEED)

# Create output directories
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)


# ---------- UTILITY FUNCTIONS ----------
def world_to_voxel(world_coord, origin, spacing):
    """
    Convert world coordinates (mm) to voxel indices.
    Assumes near-identity direction matrix (common in LUNA16).
    Returns (z, y, x) as integers.
    """
    idx = (np.array(world_coord) - np.array(origin)) / np.array(spacing)
    return np.round(idx).astype(int)


def create_yolo_annotation(img_h, img_w, center_y, center_x, diameter_pixels):
    """
    Generate YOLO format line: class x_center y_center width height (normalized).
    Bounding box is square of side = diameter_pixels, centered at (center_y, center_x).
    Returns None if bounding box is invalid.
    """
    half_size = diameter_pixels / 2.0
    x_min = max(0, center_x - half_size)
    y_min = max(0, center_y - half_size)
    x_max = min(img_w - 1, center_x + half_size)
    y_max = min(img_h - 1, center_y + half_size)

    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    if bbox_width <= 0 or bbox_height <= 0:
        return None

    x_center = (x_min + x_max) / 2.0 / img_w
    y_center = (y_min + y_max) / 2.0 / img_h
    width = bbox_width / img_w
    height = bbox_height / img_h

    return f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


# ---------- LOAD NODULE ANNOTATIONS ----------
annotations_path = os.path.join(RAW_DATA_DIR, "annotations.csv")
nodules_by_series = defaultdict(list)

print("Loading annotations.csv...")
with open(annotations_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        seriesuid = row['seriesuid']
        coord_x = float(row['coordX'])
        coord_y = float(row['coordY'])
        coord_z = float(row['coordZ'])
        diameter_mm = float(row['diameter_mm'])
        nodules_by_series[seriesuid].append({
            'coord': (coord_x, coord_y, coord_z),
            'diameter_mm': diameter_mm
        })

print(f"Found {len(nodules_by_series)} series with nodules.")


# ---------- PROCESS EACH SERIES ----------
image_label_pairs = []  # list of (image_filename, label_filename) for all generated samples

# Find all subset directories
subset_dirs = [d for d in os.listdir(RAW_DATA_DIR) if d.startswith("subset") and os.path.isdir(os.path.join(RAW_DATA_DIR, d))]

for subset_dir in sorted(subset_dirs):
    subset_path = os.path.join(RAW_DATA_DIR, subset_dir)
    mhd_files = [f for f in os.listdir(subset_path) if f.endswith('.mhd')]

    for mhd_file in tqdm(mhd_files, desc=f"Processing {subset_dir}"):
        seriesuid = mhd_file.replace('.mhd', '')
        if seriesuid not in nodules_by_series:
            continue  # skip series with no nodules

        # Load CT volume
        mhd_path = os.path.join(subset_path, mhd_file)
        img_sitk = sitk.ReadImage(mhd_path)
        volume = sitk.GetArrayFromImage(img_sitk)  # shape (z, y, x)
        origin = img_sitk.GetOrigin()
        spacing = img_sitk.GetSpacing()  # (z, y, x) spacing in mm

        # Group nodules by slice index (z)
        nodules_by_slice = defaultdict(list)
        for nodule in nodules_by_series[seriesuid]:
            coord_world = nodule['coord']
            diameter_mm = nodule['diameter_mm']
            voxel_idx = world_to_voxel(coord_world, origin, spacing)
            slice_idx = voxel_idx[0]  # z coordinate
            if 0 <= slice_idx < volume.shape[0]:
                nodules_by_slice[slice_idx].append({
                    'voxel_center': (voxel_idx[1], voxel_idx[2]),  # (y, x)
                    'diameter_mm': diameter_mm
                })

        # Process each slice that contains at least one nodule
        for slice_idx, nodules_in_slice in nodules_by_slice.items():
            # Extract the 2D slice
            slice_img = volume[slice_idx, :, :]  # shape (y, x)
            # Apply lung window and normalize to 0-255 uint8
            slice_img = np.clip(slice_img, LOW, HIGH)
            slice_img = ((slice_img - LOW) / (HIGH - LOW)) * 255
            slice_img = slice_img.astype(np.uint8)

            img_h, img_w = slice_img.shape

            # Determine average pixel spacing for bounding box size
            # spacing[1] = y spacing (mm/pixel), spacing[2] = x spacing
            pixel_spacing_y = spacing[1]
            pixel_spacing_x = spacing[2]
            pixel_spacing_avg = (pixel_spacing_y + pixel_spacing_x) / 2.0

            # Build YOLO annotation lines for all nodules in this slice
            yolo_lines = []
            for nod in nodules_in_slice:
                center_y, center_x = nod['voxel_center']
                diameter_px = nod['diameter_mm'] / pixel_spacing_avg
                yolo_line = create_yolo_annotation(img_h, img_w, center_y, center_x, diameter_px)
                if yolo_line:
                    yolo_lines.append(yolo_line)

            if not yolo_lines:
                continue

            # Save the image as PNG
            img_filename = f"{seriesuid}_slice{slice_idx:04d}.png"
            img_path = os.path.join(IMAGES_DIR, img_filename)
            img_pil = Image.fromarray(slice_img, mode='L')
            img_pil.save(img_path)

            # Save the annotations as a .txt file (same base name)
            label_filename = img_filename.replace('.png', '.txt')
            label_path = os.path.join(LABELS_DIR, label_filename)
            with open(label_path, 'w') as lf:
                lf.write('\n'.join(yolo_lines))

            image_label_pairs.append((img_filename, label_filename))

print(f"Total images (slices with nodules) generated: {len(image_label_pairs)}")


# ---------- CREATE TRAIN/VAL/TEST SPLITS ----------
# Shuffle the list
random.shuffle(image_label_pairs)

num_total = len(image_label_pairs)
num_train = int(TRAIN_VAL_TEST_SPLIT[0] * num_total)
num_val = int(TRAIN_VAL_TEST_SPLIT[1] * num_total)
num_test = num_total - num_train - num_val

train_pairs = image_label_pairs[:num_train]
val_pairs = image_label_pairs[num_train:num_train + num_val]
test_pairs = image_label_pairs[num_train + num_val:]

# Write split files (full path to each image)
def write_split_file(pairs, split_name):
    split_path = os.path.join(PROCESSED_DIR, f"{split_name}.txt")
    with open(split_path, 'w') as f:
        for img_fn, _ in pairs:
            abs_img_path = os.path.abspath(os.path.join(IMAGES_DIR, img_fn))
            f.write(abs_img_path + '\n')

write_split_file(train_pairs, "train")
write_split_file(val_pairs, "val")
write_split_file(test_pairs, "test")

print(f"Split: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")
print("Preprocessing complete. Data is ready for YOLO training.")