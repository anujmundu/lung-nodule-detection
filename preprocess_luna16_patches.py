"""
preprocess_luna16_patches.py
Extract 256x256 patches centered on each lung nodule from LUNA16 CT scans.
Each patch is saved as a PNG, and the nodule bounding box (scaled to patch size) is saved in YOLO format.
The patch size can be adjusted (e.g., 256 or 320). 
This approach makes the nodule occupy ~10-30% of the patch, enabling YOLO to learn small objects.
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
PROCESSED_DIR = "data/processed_patches"   # new directory for patches
IMAGES_DIR = os.path.join(PROCESSED_DIR, "images")
LABELS_DIR = os.path.join(PROCESSED_DIR, "labels")
PATCH_SIZE = 256   # width and height in pixels (can also try 320)
TRAIN_VAL_TEST_SPLIT = [0.7, 0.15, 0.15]  # train, val, test
RANDOM_SEED = 42
CLASS_ID = 0

# Lung window for CT (center -600, width 1500) to enhance contrast
LUNG_CENTER = -600
LUNG_WIDTH = 1500
LOW = LUNG_CENTER - LUNG_WIDTH / 2
HIGH = LUNG_CENTER + LUNG_WIDTH / 2

random.seed(RANDOM_SEED)

# Create output directories
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)


# ---------- UTILITY FUNCTIONS ----------
def world_to_voxel(world_coord, origin, spacing):
    """Convert world coordinates (mm) to voxel indices (z,y,x)."""
    idx = (np.array(world_coord) - np.array(origin)) / np.array(spacing)
    return np.round(idx).astype(int)


def extract_patch(volume, center_voxel, patch_size, spacing):
    """
    Extract a square patch of size patch_size x patch_size centered at center_voxel (y,x).
    Returns the patch as 2D numpy array (uint8) and the bounding box coordinates (x_min, y_min, x_max, y_max) in patch coordinates.
    """
    cy, cx = center_voxel
    half = patch_size // 2
    y_min = cy - half
    y_max = cy + half
    x_min = cx - half
    x_max = cx + half
    
    # Get image dimensions
    h, w = volume.shape
    
    # Calculate padding if needed
    pad_top = max(0, -y_min)
    pad_bottom = max(0, y_max - h + 1)
    pad_left = max(0, -x_min)
    pad_right = max(0, x_max - w + 1)
    
    # Adjust crop coordinates to be within image
    y_min = max(0, y_min)
    y_max = min(h, y_max)
    x_min = max(0, x_min)
    x_max = min(w, x_max)
    
    # Crop the patch
    patch = volume[y_min:y_max, x_min:x_max]
    
    # Pad if necessary
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        patch = np.pad(patch, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    
    # Ensure patch size exactly patch_size x patch_size
    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        # Resize if needed (should not happen if padding correct)
        patch = np.resize(patch, (patch_size, patch_size))
    
    # Compute bounding box in patch coordinates (original nodule center should be at patch center)
    # The bounding box is the nodule diameter in pixels, centered at patch center.
    # We'll compute actual bounding box coordinates relative to patch.
    # Since the patch is centered at the nodule center, the nodule center in patch coordinates is (patch_size/2, patch_size/2)
    # But due to padding/cropping, we need to compute the exact mapping.
    # Instead, compute the original bounding box in full image coordinates and then translate to patch.
    # However simpler: We know the nodule center in voxel coordinates (cy, cx). The patch's top-left corner in original image is (y_min, x_min).
    # So center in patch = (cy - y_min, cx - x_min). We'll use that.
    center_y_patch = cy - y_min
    center_x_patch = cx - x_min
    # Compute diameter in pixels (using average spacing)
    pixel_spacing_y = spacing[1]
    pixel_spacing_x = spacing[2]
    pixel_spacing_avg = (pixel_spacing_y + pixel_spacing_x) / 2.0
    # diameter_mm comes from annotation; we have it passed as argument
    # We'll compute later
    return patch, (y_min, x_min)  # return patch and top-left corner


def create_yolo_annotation(patch_size, center_y, center_x, diameter_px):
    """
    Generate YOLO format line: class x_center y_center width height (normalized).
    Bounding box is square of side = diameter_px.
    """
    half_size = diameter_px / 2.0
    x_min = max(0, center_x - half_size)
    y_min = max(0, center_y - half_size)
    x_max = min(patch_size - 1, center_x + half_size)
    y_max = min(patch_size - 1, center_y + half_size)
    
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    if bbox_width <= 0 or bbox_height <= 0:
        return None
    
    x_center = (x_min + x_max) / 2.0 / patch_size
    y_center = (y_min + y_max) / 2.0 / patch_size
    width = bbox_width / patch_size
    height = bbox_height / patch_size
    
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
        # Only keep nodules with diameter >= 3mm (as per paper)
        if diameter_mm >= 3.0:
            nodules_by_series[seriesuid].append({
                'coord': (coord_x, coord_y, coord_z),
                'diameter_mm': diameter_mm
            })

print(f"Found {len(nodules_by_series)} series with nodules.")


# ---------- PROCESS EACH SERIES ----------
image_label_pairs = []  # list of (image_filename, label_filename)

subset_dirs = [d for d in os.listdir(RAW_DATA_DIR) if d.startswith("subset") and os.path.isdir(os.path.join(RAW_DATA_DIR, d))]

for subset_dir in sorted(subset_dirs):
    subset_path = os.path.join(RAW_DATA_DIR, subset_dir)
    mhd_files = [f for f in os.listdir(subset_path) if f.endswith('.mhd')]
    for mhd_file in tqdm(mhd_files, desc=f"Processing {subset_dir}"):
        seriesuid = mhd_file.replace('.mhd', '')
        if seriesuid not in nodules_by_series:
            continue
        
        # Load CT volume
        mhd_path = os.path.join(subset_path, mhd_file)
        img_sitk = sitk.ReadImage(mhd_path)
        volume = sitk.GetArrayFromImage(img_sitk)  # shape (z, y, x)
        origin = img_sitk.GetOrigin()
        spacing = img_sitk.GetSpacing()
        
        # Process each nodule in this series
        for nodule in nodules_by_series[seriesuid]:
            coord_world = nodule['coord']
            diameter_mm = nodule['diameter_mm']
            voxel_idx = world_to_voxel(coord_world, origin, spacing)
            slice_idx = voxel_idx[0]  # z
            if slice_idx < 0 or slice_idx >= volume.shape[0]:
                continue
            
            # Extract the axial slice
            slice_img = volume[slice_idx, :, :]  # shape (y, x)
            # Apply lung window
            slice_img = np.clip(slice_img, LOW, HIGH)
            slice_img = ((slice_img - LOW) / (HIGH - LOW)) * 255
            slice_img = slice_img.astype(np.uint8)
            
            # Center of nodule in voxel (y,x)
            center_y, center_x = voxel_idx[1], voxel_idx[2]
            
            # Extract patch
            half = PATCH_SIZE // 2
            y_min = max(0, center_y - half)
            y_max = min(slice_img.shape[0], center_y + half)
            x_min = max(0, center_x - half)
            x_max = min(slice_img.shape[1], center_x + half)
            
            # Extract patch
            patch = slice_img[y_min:y_max, x_min:x_max]
            
            # Pad if necessary to get exact PATCH_SIZE
            pad_top = max(0, half - center_y)
            pad_bottom = max(0, (center_y + half) - slice_img.shape[0])
            pad_left = max(0, half - center_x)
            pad_right = max(0, (center_x + half) - slice_img.shape[1])
            if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                patch = np.pad(patch, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
            
            # Ensure patch size is exactly PATCH_SIZE x PATCH_SIZE
            if patch.shape[0] != PATCH_SIZE or patch.shape[1] != PATCH_SIZE:
                # Resize as last resort
                from skimage.transform import resize
                patch = (resize(patch, (PATCH_SIZE, PATCH_SIZE)) * 255).astype(np.uint8)
            
            # Compute bounding box in patch coordinates
            # The nodule center in patch coordinates:
            center_y_patch = (center_y - y_min) + pad_top
            center_x_patch = (center_x - x_min) + pad_left
            # Compute diameter in pixels
            pixel_spacing_y = spacing[1]
            pixel_spacing_x = spacing[2]
            pixel_spacing_avg = (pixel_spacing_y + pixel_spacing_x) / 2.0
            diameter_px = diameter_mm / pixel_spacing_avg
            
            yolo_line = create_yolo_annotation(PATCH_SIZE, center_y_patch, center_x_patch, diameter_px)
            if yolo_line is None:
                continue
            
            # Save patch image
            img_filename = f"{seriesuid}_slice{slice_idx:04d}_nodule_{center_y}_{center_x}.png"
            img_path = os.path.join(IMAGES_DIR, img_filename)
            Image.fromarray(patch, mode='L').save(img_path)
            
            # Save label
            label_filename = img_filename.replace('.png', '.txt')
            label_path = os.path.join(LABELS_DIR, label_filename)
            with open(label_path, 'w') as lf:
                lf.write(yolo_line + '\n')
            
            image_label_pairs.append((img_filename, label_filename))

print(f"Total patches generated: {len(image_label_pairs)}")


# ---------- CREATE TRAIN/VAL/TEST SPLITS ----------
random.shuffle(image_label_pairs)
num_total = len(image_label_pairs)
num_train = int(TRAIN_VAL_TEST_SPLIT[0] * num_total)
num_val = int(TRAIN_VAL_TEST_SPLIT[1] * num_total)
num_test = num_total - num_train - num_val

train_pairs = image_label_pairs[:num_train]
val_pairs = image_label_pairs[num_train:num_train + num_val]
test_pairs = image_label_pairs[num_train + num_val:]

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
print("Preprocessing complete. Patch dataset ready for YOLO training.")