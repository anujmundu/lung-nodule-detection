# create_synthetic_mri_detection.py
import os
import shutil
import random
from pathlib import Path
from PIL import Image

# Configuration
SOURCE_DIR = Path('data/mri_dataset/train')
CANCER_DIR = SOURCE_DIR / 'cancer'
NO_CANCER_DIR = SOURCE_DIR / 'no_cancer'

DEST_DIR = Path('data/mri_detection_synthetic')
TRAIN_IMG = DEST_DIR / 'images' / 'train'
TRAIN_LBL = DEST_DIR / 'labels' / 'train'
VAL_IMG = DEST_DIR / 'images' / 'val'
VAL_LBL = DEST_DIR / 'labels' / 'val'

# Create directories
for d in [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL]:
    d.mkdir(parents=True, exist_ok=True)

# Get image lists
cancer_imgs = list(CANCER_DIR.glob('*.jpg')) + list(CANCER_DIR.glob('*.png')) + list(CANCER_DIR.glob('*.jpeg'))
no_cancer_imgs = list(NO_CANCER_DIR.glob('*.jpg')) + list(NO_CANCER_DIR.glob('*.png')) + list(NO_CANCER_DIR.glob('*.jpeg'))

print(f"Found {len(cancer_imgs)} cancer, {len(no_cancer_imgs)} no_cancer images")

random.seed(42)
random.shuffle(cancer_imgs)
random.shuffle(no_cancer_imgs)

# Split: 30 cancer + 10 no_cancer for train; 8 cancer + 8 no_cancer for val
train_cancer = cancer_imgs[:30]
val_cancer = cancer_imgs[30:38]
train_no_cancer = no_cancer_imgs[:10]
val_no_cancer = no_cancer_imgs[10:18]

def process_images(img_list, img_dest, lbl_dest, has_nodule=True):
    for img_path in img_list:
        # Copy image
        shutil.copy(img_path, img_dest / img_path.name)
        
        # Create label file
        label_path = lbl_dest / (img_path.stem + '.txt')
        if has_nodule:
            # Synthetic box: centered, covering ~30% of image
            # YOLO format: class x_center y_center width height (normalized)
            with open(label_path, 'w') as f:
                f.write("0 0.5 0.5 0.3 0.3\n")
        else:
            # Empty file for no_cancer
            open(label_path, 'w').close()

# Process training set
process_images(train_cancer, TRAIN_IMG, TRAIN_LBL, has_nodule=True)
process_images(train_no_cancer, TRAIN_IMG, TRAIN_LBL, has_nodule=False)

# Process validation set
process_images(val_cancer, VAL_IMG, VAL_LBL, has_nodule=True)
process_images(val_no_cancer, VAL_IMG, VAL_LBL, has_nodule=False)

print(f"\nSynthetic dataset created!")
print(f"Train: {len(train_cancer)} cancer + {len(train_no_cancer)} no_cancer")
print(f"Val: {len(val_cancer)} cancer + {len(val_no_cancer)} no_cancer")