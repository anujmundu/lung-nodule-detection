# convert_to_coco.py
import json
import os
from PIL import Image

# Configuration - update these paths to match your dataset
BASE_DIR = 'data/processed_patches'
IMG_DIR = os.path.join(BASE_DIR, 'images')
LABEL_DIR = os.path.join(BASE_DIR, 'labels')

# Split files (these contain absolute or relative paths to images)
SPLITS = {
    'train': os.path.join(BASE_DIR, 'train.txt'),
    'val': os.path.join(BASE_DIR, 'val.txt'),
    'test': os.path.join(BASE_DIR, 'test.txt')
}

def convert_yolo_to_coco(split_name, split_file_path):
    """Convert YOLO labels in a split to COCO JSON format."""
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "nodule"}]
    }
    ann_id = 1
    img_id = 0
    
    with open(split_file_path, 'r') as f:
        for line in f:
            img_path = line.strip()
            # Extract just the filename (handle both absolute and relative paths)
            img_filename = os.path.basename(img_path)
            img_full_path = os.path.join(IMG_DIR, img_filename)
            
            # Check if image exists
            if not os.path.exists(img_full_path):
                print(f"Warning: Image not found: {img_full_path}")
                continue
            
            # Get image dimensions
            try:
                with Image.open(img_full_path) as img:
                    w, h = img.size
            except Exception as e:
                print(f"Error reading {img_full_path}: {e}")
                continue
            
            # Add image entry
            coco_data["images"].append({
                "id": img_id,
                "file_name": img_filename,
                "width": w,
                "height": h
            })
            
            # Load corresponding label file
            label_filename = img_filename.replace('.png', '.txt')
            label_path = os.path.join(LABEL_DIR, label_filename)
            if not os.path.exists(label_path):
                print(f"Warning: Label file not found: {label_path}")
                img_id += 1
                continue
            
            with open(label_path, 'r') as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, x_center, y_center, box_w, box_h = map(float, parts)
                    # Convert YOLO format (normalized) to absolute coordinates
                    x = (x_center - box_w/2) * w
                    y = (y_center - box_h/2) * h
                    width = box_w * w
                    height = box_h * h
                    
                    # Add annotation
                    coco_data["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "bbox": [x, y, width, height],
                        "area": width * height,
                        "category_id": 0,
                        "iscrowd": 0
                    })
                    ann_id += 1
            
            img_id += 1
    
    # Save JSON
    output_path = os.path.join(BASE_DIR, f'{split_name}_coco.json')
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    print(f"Saved {output_path} with {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations.")

if __name__ == '__main__':
    for split_name, split_file in SPLITS.items():
        if os.path.exists(split_file):
            convert_yolo_to_coco(split_name, split_file)
        else:
            print(f"Split file not found: {split_file}")
            