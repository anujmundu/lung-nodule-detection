# compare_models_detections.py
# Runs inference on multiple models and creates side‑by‑side detection comparison grids.
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import random

# Add YOLOv5 to path (adjust if needed)
sys.path.append('models/yolov5')

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes

# ============================================================================
# CONFIGURATION – CHANGE DATASET TO 'ct' OR 'xray'
# ============================================================================
MODELS = {
    'Baseline YOLOv5s': ('weights/baseline_patches_best.pt', (255, 0, 0)),      # Blue
    'YOLOv5-CASP (CT)': ('weights/casp_patches_best.pt', (0, 255, 0)),          # Green
    'YOLOv8s (CT)': ('weights/yolov8s_patches_best.pt', (0, 165, 255)),         # Orange
    'ASPP Only': ('weights/ablation_aspp_best.pt', (255, 255, 0)),              # Cyan
    'CoT3 Only': ('weights/ablation_cot3_best.pt', (255, 0, 255)),              # Magenta
    'CBAM Only': ('weights/ablation_cbam_best.pt', (128, 128, 128)),            # Gray
}

DATASET = 'ct'          # 'ct' or 'xray'
IMG_SIZE = 256 if DATASET == 'ct' else 640
CONF_THRESH = 0.25
IOU_THRESH = 0.45
NUM_SAMPLES = 4
LINE_THICKNESS = 3

if DATASET == 'ct':
    IMG_DIR = Path('data/processed_patches/images')
    LABEL_DIR = Path('data/processed_patches/labels')
    OUTPUT_DIR = Path('model_comparison_ct')
else:
    IMG_DIR = Path('data/x_nodule/test/images')
    LABEL_DIR = Path('data/x_nodule/test/labels')
    OUTPUT_DIR = Path('model_comparison_xray')

OUTPUT_DIR.mkdir(exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# FUNCTIONS
# ============================================================================
def load_model(weights_path):
    print(f"  Loading {Path(weights_path).name}...")
    model = DetectMultiBackend(weights_path, device=device)
    model.eval()
    return model

def load_and_preprocess_image(img_path, img_size):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized, ratio, pad = letterbox(img, img_size, auto=True, stride=32)
    img_tensor = img_resized.transpose((2, 0, 1))[::-1]
    img_tensor = np.ascontiguousarray(img_tensor)
    img_tensor = torch.from_numpy(img_tensor).to(device)
    img_tensor = img_tensor.float() / 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    return img_tensor, img, ratio, pad

def run_inference(model, img_tensor):
    with torch.no_grad():
        pred = model(img_tensor)
        pred = non_max_suppression(pred, CONF_THRESH, IOU_THRESH)
    return pred

def draw_boxes(img, detections, color, thickness=LINE_THICKNESS):
    img_copy = img.copy()
    for det in detections:
        if len(det):
            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
                label = f'{conf:.2f}'
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(img_copy, (x1, y1 - h - 4), (x1 + w + 4, y1), color, -1)
                cv2.putText(img_copy, label, (x1 + 2, y1 - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return img_copy

def load_ground_truth(img_path):
    label_path = LABEL_DIR / f"{img_path.stem}.txt"
    boxes = []
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])
                    img = cv2.imread(str(img_path))
                    h_img, w_img = img.shape[:2]
                    x1 = int((xc - w/2) * w_img)
                    y1 = int((yc - h/2) * h_img)
                    x2 = int((xc + w/2) * w_img)
                    y2 = int((yc + h/2) * h_img)
                    boxes.append((x1, y1, x2, y2))
    return boxes

def draw_ground_truth(img, gt_boxes, color=(255, 0, 0), thickness=2):
    img_copy = img.copy()
    for (x1, y1, x2, y2) in gt_boxes:
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(img_copy, "GT", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return img_copy

# ============================================================================
# LOAD MODELS
# ============================================================================
print("Loading models...")
models = {}
for name, (path, color) in MODELS.items():
    try:
        models[name] = (load_model(path), color)
    except Exception as e:
        print(f"  Failed to load {name}: {e}")

# ============================================================================
# SELECT TEST IMAGES
# ============================================================================
all_images = list(IMG_DIR.glob('*.png')) if DATASET == 'ct' else list(IMG_DIR.glob('*.jpg'))
valid_images = [img for img in all_images if (LABEL_DIR / f"{img.stem}.txt").exists()]
selected = random.sample(valid_images, min(NUM_SAMPLES, len(valid_images)))
print(f"Selected {len(selected)} test images with ground truth.\n")

# ============================================================================
# PROCESS EACH IMAGE
# ============================================================================
for idx, img_path in enumerate(selected, start=1):
    print(f"Processing image {idx}: {img_path.name}")
    img_name = img_path.stem
    gt_boxes = load_ground_truth(img_path)
    
    det_imgs = {}
    for name, (model, color) in models.items():
        img_tensor, original_img, ratio, pad = load_and_preprocess_image(img_path, IMG_SIZE)
        detections = run_inference(model, img_tensor)
        if len(detections[0]):
            detections[0][:, :4] = scale_boxes(img_tensor.shape[2:], detections[0][:, :4], original_img.shape).round()
        img_with_boxes = draw_boxes(original_img, detections, color)
        det_imgs[name] = img_with_boxes
        
        # Save individual image
        clean_name = f"img{idx}_{name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '')}.png"
        save_path = OUTPUT_DIR / clean_name
        cv2.imwrite(str(save_path), cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
    
    # Create grid
    model_names = list(det_imgs.keys())
    n_models = len(model_names)
    cols = 3
    rows = (n_models + 1 + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = axes.flatten() if rows*cols > 1 else [axes]
    
    # Ground truth
    img_original = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    img_gt = draw_ground_truth(img_original, gt_boxes)
    axes[0].imshow(img_gt)
    axes[0].set_title(f'Ground Truth ({len(gt_boxes)} nodules)', fontsize=10, color='red')
    axes[0].axis('off')
    
    # Models
    for i, name in enumerate(model_names):
        ax = axes[i+1] if i+1 < len(axes) else axes[i+1]
        ax.imshow(det_imgs[name])
        ax.set_title(name, fontsize=9)
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(len(model_names)+1, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Detection Comparison - Image {idx}', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'comparison_img{idx}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison_img{idx}.png and individual model images.")

print(f"\n✅ All comparisons saved to: {OUTPUT_DIR}")