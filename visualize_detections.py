# visualize_detections.py
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import sys

# Add YOLOv5 to path
sys.path.append('models/yolov5')

# YOLOv5 imports
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes

# Configuration
weights_path = 'weights/casp_x_nodule_best.pt'
test_images_dir = Path('data/x_nodule/test/images')
output_dir = Path('visualization_results')
output_dir.mkdir(exist_ok=True)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model
print(f"Loading model from: {weights_path}")
model = DetectMultiBackend(weights_path, device=device)
model.eval()
conf_threshold = 0.25
iou_threshold = 0.45
print("Model loaded successfully!\n")

def load_and_preprocess_image(img_path, img_size=640):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized, ratio, pad = letterbox(img, img_size, auto=True, stride=model.stride)
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
        pred = non_max_suppression(pred, conf_threshold, iou_threshold)
    return pred

def draw_boxes(img, detections):
    img_copy = img.copy()
    for det in detections:
        if len(det):
            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'Nodule: {conf:.2f}'
                cv2.putText(img_copy, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img_copy

# Find test images
test_images = list(test_images_dir.glob('*.jpg'))
print(f"Found {len(test_images)} test images")

# Select random images
num_samples = min(8, len(test_images))
selected_images = random.sample(test_images, num_samples)

print(f"\nProcessing {num_samples} sample images...")
print("="*60)

results = []

for i, img_path in enumerate(selected_images):
    print(f"\n{i+1}. Processing: {img_path.name}")
    
    # Run inference
    img_tensor, original_img, ratio, pad = load_and_preprocess_image(img_path)
    detections = run_inference(model, img_tensor)
    
    # Scale boxes back
    if len(detections[0]):
        detections[0][:, :4] = scale_boxes(img_tensor.shape[2:], detections[0][:, :4], original_img.shape).round()
    
    # Create side-by-side visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    # Image with detections
    if len(detections[0]):
        img_with_boxes = draw_boxes(original_img, detections)
        axes[1].imshow(img_with_boxes)
        axes[1].set_title(f'Detections: {len(detections[0])} nodules', fontsize=14)
        print(f"  ✓ Found {len(detections[0])} nodule(s)")
        results.append({'image': img_path.name, 'detections': len(detections[0]), 'status': 'Success'})
    else:
        axes[1].imshow(original_img)
        axes[1].set_title('No detections found', fontsize=14)
        print(f"  ✗ No nodules detected")
        results.append({'image': img_path.name, 'detections': 0, 'status': 'No detection'})
    
    axes[1].axis('off')
    plt.tight_layout()
    
    # Save figure
    save_path = output_dir / f'result_{img_path.stem}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()

# Print summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for r in results:
    print(f"{r['image']}: {r['detections']} nodule(s) - {r['status']}")

print(f"\n✓ All visualizations saved to: {output_dir}")
print(f"Total images processed: {len(results)}")