# train_yolov8_casp.py
from ultralytics import YOLO
import torch

print("="*60)
print("YOLOv8-CASP Training")
print("="*60)

# Train YOLOv8-CASP on X-Nodule
print("\n1. Training YOLOv8-CASP on X-Nodule...")
model = YOLO('yolov8s_casp.yaml')
results = model.train(
    data='data/x_nodule_fixed.yaml',
    epochs=100,
    imgsz=640,
    batch=8,
    device=0,
    project='runs/train',
    name='yolov8_casp_x_nodule',
    verbose=True
)

print("\n✅ Training completed!")

# Validate the model
print("\n2. Validating YOLOv8-CASP...")
val_results = model.val(
    data='data/x_nodule_fixed.yaml',
    imgsz=640,
    batch=8,
    verbose=True
)

print("\n📊 Results saved to runs/train/yolov8_casp_x_nodule")