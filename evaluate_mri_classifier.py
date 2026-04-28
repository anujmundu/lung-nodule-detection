# evaluate_mri_classifier.py
import torch
import sys
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

sys.path.append('models/yolov5')
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes
import cv2
from tqdm import tqdm

# Configuration
weights_path = 'runs/classify/mri_classifier3/weights/best.pt'
data_root = Path('data/mri_dataset')
img_size = 224
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = DetectMultiBackend(weights_path, device=device)
model.eval()

# Get class names from data folder
class_names = sorted([d.name for d in (data_root / 'val').iterdir() if d.is_dir()])
print(f"Classes: {class_names}")

# Collect all validation images and labels
val_dir = data_root / 'val'
image_paths = []
true_labels = []

for label, class_name in enumerate(class_names):
    class_dir = val_dir / class_name
    for img_path in class_dir.glob('*.*'):
        image_paths.append(img_path)
        true_labels.append(label)

print(f"Total validation images: {len(image_paths)}")

# Function to preprocess image
def preprocess_image(img_path):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize to 224x224 (simple resize, not letterbox for classification)
    img = cv2.resize(img, (img_size, img_size))
    img_tensor = torch.from_numpy(img).permute(2,0,1).float() / 255.0
    return img_tensor.unsqueeze(0).to(device)

# Run inference
predictions = []
confidences = []

print("Running inference...")
for img_path in tqdm(image_paths):
    img_tensor = preprocess_image(img_path)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
        predictions.append(pred.item())
        confidences.append(conf.item())

# Calculate metrics
print("\n" + "="*50)
print("Classification Report:")
print("="*50)
print(classification_report(true_labels, predictions, target_names=class_names))

print("\n" + "="*50)
print("Confusion Matrix:")
print("="*50)
cm = confusion_matrix(true_labels, predictions)
print(cm)

# Calculate accuracy manually
accuracy = np.sum(np.array(predictions) == np.array(true_labels)) / len(true_labels)
print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

# Per-class metrics
from sklearn.metrics import precision_score, recall_score, f1_score
precision = precision_score(true_labels, predictions, average=None)
recall = recall_score(true_labels, predictions, average=None)
f1 = f1_score(true_labels, predictions, average=None)

print("\n" + "="*50)
print("Per-class Metrics:")
print("="*50)
for i, name in enumerate(class_names):
    print(f"{name:12} | Precision: {precision[i]:.4f} | Recall: {recall[i]:.4f} | F1: {f1[i]:.4f}")
    
# Save report to file
with open('mri_classification_report.txt', 'w') as f:
    f.write("Classification Report:\n")
    f.write(classification_report(true_labels, predictions, target_names=class_names))
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(cm))
    f.write(f"\n\nOverall Accuracy: {accuracy:.4f}\n")