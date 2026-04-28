# Medical Lung Nodule Detection

## Overview

This repository contains tools, models, notebooks, and evaluation workflows for lung nodule detection across multiple imaging modalities.  
The project covers:

- CT patch detection using the LUNA16 dataset  
- Chest X‑ray detection using the X‑Nodule dataset  
- Proof‑of‑concept synthetic MRI detection  
- Model comparisons between YOLOv5‑CASP, YOLOv8, and Faster R‑CNN  
- Analysis and visualization of detection performance and failure cases  

## Key Features

- Custom YOLOv5‑CASP architecture with attention (CBAM), multi‑scale context (ASPP), and transformer‑style (CoT3) enhancements  
- Pretrained models for CT, X‑ray, and synthetic MRI detection  
- End‑to‑end detection and evaluation notebooks  
- Scripts for synthetic dataset generation and COCO conversion  
- Visual analysis tools for performance comparison and error analysis  

## Repository Structure

```
.
├── data/                          # Dataset definitions, processed images/labels
├── weights/                       # Trained model checkpoints
├── models/yolov5/                 # Bundled YOLOv5 repository
├── detection_results/             # Prediction outputs and visualizations
├── evaluation_results/            # Plots and metrics
├── notebooks/                     # Analysis notebooks
├── src/                           # Custom modules (ASPP, attention, CoT, etc.)
├── compare_models_detections.py   # Side-by-side model comparisons
├── create_synthetic_mri_detection.py  # Synthetic MRI dataset generation
├── convert_to_coco.py             # YOLO to COCO format conversion
├── generate_advanced_visualizations.py
├── generate_modality_comparison.py
├── train_yolov8_casp.py           # YOLOv8 training pipeline
├── train_faster_rcnn.py           # Faster R-CNN training script
├── evaluate_mri_classifier.py     # MRI classifier evaluation
├── detection_evaluation_complete.ipynb
├── detection_evaluation.ipynb
├── requirements.txt
└── README.md
```


## Requirements

- Python 3.9+ recommended  
- Install core dependencies:

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\Activate.ps1
# On macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
pip install -r models/yolov5/requirements.txt
```

## Usage

### Detection Evaluation

Open and run the following Jupyter notebooks:

- `detection_evaluation_complete.ipynb` – Full pipeline with all modalities
- `detection_evaluation.ipynb` – Focused evaluation workflow

These notebooks perform detection, evaluation, and generate key visualizations for CT, X-ray, and synthetic MRI.

### Run YOLOv5 Detection (X-Ray)

```bash
python detect.py \
  --weights ../../weights/casp_x_nodule_best.pt \
  --img 640 \
  --conf 0.25 \
  --source ../../data/x_nodule/test/images \
  --save-txt \
  --save-conf \
  --project ../../detection_results \
  --name xray_casp
```

### Run CT Patch Detection

```bash
cd models/yolov5
python detect.py \
  --weights ../../weights/casp_patches_best.pt \
  --img 256 \
  --conf 0.25 \
  --source ../../data/processed_patches/images \
  --save-txt \
  --save-conf \
  --project ../../detection_results \
  --name ct_casp \
  --nosave
```

### Run Synthetic MRI Detection

```bash
cd models/yolov5
python detect.py \
  --weights ../../weights/casp_mri_synthetic_best.pt \
  --img 640 \
  --conf 0.25 \
  --source ../../data/mri_synthetic/val/images \
  --save-txt \
  --save-conf \
  --project ../../detection_results \
  --name mri_synthetic \
  --exist-ok
```

### Create Synthetic MRI Detection Dataset

```bash
python create_synthetic_mri_detection.py
```

### Convert YOLO Labels to COCO

```bash
python convert_to_coco.py
```

## Notes

- The main inference code lives inside `models/yolov5/`, which is a cloned YOLOv5 repository.
- Some top-level helper scripts rely on dataset paths under `data/`.
- Use the notebooks and visualization scripts to reproduce evaluation figures.
- Large datasets and model weights are excluded from version control (see `.gitignore`).

## GitHub Push Instructions

1. **Initialize the repository** (if not already done):

```bash
git init
```

2. **Add all files and commit**:

```bash
git add .
git commit -m "Initial commit: lung nodule detection project"
```

3. **Create a repository on GitHub** (do not add README or .gitignore during creation). Then add the remote and push:

```bash
git remote add origin https://github.com/<your-username>/<your-repo>.git
git branch -M main
git push -u origin main
```

Replace `<your-username>` and `<your-repo>` with your actual GitHub username and repository name.

## Author

**Anuj Mundu**  
MCA Student, MANIT Bhopal  
Batch: 2023–2026

### License

MIT License

Copyright (c) 2025 Anuj Mundu (MANIT Bhopal)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
