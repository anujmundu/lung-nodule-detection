# Medical Lung Nodule Detection

## Overview

This repository contains tools, models, notebooks, and evaluation workflows for lung nodule detection across multiple imaging modalities.
The project covers:
- CT patch detection using the LUNA16 dataset
- Chest X-ray detection using the X-Nodule dataset
- Proof-of-concept synthetic MRI detection
- Model comparisons between YOLOv5-CASP, YOLOv8, and Faster R-CNN
- Analysis and visualization of detection performance and failure cases

## Key Features

- Custom YOLOv5-CASP architecture with attention and context enhancements
- Pretrained models for CT, X-ray, and synthetic MRI detection
- End-to-end detection and evaluation notebooks
- Scripts for synthetic dataset generation and COCO conversion
- Visual analysis tools for performance comparison and error analysis

## Repository Structure

- `data/` - dataset definitions, processed images/labels, and raw sources
- `weights/` - trained model checkpoints used by evaluation scripts
- `models/yolov5/` - bundled YOLOv5 repository used for detection and training
- `detection_results/` - prediction outputs and saved detection visualizations
- `evaluation_results/` - plots, metrics, and generated figures
- `notebooks/` - analysis notebooks and training/evaluation workflows
- `model_comparison_ct/` - model comparison visuals for CT

Top-level scripts:
- `compare_models_detections.py` - side-by-side model detection comparisons
- `create_synthetic_mri_detection.py` - build synthetic MRI detection dataset
- `convert_to_coco.py` - convert YOLO labels to COCO JSON annotations
- `generate_*` - visualization and evaluation plotting utilities
- `train_yolov8_casp.py` - YOLOv8 training pipeline for CASP datasets
- `train_faster_rcnn.py` - Faster R-CNN training script for lung nodule detection
- `evaluate_mri_classifier.py` - MRI classifier evaluation utilities

## Requirements

- Python 3.9+ recommended
- Install core dependencies:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r models/yolov5/requirements.txt
```

## Usage

### Detection evaluation
Open and run:
- `detection_evaluation_complete.ipynb`
- `detection_evaluation.ipynb`

These notebooks perform detection, evaluation, and generate key visualizations for CT, X-ray, and synthetic MRI.

### Run YOLOv5 detection (X-ray)

```bash
cd models/yolov5
python detect.py --weights ../../weights/casp_x_nodule_best.pt --img 640 --conf 0.25 --source ../../data/x_nodule/test/images --save-txt --save-conf --project ../../detection_results --name xray_casp
```

### Run CT patch detection

```bash
cd models/yolov5
python detect.py --weights ../../weights/casp_patches_best.pt --img 256 --conf 0.25 --source ../../data/processed_patches/images --save-txt --save-conf --project ../../detection_results --name ct_casp --nosave
```

### Run synthetic MRI detection

```bash
cd models/yolov5
python detect.py --weights ../../weights/casp_mri_synthetic_best.pt --img 640 --conf 0.25 --source ../../data/mri_synthetic/val/images --save-txt --save-conf --project ../../detection_results --name mri_synthetic --exist-ok
```

### Create synthetic MRI detection dataset

```bash
python create_synthetic_mri_detection.py
```

### Convert YOLO labels to COCO

```bash
python convert_to_coco.py
```

## Notes

- The main inference code lives inside `models/yolov5/`, which is a cloned YOLOv5 repository.
- Some top-level helper scripts rely on dataset paths under `data/`.
- Use the notebooks and visualization scripts to reproduce evaluation figures.

## GitHub Push Instructions

1. Initialize the repository if necessary:

```bash
git init
```
2. Add files and commit:

```bash
git add .
git commit -m "Add project README and initialize repository"
```
3. Create a GitHub repository and add the remote:

```bash
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

Replace `<your-username>` and `<your-repo>` with your GitHub account and repository name.

## License

Add your preferred license here.
