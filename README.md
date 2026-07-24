# Pneumonia Detection and Lung Region Segmentation using Deep Learning

<p align="center">

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)]()
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-green.svg)]()
[![U-Net](https://img.shields.io/badge/U--Net-Segmentation-orange.svg)]()
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-blue.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

</p>

---

# Overview

This project presents an end-to-end deep learning framework for **automatic pneumonia analysis from Chest X-ray images**.

The proposed system combines modern object detection and semantic segmentation techniques to assist radiologists in identifying pneumonia regions accurately.

The framework consists of two complementary models:

- **YOLOv8** for pneumonia localization (object detection)
- **U-Net** for lung region segmentation

Rather than relying on a single deep learning architecture, this research compares detection and segmentation performance to better understand the strengths and limitations of each approach in medical image analysis.

The project is developed as part of the MCA Thesis at

**Maulana Azad National Institute of Technology (MANIT), Bhopal**

---

# Objectives

The objectives of this work are:

- Detect pneumonia regions in Chest X-ray images
- Segment lung regions accurately
- Compare detection and segmentation approaches
- Evaluate different deep learning architectures
- Build a reproducible medical imaging pipeline
- Provide visualization and quantitative evaluation

---

# Project Pipeline

```
                Chest X-ray Image
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
     YOLOv8                     U-Net Segmentation
(Object Detection)           (Lung Segmentation)
        │                               │
        ▼                               ▼
 Bounding Boxes                 Pixel-wise Masks
        │                               │
        └───────────────┬───────────────┘
                        ▼
                Performance Evaluation
                        │
                        ▼
           Visualization & Comparative Study
```

---

# Features

- YOLOv8 based pneumonia detection
- U-Net based lung segmentation
- Automated preprocessing pipeline
- Data augmentation
- Training and validation scripts
- Evaluation metrics
- Inference utilities
- Visualization notebooks
- Modular project structure
- Easy dataset integration

---

# Repository Structure

```text
MEDICAL_LUNG_NODULE_DETECTION/
│
├── data/                                   # Datasets and processed data
│
├── detection_results/                      # Detection outputs
│
├── evaluation_results/                     # Evaluation metrics and reports
│
├── models/                                 # Model architectures and YOLOv5 repository
│
├── model_comparison_ct/                    # CT model comparison outputs
│
├── notebooks/                              # Jupyter notebooks
│
├── runs/                                   # YOLO training runs
│
├── src/                                    # Custom source modules
│
├── visualization_results/                  # Generated plots and visualizations
│
├── weights/                                # Pretrained model weights
│
├── compare_models_detections.py            # Compare model predictions
├── convert_to_coco.py                      # Convert annotations to COCO format
├── create_synthetic_mri_detection.py       # Generate synthetic MRI detection dataset
├── custom_modules.py                       # Custom deep learning modules
├── detect.py                               # Detection entry point
├── detection_evaluation.ipynb              # Detection evaluation notebook
├── detection_evaluation_complete.ipynb     # Complete evaluation workflow
├── environment.yml                         # Conda environment configuration
├── evaluate_mri_classifier.py              # MRI classifier evaluation
├── fix_paths.py                            # Dataset path correction utility
├── fix_paths_absolute.py                   # Absolute path correction utility
├── generate_advanced_visualizations.py     # Advanced visualization generation
├── generate_all_plots.py                   # Generate all evaluation plots
├── generate_modality_comparison.py         # Cross-modality comparison
├── generate_mri_synthetic_figure.py        # MRI synthetic visualization
├── preprocess_luna16.py                    # LUNA16 preprocessing
├── preprocess_luna16_patches.py            # CT patch generation
├── requirements.txt                        # Python dependencies
├── train.py                                # General training entry point
├── train_faster_rcnn.py                    # Faster R-CNN training
├── train_yolov8_casp.py                    # YOLOv8-CASP training
├── visualize_detections.py                 # Detection visualization
├── yolov8s_casp.yaml                       # YOLOv8-CASP configuration
├── README.md
└── .gitignore
```

---

# Technologies Used

- Python
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Albumentations
- scikit-image
- scikit-learn
- tqdm

---

# Datasets
Due to size and licensing, datasets are not stored in this repository.  
Download them from the following sources and place them under `data/`:

- [LUNA16 CT dataset](https://luna16.grand-challenge.org/) → `data/raw/` and `data/processed_patches/`
- [X‑Nodule chest X‑ray dataset](link) → `data/x_nodule/`
- Synthetic MRI dataset (provided via [Google Drive](link)) → `data/mri_detection_synthetic/`

Ensure the folder structure matches the repository layout before running scripts.


## Detection Dataset

Chest X-ray Pneumonia Dataset

Contains:

- Pneumonia
- Normal

Used for:

- YOLOv8 Training

---

## Segmentation Dataset

Montgomery + Shenzhen Chest X-ray Dataset

Contains:

- Chest X-ray Images
- Ground Truth Lung Masks

Used for:

- U-Net Training

---

# Dataset Directory

```
dataset/

├── detection/
│      ├── train/
│      ├── valid/
│      └── test/
│
├── segmentation/
│      ├── images/
│      └── masks/
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/<username>/medical-object-detection.git

cd medical-object-detection
```

Create virtual environment

```bash
python -m venv medod
```

Windows

```bash
medod\Scripts\activate
```

Linux/macOS

```bash
source medod/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Training

## Train YOLOv8

```bash
python scripts/train_yolo.py
```

---

## Train U-Net

```bash
python scripts/train_unet.py
```

---

# Inference

```bash
python scripts/inference.py
```

---

# Evaluation

```bash
python scripts/evaluate.py
```

---

# Evaluation Metrics

For Detection

- mAP@0.5
- Precision
- Recall
- F1 Score

For Segmentation

- Dice Coefficient
- IoU
- Precision
- Recall

---

# Results

The framework evaluates

- Detection accuracy
- Segmentation quality
- Localization performance
- Failure cases
- Comparative analysis

Visual outputs include

- Bounding box predictions
- Segmentation masks
- Overlay visualizations
- Metric plots

---

# Future Improvements

Potential extensions include:

- Attention U-Net
- YOLO11
- Swin Transformer
- SAM-assisted segmentation
- Multi-class thoracic disease detection
- Explainable AI (Grad-CAM)
- Clinical deployment pipeline

---

# Thesis Information

**Title**

> Pneumonia Detection and Lung Region Segmentation f

**Anuj Mundu**  
MCA Student, MANIT Bhopal  
Batch: 2023–2026

Copyright (c) 2026 Anuj Mundu (MANIT Bhopal)

# Future Improvements

Potential extensions include:

- Attention U-Net
- YOLO11
- Swin Transformer
- SAM-assisted segmentation
- Multi-class thoracic disease detection
- Explainable AI (Grad-CAM)
- Clinical deployment pipeline

---

# Thesis Information

**Title**

> Pneumonia Detection and Lung Region Segmentation from Chest X-ray Images using Deep Learning

**Degree**

Master of Computer Applications (MCA)

**Institute**

Maulana Azad National Institute of Technology (MANIT), Bhopal

Batch

2023–2026

---

# Author

**Anuj Mundu**

MCA Student

Maulana Azad National Institute of Technology (MANIT), Bhopal

GitHub:
https://github.com/<username>

LinkedIn:
https://linkedin.com/in/<username>

---

# License

MIT License

Copyright (c) 2026 Anuj Mundu


Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software.


The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.

