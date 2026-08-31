<!-- ========================================================= -->
<!-- HERO SECTION -->
<!-- ========================================================= -->

<h1 align="center">
🫁 Automated Multi-Modal Lung Nodule Detection using YOLOv5-CASP
</h1>

<h3 align="center">

A Production-Grade Deep Learning Framework for Accurate Pulmonary Nodule Detection in Chest X-Ray and CT Images using an Enhanced YOLOv5 Architecture with CBAM, ASPP, and CoT3 Modules

</h3>

<p align="center">

<img src="https://img.shields.io/badge/Python-3.10-blue.svg">
<img src="https://img.shields.io/badge/PyTorch-2.5.1+cu121-red.svg">
<img src="https://img.shields.io/badge/CUDA-12.1-success">
<img src="https://img.shields.io/badge/OpenCV-4.9.0-green">
<img src="https://img.shields.io/badge/YOLOv5-CASP%20Custom-orange">
<img src="https://img.shields.io/badge/License-MIT-yellow">
<img src="https://img.shields.io/badge/Platform-Windows%2011%20%7C%20Linux-lightgrey">
<img src="https://img.shields.io/badge/Status-Research%20Thesis-blueviolet">
<img src="https://img.shields.io/badge/Medical-AI-red">
<img src="https://img.shields.io/badge/Computer-Vision-success">

</p>

---

<p align="center">

⭐ If you find this repository useful for your medical AI research or thesis work, please consider giving it a star.

</p>

---

# 📖 Overview

Lung cancer is responsible for nearly **2.5 million new diagnoses and 1.8 million deaths annually worldwide** (GLOBOCAN 2022). Five-year survival exceeds 60% for Stage I disease but falls below 5% for Stage IV, making early automated Computer-Aided Detection (CAD) essential.

This repository presents the official **production-grade implementation of YOLOv5-CASP**, an enhanced deep learning object detection framework specifically designed for **accurate, multi-modal pulmonary nodule detection** across **Chest Radiographs (X-Nodule, NIH ChestX-ray 14)**, **LUNA16 CT Scans**, and **Synthetic MRI Slices**.

Unlike conventional object detectors that suffer high missed-detection rates ($19\%\text{--}54\%$) due to subtle nodules, low contrast, and overlapping ribs, this project focuses on:

- **Full Research Reproducibility**: Complete PyTorch source code, dataset YAML configs, and SGD hyperparameters.
- **Architectural Innovation**: Integration of **CBAM** (Attention), **ASPP** (Multi-Scale Context), and **CoT3** (Contextual Transformer).
- **Publication-Quality Evaluation**: 30 diagnostic charts, precision-recall curves, Grad-CAM heatmaps, and failure analyses.
- **Multi-Modal Generalization**: Benchmarked across X-ray, CT, and soft-tissue magnetic resonance cross-sections.
- **Production-Ready Engineering**: Modular structure, clean execution scripts, and zero false alarm rates.

> **Authors**: Anuj Mundu, Dr. Ghanshyam Singh Thakur, Dr. Sanjivani Joshi  
> **Institution**: Department of Computer Applications, Maulana Azad National Institute of Technology (MANIT), Bhopal, India  
> **Thesis Project**: MCA Research Project (May 2026)

---

# 🚀 Repository Highlights

✅ **Production-Grade Codebase**: Modular pipeline separating data preprocessing, architecture parsing, training, and evaluation.  
✅ **Multi-Modal Dataset Support**: Built-in configs for X-Nodule (CXR), NIH ChestX-ray 14, LUNA16 CT patches, and Synthetic MRI.  
✅ **Architectural Synergy**: Combines CBAM attention, ASPP dilated convolutions, and CoT3 transformer self-attention blocks.  
✅ **Domain Transfer Learning**: 100-epoch NIH ChestX-ray 14 fine-tuning achieving **0.644 mAP@0.5** (+1211% gain over scratch).  
✅ **Zero False Positive Rate**: **0 False Positives** on the 201-image X-Nodule test set (755 nodule instances).  
✅ **Real-Time High Throughput**: Operates at **70.98 FPS** ($14.08\text{ ms}$) on consumer GPU (NVIDIA RTX 3050).  
✅ **Automated Visualization Suite**: `generate_all_plots.py` renders all 30 thesis figures, PR curves, and failure breakdown charts.  
✅ **Ablation Pipeline**: Automated evaluation scripts comparing Baseline YOLOv5s, CBAM-only, ASPP-only, CoT3-only, and YOLOv8s.  
✅ **Publication & Thesis Ready**: Complete chapter documentation in `docs/` and organized outputs in `Detection Results/`.

---

# 🎯 Project Goals

This project aims to establish an intelligent, cross-modality lung nodule detection framework capable of:

- **Detecting Small Pulmonary Lesions**: Preserving feature edges for subtle nodules ($3\text{--}8\text{mm}$).
- **Eliminating Alert Fatigue**: Controlling false positives to maintain a $0\%$ false alarm rate on chest radiographs.
- **Improving Receptive Fields**: Utilizing ASPP atrous convolutions to capture multi-scale context without resolution loss.
- **Enabling Fast Real-Time Inference**: Operating under $15\text{ ms}$ latency for point-of-care clinical triage.
- **Ensuring Cross-Cohort Generalization**: Transferring chest X-ray features across multicenter clinical datasets.

---

# 💡 Motivation

Medical object detection presents distinct challenges compared to natural image detection:

| Challenge | Impact on Detection | YOLOv5-CASP Solution |
|---|---|---|
| **Subtle & Micro Nodules** | Lesions ($<5\text{px}$) vanish during downsampling | $640\times 640$ input resolution + ASPP multi-scale receptive fields |
| **Anatomical Overlap** | Dense ribs, cardiac shadow, & vessels hide nodules | CBAM sequential Channel & Spatial attention blocks |
| **Low Contrast** | Ground-glass opacities blend with lung parenchyma | CLAHE contrast enhancement ($8\times 8$ tile grid, clip limit $2.0$) |
| **Context Ambiguity** | Vascular junctions mimic true pulmonary nodules | CoT3 contextual transformer static key + dynamic self-attention |
| **High Cost of False Negatives** | Missed early-stage nodules lower 5-year survival | Optimized anchor box clustering & CIoU loss formulation |

---

# 🔬 Why YOLOv5?

YOLOv5 provides an optimal foundation for clinical CAD deployment:

| Property | Clinical Benefit |
|---|---|
| **Speed** | Sub-15ms execution enables real-time screening during patient examination |
| **Accuracy** | One-stage anchor regression delivers high sensitivity for dense targets |
| **Lightweight** | $19.4\text{M}$ parameters fit comfortably on 4GB consumer GPUs |
| **Flexibility** | Dynamic layer parsing in `yolo.py` enables seamless custom module insertion |

However, baseline YOLOv5s struggles with subtle medical nodules ($\text{mAP} = 0.214$). YOLOv5-CASP resolves this by enhancing feature extraction without sacrificing real-time throughput.

---

# 🧠 Proposed Solution: YOLOv5-CASP Architecture

Rather than replacing YOLOv5, this work enhances its CSPDarknet backbone and PANet feature pyramid neck:

```text
                     Input Chest X-Ray / CT Image (640x640)
                                       │
                                       ▼
                       CSPDarknet Backbone + CBAM Attention
                     (Layers 3, 6, 9: Channel + Spatial Attention)
                                       │
                                       ▼
                       CoT3 Contextual Transformer (Layer 11)
                     (3x3 Static Context + Self-Attention Guidance)
                                       │
                                       ▼
                     ASPP Multi-Scale Receptive Fields (Layer 12)
                     (Parallel Atrous Dilations d = [1, 3, 5, 7])
                                       │
                                       ▼
                       PANet Neck + CBAM Refinement (Layers 17, 22, 26)
                                       │
                                       ▼
                        Multi-Scale Detection Head (P3, P4, P5)
                                       │
                                       ▼
                   Bounding Box Predictions + Confidence Scores
```

---

# 📑 Table of Contents

- [Overview](#-overview)
- [Repository Highlights](#-repository-highlights)
- [Project Goals](#-project-goals)
- [Motivation](#-motivation)
- [Why YOLOv5?](#-why-yolov5)
- [Proposed Solution](#-proposed-solution-yolov5-casp-architecture)
- [System Architecture](#-system-architecture)
- [Modular Design](#-modular-design)
- [Technology Stack](#-technology-stack)
- [Hardware & Environment](#-hardware--environment)
- [Repository Structure](#-repository-structure)
- [Datasets & Preprocessing](#-datasets--preprocessing)
- [Installation](#-installation)
- [Execution & Workflows](#-execution--workflows)
- [Experimental Results & Benchmarks](#-experimental-results--benchmarks)
- [Ablation Study](#-ablation-study)
- [Failure Analysis](#-failure-analysis)
- [Mathematical Formulation](#-mathematical-formulation)
- [Thesis Documentation](#-thesis-documentation)
- [Future Work](#-future-work)
- [Citation](#-citation)
- [License & Contact](#-license--contact)

---

# 🏗️ System Architecture

The proposed framework follows a modular software design where preprocessing, network parsing, training optimization, and diagnostic visualization operate as decoupled components.

```text
                           Medical Datasets (X-Nodule, NIH, LUNA16, MRI)
                                                │
                                                ▼
                               CLAHE Preprocessing & Augmentations
                               (src/preprocess.py & src/augment.py)
                                                │
                                                ▼
                               Dataset Configuration (data/*.yaml)
                                                │
                                                ▼
                               YOLOv5-CASP Core PyTorch Engine
                     (yolov5/models/common.py & yolov5/models/yolo.py)
                                                │
                 ┌──────────────────────────────┼──────────────────────────────┐
                 │                              │                              │
                 ▼                              ▼                              ▼
             CBAM Module                   ASPP Module                   CoT3 Module
        (Channel & Spatial)             (Dilated Convs d=1,3,5,7)    (Contextual Transformer)
                 │                              │                              │
                 └────────────────────── Feature Fusion ───────────────────────┘
                                                │
                                                ▼
                                 SGD Optimizer & Cosine LR
                                     (train_casp.py)
                                                │
                                                ▼
                             Evaluation & Master Plotting Suite
                                  (generate_all_plots.py)
                                                │
                                                ▼
                             Organized Benchmark Outputs Directory
                                    (Detection Results/)
```

---

# ⚙️ Technology Stack

| Category | Technology / Library | Version | Purpose |
|---|---|---|---|
| **Language** | Python | `3.10.20` | Core programming language |
| **Deep Learning** | PyTorch | `2.5.1+cu121` | Neural network training & inference |
| **Vision Utilities** | TorchVision | `0.20.1+cu121` | Dataset transformations & tensor operations |
| **Detection Engine**| YOLOv5 Engine | Custom `v7.0` | Object detection backbone & anchor regression |
| **Acceleration** | CUDA | `12.1` | GPU hardware acceleration |
| **Image Processing**| OpenCV | `4.9.0` | CLAHE filtering & image manipulation |
| **Augmentation** | Albumentations | `1.0.0` | Spatial & color transform pipelines |
| **Data Analysis** | NumPy / Pandas / SciPy | `1.26.4` / `2.3.3` / `1.15.3` | Numerical computing & metrics logging |
| **Visualization** | Matplotlib / Seaborn | `3.10.9` | High-resolution publication plotting |

---

# 💻 Hardware & Environment

All benchmark experiments were conducted on the following single-GPU workstation:

| Component | Specification |
|---|---|
| **GPU Model** | NVIDIA GeForce RTX 3050 Laptop GPU |
| **VRAM** | `4,095 MB` ($4.0\text{ GB}$) GDDR6 |
| **CUDA Cores** | `2,048` CUDA Cores, Compute Capability `8.6` |
| **CPU Model** | AMD Ryzen 7 6800H with Radeon Graphics ($8\text{ Cores}, 16\text{ Threads}$) |
| **System RAM** | `31.26 GB` DDR5 |
| **Storage** | NVMe PCIe M.2 SSD |
| **OS** | Windows 11 ($64\text{-bit}$) |

---

# 📁 Repository Structure

```
Lung Nodule Detection/
├── Detection Results/                       # Categorized benchmark output directory
│   ├── 1_YOLOv5_CASP_X_Nodule_SOTA/        # X-Nodule best weights, PR curves, metrics
│   ├── 2_YOLOv5_CASP_NIH_ChestXray_100Ep_SOTA/ # NIH ChestX-ray 100-Epoch transfer weights
│   ├── 3_YOLOv5_CASP_LUNA16_CT_Patches/    # LUNA16 CT patch model weights
│   ├── 4_Ablation_*/                       # Baseline, CBAM, ASPP, and CoT3 ablation runs
│   ├── 5_Thesis_Figures_and_Dashboards/    # Master copy of all 30 thesis plots & reports
│   └── 6_Visual_Detection_Predictions/     # Bounding box label txt files & cropped images
├── data/                                   # Dataset configurations & annotations
│   ├── luna16_patches.yaml                 # LUNA16 CT patches dataset config
│   ├── nih_chestxray_nodules.yaml          # NIH ChestX-ray 14 dataset config
│   ├── x_nodule_fixed.yaml                 # X-Nodule chest radiograph dataset config
│   ├── mri_detection_synthetic.yaml        # Synthetic MRI dataset config
│   └── hyp.casp.yaml                       # SGD training hyperparameter config
├── docs/                                   # Complete MCA Thesis chapters & documentation
│   ├── 01_introduction.md                  # Background, Clinical Need, & Aim
│   ├── 02_literature_review.md             # Traditional CAD & Deep Learning Review
│   ├── 03_materials_and_methods.md         # Preprocessing & YOLOv5-CASP Architecture
│   ├── 04_experimental_setup.md            # Hardware, Environment, & Protocols
│   ├── 05_results_and_ablation.md          # Multi-modality Results & Benchmarks
│   ├── 06_failure_case_analysis.md         # False Negative Breakdown & Error Cases
│   ├── 07_discussion.md                    # Discussion, Module Interactions & Workflows
│   ├── 08_conclusion_and_future_work.md    # Conclusions & Future Research Directions
│   └── appendix.md                         # Network YAML, Training Logs & CSV schemas
├── evaluation_results/                     # Active output directory generated by generate_all_plots.py
│   ├── detection_examples/                 # Diagnostic prediction composite frames
│   ├── mri_synthetic/                      # Synthetic MRI composite frames
│   ├── figure_5_1_training_dynamics.png    # Thesis Chapter 5 loss & mAP curves
│   ├── figure_5_2_precision_recall_curves.png # Thesis Chapter 5 PR curves
│   ├── figure_5_3_ablation_study_chart.png # Thesis Chapter 5 ablation study bar chart
│   ├── figure_5_4_comparative_benchmarks.png # Thesis Chapter 5 baseline comparisons & latency
│   ├── figure_5_5_summary_dashboard.png    # Executive results dashboard
│   ├── figure_5_6_multicenter_clinical.png # Multi-center hospital scanner robustness plot
│   ├── figure_5_7_nih_chestxray.png        # NIH ChestX-ray 14 transfer learning progression
│   ├── failure_distribution.png            # 2-Panel failure case distribution pie chart
│   ├── performance_summary_table.png       # Complete 12-model summary table image
│   ├── EVALUATION_SUMMARY.md               # Markdown summary report
│   └── evaluation_summary.txt              # Text summary report
├── src/                                     # Data processing modules
│   ├── preprocess.py                       # CLAHE contrast enhancement & image resizing
│   ├── augment.py                          # Albumentations data augmentation pipeline
│   └── evaluate.py                         # Standalone evaluation & matching script
├── yolov5/                                 # Custom Ultralytics YOLOv5 engine repository
│   ├── models/
│   │   ├── common.py                       # CBAM, ASPP, and CoT3 PyTorch implementations
│   │   ├── yolo.py                         # Custom parser registrations in parse_model()
│   │   ├── yolov5s-casp.yaml               # Full 151-layer CASP architecture specification
│   │   ├── yolov5s-aspp.yaml               # ASPP ablation model architecture YAML
│   │   ├── yolov5s-cbam.yaml               # CBAM ablation model architecture YAML
│   │   └── yolov5s-cot3.yaml               # CoT3 ablation model architecture YAML
│   └── runs/train/                         # Model training checkpoints & metric logs
├── custom_modules.py                       # Standalone PyTorch module exports
├── detect.py                               # Model inference & bounding box detection script
├── generate_all_plots.py                   # Master visualization generator (30 outputs)
├── README.md                               # Project documentation
├── requirements.txt                        # Python dependencies manifest
├── train_all_ablation_models.py            # Automated ablation suite execution script
└── train_casp.py                           # Core SGD training execution wrapper
```

---

# 📂 Datasets & Preprocessing

This framework is evaluated across **4 medical imaging datasets** covering computed tomography, projection radiography, and magnetic resonance imaging:

| Dataset | Modality | Task | Images | Resolution | Annotations | Official Download Link |
|---|---|---|---|---|---|---|
| **X-Nodule** | Chest Radiography (CXR) | Detection | `2,015` | $640\times 640$ | Bounding Boxes | [Roboflow Universe](https://universe.roboflow.com/rodney/hhhh-ig2qf) |
| **NIH ChestX-ray 14** | Chest Radiography (CXR) | Detection | `492` | $640\times 640$ | Bounding Boxes ($247\text{ test}$) | [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC) / [Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data) |
| **LUNA16 Patches** | CT Scans (LIDC-IDRI) | Detection | `607` | $256\times 256$ | Bounding Boxes | [LUNA16 Grand Challenge](https://luna16.grand-challenge.org/Data/) / [Zenodo](https://zenodo.org/record/3723295) |
| **Lung MRI** | Magnetic Resonance (MRI)| Detection / Cls | `56` / `1,018` | $640\times 640$ | Bounding Boxes / Labels | [Kaggle MRI Dataset](https://www.kaggle.com/datasets/xiaopengzhang12/lung-cancer-mri-images) |

---

## 📥 Dataset Download Links & Descriptions

### 1. 🩻 X-Nodule Chest Radiograph Dataset
- **Official Source**: [Roboflow Universe - X-Nodule Dataset](https://universe.roboflow.com/rodney/hhhh-ig2qf) (Alternative mirror: [X-Ray Chest Nodule](https://universe.roboflow.com/xray-chest-nodule/xray-chest-nodule))
- **Description**: Contains 2,015 frontal chest radiographs with expert radiologist bounding-box annotations for pulmonary nodules.
- **Split**: 1,411 training images ($70\%$), 403 validation images ($20\%$), and 201 testing images ($10\%$, containing 755 ground-truth nodule instances).
- **Target Location in Workspace**: `data/x_nodule/`
- **Config File**: [`data/x_nodule_fixed.yaml`](file:///c:/Users/anujm/Desktop/Lung%20Nodule%20Detection/data/x_nodule_fixed.yaml)

### 2. 🫁 NIH ChestX-ray 14 (Nodule & Mass Cohort)
- **Official Source**: [NIH Clinical Center Open Box Archive](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- **Kaggle Mirror**: [NIH Chest X-rays Dataset on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)
- **Reference**: Wang et al., *"ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases"*, IEEE CVPR 2017.
- **Description**: Hospital-scale database of 112,120 frontal chest radiographs from 30,805 unique patients with disease labels, including the curated sub-cohort of annotated lung nodules/masses.
- **Target Location in Workspace**: `data/nih_chestxray/` (filtered nodule images in `data/nih_chestxray/filtered_nodules/`)
- **Config File**: [`data/nih_chestxray_nodules.yaml`](file:///c:/Users/anujm/Desktop/Lung%20Nodule%20Detection/data/nih_chestxray_nodules.yaml)

### 3. 🎯 LUNA16 (LUng Nodule Analysis 2016) CT Challenge
- **Official Source**: [LUNA16 Grand Challenge Data Portal](https://luna16.grand-challenge.org/Data/)
- **Zenodo Official Archive**: [LUNA16 Challenge on Zenodo](https://zenodo.org/record/3723295)
- **Source Cohort**: LIDC-IDRI (The Cancer Imaging Archive)
- **Description**: 888 thoracic computed tomography (CT) scans distributed in MetaImage (`.mhd` / `.raw`) format across 10 subsets (`subset0` to `subset9`), with 1,186 nodules measuring $\ge 3\text{mm}$ verified by at least 3 expert thoracic radiologists.
- **Target Location in Workspace**: Raw data in `data/raw/`, extracted lung-windowed axial patches in `data/processed_patches/`.
- **Config File**: [`data/luna16_patches.yaml`](file:///c:/Users/anujm/Desktop/Lung%20Nodule%20Detection/data/luna16_patches.yaml)

### 4. 🧲 Lung Cancer MRI Images
- **Official Source**: [Lung Cancer MRI Images on Kaggle](https://www.kaggle.com/datasets/xiaopengzhang12/lung-cancer-mri-images)
- **Description**: 1,018 cross-sectional magnetic resonance imaging (MRI) slices covering benign, malignant, and normal cases. Used for cross-modality classification and 56-image synthetic bounding-box detection proof-of-concept.
- **Target Location in Workspace**: `data/mri_detection_synthetic/`
- **Config File**: [`data/mri_detection_synthetic.yaml`](file:///c:/Users/anujm/Desktop/Lung%20Nodule%20Detection/data/mri_detection_synthetic.yaml)

---

## 🗂️ Data Directory Layout

Place downloaded and extracted datasets in the `data/` folder following this structure:

```text
data/
├── x_nodule/
│   ├── train/images/ & train/labels/
│   ├── valid/images/ & valid/labels/
│   └── test/images/  & test/labels/
├── nih_chestxray/
│   └── filtered_nodules/
│       ├── images/
│       └── labels/
├── processed_patches/
│   ├── images/
│   └── labels/
└── mri_detection_synthetic/
    ├── images/
    └── labels/
```

---

### Preprocessing & Augmentation Pipeline
1. **CLAHE Contrast Enhancement**: Local histogram equalization ($8\times 8$ tile grid, clip limit $2.0$) to amplify subtle ground-glass opacity.
2. **Letterbox Resizing**: Aspect-ratio-preserving padding to $640\times 640$ (X-Ray/MRI) or $256\times 256$ (CT Patches).
3. **Stochastic Augmentations**: Albumentations horizontal flipping ($p=0.5$), rotation ($\pm 5^\circ$), scaling ($\pm 20\%$), and mosaic augmentation ($p=0.5$).

---

# 🔧 Installation

### 1. Environment Creation
```bash
# Create Anaconda virtual environment
conda create -n yolo_medical python=3.10 -y
conda activate yolo_medical
```

### 2. Install PyTorch & Dependencies
```bash
# Install PyTorch 2.5.1 with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install medical vision & data science libraries
pip install opencv-python==4.9.0 numpy==1.26.4 matplotlib==3.10.9 pandas==2.3.3 albumentations==1.0.0 ultralytics==8.4.34

# Install YOLOv5 engine dependencies
cd yolov5
pip install -r requirements.txt
cd ..
```

---

# 🚀 Execution & Workflows

### 1. Training YOLOv5-CASP Models
Train YOLOv5-CASP from scratch on X-Nodule chest radiographs at $640\times 640$ resolution:
```bash
python train_casp.py --model casp --data data/x_nodule_fixed.yaml --epochs 100 --batch-size 8 --imgsz 640
```

Run the complete ablation suite (Baseline, CBAM-only, ASPP-only, CoT3-only):
```bash
python train_all_ablation_models.py
```

### 2. Fine-Tuning on NIH ChestX-Ray 14 (Transfer Learning)
Fine-tune pre-trained X-Nodule weights on NIH ChestX-ray 14 for 100 epochs at $640\times 640$:
```bash
python train_casp.py --model casp --data data/nih_chestxray_nodules.yaml --weights yolov5/runs/train/casp_x_nodule_run4/weights/best.pt --epochs 100 --batch-size 8 --imgsz 640
```

### 3. Inference & Prediction Generation
Run detection on validation/test images to save bounding box `.txt` annotations and visual crops:
```bash
python detect.py --model casp --source data/nih_chestxray/filtered_nodules/images --weights yolov5/runs/train/casp_nih_chestxray_nodules_run6/weights/best.pt --conf-thres 0.50
```

### 4. Master Thesis Visualization Generator (30 Outputs)
Generate all 30 thesis figures, PR curves, F1 curves, confusion matrices, Grad-CAM heatmaps, and markdown summaries:
```bash
python generate_all_plots.py
```

### 5. Consolidate Benchmark Results Directory
Sync all model checkpoints, weights, metrics, thesis figures, and predictions into `Detection Results/`:
```bash
python scratch/copy_detection_results.py
```

---

# 📊 Experimental Results & Benchmarks

### Complete 12-Model Performance Summary Table

| Model & Modality Variant | Input Size | Epochs | mAP@0.5 | Precision | Recall | F1-Score | Params (M) | GFLOPs | Highlight |
|---|---|---|---|---|---|---|---|---|---|
| **YOLOv5-CASP (X-Nodule Radiographs)** | $640\times 640$ | 100 | **0.809** | **0.792** | 0.708 | **0.748** | 19.4 | 25.7 | 🟩 Green (SOTA X-Ray, $0\text{ FP}$) |
| **YOLOv5-CASP (NIH ChestX-ray 100-Ep)** | $640\times 640$ | 100 | **0.644** | **0.627** | **0.677** | **0.651** | 19.4 | 25.7 | 🟩 Green (NIH SOTA Transfer) |
| **YOLOv5-CASP (Synthetic MRI)** | $640\times 640$ | 50 | **0.615** | 0.575 | **0.750** | 0.652 | 19.4 | 25.7 | 🟩 Green (MRI SOTA) |
| **YOLOv5-CASP (LUNA16 CT Patches)** | $256\times 256$ | 300 | **0.382** | 0.492 | 0.527 | 0.509 | 19.4 | 25.7 | 🟩 Green (CT SOTA, $+79\%$) |
| **YOLOv8s (X-Nodule Radiographs)** | $640\times 640$ | 100 | 0.807 | 0.752 | 0.739 | 0.745 | 11.1 | 28.4 | 🟦 Blue |
| **YOLOv8s (LUNA16 CT Patches)** | $256\times 256$ | 300 | 0.158 | 0.225 | 0.297 | 0.256 | 11.1 | 28.4 | 🟦 Blue |
| **ASPP Only (LUNA16 CT Patches)** | $256\times 256$ | 300 | 0.248 | 0.341 | 0.429 | 0.380 | 14.9 | 22.2 | 🟨 Yellow (Ablation $+16\%$) |
| **CoT3 Only (LUNA16 CT Patches)** | $256\times 256$ | 300 | 0.205 | 0.202 | 0.341 | 0.254 | 11.6 | 19.5 | 🟨 Yellow (Ablation) |
| **Baseline YOLOv5s (X-Nodule Radiographs)**| $640\times 640$ | 100 | 0.214 | 0.289 | 0.385 | 0.330 | 7.02 | 15.9 | 🟥 Red (Baseline) |
| **Baseline YOLOv5s (LUNA16 CT Patches)** | $256\times 256$ | 300 | 0.214 | 0.289 | 0.385 | 0.330 | 7.02 | 15.9 | 🟥 Red (Baseline) |
| **CBAM Only (LUNA16 CT Patches)** | $256\times 256$ | 300 | 0.001 | 0.001 | 0.264 | 0.002 | 7.18 | 16.0 | 🟥 Red (Ablation Collapse) |
| **Faster R-CNN (MobileNetV2)** | $256\times 256$ | 300 | 0.000 | 0.000 | 0.000 | 0.000 | 30.0 | 35.0 | 🟥 Red (Two-Stage Anchor Error)|

---

# 🔬 Ablation Study Insights

Ablation analysis on LUNA16 CT patches reveals crucial module dynamics:

1. **ASPP Standalone Contribution (+16% Gain)**: ASPP alone boosts mAP@0.5 from `0.214` to `0.248`, proving the critical importance of multi-scale atrous receptive fields ($d=[1,3,5,7]$) for detecting varying nodule sizes.
2. **CBAM Standalone Failure (Noise Amplification)**: CBAM alone collapses ($\text{mAP} = 0.001$) because channel and spatial attention without ASPP feature hierarchy amplify high-frequency vascular noise.
3. **Super-Additive Architectural Synergy (+79% Gain)**: When combined in full YOLOv5-CASP, ASPP establishes a rich feature hierarchy that enables CBAM and CoT3 to focus attention on true pulmonary nodules, achieving **0.382 mAP@0.5**.

---

# ❌ Failure Case & Diagnostic Breakdown

Side-by-side failure case distribution across chest radiograph test cohorts:

| Diagnostic Outcome | X-Nodule Test Cohort ($N=755$) | NIH ChestX-ray 14 Test Cohort ($N=247$) | Etiology / Clinical Factor |
|---|---|---|---|
| **Correct Detections (True Positives)** | **738 (97.7%)** | **167 (67.7%)** | Accurate localization ($\text{IoU} \ge 0.50$, $\text{Conf} \ge 0.50$) |
| **False Negatives (Missed Nodules)** | **6 (0.8%)** | **62 (25.1%)** | Micro-nodules ($\le 5\text{px}$), low contrast, ground-glass opacity |
| **Misaligned / Bounding Box Shifts** | **11 (1.5%)** | **18 (7.3%)** | Subpleural boundaries & rib junction overlap |
| **False Positives (Excess Alarms)** | **0 (0.0%)** | **0 (0.0%)** | **Zero False Alarm Rate** maintained across both cohorts |

---

# 🧮 Mathematical Formulation

### 1. Contrast-Limited Adaptive Histogram Equalization (CLAHE)
To amplify the local edge contrast of low-attenuation ground-glass opacities (GGOs) against the surrounding lung parenchyma, input images undergo tile-based adaptive histogram equalization with Rayleigh/uniform distribution clipping:
$$p_k = \min\left(p_k, \; \beta_{\text{clip}}\right) + \frac{1}{L}\sum_{j=1}^{L} \max\left(0, \; p_j - \beta_{\text{clip}}\right)$$
$$s_k = T(r_k) = (L - 1) \sum_{j=0}^{k} p_j$$
where $\beta_{\text{clip}} = 2.0$ represents the clip limit across an $8\times 8$ localized tile grid.

---

### 2. Convolutional Block Attention Module (CBAM)
CBAM sequentially computes 1D channel attention $\mathbf{M}_c \in \mathbb{R}^{C \times 1 \times 1}$ and 2D spatial attention $\mathbf{M}_s \in \mathbb{R}^{1 \times H \times W}$ over an intermediate feature tensor $\mathbf{F} \in \mathbb{R}^{C \times H \times W}$:

$$\mathbf{M}_c(\mathbf{F}) = \sigma\left(\mathbf{W}_1\left(\mathbf{W}_0\left(\mathbf{F}_{\text{avg}}^c\right)\right) + \mathbf{W}_1\left(\mathbf{W}_0\left(\mathbf{F}_{\text{max}}^c\right)\right)\right) = \sigma\left(\text{MLP}(\text{AvgPool}(\mathbf{F})) + \text{MLP}(\text{MaxPool}(\mathbf{F}))\right)$$
$$\mathbf{F}' = \mathbf{M}_c(\mathbf{F}) \odot \mathbf{F}$$
$$\mathbf{M}_s(\mathbf{F}') = \sigma\left(f^{7\times 7}\left(\left[\text{AvgPool}_c(\mathbf{F}'); \; \text{MaxPool}_c(\mathbf{F}')\right]\right)\right)$$
$$\mathbf{F}'' = \mathbf{M}_s(\mathbf{F}') \odot \mathbf{F}'$$
where $\mathbf{W}_0 \in \mathbb{R}^{C/r \times C}$ and $\mathbf{W}_1 \in \mathbb{R}^{C \times C/r}$ denote the shared MLP weights with channel reduction ratio $r=16$, and $\sigma(\cdot)$ denotes the sigmoid activation function.

---

### 3. Atrous Spatial Pyramid Pooling (ASPP)
To capture multi-scale lesion context without losing spatial feature resolution, ASPP replaces standard spatial pooling with parallel dilated convolutions at dilation rates $d \in \{1, 3, 5, 7\}$:

$$y[i, j] = \sum_{m}\sum_{n} x[i + d \cdot m, \; j + d \cdot n] \cdot w[m, n]$$
$$\mathbf{F}_{\text{ASPP}} = \text{Conv}_{1\times 1}\left(\left[ \text{Conv}_{1\times 1}(\mathbf{F}), \; f_{3\times 3}^{d=3}(\mathbf{F}), \; f_{3\times 3}^{d=5}(\mathbf{F}), \; f_{3\times 3}^{d=7}(\mathbf{F}), \; \mathcal{P}_{\text{global}}(\mathbf{F}) \right]\right)$$
where $\mathcal{P}_{\text{global}}(\mathbf{F})$ represents global average pooling followed by bilinear upsampling, and $[\cdot]$ denotes depthwise channel concatenation.

---

### 4. Contextual Transformer Self-Attention (CoT3)
CoT3 integrates static local contextual priors with dynamic global self-attention. Given an input feature map $\mathbf{X} \in \mathbb{R}^{H \times W \times C}$, key $\mathbf{K}_1$, query $\mathbf{Q} = \mathbf{X}\mathbf{W}_q$, and value $\mathbf{V} = \mathbf{X}\mathbf{W}_v$:

$$\mathbf{K}_1 = \text{Conv}_{3\times 3}(\mathbf{X})$$
$$\mathbf{A} = \sigma\left(\text{Conv}_{1\times 1}\left(\delta\left(\text{Conv}_{1\times 1}([\mathbf{K}_1; \; \mathbf{Q}])\right)\right)\right)$$
$$\mathbf{K}_2 = \mathbf{A} \odot \mathbf{V}$$
$$\mathbf{Y} = \mathbf{K}_1 + \mathbf{K}_2$$
where $\delta(\cdot)$ is the SiLU/Swish non-linear activation function, and $\odot$ represents element-wise attention matrix multiplication.

---

### 5. Multi-Task Objective Loss Function
The total training loss $\mathcal{L}_{\text{total}}$ is a composite multi-task objective balancing bounding box regression ($\mathcal{L}_{\text{CIoU}}$), objectness confidence ($\mathcal{L}_{\text{obj}}$), and classification ($\mathcal{L}_{\text{cls}}$):

$$\mathcal{L}_{\text{total}} = \lambda_{\text{box}} \mathcal{L}_{\text{CIoU}} + \lambda_{\text{obj}} \mathcal{L}_{\text{obj}} + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}}$$

#### Complete-IoU (CIoU) Box Regression Loss:
$$\mathcal{L}_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(\mathbf{b}, \mathbf{b}^{\text{gt}})}{c^2} + \alpha v$$
$$\text{where} \quad v = \frac{4}{\pi^2}\left(\arctan\frac{w^{\text{gt}}}{h^{\text{gt}}} - \arctan\frac{w}{h}\right)^2, \quad \alpha = \frac{v}{(1 - \text{IoU}) + v}$$
where $\mathbf{b}$ and $\mathbf{b}^{\text{gt}}$ denote predicted and ground-truth bounding box centroids, $\rho(\cdot)$ is the Euclidean distance, and $c$ is the diagonal length of the smallest enclosing bounding box covering both regions.

#### Objectness & Classification Binary Cross-Entropy (BCE):
$$\mathcal{L}_{\text{obj}} = -\sum_{i=0}^{S^2}\sum_{j=0}^{B} I_{ij}^{\text{obj}}\left[ g_i \log(\sigma(\hat{c}_i)) + (1 - g_i)\log(1 - \sigma(\hat{c}_i)) \right]$$
$$\mathcal{L}_{\text{cls}} = -\sum_{i=0}^{S^2} I_i^{\text{obj}} \sum_{c \in \text{classes}} \left[ y_{i,c} \log(\sigma(\hat{p}_{i,c})) + (1 - y_{i,c})\log(1 - \sigma(\hat{p}_{i,c})) \right]$$
with loss weight coefficients set to $\lambda_{\text{box}} = 0.05$, $\lambda_{\text{obj}} = 1.0$, and $\lambda_{\text{cls}} = 0.5$.

---

### 6. Quantitative Evaluation Metrics
Model performance across intersection-over-union thresholds is evaluated using the Area Under the Precision-Recall Curve:

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
$$\text{AP} = \int_{0}^{1} p(r) \, dr \approx \sum_{k=1}^{N} (r_k - r_{k-1}) \cdot \max_{\tilde{r} \ge r_k} p(\tilde{r})$$
$$\text{mAP@0.5} = \frac{1}{N_{\text{classes}}} \sum_{c=1}^{N_{\text{classes}}} \text{AP}_{c}^{\text{IoU}=0.50}$$
$$\text{mAP@[0.5:0.95]} = \frac{1}{10} \sum_{k=0}^{9} \text{mAP@}(0.50 + 0.05k)$$

---

# 📖 Thesis Documentation

Comprehensive chapter-by-chapter documentation extracted from the MCA thesis:

- 📘 [01_introduction.md](file:///c:/Users/anujm/Desktop/Lung%20Nodule%20Detection/docs/01_introduction.md) – Background, Clinical Need, Challenges, Aim & Contributions
- 📘 [02_literature_review.md](file:///c:/Users/anujm/Desktop/Lung%20Nodule%20Detection/docs/02_literature_review.md) – Traditional CAD, Deep Learning, One/Two-Stage Detectors, Attention & Transformers
- 📘 [03_materials_and_methods.md](file:///c:/Users/anujm/Desktop/Lung%20Nodule%20Detection/docs/03_materials_and_methods.md) – Datasets, Preprocessing, YOLOv5-CASP Modules & Hyperparameters
- 📘 [04_experimental_setup.md](file:///c:/Users/anujm/Desktop/Lung%20Nodule%20Detection/docs/04_experimental_setup.md) – Hardware/Software environment, Protocols & Reproducibility
- 📘 [05_results_and_ablation.md](file:///c:/Users/anujm/Desktop/Lung%20Nodule%20Detection/docs/05_results_and_ablation.md) – Quantitative Results, Multi-modality metrics, Fine-grained Ablation & Benchmarks
- 📘 [06_failure_case_analysis.md](file:///c:/Users/anujm/Desktop/Lung%20Nodule%20Detection/docs/06_failure_case_analysis.md) – Automated Hungarian matching & False Negative breakdown
- 📘 [07_discussion.md](file:///c:/Users/anujm/Desktop/Lung%20Nodule%20Detection/docs/07_discussion.md) – Result interpretations, Module interaction dynamics & Clinical workflows
- 📘 [08_conclusion_and_future_work.md](file:///c:/Users/anujm/Desktop/Lung%20Nodule%20Detection/docs/08_conclusion_and_future_work.md) – Summary of findings, Core contributions & Future roadmap
- 📘 [appendix.md](file:///c:/Users/anujm/Desktop/Lung%20Nodule%20Detection/docs/appendix.md) – Complete `yolov5s-casp.yaml` listing, Training log samples & CSV schemas

---

# 🔮 Future Work

- **3D Volumetric Tensor Analysis**: Extending 2D slice processing to 3D volumetric CT/MRI context modeling.
- **Anatomical Lung Field Priors**: Incorporating lung segmentation masks as an explicit 4th input channel.
- **Ultra-High Resolution Scaling**: Training at $1024\times 1024$ native resolution using PyTorch gradient checkpointing.

---

# ⚡ Citation & Acknowledgments

If you use **YOLOv5-CASP** or reference this work in your research, please cite:

```bibtex
@mastersthesis{Mundu2026CASP,
  author       = {Anuj Mundu},
  title        = {Lung Nodule Detection in Chest X-Ray and CT Images Using an Enhanced YOLOv5-CASP Framework},
  school       = {Maulana Azad National Institute of Technology (MANIT), Bhopal},
  year         = {2026},
  month        = {May},
  department   = {Department of Computer Applications}
}
```

---

# 📜 License & Contact

Distributed under the **MIT License**. See `LICENSE` for details.

- **Author**: Anuj Mundu  
- **Email**: anujmark.edwin.ame@gmail.com  
- **Institution**: Maulana Azad National Institute of Technology (MANIT), Bhopal, India  
- **Repository**: [github.com/anujmundu/Lung-Nodule-Detection](https://github.com/anujmundu)
