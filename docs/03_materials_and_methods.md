# Chapter 3: Materials and Methods

## 3.1 Overview

This chapter detailed dataset compositions, image preprocessing, data augmentation pipelines, baseline YOLOv5 architecture, proposed **YOLOv5-CASP** enhancements, loss functions, and evaluation metrics.

---

## 3.2 Dataset Descriptions

### 3.2.1 LUNA16 Dataset (CT Scans)
Extracted from LIDC-IDRI database comprising 888 thoracic CT scans with 1,186 radiologist-confirmed nodules ($>3\text{ mm}$).
- **Windowing Normalization**: Applied Hounsfield Unit (HU) clipping:
  $$I_{norm}(x,y) = \frac{\text{clip}(I(x,y), -1350, 150) + 1350}{1500} \times 255$$
- **Patch Extraction**: Extracted $256 \times 256$ 2D patches centered on nodule coordinates.
- **Dataset Split**:

| Subset | Images | Nodules | Purpose |
| :--- | :---: | :---: | :--- |
| **Training** | 424 | 424 | Model Optimization |
| **Validation** | 91 | 91 | Hyperparameter Tuning |
| **Testing** | 92 | 92 | Benchmark Evaluation |

### 3.2.2 X-Nodule Dataset (Chest Radiographs)
Consists of 2,015 frontal chest X-rays with bounding box annotations from multi-center clinical systems.
- **Dataset Split**:

| Subset | Images | Nodules | Purpose |
| :--- | :---: | :---: | :--- |
| **Training** | 1,411 | 4,996 | Model Optimization |
| **Validation** | 403 | 1,450 | Validation & Tuning |
| **Testing** | 201 | 755 | Benchmark & Failure Analysis |

### 3.2.3 Synthetic Kaggle Lung MRI Dataset
Contains 2,436 grayscale MRI slices (cancer vs. non-cancer). Synthetic bounding boxes ($30\%$ central area) were generated for cancer cases to enable preliminary multi-modality localization testing.

---

## 3.3 Preprocessing & Augmentation Pipelines

### 3.3.1 Image Resizing & Letterboxing
All input images are resized using aspect-ratio preserving letterboxing with gray border padding ($114, 114, 114$):
- CT Patches: $256 \times 256$
- Chest X-Rays & MRI: $640 \times 640$

### 3.3.2 Contrast Enhancement (CLAHE)
Applied to chest radiographs to boost local tissue contrast in low-density lung zones:
- Tile grid size: $(8 \times 8)$
- Clip limit: $2.0$

### 3.3.3 Albumentations Data Augmentation
- Horizontal Flipping ($p=0.5$)
- ShiftScaleRotate: Shift limit $0.0$, Scale limit $\pm 20\%$, Rotation limit $\pm 5^\circ$ ($p=0.5$)
- Random Brightness & Contrast ($\pm 20\%$, $p=0.5$)
- Mosaic Augmentation (4-image composite during training)

---

## 3.4 Baseline YOLOv5 Architecture

- **Backbone**: CSPDarknet53 (Cross Stage Partial connections for gradient flow).
- **Neck**: PANet (Path Aggregation Network for bottom-up and top-down feature fusion).
- **Head**: 3-scale Detect head ($P_3/8$, $P_4/16$, $P_5/32$).
- **Loss Function**:
  $$\mathcal{L}_{total} = \lambda_{box} \mathcal{L}_{CIoU} + \lambda_{obj} \mathcal{L}_{obj} + \lambda_{cls} \mathcal{L}_{cls}$$

---

## 3.5 Proposed YOLOv5-CASP Enhancements

### 3.5.1 CBAM (Convolutional Block Attention Module)
Appended after C3 blocks at backbone layers 3, 6, 9 and neck layers 17, 22, 26:
- **Channel Attention**: Dual AvgPool + MaxPool mapped through shared MLP ($ratio=16$).
- **Spatial Attention**: Dual spatial pooling mapped through $7 \times 7$ Conv.

### 3.5.2 ASPP (Atrous Spatial Pyramid Pooling)
Replaces the SPPF module at layer 12 with 5 parallel dilated convolution branches:
- $1 \times 1$ Conv ($rate=1$)
- $3 \times 3$ Dilated Conv ($rate=3$)
- $3 \times 3$ Dilated Conv ($rate=5$)
- $3 \times 3$ Dilated Conv ($rate=7$)
- Global Average Pooling $\rightarrow 1 \times 1$ Conv $\rightarrow$ Bilinear Interpolation
Output channels $c_2$ are projected via $1 \times 1$ Conv.

### 3.5.3 CoT3 (Contextual Transformer in CSP Bottleneck)
Replaces standard C3 bottleneck at terminal backbone layer 11 and detection head layer 29:
- Static key context $K_1 = \text{Conv}_{3 \times 3}(X)$
- Queries $Q = \text{Conv}_{1 \times 1}(X)$, Values $V = \text{Conv}_{1 \times 1}(X)$
- Guidance matrix $A = \text{Softmax}(\text{Conv}_{1 \times 1}(\text{ReLU}(\text{Conv}_{1 \times 1}([K_1, Q]))))$
- Output $Y = K_1 + (A \otimes V)$

---

## 3.6 Hyperparameters & Training Configuration

- **Optimizer**: SGD with Momentum
- **Initial Learning Rate ($lr_0$)**: $0.1$
- **Momentum**: $0.937$
- **Weight Decay**: $0.0005$
- **Epochs**: $100$ (X-Nodule), $300$ (CT Patches), $50$ (MRI)
- **Batch Size**: $8$ (X-Nodule), $16$ (CT Patches), $4$ (MRI)
