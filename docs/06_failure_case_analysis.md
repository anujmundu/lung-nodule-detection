# Chapter 6: Failure Case Analysis

## 6.1 Overview

To evaluate model reliability and understand operational boundaries, an automated and manual failure case analysis was performed on the X-Nodule test dataset (201 images, 755 ground-truth annotated nodules).

---

## 6.2 Evaluation Methodology

Predicted bounding boxes were matched to ground-truth annotations using an Intersection over Union threshold ($\text{IoU} \ge 0.5$) and the Hungarian matching algorithm. Categorization rules:
- **Correct Detection (TP)**: $\text{IoU} \ge 0.5$ matching ground truth.
- **False Negative (FN)**: Ground truth nodule missed or predicted with $\text{IoU} < 0.5$.
- **Misaligned Detection (MIS)**: Nodule detected ($\text{IoU} > 0.1$), but localization inaccuracy kept $\text{IoU} < 0.5$.
- **False Positive (FP)**: Bounding box predicted on non-nodular anatomy.

---

## 6.3 Failure Distribution (X-Nodule Test Set)

| Category | Count | Percentage |
| :--- | :---: | :---: |
| **Correct Detections (TP)** | 738 | **97.7%** |
| **False Negatives (FN)** | 6 | **0.8%** |
| **Misaligned Detections (MIS)** | 11 | **1.5%** |
| **False Positives (FP)** | 0 | **0.0%** |
| **Total Test Nodules** | 755 | 100.0% |

---

## 6.4 False Negative Analysis (6 Missed Cases)

Exhaustive breakdown of the 6 missed nodule cases:

| Image ID | Nodule Size (px) | Anatomical Location | Contrast | Primary Root Cause |
| :--- | :---: | :--- | :--- | :--- |
| `00018003_002.jpg` | 3 px | Peripheral | Low | Extremely small nodule ($<5\text{ px}$) |
| `00019643_013.jpg` | 4 px | Central | Very Low | Small size + faint contrast |
| `00019682_000.jpg` | 3 px | Subpleural | Low | Subpleural border location |
| `00022065_010.jpg` | 4 px | Peripheral | Medium | Rib bone shadow overlap |
| `00024313_002.jpg` | 3 px | Central | Low | Extremely small nodule ($<5\text{ px}$) |
| `00025448_001.jpg` | 5 px | Peripheral | Low | Lung boundary location |

---

## 6.5 Misaligned Detection Analysis (11 Misaligned Cases)

Exhaustive breakdown of the 11 spatial misalignment cases:

| Image ID | Best IoU | Primary Cause | Secondary Cause |
| :--- | :---: | :--- | :--- |
| `00004523_012.jpg` | 0.42 | Irregular Shape | Spiculated margins |
| `00008008_021.jpg` | 0.38 | Boundary Location | Coastal margin proximity |
| `00008897_002.jpg` | 0.45 | Irregular Shape | Spiculated margins |
| `00010980_000.jpg` | 0.35 | Boundary Location | Subpleural position |
| `00012973_008.jpg` | 0.41 | Boundary Location | Rib overlap |
| `00014274_008.jpg` | 0.39 | Low Contrast | Faint density gradient |
| `00015268_000.jpg` | 0.36 | Small Size | Boundary location |
| `00015507_002.jpg` | 0.44 | Irregular Shape | Asymmetric borders |
| `00021086_008.jpg` | 0.40 | Boundary Location | Diaphragmatic edge |
| `00026478_003.jpg` | 0.43 | Irregular Shape | Spiculated margins |
| `00026545_000.jpg` | 0.37 | Boundary Location | Subpleural position |

---

## 6.6 False Positive Comparison

| Model Architecture | False Negatives | False Positives | Misaligned Detections |
| :--- | :---: | :---: | :---: |
| **Baseline YOLOv5s** | 47 (6.2%) | 23 | 28 (3.7%) |
| **YOLOv5-CASP (Proposed)** | **6 (0.8%)** | **0 (0.0%)** | **11 (1.5%)** |

---

## 6.7 Primary Failure Root Causes & Mitigation Roadmap

1. **Tiny Nodule Resolution Limit ($<5\text{ px}$)**:
   - *Mitigation*: Increase training image resolution to $1024 \times 1024$ or add a $P_2/4$ high-resolution detection head.
2. **Subpleural & Lung Boundary Obscuration**:
   - *Mitigation*: Integrate a dual-stage segmentation network providing a 4th input channel (binary lung field mask).
3. **Spiculated & Irregular Boundaries**:
   - *Mitigation*: Implement Rotated Bounding Box (OBB) detection heads to accommodate non-rectangular nodule geometries.
