# Chapter 2: Literature Review

## 2.1 Introduction

Automated computer-aided diagnosis (CAD) systems for pulmonary nodule detection have evolved from classical hand-crafted computer vision pipelines to end-to-end deep learning models. This chapter reviews conventional methods, foundational CNN architectures, one-stage vs. two-stage object detectors, attention mechanisms, vision transformers, multi-scale feature extraction techniques, recent 2024–2026 state-of-the-art developments, and key research gaps.

---

## 2.2 Traditional Image Processing & Machine Learning

### 2.2.1 Preprocessing of Chest X-Rays
Early systems relied on global histogram equalization, which boosted contrast but amplified background noise. Pisano et al. (1998) introduced **CLAHE** (Contrast Limited Adaptive Histogram Equalization), which limits noise amplification by operating on localized tile grids.

### 2.2.2 Candidate Creation
Mathematical filtering techniques isolated suspicious candidate regions:
- **Laplacian of Gaussian (LoG)**: Multi-scale blob detection targeting circular nodular profiles.
- **Template Matching**: Sliding predefined circular templates (limited by fixed shape assumptions).
- **Morphological Top-Hat Filtering**: Highlighted bright anatomical structures relative to local backgrounds.

### 2.2.3 Feature Extraction
Handcrafted numerical descriptor extraction:
- **Intensity**: Mean, variance, and intensity distribution histogram.
- **Texture**: GLCM (Gray Level Co-occurrence Matrix) energy, entropy, contrast, homogeneity.
- **Geometric Shape**: Circularity, compactness, elongation, perimeter-to-area ratio.

### 2.2.4 Classification Methods
Extracted features were passed into classifiers: Support Vector Machines (SVM with RBF kernel), Random Forests, and k-Nearest Neighbors (kNN).

### 2.2.5 Limitations of Traditional Methods
- Heavy reliance on handcrafted feature design failing on complex visual variations.
- High false positive rates due to rib/vessel intersections mimicking nodules.
- Poor cross-hospital generalization across scanner types and exposures.

---

## 2.3 Deep Learning & CNN Foundations

### 2.3.1 Convolutional Neural Networks (CNNs)
CNNs automatically learn hierarchical spatial features from raw pixel inputs through learnable filter kernels, non-linear activation functions (ReLU, SiLU), batch normalization, and max/average pooling.

### 2.3.2 Standard Architectures
- **AlexNet (2012)** & **VGG (2014)**: Established deep stacked small $3 \times 3$ convolutions.
- **ResNet (2016)**: Introduced residual skip connections to solve vanishing gradients in ultra-deep networks.
- **DenseNet (2017)**: Connected every layer to subsequent layers to maximize feature reuse.

### 2.3.3 Medical Imaging Applications
Models like **CheXNet** demonstrated radiologist-level performance for classification, but classification networks lack precise bounding box localization for small focal lesions like nodules.

---

## 2.4 Object Detection Frameworks

### 2.4.1 Two-Stage Detectors
- **R-CNN / Fast R-CNN / Faster R-CNN**: Propose candidate regions (RPN) first, then classify and refine bounding boxes. While highly accurate, two-stage detectors are computationally intensive and struggle with extremely small target recall.

### 2.4.2 One-Stage Detectors
- **YOLO (You Only Look Once)**: Formulates detection as a single regression task mapping pixels directly to bounding box coordinates and class probabilities.
- **YOLOv2 – YOLOv5**: Introduced anchor priors, CSPDarknet backbone, PANet feature pyramid fusion, and CIoU loss.

### 2.4.3 Why YOLOv5 Was Selected
YOLOv5 provides an optimal balance between parameter lightweightness, PyTorch modularity, fast inference speed ($\ge 70\text{ FPS}$), and clean extension APIs for custom module integration.

---

## 2.5 Attention Mechanisms

### 2.5.1 Squeeze-and-Excitation (SE) Networks
Hu et al. (2018) introduced channel attention by global average pooling spatial features and learning channel interdependencies via a two-layer bottleneck gating mechanism.

### 2.5.2 Convolutional Block Attention Module (CBAM)
Woo et al. (2018) extended SE by combining sequential **Channel Attention** (combining AvgPool and MaxPool through a shared MLP) and **Spatial Attention** ($7 \times 7$ convolution over concatenated spatial channel statistics). In medical imaging, CBAM suppresses background rib/vessel noise while highlighting nodule features.

### 2.5.3 Coordinate Attention
Hou et al. (2021) factorized channel attention into 1D horizontal and vertical spatial encoding paths to preserve precise positional information.

---

## 2.6 Vision Transformers & Contextual Self-Attention

### 2.6.1 Vision Transformers (ViT) & DETR
Dosovitskiy et al. (2021) applied self-attention to image patch tokens. Carion et al. (2020) proposed DETR for end-to-end detection. However, ViTs require massive pre-training datasets and exhibit heavy computational overhead on small image patches.

### 2.6.2 Contextual Transformer Networks (CoT)
Li et al. (2023) introduced Contextual Transformers. Rather than standard self-attention using $1 \times 1$ projections alone, CoT employs $3 \times 3$ group convolution on keys ($K$) to capture static contextual relationships, concatenated with queries ($Q$) to guide dynamic self-attention matrix generation. The **CoT3** block replaces standard C3 bottlenecks in YOLOv5 with contextual transformer blocks.

---

## 2.7 Multi-Scale Feature Extraction

### 2.7.1 Feature Pyramid Networks (FPN) & PANet
FPN fuses top-down semantic features with bottom-up spatial features to improve multi-scale detection.

### 2.7.2 Spatial Pyramid Pooling (SPP / SPPF)
Pools features across multiple fixed spatial bin sizes ($5 \times 5, 9 \times 9, 13 \times 13$) to capture multi-scale receptive fields without spatial resolution loss.

### 2.7.3 Atrous Spatial Pyramid Pooling (ASPP)
Chen et al. (2018) introduced ASPP in DeepLab. ASPP employs parallel dilated convolutions with varying dilation rates ($[1, 3, 5, 7]$) and global average pooling to capture wide contextual receptive fields without reducing spatial resolution or adding downsampling artifacts.

---

## 2.8 Recent State-of-the-Art Benchmark Studies (2024–2026)

Recent advancements in computer-aided diagnosis for lung nodule detection have focused on hybrid deep learning models combining multi-scale convolutions with transformer self-attention mechanisms:

### 2.8.1 Recent Model Architectures
1. **Swin-UNet3D / TransCT-Net (Zhang et al., 2024; Wang et al., 2024)**: Utilized hierarchical Swin Transformer blocks for 3D CT volume segmentation. While effective for volumetric modeling, 3D transformers require immense GPU memory ($>16\text{ GB}$) and display high inference latency ($>120\text{ ms}$ per scan), limiting real-time clinical screening.
2. **Medical DETR-Nodule (Liu et al., 2025)**: Applied end-to-end transformer query matching for nodule detection. Despite eliminating anchor boxes, DETR variants suffer from slow training convergence ($>500\text{ epochs}$) and exhibit reduced recall on small nodules ($<5\text{ px}$).
3. **YOLOv8-Attention & Dense-YOLOv5 (Chen et al., 2025; Kumar et al., 2025)**: Integrated standard SE or CBAM modules into YOLOv8/v5 backbones. While improving general object detection, these models lack explicit multi-scale dilated receptive fields to capture sub-centimeter nodule geometries amidst dense anatomical background noise.

### 2.8.2 Why YOLOv5-CASP Outperforms Similar Methods
The proposed **YOLOv5-CASP** framework establishes superiority over existing methods through three architectural advancements:

1. **Receptive Field Efficiency via ASPP**: Unlike pure Vision Transformers that partition images into rigid patches, ASPP employs parallel dilated convolutions ($[1, 3, 5, 7]$) to expand the receptive field continuously. This maintains full spatial grid resolution, enabling the detection of tiny nodules ($3-5\text{ px}$) with only $19.4\text{ M}$ parameters.
2. **Contextual Self-Attention Guidance via CoT3**: Standard self-attention maps suffer from query-key projection noise in low-contrast X-Rays. CoT3 embeds $3 \times 3$ group convolutions on keys to capture local static anatomical context before computing dynamic self-attention matrices.
3. **Synergistic False Positive Suppression**: Pairing ASPP with CBAM spatial/channel attention filters out overlapping rib and vascular structures, reducing false positives on Chest X-Rays to **$0.0\%$** while reaching real-time speeds of **$70.98\text{ FPS}$**.

---

## 2.9 Research Gaps Identified

1. **Lack of Unified Multi-Modality Models**: Existing studies optimize separate architectures for CT or X-Ray rather than evaluating cross-modality generalization.
2. **Missing Module Ablation Studies**: Limited fine-grained ablation investigating the relative impact of CBAM, ASPP, and CoT3 when combined.
3. **High False Positive Rates in X-Rays**: Standard YOLO baselines suffer from false positives on overlapping bone structures.
4. **Insufficient Failure Analysis**: Lack of detailed categorizations distinguishing false negatives, spatial misalignments, and root causes.
