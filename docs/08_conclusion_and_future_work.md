# Chapter 8: Conclusion and Future Work

## 8.1 Summary of Findings

This research developed and validated **YOLOv5-CASP**, an enhanced single-stage object detection framework for lung nodule detection in multi-modality medical images. By integrating Convolutional Block Attention Module (**CBAM**), Atrous Spatial Pyramid Pooling (**ASPP**), and Contextual Transformer (**CoT3**) modules, the framework addresses key challenges including small nodule sizes, low contrast, and overlapping anatomical structures.

### Key Performance Accomplishments:
- **Chest Radiographs (X-Nodule)**: Achieved **0.809 mAP@0.5** (vs. 0.214 baseline) with **0.0% false positives**.
- **CT Patches (LUNA16)**: Achieved **0.382 mAP@0.5** (vs. 0.214 baseline), a $+78.5\%$ relative improvement over standard YOLOv5s and outperforming recent Swin-UNet/YOLOv8 baselines.
- **Inference Speed**: Reached **70.98 FPS** on an NVIDIA RTX 3050 GPU and **26.94 FPS** on an AMD Ryzen 7 CPU.

---

## 8.2 Summary of Contributions

1. **Integrated YOLOv5-CASP Architecture**: Designed a novel 31-layer detection network combining spatial attention, dilated multi-scale pooling, and contextual transformer self-attention.
2. **Multi-Modality Validation**: Demonstrated generalization across CT patches, Chest X-Rays, and MRI scans.
3. **Comprehensive Ablation Analysis**: Proven empirically that ASPP is the primary driver of multi-scale improvement (+15.9%), while full CASP integration produces a synergistic +78.5% boost.
4. **Automated & Manual Failure Case Categorization**: Identified that 97.7% of nodules are correctly detected, with residual failures constrained to ultra-small size ($<5\text{ px}$) or subpleural boundary positions.
5. **Open Reproducibility**: Released full codebase, PyTorch custom modules, model YAML configurations, preprocessing tools, and training scripts.

---

## 8.3 Future Work Directions

1. **Multi-Center Clinical Cohort Validation**:
   - Conduct cross-hospital validation on multi-center clinical datasets (e.g., NIH CXR-14, CheXpert, DeepLesion, and multi-site LIDC-IDRI splits) to benchmark generalization against diverse scanner manufacturers, protocol variations, and patient demographics.
2. **Higher Resolution & Small-Object Detection ($1024 \times 1024$)**:
   - Scale input resolution from $640 \times 640$ to $1024 \times 1024$ and incorporate a high-resolution $P_2/4$ feature map detection head to capture nodules $<5\text{ pixels}$.
3. **Lung Field Mask Input Channel**:
   - Train a preliminary U-Net segmentation model to generate binary lung field masks and feed them as a 4th input channel to eliminate subpleural boundary misalignments.
4. **Rotated Bounding Boxes (OBB)**:
   - Transition from axis-aligned bounding boxes to oriented bounding boxes to fit spiculated and irregular nodule geometries better.
5. **Self-Supervised Pre-Training**:
   - Pre-train the CASP backbone on large unlabeled medical imaging archives using self-supervised contrastive learning (SimCLR/DINO).
6. **Explainability & PACS Clinical Integration**:
   - Integrate Grad-CAM/Eigen-CAM saliency visualizations into a web-based PACS viewer interface for clinical reader studies with practicing radiologists.
