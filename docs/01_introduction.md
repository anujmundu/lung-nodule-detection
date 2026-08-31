# Chapter 1: Introduction

## 1.1 Why This Research Matters

Lung cancer continues to be one of the leading causes of cancer deaths globally. According to the 2022 GLOBOCAN report, there are approximately 1.5 million new cases and over 1.8 million deaths each year. In India, lung cancer accounts for 5.9% of all cancer cases and 8.1% of cancer-related mortality. 

The primary reason for low 5-year survival rates is late-stage diagnosis. Patients diagnosed in early stages (Stage I/II) have significantly higher survival rates. Chest X-rays are the primary screening tool due to their low cost and wide accessibility. However, early-stage lung nodules are often extremely small, low-contrast, or obscured by anatomical structures such as ribs, clavicles, and vascular shadows, leading to high miss rates even among experienced radiologists.

### 1.1.1 Chest X-Rays in Clinical Practice
Chest X-rays (CXRs) are quick, cheap, and expose patients to low radiation doses. In high-volume hospitals, radiologists process hundreds of scans daily, increasing diagnostic fatigue and the risk of missing subtle nodular lesions ($<10\text{ mm}$).

### 1.1.2 CT and MRI Imaging
- **CT Scans**: Provide high-resolution 3D anatomical details and are highly sensitive for nodule detection (e.g., LUNA16 benchmark). However, CT scans are expensive, involve higher radiation exposure, and have lower availability in rural centers.
- **MRI Scans**: Avoid ionizing radiation but possess lower spatial resolution for pulmonary parenchyma and lack large public annotated nodule benchmarks.

### 1.1.3 Aim of This Research
The primary objective of this project is to develop a unified, lightweight, real-time deep learning framework—**YOLOv5-CASP**—capable of accurately detecting lung nodules across multiple imaging modalities (Chest X-Rays, CT patches, and MRI scans).

---

## 1.2 Challenges in Lung Nodule Detection

1. **Small Nodule Size**: Early-stage nodules occupy only a tiny fraction of the image pixels ($3 \times 3$ to $10 \times 10$ pixels on $640 \times 640$ radiographs), making feature retention across downsampling stages difficult.
2. **Low Contrast & Overlapping Anatomy**: Nodules blend into surrounding lung parenchyma and are frequently obscured by ribs, cardiac structures, or diaphragm boundaries.
3. **Variation in Appearance**: Nodules exhibit significant morphological heterogeneity (round, oval, irregular, ground-glass, or spiculated borders).
4. **Class Imbalance & False Positives**: Vessel intersections and bone structures mimic nodular shapes, generating high false positive rates in standard object detectors.
5. **Real-Time Utility**: Clinical integration requires low latency ($\ge 30\text{ FPS}$) on standard hospital workstation hardware.

---

## 1.3 Purpose of the Study

1. Propose the **YOLOv5-CASP** framework by integrating **CBAM** (attention refinement), **ASPP** (multi-scale feature extraction), and **CoT3** (contextual transformer self-attention) into YOLOv5.
2. Evaluate performance across Chest X-Ray (X-Nodule), CT patch (LUNA16), and MRI modalities.
3. Conduct comprehensive ablation studies quantifying individual and combined module contributions.
4. Benchmark against baseline YOLOv5s, YOLOv8s, and Faster R-CNN.
5. Perform automated and manual failure case analysis to identify root causes of false negatives and misalignment errors.
6. Evaluate inference throughput (FPS) on consumer GPU (NVIDIA RTX 3050) and CPU hardware.

---

## 1.4 Major Contributions

- **Novel Architecture**: Designed YOLOv5-CASP, integrating CBAM at 6 strategic layers, replacing terminal C3 with CoT3, and swapping SPPF for ASPP with dilated rates $[1, 3, 5, 7]$.
- **Superior Detection Performance**: Achieved **0.809 mAP@0.5** on X-Nodule chest X-rays (vs. 0.214 baseline) and **0.382 mAP@0.5** on LUNA16 CT patches (vs. 0.214 baseline).
- **Zero False Positives**: Eliminated false positives on the X-Nodule test set (0.0% FP rate vs. 23 FPs in baseline YOLOv5s).
- **Ablation Insights**: Demonstrated that ASPP provides the single largest gain (+15.9% mAP), while full CASP integration achieves a synergistic +78.5% improvement.
- **Real-Time Efficiency**: Attained **70.98 FPS** on RTX 3050 GPU and **26.94 FPS** on CPU.

---

## 1.5 Organization of the Thesis

- **Chapter 1**: Introduction & Research Objectives
- **Chapter 2**: Literature Review & Related Work
- **Chapter 3**: Materials, Methods & YOLOv5-CASP Architecture
- **Chapter 4**: Experimental Setup & Implementation Details
- **Chapter 5**: Experimental Results & Ablation Analysis
- **Chapter 6**: Failure Case Analysis
- **Chapter 7**: Discussion & Clinical Relevance
- **Chapter 8**: Conclusion & Future Work
