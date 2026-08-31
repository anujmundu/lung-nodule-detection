# Chapter 4: Experimental Setup

## 4.1 Hardware & Software Specifications

### Hardware Configuration
- **Processor**: AMD Ryzen 7 6800H (8 Cores, 16 Threads, up to 4.7 GHz)
- **System Memory**: 32 GB DDR5 RAM
- **Graphics Card**: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB GDDR6 VRAM)
- **Storage**: 1 TB NVMe PCIe SSD
- **Operating System**: Windows 11 Pro 64-bit

### Software Environment
- **Conda Environment**: `YOLO_MEDICAL`
- **Python**: `3.10.20`
- **PyTorch**: `2.5.1+cu121`
- **CUDA / cuDNN**: CUDA 12.1
- **OpenCV**: `4.9.0.80`
- **Albumentations**: `1.0.0`
- **Ultralytics**: `8.4.120`

---

## 4.2 Implementation & Repository Organization

The repository structure follows modular standards:

```
MEDICAL_LUNG_NODULE_DETECTION/
├── yolov5/                     # Custom Modified YOLOv5 Framework
│   ├── models/
│   │   ├── common.py           # Contains CBAM, ASPP, CoT3 implementations
│   │   ├── yolo.py             # Parser logic for custom modules
│   │   └── yolov5s-casp.yaml   # 31-layer architecture config
│   ├── train.py
│   └── detect.py
├── src/
│   ├── preprocess.py           # CLAHE & letterbox functions
│   ├── augment.py              # Albumentations pipeline
│   ├── evaluate.py             # Benchmark & mAP evaluation routines
│   └── analyze_failures.py     # Hungarian matching failure categorization
├── custom_modules.py           # Exported PyTorch modules
├── train_casp.py               # SGD training pipeline script
└── docs/                       # Comprehensive thesis chapter markdown files
```

---

## 4.3 Experimental Protocols

### 4.3.1 Baseline Training Protocol
Standard YOLOv5s models were trained on pretrained COCO weights using identical image sizes ($640 \times 640$ for X-Ray/MRI, $256 \times 256$ for CT patches) and SGD hyperparameters for benchmark comparison.

### 4.3.2 Ablation Study Protocol
To isolate individual module contributions, four separate models were trained under identical settings on LUNA16 CT patches:
1. Baseline YOLOv5s
2. Baseline + CBAM only (`yolov5s-cbam.yaml`)
3. Baseline + ASPP only (`yolov5s-aspp.yaml`)
4. Baseline + CoT3 only (`yolov5s-cot3.yaml`)
5. Full YOLOv5-CASP (`yolov5s-casp.yaml`)

### 4.3.3 Comparative Benchmarking
- **YOLOv8s**: Trained using standard Ultralytics implementation with equivalent epoch counts and input resolutions.
- **Faster R-CNN**: Two-stage baseline evaluated using MobileNetV2 backbone.

### 4.3.4 Test-Time Augmentation (TTA)
CT patch evaluation incorporated TTA (horizontal flip, vertical flip, small scale variations combined via Weighted Box Fusion).

### 4.3.5 Inference Speed Benchmark Protocol
Inference throughput (FPS) was benchmarked on CPU (AMD Ryzen 7 6800H) and GPU (NVIDIA RTX 3050) using batch size 1 with 100 warmup iterations followed by 500 timed passes.

---

## 4.4 Reproducibility Measures

- Fixed random seeds across Python (`random.seed(42)`), NumPy (`np.random.seed(42)`), and PyTorch (`torch.manual_seed(42)`).
- Automatic logging of hyperparameter YAML configurations and PyTorch `.pt` checkpoints.
- Environment export via `environment.yml` and pinned `requirements.txt`.
