# Appendix A: Architecture, Training Logs, & Code Repository Structure

## A.1 Complete Architecture Listing: `yolov5s-casp.yaml`

```yaml
# YOLOv5s-CASP Configuration
nc: 1 # number of classes (nodule)
depth_multiple: 0.33 # model depth multiple
width_multiple: 0.50 # layer channel multiple
anchors:
  - [10,13, 16,30, 33,23] # P3/8
  - [30,61, 62,45, 59,119] # P4/16
  - [116,90, 156,198, 373,326] # P5/32

# Backbone
backbone:
  # [from, number, module, args]
  - [-1, 1, Conv, [64, 6, 2, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 3, C3, [128]] # 2
  - [-1, 1, CBAM, [128]] # 3: CBAM inserted after first C3 stage
  - [-1, 1, Conv, [256, 3, 2]] # 4-P3/8
  - [-1, 6, C3, [256]] # 5
  - [-1, 1, CBAM, [256]] # 6: attention refinement stage
  - [-1, 1, Conv, [512, 3, 2]] # 7-P4/16
  - [-1, 9, C3, [512]] # 8
  - [-1, 1, CBAM, [512]] # 9: feature attention enhancement
  - [-1, 1, Conv, [1024, 3, 2]] # 10-P5/32
  - [-1, 3, CoT3, [1024, 1, False]] # 11: CoT3 used instead of terminal C3 block
  - [-1, 1, ASPP, [1024, [1, 3, 5, 7]]] # 12: ASPP substituted for SPPF module

# Head
head:
  - [-1, 1, Conv, [512, 1, 1]] # 13
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # 14
  - [[-1, 9], 1, Concat, [1]] # 15: fuse with backbone P4 features
  - [-1, 3, C3, [512, False]] # 16
  - [-1, 1, CBAM, [512]] # 17: attention refinement in neck
  - [-1, 1, Conv, [256, 1, 1]] # 18
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # 19
  - [[-1, 6], 1, Concat, [1]] # 20: fuse with backbone P3 features
  - [-1, 3, C3, [256, False]] # 21
  - [-1, 1, CBAM, [256]] # 22: attention enhancement in neck
  - [-1, 1, Conv, [256, 3, 2]] # 23
  - [[-1, 17], 1, Concat, [1]] # 24: merge with P4 head features
  - [-1, 3, C3, [512, False]] # 25
  - [-1, 1, CBAM, [512]] # 26: additional attention block
  - [-1, 1, Conv, [512, 3, 2]] # 27
  - [[-1, 12], 1, Concat, [1]] # 28: merge with P5 features
  - [-1, 3, CoT3, [1024, 1, False]] # 29: contextual transformer block
  - [[22, 26, 29], 1, Detect, [nc, anchors]] # 30: multi-scale detection head
```

---

## A.2 Training Log Sample (X-Nodule Dataset)

```text
Epoch GPU_mem box_loss obj_loss cls_loss Instances Size
 0/99  0.923G   0.1086   0.0629        0       112  640
10/99  4.74G    0.0538   0.0429        0        41  640
20/99  4.74G    0.0443   0.0366        0        25  640
30/99  4.74G    0.0414   0.0345        0        21  640
40/99  4.74G    0.0397   0.0341        0        18  640
50/99  4.74G    0.0379   0.0337        0        19  640
60/99  4.74G    0.0368   0.0328        0         9  640
70/99  4.74G    0.0350   0.0298        0         5  640
80/99  4.74G    0.0345   0.0318        0        40  640
90/99  4.74G    0.0334   0.0289        0        12  640
99/99  4.74G    0.0326   0.0283        0        21  640

Validation metrics:
Epoch   Precision   Recall   mAP@0.5   mAP@0.5:0.95
 0/99     0.125      0.264    0.100       0.024
10/99     0.456      0.467    0.441       0.204
20/99     0.623      0.587    0.611       0.311
30/99     0.701      0.642    0.688       0.370
40/99     0.734      0.672    0.727       0.401
50/99     0.755      0.688    0.753       0.423
60/99     0.768      0.696    0.772       0.439
70/99     0.777      0.702    0.788       0.452
80/99     0.785      0.706    0.798       0.460
90/99     0.790      0.708    0.806       0.465
99/99     0.792      0.708    0.809       0.467
```

---

## A.3 Failure Analysis Report Excerpt (`failure_analysis.csv`)

```csv
Image,Failure Type,Best IoU,Ground Truth Count,Prediction Count,Description
00018003_002.jpg,False Negative,0.000,1,0,Very small nodule (3 px)
00019643_013.jpg,False Negative,0.000,1,0,Low contrast + small
00019682_000.jpg,False Negative,0.000,1,0,Subpleural + small
00004523_012.jpg,Misaligned,0.422,1,1,Irregular shape
00008008_021.jpg,Misaligned,0.384,1,1,Boundary location
00008897_002.jpg,Misaligned,0.449,1,1,Spiculated nodule
00010980_000.jpg,Misaligned,0.354,1,1,Boundary + subpleural
00012973_008.jpg,Misaligned,0.412,1,1,Boundary + rib overlap
00014274_008.jpg,Misaligned,0.391,1,1,Low contrast
```

---

## A.4 Code Repository Structure

```text
MEDICAL_LUNG_NODULE_DETECTION/
├── yolov5/                     # Custom Modified YOLOv5 Repository
│   ├── models/
│   │   ├── common.py           # CBAM, ASPP, CoT3 implementations
│   │   ├── yolo.py             # parse_model parser registration
│   │   └── yolov5s-casp.yaml   # 31-layer architecture definition
│   ├── train.py
│   └── detect.py
├── data/
│   ├── raw/                    # LUNA16 raw CT scans
│   ├── processed_patches/      # LUNA16 2D CT patches
│   ├── x_nodule/               # Chest X-Ray images & YOLO format labels
│   └── mri_dataset/            # Kaggle Lung MRI dataset
├── docs/                       # Complete Chapter-by-Chapter Markdown documentation
│   ├── 01_introduction.md
│   ├── 02_literature_review.md
│   ├── 03_materials_and_methods.md
│   ├── 04_experimental_setup.md
│   ├── 05_results_and_ablation.md
│   ├── 06_failure_case_analysis.md
│   ├── 07_discussion.md
│   ├── 08_conclusion_and_future_work.md
│   └── appendix.md
├── src/
│   ├── preprocess.py           # CLAHE contrast enhancement & letterboxing
│   └── augment.py              # Albumentations data augmentation pipeline
├── custom_modules.py           # Standalone PyTorch module exports
├── train_casp.py               # Multi-modality SGD training launcher
├── requirements.txt            # Environment dependency list
└── README.md                   # Project overview & quickstart guide
```
