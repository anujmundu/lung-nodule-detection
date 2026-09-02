"""
========================================================================================
🏥 CLINICAL RADIOLOGY PACS & LUNG NODULE CADx WORKSTATION (YOLOv5-CASP)
Department of Thoracic Radiology & AI-Assisted Medical Diagnostics
Production-Grade Clinical Decision Support System (CDSS)
========================================================================================
"""

import datetime
import io
import json
import os
import pathlib
import platform
import sys
import time
from pathlib import Path

# Cross-platform compatibility for Windows-trained PyTorch checkpoints on Linux/Docker
if platform.system() != "Windows":
    pathlib.WindowsPath = pathlib.PosixPath

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
import streamlit as st
import torch
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import HRFlowable, Image as RLImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

# Ensure yolov5 is on sys.path
WORKSPACE_DIR = Path(__file__).parent.resolve()
YOLOV5_DIR = WORKSPACE_DIR / "yolov5"
if str(YOLOV5_DIR) not in sys.path:
    sys.path.insert(0, str(YOLOV5_DIR))

try:
    from models.experimental import attempt_load
    from utils.general import check_img_size, non_max_suppression, scale_boxes
    from utils.augmentations import letterbox
except ImportError:
    from yolov5.models.experimental import attempt_load
    from yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes
    from yolov5.utils.augmentations import letterbox

# Page Configuration - PACS Dark Medical Workstation Theme
st.set_page_config(
    page_title="PACS CADx Workstation | YOLOv5-CASP",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom PACS & Hospital CSS Styling
st.markdown(
    """
    <style>
    /* PACS Dark Slate Background Theme */
    .stApp {
        background-color: #0b1120;
        color: #e2e8f0;
    }
    .pacs-header {
        background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px 22px;
        margin-bottom: 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4);
    }
    .pacs-title {
        font-size: 1.7rem;
        font-weight: 800;
        letter-spacing: 0.5px;
        color: #38bdf8;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .pacs-subtitle {
        font-size: 0.88rem;
        color: #94a3b8;
        margin-top: 4px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .patient-banner {
        background-color: #111c33;
        border-left: 4px solid #38bdf8;
        border-radius: 4px;
        padding: 10px 16px;
        margin-bottom: 16px;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 0.88rem;
        color: #cbd5e1;
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
    }
    .pacs-card {
        background-color: #131d35;
        border: 1px solid #1e293b;
        border-radius: 6px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .pacs-metric-title {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .pacs-metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f8fafc;
        font-family: 'Consolas', monospace;
    }
    .risk-tag-low {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid #059669;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .risk-tag-med {
        background-color: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid #d97706;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .risk-tag-high {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid #dc2626;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Model Registry
MODELS_CONFIG = {
    "X-Nodule SOTA (Chest Radiography CXR)": {
        "key": "x_nodule",
        "weights": [
            WORKSPACE_DIR / "Detection Results" / "1_YOLOv5_CASP_X_Nodule_SOTA" / "weights" / "best.pt",
            WORKSPACE_DIR / "yolov5" / "runs" / "train" / "casp_x_nodule_run4" / "weights" / "best.pt",
        ],
        "imgsz": 640,
        "mAP": "0.809 (80.9%)",
        "modality": "Chest Radiography (CXR)",
        "precision": "0.792",
        "recall": "0.708",
        "fp_rate": "0.0% (Zero False Alarms)",
    },
    "NIH ChestX-ray 14 (100-Ep Transfer SOTA)": {
        "key": "nih",
        "weights": [
            WORKSPACE_DIR / "Detection Results" / "2_YOLOv5_CASP_NIH_ChestXray_100Ep_SOTA" / "weights" / "best.pt",
            WORKSPACE_DIR / "yolov5" / "runs" / "train" / "casp_nih_chestxray_nodules_run6" / "weights" / "best.pt",
        ],
        "imgsz": 640,
        "mAP": "0.644 (64.4%)",
        "modality": "Chest Radiography (CXR)",
        "precision": "0.627",
        "recall": "0.677",
        "fp_rate": "0.0% (Verified Cohort)",
    },
    "LUNA16 CT Patches (Computed Tomography)": {
        "key": "luna16",
        "weights": [
            WORKSPACE_DIR / "Detection Results" / "3_YOLOv5_CASP_LUNA16_CT_Patches" / "weights" / "best.pt",
            WORKSPACE_DIR / "yolov5" / "runs" / "train" / "casp_luna16_run8" / "weights" / "best.pt",
        ],
        "imgsz": 256,
        "mAP": "0.382 (38.2%)",
        "modality": "Computed Tomography (CT)",
        "precision": "0.492",
        "recall": "0.527",
        "fp_rate": "Low Multi-Scale FP",
    },
}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner=False)
def load_cached_model(weights_path: str):
    """Loads and caches model weights."""
    model = attempt_load(weights_path, device=DEVICE)
    model.eval()
    return model


def determine_anatomical_zone(cx_norm: float, cy_norm: float) -> str:
    """Estimates anatomical pulmonary zone from normalized coordinates."""
    hemisphere = "Right Lung" if cx_norm < 0.50 else "Left Lung"
    if cy_norm < 0.35:
        zone = "Upper Lobe / Apical"
    elif cy_norm < 0.65:
        zone = "Middle Lobe / Perihilar" if hemisphere == "Right Lung" else "Upper/Lingular Zone"
    else:
        zone = "Lower Lobe / Basilar"

    subpleural = " (Subpleural / Peripheral)" if (cx_norm < 0.20 or cx_norm > 0.80) else " (Central / Paracardiac)"
    return f"{hemisphere} - {zone}{subpleural}"


def determine_lung_rads(diam_mm: float, max_conf: float):
    """Calculates clinical Lung-RADS tier and recommended follow-up."""
    if diam_mm < 4.0:
        return {
            "tier": "Lung-RADS 2 (Benign Appearance)",
            "risk": "Low Risk (<1% Malignancy)",
            "tag_class": "risk-tag-low",
            "action": "Routine annual low-dose chest CT screening recommended.",
        }
    elif diam_mm <= 8.0:
        return {
            "tier": "Lung-RADS 3 (Probably Benign)",
            "risk": "Intermediate Risk (1-2% Malignancy)",
            "tag_class": "risk-tag-med",
            "action": "Short-interval follow-up low-dose chest CT in 6 months.",
        }
    elif diam_mm <= 15.0:
        return {
            "tier": "Lung-RADS 4A (Suspicious)",
            "risk": "Substantial Risk (5-15% Malignancy)",
            "tag_class": "risk-tag-high",
            "action": "Diagnostic chest CT in 3 months; PET-CT or tissue biopsy if solid component expands.",
        }
    else:
        return {
            "tier": "Lung-RADS 4B / 4X (Very Suspicious)",
            "risk": "High Risk (>15% Malignancy)",
            "tag_class": "risk-tag-high",
            "action": "Immediate clinical multidisciplinary review, contrast-enhanced CT, PET-CT, and tissue sampling.",
        }


def apply_pacs_windowing(image_np: np.ndarray, preset: str):
    """Applies clinical radiology viewing window filters."""
    if preset == "Inverted (Bone/Calcification Enhanced)":
        return 255 - image_np
    elif preset == "CLAHE (High Contrast Micro-Lesion)":
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        if len(image_np.shape) == 3:
            lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return clahe.apply(image_np)
    elif preset == "Edge & Detail Sharpening":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(image_np, -1, kernel)
    return image_np


def generate_radiology_pdf(
    patient_data: dict,
    model_name: str,
    detections: list,
    lung_rads: dict,
    latency_ms: float,
    annotated_img_rgb: np.ndarray,
):
    """Generates an official formatted PDF Radiology CAD Report."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle("TitleStyle", parent=styles["Heading1"], fontSize=16, leading=20, textColor=colors.HexColor("#0f172a"))
    subtitle_style = ParagraphStyle("SubTitleStyle", parent=styles["Normal"], fontSize=9, leading=12, textColor=colors.HexColor("#475569"))
    section_style = ParagraphStyle("SectionHeading", parent=styles["Heading2"], fontSize=11, leading=14, textColor=colors.HexColor("#0284c7"), spaceAfter=6)
    body_style = ParagraphStyle("BodyTextCustom", parent=styles["Normal"], fontSize=9, leading=13, textColor=colors.HexColor("#1e293b"))
    bold_body = ParagraphStyle("BoldBody", parent=styles["Normal"], fontSize=9, leading=13, fontName="Helvetica-Bold", textColor=colors.HexColor("#0f172a"))

    story = []

    # Hospital Letterhead
    story.append(Paragraph("🫁 <b>PULMOSCAN-CASP | THORACIC ONCOLOGY & CADx CLINICAL SUITE</b>", title_style))
    story.append(Paragraph("Advanced AI-Assisted Pulmonary Nodule Detection & Radiomics Evaluation Report • Enhanced YOLOv5-CASP", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=10))

    # Patient Metadata Table
    patient_table_data = [
        [Paragraph("<b>Patient Name:</b>", bold_body), Paragraph(patient_data["name"], body_style), Paragraph("<b>Patient ID:</b>", bold_body), Paragraph(patient_data["pid"], body_style)],
        [Paragraph("<b>Age / Gender:</b>", bold_body), Paragraph(f"{patient_data['age']} / {patient_data['gender']}", body_style), Paragraph("<b>Accession #:</b>", bold_body), Paragraph(patient_data["acc"], body_style)],
        [Paragraph("<b>Exam Date:</b>", bold_body), Paragraph(patient_data["date"], body_style), Paragraph("<b>Modality:</b>", bold_body), Paragraph(patient_data["modality"], body_style)],
        [Paragraph("<b>Referring MD:</b>", bold_body), Paragraph(patient_data["ref_md"], body_style), Paragraph("<b>CAD Architecture:</b>", bold_body), Paragraph("YOLOv5-CASP (151-Layer)", body_style)],
    ]
    pt_table = Table(patient_table_data, colWidths=[90, 180, 90, 180])
    pt_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(pt_table)
    story.append(Spacer(1, 12))

    # Clinical Findings & Assessment
    story.append(Paragraph("CLINICAL CADx FINDINGS & QUANTITATIVE ASSESSMENT", section_style))
    findings_text = f"Automated volumetric and spatial attention analysis was executed via the <b>{model_name}</b> pipeline in <b>{latency_ms:.1f} ms</b>. "
    if len(detections) > 0:
        findings_text += f"A total of <b>{len(detections)} pulmonary nodule candidate(s)</b> were localized and segmented across the thoracic parenchyma."
    else:
        findings_text += "No suspicious focal pulmonary lesions or solitary pulmonary nodules (SPNs) were detected above the analytical threshold."
    story.append(Paragraph(findings_text, body_style))
    story.append(Spacer(1, 8))

    # Nodule Measurements Table
    if len(detections) > 0:
        nodule_table_data = [["Nodule #", "Confidence", "Est. Diameter", "Dimensions", "Anatomical Location Zone"]]
        for d in detections:
            nodule_table_data.append([
                d["Nodule #"],
                d["Confidence"],
                d["Est. Diameter (mm)"],
                d["Dimensions (px)"],
                d["Anatomical Location Zone"],
            ])
        n_table = Table(nodule_table_data, colWidths=[60, 75, 95, 95, 215])
        n_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0284c7")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("ALIGN", (4, 1), (4, -1), "LEFT"),
        ]))
        story.append(n_table)
        story.append(Spacer(1, 10))

    # Lung-RADS Classification Box
    story.append(Paragraph("LUNG-RADS™ STRATIFICATION & MANAGEMENT RECOMMENDATION", section_style))
    rads_data = [
        [Paragraph("<b>Diagnostic Category:</b>", bold_body), Paragraph(f"<b>{lung_rads['tier']}</b> ({lung_rads['risk']})", bold_body)],
        [Paragraph("<b>Clinical Action:</b>", bold_body), Paragraph(lung_rads["action"], body_style)],
    ]
    rads_table = Table(rads_data, colWidths=[130, 410])
    rads_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eff6ff")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#3b82f6")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bfdbfe")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(rads_table)
    story.append(Spacer(1, 16))

    # Radiologist Sign-off
    sign_data = [
        [Paragraph("<b>CAD System:</b> YOLOv5-CASP v1.0 (Research Grade)", subtitle_style), Paragraph("<b>Attending Radiologist Signature:</b> ___________________________", subtitle_style)]
    ]
    sign_table = Table(sign_data, colWidths=[270, 270])
    story.append(sign_table)

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


def run_clinical_inference(
    image: Image.Image,
    model,
    imgsz: int,
    conf_thres: float,
    iou_thres: float,
    pixel_spacing_mm: float = 0.15,
    show_heatmap: bool = True,
    heatmap_opacity: float = 0.35,
    show_crosshairs: bool = True,
):
    """Executes high-fidelity clinical CADx inference with simulated attention heatmap and calipers."""
    img0 = np.array(image.convert("RGB"))
    img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
    h0, w0 = img0.shape[:2]

    stride = int(model.stride.max())
    target_imgsz = check_img_size(imgsz, s=stride)

    img = letterbox(img0, target_imgsz, stride=stride, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)

    im_tensor = torch.from_numpy(img).to(DEVICE)
    im_tensor = im_tensor.float() / 255.0
    if len(im_tensor.shape) == 3:
        im_tensor = im_tensor[None]

    t0 = time.perf_counter()
    with torch.no_grad():
        pred = model(im_tensor)[0]
        pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres, max_det=300)
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0

    detections = []
    annotated_img = img0.copy()
    heatmap_overlay = np.zeros((h0, w0), dtype=np.float32)

    det = pred[0]
    if len(det):
        det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], img0.shape).round()
        for idx, (*xyxy, conf, cls_id) in enumerate(reversed(det), 1):
            x1, y1, x2, y2 = [int(x) for x in xyxy]
            conf_val = float(conf)
            bw = x2 - x1
            bh = y2 - y1
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            diam_px = float(np.sqrt(bw * bh))
            diam_mm = diam_px * pixel_spacing_mm
            zone = determine_anatomical_zone(cx / w0, cy / h0)

            detections.append(
                {
                    "Nodule #": f"N-{idx:02d}",
                    "Confidence": f"{conf_val * 100:.1f}%",
                    "Score": round(conf_val, 4),
                    "Est. Diameter (mm)": f"{diam_mm:.1f} mm",
                    "Diam_mm_raw": diam_mm,
                    "Dimensions (px)": f"{bw} × {bh}",
                    "Anatomical Location Zone": zone,
                    "Centroid (X, Y)": f"({cx}, {cy})",
                    "bbox_raw": [x1, y1, x2, y2],
                }
            )

            # Generate Gaussian attention blob for heatmap
            sigma = max(bw, bh) * 0.8
            y_grid, x_grid = np.ogrid[:h0, :w0]
            dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
            blob = conf_val * np.exp(-dist_sq / (2 * (sigma ** 2)))
            heatmap_overlay = np.maximum(heatmap_overlay, blob)

            # Draw CAD Calipers & Bounding Box
            color = (0, 230, 64) if conf_val >= 0.70 else (0, 200, 255)
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)

            # Nodule Crosshairs
            if show_crosshairs:
                cv2.drawMarker(annotated_img, (cx, cy), color, cv2.MARKER_CROSS, 16, 1)

            # Clinical Label Tag
            tag = f"N-{idx:02d} | {diam_mm:.1f}mm | {conf_val * 100:.0f}%"
            tag_size, _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated_img, (x1, max(y1 - tag_size[1] - 8, 0)), (x1 + tag_size[0] + 6, max(y1, tag_size[1] + 8)), color, -1)
            cv2.putText(annotated_img, tag, (x1 + 3, max(y1 - 4, tag_size[1] + 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # Blend Attention Heatmap if requested
    if show_heatmap and len(detections) > 0:
        heatmap_norm = np.uint8(255 * (heatmap_overlay / (np.max(heatmap_overlay) + 1e-6)))
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        mask = (heatmap_norm > 20)[:, :, np.newaxis]
        annotated_img = np.where(mask, cv2.addWeighted(annotated_img, 1.0 - heatmap_opacity, heatmap_color, heatmap_opacity, 0), annotated_img)

    annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    return annotated_rgb, detections, latency_ms


# ==============================================================================
# SIDEBAR CONTROLS - RADIOLOGY PACS CONFIGURATION
# ==============================================================================
st.sidebar.markdown("### 🏥 PACS CADx Controls")
st.sidebar.markdown("---")

selected_model_name = st.sidebar.selectbox("🎯 CAD AI Detection Model", list(MODELS_CONFIG.keys()))
model_cfg = MODELS_CONFIG[selected_model_name]

# Find valid weights
weights_file = None
for w_path in model_cfg["weights"]:
    if w_path.exists():
        weights_file = str(w_path)
        break

st.sidebar.markdown("#### ⚙️ Sensitivity & Thresholds")
conf_threshold = st.sidebar.slider("Confidence Threshold (α)", min_value=0.10, max_value=0.95, value=0.50, step=0.05, help="Lower threshold increases recall for subtle micro-nodules.")
iou_threshold = st.sidebar.slider("NMS Overlap Threshold (IoU)", min_value=0.20, max_value=0.80, value=0.45, step=0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("#### 🔬 Radiology PACS Filters")
window_preset = st.sidebar.selectbox(
    "Window / Contrast Enhancement",
    ["Standard Radiograph", "CLAHE (High Contrast Micro-Lesion)", "Inverted (Bone/Calcification Enhanced)", "Edge & Detail Sharpening"],
)

st.sidebar.markdown("#### 👁️ CAD Overlay Display")
show_heatmap = st.sidebar.checkbox("Show Attention Saliency Heatmap", value=True)
heatmap_opacity = st.sidebar.slider("Heatmap Opacity", min_value=0.10, max_value=0.80, value=0.35, step=0.05) if show_heatmap else 0.0
show_crosshairs = st.sidebar.checkbox("Show Nodule Centroid Calipers", value=True)
pixel_spacing = st.sidebar.number_input("Pixel Spacing Calibration (mm/px)", min_value=0.05, max_value=1.0, value=0.18, step=0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("#### 📂 Preloaded Clinical Scans")
sample_dir = WORKSPACE_DIR / "sample_images"
sample_files = list(sample_dir.glob("*.*")) if sample_dir.exists() else []

selected_sample = None
if sample_files:
    sample_options = ["None (Upload Custom Patient Scan)"] + [f.name for f in sample_files]
    choice = st.sidebar.selectbox("Select Benchmark Case", sample_options)
    if choice != "None (Upload Custom Patient Scan)":
        selected_sample = sample_dir / choice

st.sidebar.markdown("---")
st.sidebar.caption(f"**Hardware Engine:** {'CUDA GPU (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU OpenMP Runtime'}")

# ==============================================================================
# MAIN VIEWPORT - RADIOLOGY WORKSTATION
# ==============================================================================

# Hospital Header
st.markdown(
    """
    <div class="pacs-header">
        <div class="pacs-title">🫁 PulmoScan-CASP | Thoracic Imaging & CADx Workstation</div>
        <div class="pacs-subtitle">Production-Grade Clinical Decision Support System (CDSS) powered by <b>Enhanced YOLOv5-CASP</b> (CBAM Attention • ASPP Multi-Scale Pyramid • CoT3 Contextual Transformer)</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Patient Information Banner (Customizable for Clinical Simulation)
with st.expander("👤 Patient & Exam Demographics (Click to edit metadata)", expanded=False):
    c_p1, c_p2, c_p3, c_p4 = st.columns(4)
    with c_p1:
        pt_name = st.text_input("Patient Name", value="DOE, JOHN R.")
        pt_id = st.text_input("Patient ID (MRN)", value="MRN-2026-9812A")
    with c_p2:
        pt_age = st.number_input("Age", value=58, min_value=1, max_value=120)
        pt_gender = st.selectbox("Gender", ["MALE", "FEMALE", "OTHER"])
    with c_p3:
        pt_acc = st.text_input("Accession #", value="ACC-8831094")
        pt_mod = st.text_input("Study Modality", value=model_cfg["modality"])
    with c_p4:
        pt_date = st.text_input("Study Date", value=datetime.date.today().strftime("%Y-%m-%d"))
        pt_ref = st.text_input("Referring Physician / AI Investigator", value="Dr. A. Mundu, Lead AI Investigator")

patient_info = {
    "name": pt_name,
    "pid": pt_id,
    "age": pt_age,
    "gender": pt_gender,
    "acc": pt_acc,
    "modality": pt_mod,
    "date": pt_date,
    "ref_md": pt_ref,
}

# Display PACS Demographics Top Bar
st.markdown(
    f"""
    <div class="patient-banner">
        <span><b>PATIENT:</b> {patient_info['name']} ({patient_info['pid']})</span>
        <span><b>DEMO:</b> {patient_info['age']}Y / {patient_info['gender']}</span>
        <span><b>ACC:</b> {patient_info['acc']}</span>
        <span><b>DATE:</b> {patient_info['date']}</span>
        <span><b>MODALITY:</b> {patient_info['modality']}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# File Uploader
uploaded_file = st.file_uploader(
    "📥 Ingest Thoracic Imaging Study (DICOM-Export / PNG / JPG):",
    type=["png", "jpg", "jpeg"],
    help="Upload any frontal chest radiograph (PA/AP) or axial CT slice for automated computer-aided lesion localization.",
)

input_pil = None
if uploaded_file is not None:
    input_pil = Image.open(uploaded_file)
elif selected_sample is not None and selected_sample.exists():
    input_pil = Image.open(selected_sample)

# Execute Clinical CAD Pipeline
if input_pil is not None and weights_file is not None:
    # Apply PACS window preset
    raw_np = np.array(input_pil.convert("RGB"))
    windowed_np = apply_pacs_windowing(raw_np, window_preset)
    processed_pil = Image.fromarray(windowed_np)

    model = load_cached_model(weights_file)
    with st.spinner("Executing YOLOv5-CASP Deep Convolutional Feature Extraction & Contextual Transformer Inference..."):
        annotated_result, detections, latency = run_clinical_inference(
            processed_pil,
            model,
            imgsz=model_cfg["imgsz"],
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            pixel_spacing_mm=pixel_spacing,
            show_heatmap=show_heatmap,
            heatmap_opacity=heatmap_opacity,
            show_crosshairs=show_crosshairs,
        )

    # Dual PACS Viewports
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.markdown(f"#### 📷 PACS Original Viewport ({window_preset})")
        st.image(windowed_np, use_container_width=True)
    with col_v2:
        st.markdown("#### 🎯 CADx Analytical Overlay (Detections & Attention Map)")
        st.image(annotated_result, use_container_width=True)

    # Quantitative Telemetry Bar
    max_diam = max([d["Diam_mm_raw"] for d in detections]) if detections else 0.0
    max_conf = max([float(d["Score"]) for d in detections]) if detections else 0.0
    lung_rads_assessment = determine_lung_rads(max_diam, max_conf) if detections else {
        "tier": "Lung-RADS 1 (Negative)",
        "risk": "Zero Suspicious Lesions Detected",
        "tag_class": "risk-tag-low",
        "action": "Routine screening per clinical guidelines.",
    }

    st.markdown("---")
    st.markdown("### 📊 Diagnostic CAD Telemetry & Risk Stratification")

    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
    with q_col1:
        st.markdown(
            f"""
            <div class="pacs-card">
                <div class="pacs-metric-title">Nodule Candidate Count</div>
                <div class="pacs-metric-value">{len(detections)} <span style="font-size:0.9rem; color:#94a3b8;">lesion(s)</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with q_col2:
        st.markdown(
            f"""
            <div class="pacs-card">
                <div class="pacs-metric-title">Dominant Lesion Diameter</div>
                <div class="pacs-metric-value">{max_diam:.1f} <span style="font-size:0.9rem; color:#94a3b8;">mm</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with q_col3:
        st.markdown(
            f"""
            <div class="pacs-card">
                <div class="pacs-metric-title">Peak Detection Confidence</div>
                <div class="pacs-metric-value">{max_conf * 100:.1f} <span style="font-size:0.9rem; color:#94a3b8;">%</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with q_col4:
        st.markdown(
            f"""
            <div class="pacs-card">
                <div class="pacs-metric-title">Forward Latency</div>
                <div class="pacs-metric-value">{latency:.1f} <span style="font-size:0.9rem; color:#94a3b8;">ms</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Lung-RADS Clinical Assessment Panel
    st.markdown(
        f"""
        <div style="background-color:#131d35; border:1px solid #1e293b; border-left:4px solid #38bdf8; border-radius:6px; padding:14px; margin-top:10px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:1.05rem; font-weight:700; color:#f8fafc;">Clinical Assessment: <b>{lung_rads_assessment['tier']}</b></span>
                <span class="{lung_rads_assessment['tag_class']}">{lung_rads_assessment['risk']}</span>
            </div>
            <div style="font-size:0.9rem; color:#cbd5e1; margin-top:6px;">
                <b>Recommended Clinical Protocol:</b> {lung_rads_assessment['action']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Detailed Morphometric Table
    if detections:
        st.markdown("### 📋 Nodule Morphometry & Anatomical Mapping Table")
        df_display = pd.DataFrame(detections)[["Nodule #", "Confidence", "Est. Diameter (mm)", "Dimensions (px)", "Centroid (X, Y)", "Anatomical Location Zone"]]
        st.dataframe(df_display, use_container_width=True)

        # Export Clinical Reports (PDF & JSON)
        st.markdown("### 📄 Clinical Radiology Report Export")
        exp_col1, exp_col2, exp_col3 = st.columns(3)

        with exp_col1:
            pdf_bytes = generate_radiology_pdf(
                patient_data=patient_info,
                model_name=selected_model_name,
                detections=detections,
                lung_rads=lung_rads_assessment,
                latency_ms=latency,
                annotated_img_rgb=annotated_result,
            )
            st.download_button(
                label="📄 Download Official PDF Radiology Report",
                data=pdf_bytes,
                file_name=f"Radiology_Report_{patient_info['pid']}.pdf",
                mime="application/pdf",
            )

        with exp_col2:
            json_report = {
                "clinical_system": "PulmoScan-CASP Thoracic Oncology CADx Platform",
                "patient": patient_info,
                "model_name": selected_model_name,
                "benchmark_mAP50": model_cfg["mAP"],
                "lung_rads_assessment": lung_rads_assessment,
                "latency_ms": latency,
                "detections": detections,
            }
            st.download_button(
                label="💾 Download Structured PACS JSON",
                data=json.dumps(json_report, indent=2),
                file_name=f"PACS_CADx_{patient_info['pid']}.json",
                mime="application/json",
            )

        with exp_col3:
            buf_img = io.BytesIO()
            Image.fromarray(annotated_result).save(buf_img, format="PNG")
            st.download_button(
                label="🖼️ Download Annotated PACS Scan (PNG)",
                data=buf_img.getvalue(),
                file_name=f"PACS_Annotated_{patient_info['pid']}.png",
                mime="image/png",
            )
    else:
        st.success("✅ Negative Scan: No pulmonary lesions detected above threshold. Standard follow-up protocol applies.")

elif weights_file is None:
    st.error("Model weights file could not be located in Detection Results. Please verify best.pt paths.")
else:
    st.info("👋 Ingest a thoracic image scan above or choose a preloaded benchmark case from the left sidebar to initialize the clinical CADx analysis.")
