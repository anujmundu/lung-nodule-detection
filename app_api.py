"""
Production-Grade FastAPI Inference Service for YOLOv5-CASP.
Provides high-throughput asynchronous REST API endpoints for pulmonary nodule detection.
"""

import io
import os
import pathlib
import platform
import sys
import time
from pathlib import Path
from typing import List, Optional

# Cross-platform compatibility for Windows-trained PyTorch checkpoints on Linux/Docker
if platform.system() != "Windows":
    pathlib.WindowsPath = pathlib.PosixPath

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

# Ensure yolov5 root is on sys.path
WORKSPACE_DIR = Path(__file__).parent.resolve()
YOLOV5_DIR = WORKSPACE_DIR / "yolov5"
if str(YOLOV5_DIR) not in sys.path:
    sys.path.insert(0, str(YOLOV5_DIR))

# Import YOLOv5 core utilities
try:
    from models.experimental import attempt_load
    from utils.general import check_img_size, non_max_suppression, scale_boxes
    from utils.augmentations import letterbox
except ImportError:
    # Fallback to importing directly from yolov5 package
    from yolov5.models.experimental import attempt_load
    from yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes
    from yolov5.utils.augmentations import letterbox

# Initialize FastAPI App
app = FastAPI(
    title="🫁 YOLOv5-CASP Medical AI Engine",
    description="Production-grade asynchronous CAD inference API for automated pulmonary nodule detection in Chest X-Ray and CT images.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for cross-origin web apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Available Model Configurations
AVAILABLE_MODELS = {
    "x_nodule_sota": {
        "name": "YOLOv5-CASP (X-Nodule Radiographs)",
        "modality": "Chest Radiography (CXR)",
        "weights": [
            WORKSPACE_DIR / "Detection Results" / "1_YOLOv5_CASP_X_Nodule_SOTA" / "weights" / "best.pt",
            WORKSPACE_DIR / "yolov5" / "runs" / "train" / "casp_x_nodule_run4" / "weights" / "best.pt",
        ],
        "default_imgsz": 640,
        "mAP50": 0.809,
    },
    "nih_chestxray_sota": {
        "name": "YOLOv5-CASP (NIH ChestX-ray 14 100-Ep Transfer)",
        "modality": "Chest Radiography (CXR)",
        "weights": [
            WORKSPACE_DIR / "Detection Results" / "2_YOLOv5_CASP_NIH_ChestXray_100Ep_SOTA" / "weights" / "best.pt",
            WORKSPACE_DIR / "yolov5" / "runs" / "train" / "casp_nih_chestxray_nodules_run6" / "weights" / "best.pt",
        ],
        "default_imgsz": 640,
        "mAP50": 0.644,
    },
    "luna16_ct_sota": {
        "name": "YOLOv5-CASP (LUNA16 CT Patches)",
        "modality": "Computed Tomography (CT)",
        "weights": [
            WORKSPACE_DIR / "Detection Results" / "3_YOLOv5_CASP_LUNA16_CT_Patches" / "weights" / "best.pt",
            WORKSPACE_DIR / "yolov5" / "runs" / "train" / "casp_luna16_run8" / "weights" / "best.pt",
        ],
        "default_imgsz": 256,
        "mAP50": 0.382,
    },
}

# In-Memory Model Cache
MODEL_CACHE = {}
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_model(model_key: str = "x_nodule_sota"):
    """Load and cache YOLOv5-CASP model weights in memory."""
    if model_key not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model key '{model_key}'. Available: {list(AVAILABLE_MODELS.keys())}")

    if model_key in MODEL_CACHE:
        return MODEL_CACHE[model_key]

    config = AVAILABLE_MODELS[model_key]
    weights_path = None
    for candidate in config["weights"]:
        if candidate.exists():
            weights_path = str(candidate)
            break

    if not weights_path:
        raise FileNotFoundError(f"Model checkpoint for '{model_key}' not found in expected paths.")

    # Load model
    model = attempt_load(weights_path, device=DEVICE)
    model.eval()
    MODEL_CACHE[model_key] = model
    return model


# Pydantic Response Schemas
class BoundingBox(BaseModel):
    xmin: float = Field(..., description="Top-left X coordinate in pixels")
    ymin: float = Field(..., description="Top-left Y coordinate in pixels")
    xmax: float = Field(..., description="Bottom-right X coordinate in pixels")
    ymax: float = Field(..., description="Bottom-right Y coordinate in pixels")


class DetectionItem(BaseModel):
    class_id: int = Field(0, description="Class index (0 for nodule)")
    class_name: str = Field("nodule", description="Predicted class name")
    confidence: float = Field(..., description="Prediction confidence score between 0.0 and 1.0")
    bbox: BoundingBox
    center_normalized: List[float] = Field(..., description="Normalized [X_center, Y_center] coordinates")
    estimated_diameter_px: float = Field(..., description="Estimated nodule diameter in pixels")


class PredictionResponse(BaseModel):
    status: str = Field("success", description="Response status")
    model_name: str
    modality: str
    image_shape_original: List[int] = Field(..., description="[Height, Width] of original input")
    detections_count: int = Field(..., description="Number of detected pulmonary nodules")
    inference_time_ms: float = Field(..., description="Forward pass + NMS latency in milliseconds")
    detections: List[DetectionItem]


class HealthResponse(BaseModel):
    status: str
    device: str
    cuda_available: bool
    gpu_name: Optional[str]
    loaded_models: List[str]
    timestamp: float


# Endpoints
@app.get("/health", response_model=HealthResponse, tags=["Health & Diagnostics"])
def health_check():
    """Returns runtime system status and GPU telemetry."""
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return HealthResponse(
        status="healthy",
        device=str(DEVICE),
        cuda_available=torch.cuda.is_available(),
        gpu_name=gpu_name,
        loaded_models=list(MODEL_CACHE.keys()),
        timestamp=time.time(),
    )


@app.get("/v1/models", tags=["Model Registry"])
def list_models():
    """Lists available YOLOv5-CASP models and verified benchmark statistics."""
    models_info = []
    for key, val in AVAILABLE_MODELS.items():
        models_info.append(
            {
                "model_key": key,
                "name": val["name"],
                "modality": val["modality"],
                "default_imgsz": val["default_imgsz"],
                "mAP50_benchmark": val["mAP50"],
                "is_cached": key in MODEL_CACHE,
            }
        )
    return {"status": "success", "models": models_info}


def run_inference_on_bytes(
    image_bytes: bytes,
    model_key: str = "x_nodule_sota",
    conf_thres: float = 0.50,
    iou_thres: float = 0.45,
    imgsz: Optional[int] = None,
):
    """Core preprocessing, inference, and postprocessing pipeline."""
    # Decode image bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    img0 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img0 is None:
        raise HTTPException(status_code=400, detail="Invalid image payload. Unable to decode image.")

    h0, w0 = img0.shape[:2]

    # Get model and settings
    model = get_model(model_key)
    target_imgsz = imgsz or AVAILABLE_MODELS[model_key]["default_imgsz"]
    stride = int(model.stride.max())
    target_imgsz = check_img_size(target_imgsz, s=stride)

    # Preprocessing (Letterbox + Normalize)
    img = letterbox(img0, target_imgsz, stride=stride, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    im_tensor = torch.from_numpy(img).to(DEVICE)
    im_tensor = im_tensor.float() / 255.0
    if len(im_tensor.shape) == 3:
        im_tensor = im_tensor[None]  # Add batch dimension

    # Inference timing
    t0 = time.perf_counter()
    with torch.no_grad():
        pred = model(im_tensor)[0]
        pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres, max_det=300)
    t1 = time.perf_counter()
    latency_ms = round((t1 - t0) * 1000.0, 2)

    # Process detections
    detections = []
    det = pred[0]
    if len(det):
        det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], img0.shape).round()
        for *xyxy, conf, cls_id in reversed(det):
            x1, y1, x2, y2 = [float(x) for x in xyxy]
            conf_val = float(conf)
            class_idx = int(cls_id)
            bw = x2 - x1
            bh = y2 - y1
            cx_norm = round(((x1 + x2) / 2.0) / w0, 4)
            cy_norm = round(((y1 + y2) / 2.0) / h0, 4)
            diam_px = round(float(np.sqrt(bw * bh)), 2)

            detections.append(
                DetectionItem(
                    class_id=class_idx,
                    class_name="nodule",
                    confidence=round(conf_val, 4),
                    bbox=BoundingBox(
                        xmin=round(x1, 2),
                        ymin=round(y1, 2),
                        xmax=round(x2, 2),
                        ymax=round(y2, 2),
                    ),
                    center_normalized=[cx_norm, cy_norm],
                    estimated_diameter_px=diam_px,
                )
            )

    return img0, detections, latency_ms, h0, w0


@app.post("/v1/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_nodules(
    file: UploadFile = File(...),
    model: str = Query("x_nodule_sota", description="Model variant key"),
    conf_thres: float = Query(0.50, ge=0.01, le=0.99, description="Confidence threshold"),
    iou_thres: float = Query(0.45, ge=0.01, le=0.99, description="NMS IoU threshold"),
    imgsz: Optional[int] = Query(None, description="Inference resolution"),
):
    """
    Asynchronously detects pulmonary nodules in uploaded Chest X-ray or CT scans.
    Returns structured JSON with bounding box coordinates, confidence scores, and latency.
    """
    if model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model key '{model}'. Valid: {list(AVAILABLE_MODELS.keys())}")

    image_bytes = await file.read()
    _, detections, latency_ms, h0, w0 = run_inference_on_bytes(
        image_bytes,
        model_key=model,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        imgsz=imgsz,
    )

    return PredictionResponse(
        status="success",
        model_name=AVAILABLE_MODELS[model]["name"],
        modality=AVAILABLE_MODELS[model]["modality"],
        image_shape_original=[h0, w0],
        detections_count=len(detections),
        inference_time_ms=latency_ms,
        detections=detections,
    )


@app.post("/v1/predict/visual", tags=["Inference"])
async def predict_visual(
    file: UploadFile = File(...),
    model: str = Query("x_nodule_sota", description="Model variant key"),
    conf_thres: float = Query(0.50, ge=0.01, le=0.99),
    iou_thres: float = Query(0.45, ge=0.01, le=0.99),
):
    """
    Runs nodule detection and returns the annotated image directly with visual bounding boxes.
    """
    image_bytes = await file.read()
    img0, detections, _, _, _ = run_inference_on_bytes(
        image_bytes,
        model_key=model,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
    )

    # Draw boxes
    for item in detections:
        b = item.bbox
        x1, y1, x2, y2 = int(b.xmin), int(b.ymin), int(b.xmax), int(b.ymax)
        cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Nodule {item.confidence:.2f}"
        cv2.putText(img0, label, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    _, encoded_img = cv2.imencode(".png", img0)
    return Response(content=encoded_img.tobytes(), media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_api:app", host="0.0.0.0", port=8000, reload=True)
