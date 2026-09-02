"""
Integration Tests for FastAPI Endpoints.
"""

import io
import pathlib
import platform
import sys
from pathlib import Path

# Cross-platform compatibility for Windows-trained PyTorch checkpoints on Linux/Docker
if platform.system() != "Windows":
    pathlib.WindowsPath = pathlib.PosixPath

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

WORKSPACE_DIR = Path(__file__).parent.parent.resolve()
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from app_api import app

client = TestClient(app)


def test_health_endpoint():
    """Verify that /health returns HTTP 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "device" in data
    assert "cuda_available" in data


def test_list_models_endpoint():
    """Verify that /v1/models lists available model variants."""
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["models"]) >= 3
    model_keys = [m["model_key"] for m in data["models"]]
    assert "x_nodule_sota" in model_keys


def test_predict_endpoint_with_dummy_image():
    """Verify that /v1/predict accepts an image and returns valid detection schema."""
    # Create a synthetic 640x640 test image
    dummy_img = np.ones((640, 640, 3), dtype=np.uint8) * 128
    _, encoded = cv2.imencode(".png", dummy_img)
    image_bytes = io.BytesIO(encoded.tobytes())

    response = client.post(
        "/v1/predict",
        files={"file": ("test_xray.png", image_bytes, "image/png")},
        params={"model": "x_nodule_sota", "conf_thres": 0.50},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "detections_count" in data
    assert "inference_time_ms" in data
    assert isinstance(data["detections"], list)
