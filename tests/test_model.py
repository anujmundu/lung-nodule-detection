"""
Unit Tests for Custom YOLOv5-CASP Architectural Modules & Forward Pass.
"""

import pathlib
import platform
import sys
from pathlib import Path

# Cross-platform compatibility for Windows-trained PyTorch checkpoints on Linux/Docker
if platform.system() != "Windows":
    pathlib.WindowsPath = pathlib.PosixPath

import pytest
import torch

WORKSPACE_DIR = Path(__file__).parent.parent.resolve()
YOLOV5_DIR = WORKSPACE_DIR / "yolov5"
if str(YOLOV5_DIR) not in sys.path:
    sys.path.insert(0, str(YOLOV5_DIR))

from models.common import CBAM, ASPP, CoT3, ChannelAttention, SpatialAttention
from models.experimental import attempt_load


class TestCustomModules:
    """Test individual custom PyTorch attention & pyramid modules."""

    def test_cbam_channel_attention(self):
        ca = ChannelAttention(channels=64, reduction=16)
        x = torch.randn(2, 64, 32, 32)
        out = ca(x)
        assert out.shape == (2, 64, 32, 32), f"Expected shape (2, 64, 32, 32), got {out.shape}"

    def test_cbam_spatial_attention(self):
        sa = SpatialAttention(kernel_size=7)
        x = torch.randn(2, 64, 32, 32)
        out = sa(x)
        assert out.shape == (2, 64, 32, 32), f"Expected shape (2, 64, 32, 32), got {out.shape}"

    def test_cbam_full_module(self):
        cbam = CBAM(c1=128, ratio=16, kernel_size=7)
        x = torch.randn(2, 128, 40, 40)
        out = cbam(x)
        assert out.shape == x.shape, f"CBAM must preserve input dimensions. Got {out.shape} for input {x.shape}"

    def test_aspp_module(self):
        aspp = ASPP(c1=256, c2=256, rates=[1, 3, 5, 7])
        x = torch.randn(2, 256, 20, 20)
        out = aspp(x)
        assert out.shape == (2, 256, 20, 20), f"ASPP output shape mismatch: {out.shape}"

    def test_cot3_module(self):
        cot = CoT3(c1=128, c2=128, n=1, shortcut=True, g=1, e=0.5)
        x = torch.randn(2, 128, 32, 32)
        out = cot(x)
        assert out.shape == (2, 128, 32, 32), f"CoT3 output shape mismatch: {out.shape}"


class TestModelLoading:
    """Test full YOLOv5-CASP checkpoint loading and inference."""

    def test_x_nodule_checkpoint_load(self):
        weights_path = WORKSPACE_DIR / "Detection Results" / "1_YOLOv5_CASP_X_Nodule_SOTA" / "weights" / "best.pt"
        if not weights_path.exists():
            weights_path = WORKSPACE_DIR / "yolov5" / "runs" / "train" / "casp_x_nodule_run4" / "weights" / "best.pt"

        assert weights_path.exists(), f"Checkpoint not found at {weights_path}"
        model = attempt_load(str(weights_path), device=torch.device("cpu"))
        model.eval()

        dummy = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            pred = model(dummy)

        assert pred is not None, "Model prediction returned None"
        assert len(pred) > 0, "Model prediction tensor is empty"
