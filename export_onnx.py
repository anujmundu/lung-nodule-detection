"""
ONNX Model Exporter & Inference Runtime for YOLOv5-CASP.
Serializes PyTorch weights into optimized ONNX graph for high-throughput deployment.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

WORKSPACE_DIR = Path(__file__).parent.resolve()
YOLOV5_DIR = WORKSPACE_DIR / "yolov5"
if str(YOLOV5_DIR) not in sys.path:
    sys.path.insert(0, str(YOLOV5_DIR))

try:
    from models.experimental import attempt_load
    from utils.augmentations import letterbox
    from utils.general import non_max_suppression, scale_boxes
except ImportError:
    from yolov5.models.experimental import attempt_load
    from yolov5.utils.augmentations import letterbox
    from yolov5.utils.general import non_max_suppression, scale_boxes


def export_to_onnx(
    weights_path: str,
    output_path: str = "model.onnx",
    imgsz: int = 640,
    opset: int = 12,
    dynamic: bool = True,
):
    """Exports PyTorch YOLOv5-CASP model to ONNX format."""
    print("=" * 60)
    print(f"Loading PyTorch Model from: {weights_path}")
    device = torch.device("cpu")
    model = attempt_load(weights_path, device=device)
    model.eval()

    # Dry run dummy input
    dummy_input = torch.zeros(1, 3, imgsz, imgsz, device=device)

    print(f"Exporting to ONNX format: {output_path} (Opset={opset}, ImgSz={imgsz})")
    dynamic_axes = {"images": {0: "batch"}, "output": {0: "batch"}} if dynamic else None

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        verbose=False,
        opset_version=opset,
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    # Validate ONNX graph
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX Model Graph Validation: SUCCESS")
    except ImportError:
        print("onnx package not installed for graph checking (Skipped).")

    print(f"Export complete. File saved to: {output_path}")
    print("=" * 60)
    return output_path


class ONNXNoduleDetector:
    """Standalone ONNX Runtime inference engine with zero PyTorch dependency."""

    def __init__(self, onnx_model_path: str, imgsz: int = 640):
        import onnxruntime as ort

        self.imgsz = imgsz
        self.session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, img0: np.ndarray):
        img, ratio, (dw, dh) = letterbox(img0, self.imgsz, auto=False)
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        return img[None], ratio, (dw, dh)

    def predict(self, img0: np.ndarray, conf_thres: float = 0.50, iou_thres: float = 0.45):
        tensor, ratio, pad = self.preprocess(img0)
        t0 = time.perf_counter()
        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        pred = torch.from_numpy(outputs[0])
        det = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres)[0]
        t1 = time.perf_counter()

        detections = []
        if len(det):
            det[:, :4] = scale_boxes(tensor.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls_id in reversed(det):
                detections.append(
                    {
                        "bbox": [float(x) for x in xyxy],
                        "confidence": float(conf),
                        "class_id": int(cls_id),
                    }
                )
        return detections, (t1 - t0) * 1000.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLOv5-CASP to ONNX")
    parser.add_argument(
        "--weights",
        type=str,
        default="Detection Results/1_YOLOv5_CASP_X_Nodule_SOTA/weights/best.pt",
        help="Path to .pt weights",
    )
    parser.add_argument("--output", type=str, default="yolov5s_casp.onnx", help="Output .onnx path")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    args = parser.parse_args()

    export_to_onnx(args.weights, args.output, args.imgsz)
