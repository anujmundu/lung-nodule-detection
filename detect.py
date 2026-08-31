"""
Inference & Visualization script for YOLOv5-CASP.
Runs prediction on input images/directories and draws bounding boxes with confidence scores.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def get_python_executable():
    """
    Returns appropriate Python interpreter executable (preferring yolo_medical conda env).
    """
    possible_paths = [
        Path(r"C:\Users\anujm\anaconda3\envs\yolo_medical\python.exe"),
        Path(r"C:\Users\anujm\anaconda3\envs\YOLO_MEDICAL\python.exe"),
    ]
    for p in possible_paths:
        if p.exists():
            return str(p)
    return sys.executable


def find_best_weights(model="casp", modality="x_nodule", workspace_dir=None):
    """
    Dynamically finds the best trained weights checkpoint for specified model and modality.
    """
    if workspace_dir is None:
        workspace_dir = Path(__file__).parent.resolve()

    runs_dir = workspace_dir / "yolov5" / "runs" / "train"
    if runs_dir.exists():
        pattern = f"{model.lower()}_{modality.lower()}_run*/weights/best.pt"
        matching_weights = list(runs_dir.glob(pattern))
        if matching_weights:
            matching_weights.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return matching_weights[0]

        any_model_weights = list(runs_dir.glob(f"*{model.lower()}*/weights/best.pt"))
        if any_model_weights:
            any_model_weights.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return any_model_weights[0]

    static_weights = workspace_dir / "weights" / "casp" / "casp_xray_best.pt"
    return static_weights


def get_default_source(modality="x_nodule", workspace_dir=None):
    """
    Returns default dataset image directory based on target modality.
    """
    if workspace_dir is None:
        workspace_dir = Path(__file__).parent.resolve()

    if modality == "luna16":
        source = workspace_dir / "data" / "processed_patches" / "images" / "val"
        if not source.exists():
            source = workspace_dir / "data" / "processed_patches" / "images"
    elif modality == "mri":
        source = workspace_dir / "data" / "mri_detection_synthetic" / "images" / "val"
    else:
        source = workspace_dir / "data" / "x_nodule" / "test" / "images"

    return source


def run_detection(weights=None, source=None, conf_thres=0.25, imgsz=640, model="casp", modality="x_nodule"):
    """
    Executes YOLOv5-CASP inference pipeline.
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    workspace_dir = Path(__file__).parent.resolve()
    yolov5_dir = workspace_dir / "yolov5"

    if not yolov5_dir.exists():
        raise FileNotFoundError("yolov5 directory not found.")

    if not weights:
        weights = find_best_weights(model=model, modality=modality, workspace_dir=workspace_dir)

    if not source:
        source = get_default_source(modality=modality, workspace_dir=workspace_dir)

    weights_path = Path(weights)
    if not weights_path.exists():
        print(f"WARNING: Specified weights file not found: {weights_path}")
        print("Searching for alternative trained weights...")
        fallback = find_best_weights(model="casp", modality="x_nodule", workspace_dir=workspace_dir)
        if fallback.exists():
            print(f"Using alternative trained checkpoint: {fallback}")
            weights_path = fallback
        else:
            raise FileNotFoundError(f"No valid weights found at {weights_path}")

    python_bin = get_python_executable()
    cmd = [
        python_bin,
        str(yolov5_dir / "detect.py"),
        "--weights", str(weights_path),
        "--source", str(source),
        "--img", str(imgsz),
        "--conf-thres", str(conf_thres),
        "--name", f"{model}_{modality}_detection",
        "--save-txt",
        "--save-conf",
        "--exist-ok",
    ]

    logs_dir = workspace_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    source_stem = Path(source).stem if source else "default"
    log_file = logs_dir / f"detect_{model}_{source_stem}.log"

    print(f"\n========================================================")
    print(f"Launching YOLOv5-CASP Detection Pipeline")
    print(f"Model: [{model.upper()}] | Modality: [{modality}]")
    print(f"Python Environment: {python_bin}")
    print(f"Weights: {weights_path}")
    print(f"Source: {source}")
    print(f"Persistent Log File: {log_file}")
    print(f"Command: {' '.join(cmd)}")
    print("========================================================\n")

    with open(log_file, "a", encoding="utf-8") as f_log:
        header = f"\n=== Detection Execution Started | Model: {model} | Source: {source} ===\n"
        f_log.write(header)
        f_log.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        for line in proc.stdout:
            try:
                sys.stdout.write(line)
                sys.stdout.flush()
            except UnicodeEncodeError:
                sys.stdout.write(line.encode("ascii", errors="replace").decode("ascii"))
                sys.stdout.flush()
            f_log.write(line)
            f_log.flush()
        proc.wait()

        footer = f"\n=== Detection Completed | Return Code: {proc.returncode} | Log Saved to: {log_file} ===\n"
        sys.stdout.write(footer)
        f_log.write(footer)
        return proc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv5-CASP detection on images.")
    parser.add_argument("--model", type=str, default="casp", help="Model variant (casp, baseline, cbam, aspp, cot3)")
    parser.add_argument("--modality", type=str, default="x_nodule", help="Dataset modality (x_nodule, luna16, mri)")
    parser.add_argument("--weights", type=str, default="", help="Path to weights .pt file")
    parser.add_argument("--source", type=str, default="", help="Path to image or directory")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    args = parser.parse_args()

    run_detection(
        weights=args.weights,
        source=args.source,
        conf_thres=args.conf_thres,
        imgsz=args.imgsz,
        model=args.model,
        modality=args.modality,
    )

