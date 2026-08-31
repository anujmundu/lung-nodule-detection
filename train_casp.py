"""
Training script for YOLOv5-CASP and baseline/ablation variants on Lung Nodule Detection datasets.
Configures SGD optimizer (LR=0.1, Momentum=0.937, Weight Decay=0.0005) for 100 epochs with batch size 8.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def train_model(
    modality="x_nodule",
    model_variant="casp",
    data_path=None,
    weights="",
    epochs=100,
    batch_size=8,
    imgsz=640,
):
    """
    Executes training pipeline for specified model variant and dataset modality.
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    workspace_dir = Path(__file__).parent.resolve()
    yolov5_dir = workspace_dir / "yolov5"

    if not yolov5_dir.exists():
        raise FileNotFoundError("yolov5 directory not found. Please run environment setup first.")

    model_configs = {
        "casp": yolov5_dir / "models" / "yolov5s-casp.yaml",
        "baseline": yolov5_dir / "models" / "yolov5s.yaml",
        "cbam": yolov5_dir / "models" / "yolov5s-cbam.yaml",
        "aspp": yolov5_dir / "models" / "yolov5s-aspp.yaml",
        "cot3": yolov5_dir / "models" / "yolov5s-cot3.yaml",
    }
    model_cfg = model_configs.get(model_variant.lower(), yolov5_dir / "models" / "yolov5s-casp.yaml")

    if data_path:
        data_cfg = Path(data_path)
    else:
        if modality == "x_nodule":
            data_cfg = workspace_dir / "data" / "x_nodule_fixed.yaml"
        elif modality == "luna16":
            data_cfg = workspace_dir / "data" / "luna16_patches.yaml"
            imgsz = 256
            batch_size = 16
            epochs = 300
        elif modality == "mri":
            data_cfg = workspace_dir / "data" / "mri_detection_synthetic.yaml"
            batch_size = 4
            epochs = 50
        else:
            data_cfg = workspace_dir / "data" / "x_nodule_fixed.yaml"

    hyp_cfg = workspace_dir / "data" / "hyp.casp.yaml"
    data_stem = Path(data_cfg).stem.replace("_fixed", "").replace("_patches", "")
    run_name = f"{model_variant}_{data_stem}_run"

    cmd = [
        sys.executable,
        str(yolov5_dir / "train.py"),
        "--img",
        str(imgsz),
        "--batch",
        str(batch_size),
        "--epochs",
        str(epochs),
        "--data",
        str(data_cfg),
        "--cfg",
        str(model_cfg),
        "--hyp",
        str(hyp_cfg),
        "--weights",
        weights,
        "--optimizer",
        "SGD",
        "--workers",
        "2",
        "--name",
        run_name,
    ]

    logs_dir = workspace_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"train_{model_variant}_{data_stem}.log"

    print("\n========================================================")
    print(f"Launching Training: Model=[{model_variant.upper()}] Data=[{data_cfg.name}]")
    print(f"Config: {model_cfg.name} | Resolution: {imgsz}x{imgsz}")
    print(
        f"Hyperparameters: SGD (LR=0.1, Momentum=0.937, WeightDecay=0.0005, Epochs={epochs}, Batch={batch_size})"
    )
    print(f"Persistent Log File: {log_file}")
    print(f"Command: {' '.join(cmd)}")
    print("========================================================\n")

    with open(log_file, "a", encoding="utf-8") as f_log:
        header = f"\n=== Execution Started | Model: {model_variant} | Data: {data_cfg.name} | Epochs: {epochs} | Batch: {batch_size} ===\n"
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

        footer = f"\n=== Execution Completed | Return Code: {proc.returncode} | Log Saved to: {log_file} ===\n"
        sys.stdout.write(footer)
        f_log.write(footer)
        return proc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv5-CASP and baseline models on medical datasets.")
    parser.add_argument(
        "--modality",
        type=str,
        default="x_nodule",
        choices=["x_nodule", "luna16", "mri", "nih_chestxray", "custom"],
        help="Dataset modality",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="casp",
        choices=["casp", "baseline", "cbam", "aspp", "cot3", "mri", "luna16", "x_nodule"],
        help="Model architecture variant or dataset shortcut",
    )
    parser.add_argument("--data", type=str, default="", help="Path to custom dataset .yaml config file")
    parser.add_argument("--weights", type=str, default="", help="Pretrained weights path (optional)")
    parser.add_argument("--epochs", type=int, default=0, help="Epoch count (0 = use modality default)")
    parser.add_argument("--batch-size", type=int, default=0, help="Batch size (0 = use modality default)")
    parser.add_argument("--imgsz", type=int, default=0, help="Image resolution size (0 = use modality default)")

    args = parser.parse_args()

    if args.model in ["mri", "luna16", "x_nodule"]:
        modality = args.model
        model_variant = "casp"
    else:
        modality = args.modality
        model_variant = args.model

    kwargs = {}
    if args.epochs > 0:
        kwargs["epochs"] = args.epochs
    if args.batch_size > 0:
        kwargs["batch_size"] = args.batch_size
    if args.imgsz > 0:
        kwargs["imgsz"] = args.imgsz
    if args.data:
        kwargs["data_path"] = args.data

    train_model(
        modality=modality,
        model_variant=model_variant,
        weights=args.weights,
        **kwargs,
    )

