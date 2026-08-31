"""
Evaluation module computing mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1-Score, and FPS.
Parses trained runs from yolov5/runs/train/ and merges with thesis benchmark data.
"""

import os
import sys
import pandas as pd
from pathlib import Path


def parse_trained_run_results(run_dir):
    """
    Parses results.csv from a YOLOv5 run directory to extract final metrics.
    """
    results_csv = Path(run_dir) / "results.csv"
    if not results_csv.exists():
        return None

    try:
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()
        if len(df) == 0:
            return None
        last_row = df.iloc[-1]

        precision = float(last_row.get("metrics/precision", 0.0))
        recall = float(last_row.get("metrics/recall", 0.0))
        map50 = float(last_row.get("metrics/mAP_0.5", 0.0))
        map50_95 = float(last_row.get("metrics/mAP_0.5:0.95", 0.0))
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "mAP0.5": round(map50, 4),
            "mAP0.5_0.95": round(map50_95, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
            "Epochs": len(df),
        }
    except Exception:
        return None


def scan_all_trained_runs(runs_dir):
    """
    Scans yolov5/runs/train directory and extracts metrics for the latest completed run per model & modality.
    """
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return {}

    all_runs = {}
    for run_path in runs_dir.iterdir():
        if run_path.is_dir():
            parsed = parse_trained_run_results(run_path)
            if parsed and parsed["Epochs"] > 0:
                name = run_path.name.lower()
                # Determine model and modality from folder name (e.g. casp_luna16_run8)
                parts = name.split("_")
                if len(parts) >= 2:
                    model = parts[0].upper()
                    modality = parts[1].upper()
                    if modality == "X" and len(parts) >= 3 and parts[2] == "NODULE":
                        modality = "X-NODULE"
                    elif modality == "LUNA16":
                        modality = "LUNA16"
                    elif modality == "MRI":
                        modality = "MRI"

                    key = (model, modality)
                    # Keep the run with maximum completed epochs or latest st_mtime
                    if key not in all_runs or parsed["Epochs"] > all_runs[key]["Epochs"]:
                        all_runs[key] = {**parsed, "RunFolder": run_path.name}

    return all_runs


def run_evaluation_summary():
    """
    Prints comparative summary table across all trained models and datasets.
    """
    workspace_dir = Path(__file__).parent.parent.resolve()
    runs_dir = workspace_dir / "yolov5" / "runs" / "train"

    # Default thesis reference values
    benchmark_data = {
        ("YOLOv5-CASP (Thesis Ref)", "X-Nodule"): {"mAP0.5": 0.809, "mAP0.5_0.95": 0.467, "Precision": 0.792, "Recall": 0.708, "F1": 0.748, "GPU_FPS": 70.98},
        ("YOLOv5-CASP (Thesis Ref)", "LUNA16"): {"mAP0.5": 0.382, "mAP0.5_0.95": 0.124, "Precision": 0.492, "Recall": 0.527, "F1": 0.509, "GPU_FPS": 70.98},
        ("YOLOv5-CASP (Thesis Ref)", "MRI"): {"mAP0.5": 0.615, "mAP0.5_0.95": 0.312, "Precision": 0.575, "Recall": 0.750, "F1": 0.651, "GPU_FPS": 70.98},
        ("Baseline YOLOv5s (Thesis Ref)", "X-Nodule"): {"mAP0.5": 0.214, "mAP0.5_0.95": 0.055, "Precision": 0.289, "Recall": 0.385, "F1": 0.330},
        ("Baseline YOLOv5s (Thesis Ref)", "LUNA16"): {"mAP0.5": 0.214, "mAP0.5_0.95": 0.055, "Precision": 0.289, "Recall": 0.385, "F1": 0.330},
        ("ASPP Only (Thesis Ref)", "LUNA16"): {"mAP0.5": 0.248, "mAP0.5_0.95": 0.078, "Precision": 0.341, "Recall": 0.429, "F1": 0.380},
        ("CoT3 Only (Thesis Ref)", "LUNA16"): {"mAP0.5": 0.205, "mAP0.5_0.95": 0.051, "Precision": 0.202, "Recall": 0.341, "F1": 0.254},
        ("CBAM Only (Thesis Ref)", "LUNA16"): {"mAP0.5": 0.001, "mAP0.5_0.95": 0.000, "Precision": 0.001, "Recall": 0.264, "F1": 0.002},
        ("YOLOv8s (Thesis Ref)", "X-Nodule"): {"mAP0.5": 0.807, "mAP0.5_0.95": 0.455, "Precision": 0.752, "Recall": 0.746, "F1": 0.749},
        ("YOLOv8s (Thesis Ref)", "LUNA16"): {"mAP0.5": 0.158, "mAP0.5_0.95": 0.041, "Precision": 0.225, "Recall": 0.297, "F1": 0.256},
    }

    # Dynamically parse actual trained run directories
    trained_runs = scan_all_trained_runs(runs_dir)

    rows = []
    # Add trained runs first
    for (model, modality), met in sorted(trained_runs.items()):
        rows.append({
            "Model": f"{model} (Trained Run)",
            "Dataset": modality,
            "mAP0.5": met["mAP0.5"],
            "mAP0.5_0.95": met["mAP0.5_0.95"],
            "Precision": met["Precision"],
            "Recall": met["Recall"],
            "F1": met["F1"],
            "Epochs": met["Epochs"],
            "RunFolder": met["RunFolder"],
        })

    # Add thesis reference rows
    for (m, d), met in benchmark_data.items():
        rows.append({"Model": m, "Dataset": d, "Epochs": "-", "RunFolder": "Reference", **met})

    df = pd.DataFrame(rows)
    df = df.fillna("-")
    print("\n=================== YOLOv5-CASP Evaluation Metrics Summary ===================")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    run_evaluation_summary()
