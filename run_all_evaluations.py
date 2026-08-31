"""
Master evaluation, detection, validation, and visualization script for YOLOv5-CASP.
Executes end-to-end evaluation across all models (Baseline, CBAM, ASPP, CoT3, CASP),
all imaging modalities (X-Nodule, LUNA16, Synthetic MRI), and multi-center clinical hospital scanner vendors.
"""

import os
import sys
import subprocess
import time
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


def print_banner(text):
    print("\n" + "=" * 75)
    print(f"  {text}")
    print("=" * 75 + "\n")


def run_master_pipeline():
    start_time = time.time()
    workspace_dir = Path(__file__).parent.resolve()
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    python_bin = get_python_executable()
    print_banner(f"YOLOv5-CASP Master Evaluation & Detection Pipeline (Python: {python_bin})")

    # Step 1: Execute Detection Inference on Test Sets
    print(">>> Step 1/5: Running Object Detection Inference on Test Sets...")
    for modality in ["x_nodule", "luna16", "mri"]:
        try:
            cmd = [
                python_bin,
                str(workspace_dir / "detect.py"),
                "--model", "casp",
                "--modality", modality,
                "--conf-thres", "0.25",
            ]
            print(f"Executing: {' '.join(cmd)}")
            res = subprocess.run(cmd, check=False)
            if res.returncode == 0:
                print(f"[OK] Detection complete for modality: {modality}")
            else:
                print(f"[WARN] Detection returned non-zero code {res.returncode} for {modality}")
        except Exception as e:
            print(f"[ERROR] Failed running detection for {modality}: {e}")

    # Step 2: Execute Hungarian Matching Failure Case Analysis
    print("\n>>> Step 2/5: Running Hungarian Matching Failure Case Analysis...")
    try:
        failure_script = workspace_dir / "src" / "analyze_failures.py"
        res = subprocess.run([python_bin, str(failure_script)], check=False)
        if res.returncode == 0:
            print("[OK] Failure analysis CSV successfully generated.")
        else:
            print(f"[WARN] Failure analysis returned exit code {res.returncode}")
    except Exception as e:
        print(f"[ERROR] Failure analysis failed: {e}")

    # Step 3: Run Evaluation Summary across all 15 Model/Modality Runs
    print("\n>>> Step 3/5: Parsing Metrics Summary across All Model Variants...")
    try:
        eval_script = workspace_dir / "src" / "evaluate.py"
        res = subprocess.run([python_bin, str(eval_script)], check=False)
        if res.returncode == 0:
            print("[OK] Multi-model evaluation summary completed.")
        else:
            print(f"[WARN] Evaluation summary returned exit code {res.returncode}")
    except Exception as e:
        print(f"[ERROR] Evaluation summary failed: {e}")

    # Step 4: Run Multi-Center Clinical Hospital Dataset Evaluation
    print("\n>>> Step 4/5: Running Multi-Center Hospital Scanner & Clinical Evaluation...")
    try:
        multicenter_script = workspace_dir / "src" / "evaluate_multicenter.py"
        res = subprocess.run([python_bin, str(multicenter_script)], check=False)
        if res.returncode == 0:
            print("[OK] Multi-center hospital clinical dataset evaluation completed.")
        else:
            print(f"[WARN] Multi-center evaluation returned exit code {res.returncode}")
    except Exception as e:
        print(f"[ERROR] Multi-center evaluation failed: {e}")

    # Step 5: Generate All Thesis Figures (Figures 5.1 - 5.6)
    print("\n>>> Step 5/5: Rendering Thesis Visualization Plots (Figures 5.1 - 5.6)...")
    try:
        plot_script = workspace_dir / "generate_all_plots.py"
        res = subprocess.run([python_bin, str(plot_script)], check=False)
        if res.returncode == 0:
            print("[OK] All thesis figures (5.1 - 5.6) generated successfully in evaluation_results/")
        else:
            print(f"[WARN] Plot generator returned exit code {res.returncode}")
    except Exception as e:
        print(f"[ERROR] Plot generation failed: {e}")

    elapsed = time.time() - start_time
    print_banner(f"Master Pipeline Completed in {elapsed:.2f} seconds")
    print("Output Artifacts:")
    print("  - Detection Predictions: yolov5/runs/detect/")
    print("  - Failure Analysis Report: failure_analysis.csv")
    print("  - Generated Figures (5.1 - 5.6): evaluation_results/")
    print("=========================================================================\n")


if __name__ == "__main__":
    run_master_pipeline()
