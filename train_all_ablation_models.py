"""
Master automation script to train all missing ablation model variants
(CBAM, ASPP, CoT3, Baseline, CASP) across all 3 medical imaging modalities.
Persistently logs all stdout/stderr terminal outputs into training_logs/ directory.
"""

import os
import sys
import subprocess
import datetime
from pathlib import Path

# Master matrix of models and modalities to train
TRAINING_MATRIX = [
    # CT Patches (LUNA16)
    {"modality": "luna16", "model": "casp", "epochs": 50, "batch_size": 16},
    {"modality": "luna16", "model": "aspp", "epochs": 50, "batch_size": 16},
    {"modality": "luna16", "model": "cbam", "epochs": 50, "batch_size": 16},
    {"modality": "luna16", "model": "cot3", "epochs": 50, "batch_size": 16},
    {"modality": "luna16", "model": "baseline", "epochs": 50, "batch_size": 16},

    # Chest X-Rays (X-Nodule)
    {"modality": "x_nodule", "model": "aspp", "epochs": 50, "batch_size": 8},
    {"modality": "x_nodule", "model": "cbam", "epochs": 50, "batch_size": 8},
    {"modality": "x_nodule", "model": "cot3", "epochs": 50, "batch_size": 8},

    # Synthetic MRI
    {"modality": "mri", "model": "aspp", "epochs": 30, "batch_size": 4},
    {"modality": "mri", "model": "cbam", "epochs": 30, "batch_size": 4},
    {"modality": "mri", "model": "cot3", "epochs": 30, "batch_size": 4},
    {"modality": "mri", "model": "baseline", "epochs": 30, "batch_size": 4},
]


def run_batch_training():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    workspace_dir = Path(__file__).parent.resolve()
    runs_dir = workspace_dir / "yolov5" / "runs" / "train"
    logs_dir = workspace_dir / "training_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("=========================================================================")
    print(" Master Multi-Model Training Manager (Full File Logging Enabled)")
    print(f" Saved Logs Directory: {logs_dir}")
    print("=========================================================================\n")

    total_runs = len(TRAINING_MATRIX)
    for idx, item in enumerate(TRAINING_MATRIX, 1):
        modality = item["modality"]
        model = item["model"]
        epochs = item["epochs"]
        batch_size = item["batch_size"]

        run_folder = runs_dir / f"{model}_{modality}_run"
        best_weights = run_folder / "weights" / "best.pt"
        log_file = logs_dir / f"{model}_{modality}_run.log"

        if best_weights.exists():
            print(f"[{idx}/{total_runs}] SKIP: Model [{model.upper()}] on [{modality}] already trained at {best_weights}")
            continue

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{idx}/{total_runs}] STARTING: Model [{model.upper()}] on [{modality}] ({epochs} Epochs, Batch {batch_size})...")
        print(f"              Logging output to: {log_file}")

        cmd = [
            sys.executable,
            str(workspace_dir / "train_casp.py"),
            "--model", model,
            "--modality", modality,
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
        ]

        with open(log_file, "a", encoding="utf-8") as f_log:
            f_log.write(f"\n=== Training Start: {timestamp} | Model: {model} | Modality: {modality} ===\n")
            f_log.flush()

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            for line in process.stdout:
                try:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                except UnicodeEncodeError:
                    sys.stdout.write(line.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8"))
                    sys.stdout.flush()
                f_log.write(line)
                f_log.flush()

            process.wait()
            end_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f_log.write(f"\n=== Training Finished: {end_timestamp} | Exit Code: {process.returncode} ===\n")

        if process.returncode == 0:
            print(f"[{idx}/{total_runs}] COMPLETED: Model [{model.upper()}] on [{modality}] (Log saved: {log_file.name})\n")
        else:
            print(f"[{idx}/{total_runs}] ERROR: Model [{model.upper()}] on [{modality}] crashed with exit code {process.returncode}. Detailed log in {log_file}\n")

    print("=========================================================================")
    print(f" All Pending Runs Processed. Logs archived in: {logs_dir}")
    print("=========================================================================")


if __name__ == "__main__":
    run_batch_training()
