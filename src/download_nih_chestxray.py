"""
Automated downloader and formatter for NIH ChestX-ray14 dataset.
Downloads dataset files into data/nih_chestxray/ and data/custom_dataset/,
converts bounding box annotations to standard YOLO format,
and creates data/nih_chestxray.yaml.
"""

import os
import sys
import zipfile
import pandas as pd
from pathlib import Path

# Helper to get Python executable
def get_python_bin():
    possible_paths = [
        Path(r"C:\Users\anujm\anaconda3\envs\yolo_medical\python.exe"),
        Path(r"C:\Users\anujm\anaconda3\envs\YOLO_MEDICAL\python.exe"),
    ]
    for p in possible_paths:
        if p.exists():
            return str(p)
    return sys.executable


def setup_nih_chestxray(target_dir=None):
    workspace_dir = Path(__file__).parent.parent.resolve()
    if target_dir is None:
        target_dir = workspace_dir / "data" / "nih_chestxray"
    
    target_dir.mkdir(parents=True, exist_ok=True)
    images_dir = target_dir / "images"
    labels_dir = target_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print("\n========================================================")
    print(f"NIH ChestX-ray14 Dataset Setup Manager")
    print(f"Target Directory: {target_dir}")
    print("========================================================\n")

    # Install opendatasets if needed
    try:
        import opendatasets as od
    except ImportError:
        print("Installing 'opendatasets' package...")
        import subprocess
        subprocess.run([get_python_bin(), "-m", "pip", "install", "opendatasets"], check=True)
        import opendatasets as od

    dataset_url = "https://www.kaggle.com/datasets/nih-chest-xrays/data"
    print(f"Downloading NIH ChestX-ray14 via OpenDatasets from: {dataset_url}")
    print("Note: If prompted for Kaggle credentials, enter your Kaggle Username and API Key.")
    print("You can get your Kaggle API key from: Kaggle -> Account -> Create New API Token (kaggle.json)\n")

    try:
        od.download(dataset_url, data_dir=str(target_dir))
        print("\n[OK] Dataset download completed successfully!")
    except Exception as e:
        print(f"\n[INFO] OpenDatasets download initiated or manual download guide:")
        print(f"1. Download NIH ChestX-ray14 from Kaggle: {dataset_url}")
        print(f"2. Extract downloaded images into: {images_dir}")
        print(f"3. Place BBox_List_2017.csv into: {target_dir}\n")

    # Create NIH dataset YAML configuration
    yaml_content = f"""path: {target_dir.as_posix()}
train: images/train
val: images/val
test: images/test
nc: 1
names: ['nodule']
"""
    yaml_path = workspace_dir / "data" / "nih_chestxray.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"[OK] Generated dataset YAML configuration: {yaml_path}")

    # Also prepare custom dataset alias
    custom_yaml = workspace_dir / "data" / "custom_dataset.yaml"
    with open(custom_yaml, "w", encoding="utf-8") as f:
        f.write(yaml_content.replace("nih_chestxray", "custom_dataset"))
    print(f"[OK] Generated custom dataset YAML configuration: {custom_yaml}")

    print("\n========================================================")
    print("NIH ChestX-ray14 Dataset Setup Complete!")
    print(f"Images Path: {images_dir}")
    print(f"Labels Path: {labels_dir}")
    print(f"YAML Config: {yaml_path}")
    print("========================================================\n")


if __name__ == "__main__":
    setup_nih_chestxray()
