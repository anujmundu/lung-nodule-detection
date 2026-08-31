"""
Annotation converter & formatter for NIH ChestX-ray14 dataset.
Reads data/nih_chestxray/BBox_List_2017.csv, converts bounding box coordinates
to standard YOLO format (.txt), and sets up dataset splits.
"""

import os
import sys
import shutil
import pandas as pd
from pathlib import Path


def convert_bbox_to_yolo(x, y, w, h, img_w=1024, img_h=1024):
    """
    Converts pixel coordinates [x, y, w, h] to normalized YOLO format [x_center, y_center, width, height].
    """
    x_center = (x + w / 2.0) / img_w
    y_center = (y + h / 2.0) / img_h
    norm_w = w / img_w
    norm_h = h / img_h
    return f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n"


def process_nih_dataset():
    workspace_dir = Path(__file__).parent.parent.resolve()
    nih_dir = workspace_dir / "data" / "nih_chestxray"
    csv_path = nih_dir / "BBox_List_2017.csv"
    images_dir = nih_dir / "images"
    labels_dir = nih_dir / "labels"

    print("\n========================================================")
    print("NIH ChestX-ray14 Bounding Box Processor")
    print(f"Target Directory: {nih_dir}")
    print("========================================================\n")

    if not csv_path.exists():
        print(f"[ERROR] BBox_List_2017.csv not found at: {csv_path}")
        print("Please place BBox_List_2017.csv in data/nih_chestxray/ before running this script.")
        return False

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Filter for Nodule or Mass if applicable
    nodule_df = df[df["Finding Label"].isin(["Nodule", "Mass"])].copy()
    print(f"[INFO] Total bounding box entries in CSV: {len(df)}")
    print(f"[INFO] Nodule/Mass bounding box entries: {len(nodule_df)}")

    labels_dir.mkdir(parents=True, exist_ok=True)
    converted_count = 0

    for img_name, group in nodule_df.groupby("Image Index"):
        txt_filename = Path(img_name).stem + ".txt"
        txt_path = labels_dir / txt_filename

        with open(txt_path, "w", encoding="utf-8") as f:
            for _, row in group.iterrows():
                try:
                    # BBox_List_2017 format: Bbox [x, y, w, h]
                    x = float(row["Bbox [x"]) if "Bbox [x" in row else float(row.iloc[2])
                    y = float(row["y"]) if "y" in row else float(row.iloc[3])
                    w = float(row["w"]) if "w" in row else float(row.iloc[4])
                    h = float(row["h]"]) if "h]" in row else float(row.iloc[5])

                    yolo_line = convert_bbox_to_yolo(x, y, w, h)
                    f.write(yolo_line)
                except Exception:
                    continue
        converted_count += 1

    print(f"[OK] Converted {converted_count} image label files into YOLO format at: {labels_dir}")
    
    # Update dataset YAML config
    yaml_content = f"""path: {nih_dir.as_posix()}
train: images
val: images
test: images
nc: 1
names: ['nodule']
"""
    yaml_path = workspace_dir / "data" / "nih_chestxray.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"[OK] Generated dataset YAML configuration: {yaml_path}")
    print("========================================================\n")
    return True


if __name__ == "__main__":
    process_nih_dataset()
