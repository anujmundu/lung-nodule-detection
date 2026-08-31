"""
Creates a focused clinical research split for NIH ChestX-ray14.
Filters the 112,120 dataset down to 164 nodule/mass positive images
and a 2:1 control negative background set (492 total images).
Generates data/nih_chestxray_nodules.yaml for 50x faster training.
"""

import os
import sys
import shutil
import random
from pathlib import Path


def create_focused_nih_dataset():
    workspace_dir = Path(__file__).parent.parent.resolve()
    nih_dir = workspace_dir / "data" / "nih_chestxray"
    images_dir = nih_dir / "images"
    labels_dir = nih_dir / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        print(f"[ERROR] NIH dataset directories not found in {nih_dir}")
        return False

    label_files = list(labels_dir.glob("*.txt"))
    annotated_stems = {f.stem for f in label_files}

    print("\n========================================================")
    print("NIH ChestX-ray14 Focused Nodule Dataset Filter")
    print(f"Annotated Nodule/Mass Stems Found: {len(annotated_stems)}")
    print("========================================================\n")

    all_image_paths = list(images_dir.glob("*.png"))
    annotated_images = [p for p in all_image_paths if p.stem in annotated_stems]
    negative_images = [p for p in all_image_paths if p.stem not in annotated_stems]

    random.seed(42)
    # Sample 2:1 negative ratio
    sampled_negatives = random.sample(negative_images, min(len(negative_images), len(annotated_images) * 2))

    subset_dir = nih_dir / "filtered_nodules"
    sub_img_dir = subset_dir / "images"
    sub_lbl_dir = subset_dir / "labels"
    sub_img_dir.mkdir(parents=True, exist_ok=True)
    sub_lbl_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in annotated_images + sampled_negatives:
        dest_img = sub_img_dir / img_path.name
        if not dest_img.exists():
            try:
                os.link(str(img_path), str(dest_img))
            except Exception:
                shutil.copy(str(img_path), str(dest_img))

        lbl_src = labels_dir / f"{img_path.stem}.txt"
        if lbl_src.exists():
            dest_lbl = sub_lbl_dir / f"{img_path.stem}.txt"
            if not dest_lbl.exists():
                shutil.copy(str(lbl_src), str(dest_lbl))
        count += 1

    # Create focused YAML configuration
    yaml_content = f"""path: {subset_dir.as_posix()}
train: images
val: images
test: images
nc: 1
names: ['nodule']
"""
    yaml_path = workspace_dir / "data" / "nih_chestxray_nodules.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"[OK] Created focused research split ({count} total images) at: {subset_dir}")
    print(f"[OK] Generated dataset YAML config: {yaml_path}")
    print("========================================================\n")
    return True


if __name__ == "__main__":
    create_focused_nih_dataset()
