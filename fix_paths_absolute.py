# fix_paths_absolute.py
import os

base = r"D:\MEDICAL_LUNG_NODULE_DETECTION\data\processed"

for split in ['train', 'val', 'test']:
    file_path = os.path.join(base, f"{split}.txt")
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    abs_lines = []
    for line in lines:
        rel_path = line.strip()
        # rel_path is like "images/filename.png"
        abs_path = os.path.join(base, rel_path)
        abs_lines.append(abs_path + '\n')
    
    with open(file_path, 'w') as f:
        f.writelines(abs_lines)
    
    print(f"{split}.txt converted to absolute paths.")