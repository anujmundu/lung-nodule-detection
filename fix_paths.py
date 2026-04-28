# fix_paths.py
import os

for split in ['train', 'val', 'test']:
    file_path = f'data/processed/{split}.txt'
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Convert absolute paths to relative (images/filename.png)
    rel_lines = []
    for line in lines:
        filename = os.path.basename(line.strip())
        rel_lines.append(f'images/{filename}\n')
    
    with open(file_path, 'w') as f:
        f.writelines(rel_lines)
    
    print(f'{split}.txt converted to relative paths.') 