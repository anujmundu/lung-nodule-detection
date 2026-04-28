import matplotlib.pyplot as plt
import numpy as np

modalities = ['CT (LUNA16)', 'X‑ray (X‑Nodule)', 'MRI (Synthetic)']
mAP = [0.382, 0.809, 0.615]
colors = ['#3498db', '#2ecc71', '#e74c3c']

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(modalities, mAP, color=colors, edgecolor='black')
ax.set_ylabel('mAP@0.5', fontsize=12)
ax.set_title('YOLOv5‑CASP Detection Performance Across Modalities', fontsize=14)
ax.set_ylim(0, 1.0)
for bar, val in zip(bars, mAP):
    ax.annotate(f'{val:.3f}', xy=(bar.get_x()+bar.get_width()/2, val),
                xytext=(0, 5), textcoords='offset points', ha='center', fontsize=10)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('evaluation_results/modality_detection_comparison.png', dpi=200)
plt.show()
print("✅ Figure saved: evaluation_results/modality_detection_comparison.png")