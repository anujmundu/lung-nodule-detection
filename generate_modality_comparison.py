# generate_modality_comparison.py
# Creates a grouped bar chart comparing YOLOv5‑CASP detection performance across three modalities:
# CT (LUNA16), X‑ray (X‑Nodule), and MRI synthetic (proof‑of‑concept).
import matplotlib.pyplot as plt
import numpy as np

# Data from thesis results – all detection tasks
modalities = [
    'CT (LUNA16)\nDetection',
    'X-ray (X-Nodule)\nDetection',
    'MRI (Synthetic)\nDetection'
]

metrics = {
    'mAP@0.5':   [0.382, 0.809, 0.615],
    'Precision': [0.492, 0.792, 0.575],
    'Recall':    [0.527, 0.708, 0.750],
    'F1 Score':  [0.509, 0.748, 0.652]
}

# Publication‑style palette (same as before)
colors = ['#01696f', '#2f8fbd', '#a9d3e8', '#d9eaf2']

x = np.arange(len(modalities))
width = 0.18

fig, ax = plt.subplots(figsize=(12.5, 7.5))
fig.patch.set_facecolor('#f9f8f5')
ax.set_facecolor('white')

for i, (attribute, measurement) in enumerate(metrics.items()):
    offset = (i - 1.5) * width
    rects = ax.bar(
        x + offset,
        measurement,
        width,
        label=attribute,
        color=colors[i],
        edgecolor='#d0d7de',
        linewidth=0.8
    )
    ax.bar_label(rects, padding=3, fmt='%.3f', fontsize=9, color='#1f2937')

ax.set_ylabel('Score', fontsize=12, labelpad=10)
ax.set_title(
    'YOLOv5-CASP Detection Performance Across Modalities',
    fontsize=16,
    weight='bold',
    pad=14
)
ax.text(
    0.5, 1.001,
    'Cross‑modality detection comparison: CT patches (LUNA16), chest X‑ray (X‑Nodule), and synthetic MRI',
    transform=ax.transAxes,
    ha='center',
    va='bottom',
    fontsize=11,
    color='#6b7280'
)

ax.set_xticks(x)
ax.set_xticklabels(modalities, fontsize=11)
ax.set_ylim(0, 1.08)

ax.grid(True, axis='y', linestyle='--', alpha=0.25)
ax.set_axisbelow(True)

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.spines['left'].set_color('#c7c7c7')
ax.spines['bottom'].set_color('#c7c7c7')

ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.12),
    ncol=2,
    frameon=False,
    fontsize=10
)

plt.tight_layout()
plt.savefig(
    'evaluation_results/modality_comparison.png',
    dpi=300,
    bbox_inches='tight',
    facecolor=fig.get_facecolor()
)
plt.close()

print("Figure saved: evaluation_results/modality_comparison.png")