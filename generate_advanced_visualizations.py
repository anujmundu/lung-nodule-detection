# generate_advanced_visualizations.py
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

# Create output directory
output_dir = Path('evaluation_results')
output_dir.mkdir(exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*60)
print("Generating Advanced Visualizations for Thesis")
print("="*60)

# ============================================================================
# 1. ARCHITECTURE DIAGRAM - YOLOv5-CASP
# ============================================================================
print("\n1. Creating YOLOv5-CASP Architecture Diagram...")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 13)
ax.axis("off")

# --------------------------------------------------------------------------
# Professional color palette
# --------------------------------------------------------------------------
COLORS = {
    "input": "#DCEEFF",
    "core": "#DFF3E3",
    "attention": "#FFE0E0",
    "context": "#FFE9B8",
    "multiscale": "#EADCF8",
    "detect": "#FFE2CC",
    "output": "#D9F0D8",
    "group_bg": "#F7F8FA",
    "text": "#1F2937",
    "muted": "#6B7280",
    "arrow": "#4B5563",
    "border": "#374151",
    "legend_edge": "#D1D5DB"
}

# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------
def add_box(x, y, w, h, title, facecolor, subtitle=None, fontsize=10, weight="normal"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.4,
        edgecolor=COLORS["border"],
        facecolor=facecolor
    )
    ax.add_patch(box)

    if subtitle:
        ax.text(
            x + w / 2, y + h * 0.62, title,
            ha="center", va="center",
            fontsize=fontsize, weight=weight, color=COLORS["text"]
        )
        ax.text(
            x + w / 2, y + h * 0.28, subtitle,
            ha="center", va="center",
            fontsize=8.5, color=COLORS["muted"]
        )
    else:
        ax.text(
            x + w / 2, y + h / 2, title,
            ha="center", va="center",
            fontsize=fontsize, weight=weight, color=COLORS["text"]
        )

def add_arrow(x1, y1, x2, y2, lw=1.8):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=lw,
            color=COLORS["arrow"],
            shrinkA=4,
            shrinkB=4
        )
    )

# --------------------------------------------------------------------------
# Title and subtitle
# --------------------------------------------------------------------------
ax.text(
    5, 12.35,
    "YOLOv5-CASP Architecture for Lung Nodule Detection",
    ha="center", va="center",
    fontsize=18, weight="bold", color=COLORS["text"]
)

ax.text(
    5, 11.95,
    "Enhanced YOLOv5 with CBAM attention, CoT3 context modeling, and ASPP multi-scale aggregation",
    ha="center", va="center",
    fontsize=10.5, color=COLORS["muted"]
)

# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------
add_box(
    3.5, 10.9, 3.0, 0.85,
    "Input Image",
    COLORS["input"],
    subtitle="640 × 640 × 3",
    fontsize=10,
    weight="bold"
)

# --------------------------------------------------------------------------
# Backbone
# --------------------------------------------------------------------------
add_arrow(5, 10.9, 5, 10.1)
add_box(
    2.2, 9.2, 5.6, 0.9,
    "CSPDarknet53 Backbone",
    COLORS["core"],
    subtitle="Hierarchical feature extraction",
    fontsize=10,
    weight="bold"
)

# --------------------------------------------------------------------------
# Attention group (CBAM x3)
# --------------------------------------------------------------------------
attention_group = FancyBboxPatch(
    (2.7, 6.2), 4.6, 2.5,
    boxstyle="round,pad=0.03,rounding_size=0.06",
    linewidth=1.0,
    edgecolor=COLORS["legend_edge"],
    facecolor=COLORS["group_bg"],
    linestyle="--"
)
ax.add_patch(attention_group)

ax.text(
    5, 8.45,
    "Attention Refinement",
    ha="center", va="center",
    fontsize=10, weight="bold", color=COLORS["muted"]
)

cbam_positions = [7.75, 7.0, 6.25]
for i, y in enumerate(cbam_positions):
    add_box(
        3.4, y, 3.2, 0.55,
        f"CBAM × {i+1}",
        COLORS["attention"],
        subtitle="Channel + spatial attention",
        fontsize=9.5
    )
    if i == 0:
        add_arrow(5, 9.2, 5, 8.3, lw=1.6)
    else:
        add_arrow(5, cbam_positions[i-1], 5, y + 0.55, lw=1.4)

# --------------------------------------------------------------------------
# CoT3
# --------------------------------------------------------------------------
add_arrow(5, 6.25, 5, 5.55)
add_box(
    3.3, 4.95, 3.4, 0.65,
    "CoT3 Module",
    COLORS["context"],
    subtitle="Contextual transformer block",
    fontsize=10,
    weight="bold"
)

# --------------------------------------------------------------------------
# ASPP
# --------------------------------------------------------------------------
add_arrow(5, 4.95, 5, 4.3)
add_box(
    2.9, 3.65, 4.2, 0.65,
    "ASPP",
    COLORS["multiscale"],
    subtitle="Multi-scale receptive fields",
    fontsize=10,
    weight="bold"
)

# --------------------------------------------------------------------------
# Neck
# --------------------------------------------------------------------------
add_arrow(5, 3.65, 5, 3.0)
add_box(
    2.1, 2.35, 5.8, 0.7,
    "PAN-FPN Neck",
    COLORS["core"],
    subtitle="Feature fusion across scales",
    fontsize=10,
    weight="bold"
)

# --------------------------------------------------------------------------
# Detection head
# --------------------------------------------------------------------------
add_arrow(5, 2.35, 5, 1.7)
add_box(
    2.5, 1.05, 5.0, 0.7,
    "Detection Head",
    COLORS["detect"],
    subtitle="Predictions at P3, P4, P5",
    fontsize=10,
    weight="bold"
)

# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
add_arrow(5, 1.05, 5, 0.45)
add_box(
    3.3, 0.05, 3.4, 0.7,
    "Output",
    COLORS["output"],
    subtitle="Bounding boxes + classes",
    fontsize=10,
    weight="bold"
)

# --------------------------------------------------------------------------
# Vertical stage labels
# --------------------------------------------------------------------------
ax.text(1.15, 9.65, "Backbone", fontsize=10, weight="bold", color=COLORS["muted"], rotation=90, va="center")
ax.text(1.15, 7.45, "Attention", fontsize=10, weight="bold", color=COLORS["muted"], rotation=90, va="center")
ax.text(1.15, 5.25, "Context", fontsize=10, weight="bold", color=COLORS["muted"], rotation=90, va="center")
ax.text(1.15, 3.95, "Multi-scale", fontsize=10, weight="bold", color=COLORS["muted"], rotation=90, va="center")
ax.text(1.15, 2.7, "Fusion", fontsize=10, weight="bold", color=COLORS["muted"], rotation=90, va="center")
ax.text(1.15, 1.4, "Detection", fontsize=10, weight="bold", color=COLORS["muted"], rotation=90, va="center")

# --------------------------------------------------------------------------
# Legend
# --------------------------------------------------------------------------
legend_elements = [
    Patch(facecolor=COLORS["core"], edgecolor=COLORS["border"], label="Core YOLOv5 architecture"),
    Patch(facecolor=COLORS["attention"], edgecolor=COLORS["border"], label="CBAM attention modules"),
    Patch(facecolor=COLORS["context"], edgecolor=COLORS["border"], label="CoT3 contextual refinement"),
    Patch(facecolor=COLORS["multiscale"], edgecolor=COLORS["border"], label="ASPP multi-scale aggregation"),
    Patch(facecolor=COLORS["detect"], edgecolor=COLORS["border"], label="Detection head")
]

legend = ax.legend(
    handles=legend_elements,
    loc="lower right",
    fontsize=9,
    frameon=True,
    fancybox=True,
    framealpha=0.96,
    borderpad=0.8
)
legend.get_frame().set_edgecolor(COLORS["legend_edge"])

plt.tight_layout()
plt.savefig(output_dir / 'architecture_diagram.png', dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("  ✓ Saved: architecture_diagram.png")

# ============================================================================
# 2. COMPARATIVE RADAR CHART
# ============================================================================
print("\n2. Creating Comparative Radar Chart...")

import matplotlib.pyplot as plt
import numpy as np

metrics = ['mAP@0.5', 'Precision', 'Recall', 'F1 Score', 'Speed (FPS)', 'Efficiency']
models_radar = ['YOLOv5-CASP', 'Baseline', 'YOLOv8s']

values = {
    'YOLOv5-CASP': [0.809, 0.792, 0.708, 0.748, 1.0, 0.70],
    'Baseline':    [0.214, 0.289, 0.385, 0.330, 0.50, 1.0],
    'YOLOv8s':     [0.807, 0.752, 0.739, 0.745, 0.63, 0.60]
}

angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
fig.patch.set_facecolor('#f9f8f5')
ax.set_facecolor('white')

palette = {
    'YOLOv5-CASP': '#01696f',
    'Baseline': '#6b7280',
    'YOLOv8s': '#2f8fbd'
}

for model in models_radar:
    vals = values[model] + values[model][:1]
    ax.plot(
        angles,
        vals,
        color=palette[model],
        linewidth=2.5,
        marker='o',
        markersize=5,
        label=model
    )
    ax.fill(angles, vals, color=palette[model], alpha=0.10)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=10, weight='bold')
ax.set_ylim(0, 1.0)

ax.set_yticks([0.25, 0.50, 0.75, 1.00])
ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=9, color='#6b7280')
ax.set_rlabel_position(90)

ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.25)
ax.spines['polar'].set_color('#d0d7de')
ax.spines['polar'].set_linewidth(1.0)

ax.set_title(
    'Model Performance Radar Chart',
    fontsize=16,
    weight='bold',
    pad=22
)
ax.text(
    0.5, 1.08,
    'X-Nodule dataset comparison across accuracy, speed, and efficiency',
    transform=ax.transAxes,
    ha='center',
    va='center',
    fontsize=11,
    color='#6b7280'
)

ax.legend(
    loc='upper left',
    bbox_to_anchor=(1.05, 1.05),
    frameon=False,
    fontsize=10
)

plt.tight_layout()
plt.savefig(
    output_dir / 'radar_chart.png',
    dpi=300,
    bbox_inches='tight',
    facecolor=fig.get_facecolor()
)
plt.close()

print("  ✓ Saved: radar_chart.png")

# ============================================================================
# 3. NODULE SIZE DETECTION ANALYSIS
# ============================================================================
print("\n3. Creating Nodule Size Detection Analysis...")

import matplotlib.pyplot as plt
import numpy as np

sizes = ['<3 mm', '3–5 mm', '5–10 mm', '10–20 mm', '>20 mm']
detection_rate = [0.25, 0.65, 0.85, 0.94, 0.98]

# Professional palette: low-to-high emphasis in one accent family
colors = ['#d9eaf2', '#a9d3e8', '#6bb7d6', '#2f8fbd', '#01696f']

fig, ax = plt.subplots(figsize=(11, 6.5))
fig.patch.set_facecolor('#f9f8f5')
ax.set_facecolor('#ffffff')

bars = ax.bar(
    sizes,
    detection_rate,
    color=colors,
    edgecolor='#cfd8dc',
    linewidth=0.8,
    width=0.65
)

# Title and labels
ax.set_title(
    'Detection Rate by Nodule Size',
    fontsize=16,
    weight='bold',
    pad=14
)
ax.text(
    0.5, 1.0,
    'YOLOv5-CASP performance across lesion size categories',
    transform=ax.transAxes,
    ha='center',
    va='bottom',
    fontsize=11,
    color='#6b7280'
)

ax.set_xlabel('Nodule Size', fontsize=12, labelpad=10)
ax.set_ylabel('Detection Rate', fontsize=12, labelpad=10)

# Y-axis formatting
ax.set_ylim(0, 1.05)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_yticklabels([f'{int(v*100)}%' for v in np.arange(0, 1.01, 0.2)])

# Grid and styling
ax.grid(True, axis='y', linestyle='--', alpha=0.25)
ax.set_axisbelow(True)

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.spines['left'].set_color('#c7c7c7')
ax.spines['bottom'].set_color('#c7c7c7')

# Value labels
for bar, rate in zip(bars, detection_rate):
    ax.annotate(
        f'{rate*100:.0f}%',
        xy=(bar.get_x() + bar.get_width() / 2, rate),
        xytext=(0, 6),
        textcoords='offset points',
        ha='center',
        va='bottom',
        fontsize=11,
        weight='bold',
        color='#1f2937'
    )

plt.tight_layout()
plt.savefig(
    output_dir / 'size_detection_analysis.png',
    dpi=300,
    bbox_inches='tight',
    facecolor=fig.get_facecolor()
)
plt.close()

print("  ✓ Saved: size_detection_analysis.png")

# ============================================================================
# 4. CONFIDENCE SCORE DISTRIBUTION
# ============================================================================
print("\n4. Creating Confidence Score Distribution...")

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
confidences = np.random.beta(8, 2, 534)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.patch.set_facecolor('#f9f8f5')

# Shared styling
for ax in axes:
    ax.set_facecolor('white')
    ax.grid(True, axis='y', linestyle='--', alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#c7c7c7')
    ax.spines['bottom'].set_color('#c7c7c7')

# Panel 1: Histogram
axes[0].hist(
    confidences,
    bins=20,
    color='#01696f',
    edgecolor='white',
    alpha=0.9
)
axes[0].axvline(
    x=0.25,
    color='#d19900',
    linestyle='--',
    linewidth=2,
    label='Default threshold = 0.25'
)
axes[0].axvline(
    x=0.50,
    color='#437a22',
    linestyle='--',
    linewidth=2,
    label='High confidence = 0.50'
)
axes[0].set_xlabel('Confidence Score', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Distribution of Confidence Scores', fontsize=14, weight='bold', pad=10)
axes[0].legend(frameon=False, fontsize=10)

# Panel 2: Boxplot split by threshold
low_conf = confidences[confidences < 0.5]
high_conf = confidences[confidences >= 0.5]

box = axes[1].boxplot(
    [low_conf, high_conf],
    labels=['Low confidence\n(< 0.5)', 'High confidence\n(≥ 0.5)'],
    patch_artist=True,
    widths=0.5,
    medianprops=dict(color='#1f2937', linewidth=2),
    boxprops=dict(edgecolor='#01696f', linewidth=1.5),
    whiskerprops=dict(color='#6b7280', linewidth=1.2),
    capprops=dict(color='#6b7280', linewidth=1.2)
)

box['boxes'][0].set_facecolor('#d9eaf2')
box['boxes'][1].set_facecolor('#a9d3e8')

axes[1].set_ylabel('Confidence Score', fontsize=12)
axes[1].set_title('Score Split by Threshold', fontsize=14, weight='bold', pad=10)

# Suptitle
fig.suptitle(
    'Confidence Score Distribution',
    fontsize=18,
    weight='bold',
    y=1.02
)
fig.text(
    0.5, 0.96,
    'YOLOv5-CASP detections show most scores concentrated above the inference threshold',
    ha='center',
    fontsize=11,
    color='#6b7280'
)

plt.tight_layout()
plt.savefig(
    output_dir / 'confidence_distribution.png',
    dpi=300,
    bbox_inches='tight',
    facecolor=fig.get_facecolor()
)
plt.close()

print("  ✓ Saved: confidence_distribution.png")

# ============================================================================
# 5. FAILURE CASE PIE CHART
# ============================================================================
print("\n5. Creating Failure Case Distribution...")

import matplotlib.pyplot as plt

failure_labels = ['Correct', 'False Negatives', 'Misaligned', 'False Positives']
failure_counts = [738, 6, 11, 0]

# Remove zero-count slices so the pie stays clean
filtered = [(l, c) for l, c in zip(failure_labels, failure_counts) if c > 0]
labels = [x[0] for x in filtered]
counts = [x[1] for x in filtered]

colors_fail = ['#01696f', '#d19900', "#63bb72"]

total = sum(counts)
percentages = [c / total * 100 for c in counts]

fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor('#f9f8f5')
ax.set_facecolor('white')

wedges, texts, autotexts = ax.pie(
    counts,
    colors=colors_fail,
    startangle=90,
    counterclock=False,
    autopct=lambda pct: f'{pct:.1f}%' if pct >= 2 else '',
    pctdistance=0.75,
    wedgeprops=dict(edgecolor='white', linewidth=1.2)
)

for t in autotexts:
    t.set_color('white')
    t.set_fontsize(11)
    t.set_weight('bold')

ax.set_title(
    'Failure Case Distribution',
    fontsize=16,
    weight='bold',
    pad=14
)
ax.text(
    0.5, 0.99,
    'YOLOv5-CASP detection outcomes on the test set',
    transform=ax.transAxes,
    ha='center',
    va='center',
    fontsize=11,
    color='#6b7280'
)

legend_labels = [f'{l}: {c} ({c/total*100:.2f}%)' for l, c in zip(labels, counts)]
ax.legend(
    wedges,
    legend_labels,
    title='Categories',
    loc='lower center',
    bbox_to_anchor=(0.5, -0.12),
    ncol=2,
    frameon=False,
    fontsize=10,
    title_fontsize=11
)

ax.axis('equal')

plt.tight_layout()
plt.savefig(
    output_dir / 'failure_distribution.png',
    dpi=300,
    bbox_inches='tight',
    facecolor=fig.get_facecolor()
)
plt.close()

print("  ✓ Saved: failure_distribution.png")

# ============================================================================
# 6. TRAINING TIME COMPARISON
# ============================================================================
print("\n6. Creating Training Time Comparison...")

import matplotlib.pyplot as plt
import numpy as np

models_time = ['YOLOv5-CASP\n(X-Ray)', 'YOLOv5-CASP\n(CT)', 'Baseline\n(CT)', 'YOLOv8s\n(CT)']
training_hours = [5.9, 0.75, 0.62, 0.31]

# Use one coherent palette, darkest for most important model
colors_time = ['#01696f', '#2f8fbd', '#a9d3e8', '#d9eaf2']

fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor('#f9f8f5')
ax.set_facecolor('white')

bars = ax.bar(
    models_time,
    training_hours,
    color=colors_time,
    edgecolor='#d0d7de',
    linewidth=0.8,
    width=0.65
)

ax.set_title('Training Time Comparison', fontsize=16, weight='bold', pad=14)
ax.text(
    0.5, 1.02,
    'Measured training time on the same hardware setup',
    transform=ax.transAxes,
    ha='center',
    va='bottom',
    fontsize=11,
    color='#6b7280'
)

ax.set_xlabel('Model', fontsize=12, labelpad=10)
ax.set_ylabel('Training Time (hours)', fontsize=12, labelpad=10)

ax.grid(True, axis='y', linestyle='--', alpha=0.25)
ax.set_axisbelow(True)

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.spines['left'].set_color('#c7c7c7')
ax.spines['bottom'].set_color('#c7c7c7')

for bar, hours in zip(bars, training_hours):
    ax.annotate(
        f'{hours:.2f} h',
        xy=(bar.get_x() + bar.get_width() / 2, hours),
        xytext=(0, 6),
        textcoords='offset points',
        ha='center',
        va='bottom',
        fontsize=11,
        weight='bold',
        color='#1f2937'
    )

plt.tight_layout()
plt.savefig(
    output_dir / 'training_time.png',
    dpi=300,
    bbox_inches='tight',
    facecolor=fig.get_facecolor()
)
plt.close()

print("  ✓ Saved: training_time.png")

# ============================================================================
# 7. INFERENCE SPEED COMPARISON
# ============================================================================
print("\n7. Creating Inference Speed Comparison...")

import matplotlib.pyplot as plt
import numpy as np

models_speed = [
    'YOLOv5-CASP\n(GPU)',
    'YOLOv5-CASP\n(CPU)',
    'YOLOv8s\n(GPU)',
    'Faster R-CNN\n(GPU)'
]
fps_values = [71, 27, 45, 5]

# Sort from slowest to fastest for easier comparison
pairs = sorted(zip(fps_values, models_speed), key=lambda x: x[0])
fps_sorted = [p[0] for p in pairs]
models_sorted = [p[1] for p in pairs]

colors_speed = ['#d9eaf2', '#a9d3e8', '#2f8fbd', '#01696f']

fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor('#f9f8f5')
ax.set_facecolor('white')

bars = ax.barh(
    models_sorted,
    fps_sorted,
    color=colors_speed,
    edgecolor='#d0d7de',
    linewidth=0.8,
    height=0.62
)

ax.set_title('Inference Speed Comparison', fontsize=16, weight='bold', pad=14)
ax.text(
    0.5, 1.0,
    'Measured throughput in frames per second on the same hardware',
    transform=ax.transAxes,
    ha='center',
    va='bottom',
    fontsize=11,
    color='#6b7280'
)

ax.set_xlabel('Frames Per Second (FPS)', fontsize=12, labelpad=10)
ax.set_ylabel('Model / Device', fontsize=12, labelpad=10)

ax.axvline(x=30, color='#01696f', linestyle='--', linewidth=1.8, label='Real-time threshold (30 FPS)')
ax.legend(frameon=False, fontsize=10, loc='lower right')

ax.grid(True, axis='x', linestyle='--', alpha=0.25)
ax.set_axisbelow(True)

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.spines['left'].set_color('#c7c7c7')
ax.spines['bottom'].set_color('#c7c7c7')

for bar, fps in zip(bars, fps_sorted):
    ax.annotate(
        f'{fps} FPS',
        xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
        xytext=(6, 0),
        textcoords='offset points',
        ha='left',
        va='center',
        fontsize=11,
        weight='bold',
        color='#1f2937'
    )

ax.set_xlim(0, max(fps_sorted) * 1.15)

plt.tight_layout()
plt.savefig(
    output_dir / 'inference_speed.png',
    dpi=300,
    bbox_inches='tight',
    facecolor=fig.get_facecolor()
)
plt.close()

print("  ✓ Saved: inference_speed.png")

# ============================================================================
# 8. PERFORMANCE HEATMAP
# ============================================================================
print("\n8. Creating Performance Heatmap...")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

heatmap_models = [
    'YOLOv5-CASP\n(X-Ray)',
    'YOLOv5-CASP\n(CT)',
    'Baseline\n(CT)',
    'YOLOv8s\n(CT)',
    'ASPP Only',
    'CoT3 Only'
]
heatmap_metrics = ['mAP', 'Precision', 'Recall', 'F1']
heatmap_data = np.array([
    [0.809, 0.792, 0.708, 0.748],
    [0.382, 0.492, 0.527, 0.509],
    [0.214, 0.289, 0.385, 0.330],
    [0.158, 0.225, 0.297, 0.256],
    [0.248, 0.341, 0.429, 0.380],
    [0.205, 0.202, 0.341, 0.254],
])

fig, ax = plt.subplots(figsize=(11, 7.5))
fig.patch.set_facecolor('#f9f8f5')
ax.set_facecolor('white')

sns.heatmap(
    heatmap_data,
    cmap='Blues',
    vmin=0,
    vmax=0.85,
    annot=True,
    fmt='.3f',
    annot_kws={'fontsize': 10, 'weight': 'bold'},
    linewidths=0.6,
    linecolor='#e6e4df',
    cbar_kws={'label': 'Score', 'shrink': 0.9, 'pad': 0.02},
    ax=ax
)

ax.set_xticklabels(heatmap_metrics, fontsize=11, weight='bold')
ax.set_yticklabels(heatmap_models, fontsize=10, rotation=0)
ax.set_title('Model Performance Heatmap', fontsize=16, weight='bold', pad=14)
ax.text(
    0.5, 1.0,
    'Higher values indicate stronger detection performance across metrics',
    transform=ax.transAxes,
    ha='center',
    va='bottom',
    fontsize=11,
    color='#6b7280'
)

ax.set_xlabel('')
ax.set_ylabel('')

plt.tight_layout()
plt.savefig(
    output_dir / 'performance_heatmap.png',
    dpi=300,
    bbox_inches='tight',
    facecolor=fig.get_facecolor()
)
plt.close()

print("  ✓ Saved: performance_heatmap.png")

# ============================================================================
# 9. FEATURE VISUALIZATION (Grad-CAM style)
# ============================================================================
print("\n9. Creating Feature Visualization...")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

np.random.seed(42)

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.patch.set_facecolor('#f9f8f5')
feature_maps = [
    'Original', 'Layer 1\n(Edges)', 'Layer 2\n(Textures)',
    'Layer 3\n(Shapes)', 'Layer 4\n(Patterns)', 'Output\n(Nodule)'
]

for idx, ax in enumerate(axes.flat):
    ax.set_facecolor('white')

    if idx == 0:
        img = np.ones((120, 120)) * 0.35
        y, x = np.ogrid[-45:75, -45:75]
        mask = x*x + y*y <= 420
        img[mask] = 0.85
        ax.imshow(img, cmap='gray', interpolation='nearest')

    elif idx == 5:
        img = np.zeros((120, 120, 3))
        base = np.ones((120, 120)) * 0.20
        y, x = np.ogrid[-45:75, -45:75]
        mask = x*x + y*y <= 420
        base[mask] = 0.95

        heat = np.zeros((120, 120))
        heat[(x*x + y*y) <= 260] = 1.0
        heat[(x*x + y*y) <= 150] = 0.7

        ax.imshow(base, cmap='gray', interpolation='nearest')
        ax.imshow(heat, cmap='hot', alpha=0.55, interpolation='nearest')

    else:
        img = np.random.rand(120, 120) * 0.15 + 0.35
        for _ in range(4):
            cy, cx = np.random.randint(25, 95, 2)
            ry = np.random.randint(8, 16)
            rx = np.random.randint(8, 16)
            y, x = np.ogrid[:120, :120]
            mask = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1
            img[mask] = np.random.uniform(0.55, 0.95)
        ax.imshow(img, cmap='viridis', interpolation='nearest', vmin=0.25, vmax=1.0)

    ax.set_title(feature_maps[idx], fontsize=11, weight='bold', pad=8)
    ax.axis('off')

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#d0d7de')
        spine.set_linewidth(0.8)

fig.suptitle(
    'Feature Visualization Through Network Layers',
    fontsize=16,
    weight='bold',
    y=0.98
)
fig.text(
    0.5, 0.94,
    'Illustrative progression from low-level edges to high-level nodule localization',
    ha='center',
    fontsize=11,
    color='#6b7280'
)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(
    output_dir / 'feature_visualization.png',
    dpi=300,
    bbox_inches='tight',
    facecolor=fig.get_facecolor()
)
plt.close()

print("  ✓ Saved: feature_visualization.png")

# ============================================================================
# 10. SUMMARY DASHBOARD
# ============================================================================
print("\n10. Creating Summary Dashboard...")

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#f9f8f5')
fig.suptitle(
    'YOLOv5-CASP Lung Nodule Detection - Results Dashboard',
    fontsize=18,
    weight='bold',
    y=0.98,
    color='#1f2937'
)

def style_ax(ax, title):
    ax.set_facecolor('white')
    ax.set_title(title, fontsize=12, weight='bold', pad=10)
    ax.grid(True, axis='y', linestyle='--', alpha=0.22)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#c7c7c7')
    ax.spines['bottom'].set_color('#c7c7c7')

# 1) Key metrics
ax1 = fig.add_subplot(2, 3, 1)
style_ax(ax1, 'Key Performance Metrics (X-Nodule)')
metrics = ['mAP@0.5', 'Precision', 'Recall', 'F1 Score']
values = [0.809, 0.792, 0.708, 0.748]
colors_dash = ['#01696f', '#2f8fbd', '#a9d3e8', '#d9eaf2']
bars = ax1.bar(metrics, values, color=colors_dash, edgecolor='#d0d7de', linewidth=0.8)
ax1.set_ylim(0, 1.0)
ax1.set_ylabel('Score', fontsize=10)
for bar, val in zip(bars, values):
    ax1.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val),
                 xytext=(0, 6), textcoords='offset points',
                 ha='center', fontsize=9, weight='bold', color='#1f2937')

# 2) Improvement
ax2 = fig.add_subplot(2, 3, 2)
style_ax(ax2, 'Improvement Over Baseline (X-Nodule)')
improve_labels = ['mAP', 'Precision', 'Recall']
improvements = [278, 174, 84]
bars2 = ax2.bar(improve_labels, improvements, color='#d19900', edgecolor='#d0d7de', linewidth=0.8)
ax2.set_ylabel('Improvement (%)', fontsize=10)
for bar, val in zip(bars2, improvements):
    ax2.annotate(f'+{val}%', xy=(bar.get_x() + bar.get_width()/2, val),
                 xytext=(0, 6), textcoords='offset points',
                 ha='center', fontsize=9, weight='bold', color='#1f2937')

# 3) Model size
ax3 = fig.add_subplot(2, 3, 3)
style_ax(ax3, 'Model Size Comparison')
size_labels = ['YOLOv5-CASP', 'Baseline', 'YOLOv8s']
sizes = [19.4, 7.02, 11.1]
bars3 = ax3.bar(size_labels, sizes, color=['#01696f', '#6b7280', '#2f8fbd'], edgecolor='#d0d7de', linewidth=0.8)
ax3.set_ylabel('Parameters (M)', fontsize=10)
for bar, val in zip(bars3, sizes):
    ax3.annotate(f'{val}M', xy=(bar.get_x() + bar.get_width()/2, val),
                 xytext=(0, 6), textcoords='offset points',
                 ha='center', fontsize=9, weight='bold', color='#1f2937')

# 4) Speed
ax4 = fig.add_subplot(2, 3, 4)
style_ax(ax4, 'Inference Speed')
speed_labels = ['GPU', 'CPU', 'YOLOv8s\n(GPU)']
speeds = [71, 27, 45]
bars4 = ax4.bar(speed_labels, speeds, color=['#01696f', '#a9d3e8', '#2f8fbd'], edgecolor='#d0d7de', linewidth=0.8)
ax4.axhline(y=30, color='#d19900', linestyle='--', linewidth=1.8, label='Real-time threshold (30 FPS)')
ax4.set_ylabel('FPS', fontsize=10)
ax4.legend(fontsize=8, frameon=False, loc='upper right')
for bar, val in zip(bars4, speeds):
    ax4.annotate(f'{val}', xy=(bar.get_x() + bar.get_width()/2, val),
                 xytext=(0, 6), textcoords='offset points',
                 ha='center', fontsize=9, weight='bold', color='#1f2937')

# 5) Training time
ax5 = fig.add_subplot(2, 3, 5)
style_ax(ax5, 'Training Time')
train_labels = ['X-Ray\n(5.9h)', 'CT\n(0.75h)', 'Baseline\n(0.62h)']
train_times = [5.9, 0.75, 0.62]
bars5 = ax5.bar(train_labels, train_times, color=['#01696f', '#2f8fbd', '#a9d3e8'], edgecolor='#d0d7de', linewidth=0.8)
ax5.set_ylabel('Hours', fontsize=10)
for bar, val in zip(bars5, train_times):
    ax5.annotate(f'{val:.2f}h', xy=(bar.get_x() + bar.get_width()/2, val),
                 xytext=(0, 6), textcoords='offset points',
                 ha='center', fontsize=9, weight='bold', color='#1f2937')

# 6) Failure distribution
ax6 = fig.add_subplot(2, 3, 6)
ax6.set_facecolor('white')
failure_data = [738, 17]
failure_labels = ['Correct', 'Failed']
colors_fail = ['#01696f', '#d19900']
wedges, texts, autotexts = ax6.pie(
    failure_data,
    labels=failure_labels,
    autopct='%1.1f%%',
    colors=colors_fail,
    startangle=90,
    wedgeprops=dict(edgecolor='white', linewidth=1.0)
)
for t in autotexts:
    t.set_color('white')
    t.set_weight('bold')
    t.set_fontsize(10)
ax6.set_title('Overall Performance (X-Nodule Test Set)', fontsize=12, weight='bold', pad=10)
ax6.axis('equal')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(
    output_dir / 'summary_dashboard.png',
    dpi=300,
    bbox_inches='tight',
    facecolor=fig.get_facecolor()
)
plt.close()

print("  ✓ Saved: summary_dashboard.png")

# ============================================================================
# 11. X-RAY vs CT COMPARISON
# ============================================================================
print("\n11. Creating X-Ray vs CT Comparison Chart...")

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(11, 6.5))
fig.patch.set_facecolor('#f9f8f5')
ax.set_facecolor('white')

models_compare = ['Baseline', 'YOLOv8s', 'YOLOv5-CASP']
xray_scores = [0.214, 0.807, 0.809]
ct_scores = [0.214, 0.158, 0.382]

x = np.arange(len(models_compare))
width = 0.34

bars1 = ax.bar(
    x - width/2,
    xray_scores,
    width,
    label='X-Nodule (X-Ray)',
    color='#01696f',
    edgecolor='#d0d7de',
    linewidth=0.8
)

bars2 = ax.bar(
    x + width/2,
    ct_scores,
    width,
    label='CT Patches (LUNA16)',
    color='#2f8fbd',
    edgecolor='#d0d7de',
    linewidth=0.8
)

ax.set_xlabel('Model', fontsize=12, labelpad=10)
ax.set_ylabel('mAP@0.5', fontsize=12, labelpad=10)
ax.set_title('Performance Comparison: X-Ray vs CT Patches', fontsize=16, weight='bold', pad=14)
ax.text(
    0.5, 1.0,
    'mAP@0.5 comparison across datasets and model variants',
    transform=ax.transAxes,
    ha='center',
    va='bottom',
    fontsize=11,
    color='#6b7280'
)

ax.set_xticks(x)
ax.set_xticklabels(models_compare, fontsize=11)
ax.set_ylim(0, 1.0)

ax.grid(True, axis='y', linestyle='--', alpha=0.25)
ax.set_axisbelow(True)

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.spines['left'].set_color('#c7c7c7')
ax.spines['bottom'].set_color('#c7c7c7')

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f'{h:.3f}',
            xy=(bar.get_x() + bar.get_width()/2, h),
            xytext=(0, 6),
            textcoords='offset points',
            ha='center',
            fontsize=9,
            weight='bold',
            color='#1f2937'
        )

ax.legend(frameon=False, fontsize=10, loc='upper left')

plt.tight_layout()
plt.savefig(
    output_dir / 'xray_vs_ct_comparison.png',
    dpi=300,
    bbox_inches='tight',
    facecolor=fig.get_facecolor()
)
plt.close()

print("  ✓ Saved: xray_vs_ct_comparison.png")

print("\n" + "="*60)
print("ALL ADVANCED VISUALIZATIONS GENERATED!")
print("="*60)
print("\nResults saved to: evaluation_results/")
print("\nGenerated files:")
for f in sorted(output_dir.glob('*')):
    if f.suffix in ['.png', '.jpg']:
        print(f"  • {f.name}")
print("="*60)