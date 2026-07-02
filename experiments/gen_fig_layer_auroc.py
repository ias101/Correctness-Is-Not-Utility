"""Generate Layer-wise Correctness Signal Localization line plot (Ablation #6)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 9, 'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'figure.dpi': 300,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': False, 'axes.spines.top': False,
    'axes.spines.right': False,
})

# ── Data ──
# Layer indices in Qwen2.5-7B (28 layers total, probed last 4)
layer_nums = [24, 25, 26, 27]
auroc_vals = [0.6726, 0.6824, 0.5810, 0.6055]
auprc_vals = [0.8549, 0.8588, 0.8201, 0.8262]

# Additional reference points
final_token = 0.6726  # original final-token = layer 27
all_concat_auroc = 0.6600
all_concat_auprc = 0.8649

fig, ax1 = plt.subplots(figsize=(5.5, 3.5))

color_auroc = '#1565C0'
color_auprc = '#E53935'

# AUROC line
ax1.plot(layer_nums, auroc_vals, 'o-', color=color_auroc, linewidth=2,
         markersize=8, label='AUROC', zorder=5)
ax1.set_ylabel('AUROC', fontweight='bold', color=color_auroc)
ax1.tick_params(axis='y', labelcolor=color_auroc)
ax1.set_ylim(0.50, 0.78)

# AUPRC line on secondary axis
ax2 = ax1.twinx()
ax2.plot(layer_nums, auprc_vals, 's--', color=color_auprc, linewidth=2,
         markersize=8, label='AUPRC', zorder=4)
ax2.set_ylabel('AUPRC', fontweight='bold', color=color_auprc)
ax2.tick_params(axis='y', labelcolor=color_auprc)
ax2.set_ylim(0.78, 0.92)

# Annotations
# Peak
ax1.annotate('Peak: AUROC 0.682\n(Layer 25/28)',
            xy=(25, 0.6824), xytext=(25.3, 0.71),
            fontsize=7.5, ha='left', color=color_auroc,
            arrowprops=dict(arrowstyle='->', color=color_auroc, lw=1.2))

# Decline
ax1.annotate('$-0.10$ AUROC\ndecline',
            xy=(25.8, 0.63), xytext=(26.3, 0.55),
            fontsize=7.5, ha='center', color='#666666',
            arrowprops=dict(arrowstyle='->', color='#999999', lw=1.0))

# Region labels
ax1.axvspan(23.5, 25.5, alpha=0.08, color='#4CAF50')
ax1.axvspan(25.5, 27.5, alpha=0.08, color='#FF9800')
ax1.text(24.5, 0.52, 'Semantic\nMatching', ha='center', fontsize=7,
         color='#2E7D32', fontstyle='italic')
ax1.text(26.5, 0.52, 'Generation\nProjection', ha='center', fontsize=7,
         color='#E65100', fontstyle='italic')

ax1.set_xlabel('Transformer Layer (of 28)', fontweight='bold')
ax1.set_xticks(layer_nums)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False,
          loc='lower left', fontsize=8)

plt.tight_layout()
fig.savefig('fig_layer_auroc.pdf')
fig.savefig('fig_layer_auroc.png')
print('Saved: fig_layer_auroc.pdf + .png')
