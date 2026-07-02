"""Generate LSTM Sequential Modeling comparison bar chart (Ablation #2)."""
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
categories = ['Correctness\nAUROC', 'Correctness\nAUPRC',
              'Benefit (Delta)\nAUROC', 'Benefit (Delta)\nAUPRC']
static_vals = [0.7235, 0.8897, 0.7075, 0.3645]
lstm_vals =  [0.7224, 0.8854, 0.5674, 0.3066]

x = np.arange(len(categories))
width = 0.32

fig, ax = plt.subplots(figsize=(5.5, 3.5))

bars1 = ax.bar(x - width/2, static_vals, width, label='Static MLP',
               color='#42A5F5', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x + width/2, lstm_vals, width, label='LSTM (2-layer, bidirectional)',
               color='#EF5350', edgecolor='white', linewidth=0.5)

# Value labels
for bar, val in zip(bars1, static_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=7)
for bar, val in zip(bars2, lstm_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=7)

# Highlight the degradation
ax.annotate('LSTM DEGRADES\nbenefit prediction',
            xy=(2.16, 0.5674), xytext=(2.8, 0.68),
            fontsize=8, ha='center', color='#B71C1C', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#EF5350', lw=1.5))

ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend(frameon=False, loc='upper right', fontsize=7.5)
ax.set_ylabel('Score', fontweight='bold')
ax.set_ylim(0, 1.05)

plt.tight_layout()
fig.savefig('fig_lstm_comparison.pdf')
fig.savefig('fig_lstm_comparison.png')
print('Saved: fig_lstm_comparison.pdf + .png')
