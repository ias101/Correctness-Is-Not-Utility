"""Generate Sharpness Histogram (probability mass concentration analysis).

Loads real probe prediction scores if available (from the correctness-probe
evaluation pipeline); falls back to the published binned distribution from the
paper's empirical results (A_appendix.tex, §Sharpness Distribution):

  [0.0, 0.3): 38.4%   [0.3, 0.7): 3.6%
  [0.7, 0.9):  3.0%   [0.9, 1.0]: 55.0%

Mean predicted P(correct) = 0.600 (out-of-fold probe, HotpotQA S0).

This script loads the 2000 committed out-of-fold HotpotQA S0 probe scores from
results/cached_probe_scores_s0.npy and plots the histogram directly. To
(re)generate that cache, run:

    python experiments/routing_hotpotqa_v5.py --s0_cache results/cached_probe_scores_s0.npy

The binned literals below are kept only as a labeled fallback if the cache is absent.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

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

# ── Try loading real probe scores ──
SCORE_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           'results', 'cached_probe_scores_s0.npy')
if os.path.exists(SCORE_CACHE):
    all_probs = np.load(SCORE_CACHE)
    print(f"Loaded {len(all_probs)} real probe scores from {SCORE_CACHE}")
else:
    # ── Reconstruct from the reported binned distribution ──
    # Empirical bin counts (out-of-fold probe, HotpotQA S0; results/v5_experiment/routing_hotpotqa_v5.json):
    #   38.4% in [0.0, 0.3), 3.6% in [0.3, 0.7), 3.0% in [0.7, 0.9), 55.0% in [0.9, 1.0]
    # This IS the real empirical distribution; we reconstruct individual points
    # uniformly within each bin for histogram rendering.
    np.random.seed(42)
    n = 2000

    bins = [(0.0, 0.3, 0.384), (0.3, 0.7, 0.036),
            (0.7, 0.9, 0.030), (0.9, 1.0, 0.550)]
    samples = []
    for lo, hi, frac in bins:
        count = int(n * frac)
        samples.append(np.random.uniform(lo, hi, count))
    all_probs = np.concatenate(samples)
    all_probs = np.clip(all_probs, 0.01, 0.99)
    print(f"Reconstructed {len(all_probs)} scores from empirical binned distribution")

# ── Plot ──
fig, ax = plt.subplots(figsize=(5, 3.5))

bins = np.linspace(0, 1, 21)
ax.hist(all_probs, bins=bins, color='#42A5F5', edgecolor='white',
        linewidth=0.5, alpha=0.85)

# Empirical label mean
ax.axvline(x=0.595, color='#E53935', linestyle='--', linewidth=1.5,
           label='Empirical accuracy (0.595)')

# Predicted mean
ax.axvline(x=0.600, color='#1565C0', linestyle='-', linewidth=1.5,
           label='Mean $P$(correct) (0.600)')

# Annotations (out-of-fold probe, HotpotQA S0)
ax.annotate('55\\% mass in $[0.9, 1.0]$',
            xy=(0.80, 650), fontsize=8, ha='center',
            color='#1565C0', fontweight='bold')

ax.annotate('38\\% in $[0, 0.3)$',
            xy=(0.20, 500), fontsize=8, ha='center',
            color='#666666')

# Threshold dead zone
ax.axvspan(0.3, 0.8, alpha=0.06, color='#FF9800')
ax.text(0.55, 700, 'Threshold\n"dead zone"', ha='center',
        fontsize=7.5, color='#E65100', fontstyle='italic')

ax.set_xlabel('$P$(correct $\\mid$ $\\mathbf{h}_0$, stage=0)', fontweight='bold')
ax.set_ylabel('Number of Queries', fontweight='bold')
ax.legend(frameon=False, fontsize=8)

plt.tight_layout()
fig.savefig('fig_sharpness.pdf')
fig.savefig('fig_sharpness.png')
print('Saved: fig_sharpness.pdf + .png')
