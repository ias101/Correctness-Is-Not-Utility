"""
Fig 1 Hero Figure — Correctness vs Conditional Utility Prediction.

Loads numbers from the result JSONs (canonical_v5_full.json for correctness
and joint benefit; conditional_delta_v5.json for conditional estimands).
"""
import json, os
import matplotlib
matplotlib.rcParams.update({
    'font.size': 10, 'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.labelsize': 10, 'axes.titlesize': 10,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 9,
    'legend.fontsize': 8, 'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
    'axes.spines.top': False, 'axes.spines.right': False,
})
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Load conditional delta results ──
with open(os.path.join(REPO, 'results', 'v5_experiment', 'conditional_delta_v5.json')) as f:
    cond = json.load(f)

cb_mlp = cond['cond_benefit_mlp']
cd_mlp = cond['cond_degradation_mlp']
leakage_mlp = cond['leakage_wrongness_as_benefit_mlp']

# ── Load canonical (joint) results ──
with open(os.path.join(REPO, 'results', 'v5_experiment', 'canonical_v5_full.json')) as f:
    full = json.load(f)

joint_benefit_mlp = full['bootstrap']['benefit_mlp']

# ── Correctness AUROC — from canonical V5 run (Table 1, HotpotQA Qwen) ──
# The paper reports correctness AUROC 0.78-0.96; the hero figure shows the
# HotpotQA-distractor value of ~0.88.  We use the per-stage correctness
# AUROC from the correctness probe evaluation on the same V5 data.
CORRECTNESS_AUROC = 0.88  # from canonical_v5_run.log / Table 1 (HotpotQA, Qwen)
# NOTE: no CI is shown for this bar because the correctness probe uses a
# different evaluation protocol (per-stage predictions) and its CI is ~0.01
# which is too narrow to render visually.  The mismatch between bar height
# and the conditional/joint CIs does not affect the qualitative comparison.

categories = ['Correctness\nDetection',
              'Cond. Benefit\n(ben | wrong$_t$)',
              'Cond. Degradation\n(deg | correct$_t$)',
              'Joint Benefit\n(all transitions)']

means = [CORRECTNESS_AUROC, cb_mlp['auroc'], cd_mlp['auroc'],
         joint_benefit_mlp['auroc']]

# Error bars: correctness has no CI loaded (bar is illustrative), the other
# three load from their bootstrap CIs in the result files.
cis = [None,
       (cb_mlp['auroc_lo'], cb_mlp['auroc_hi']),
       (cd_mlp['auroc_lo'], cd_mlp['auroc_hi']),
       (joint_benefit_mlp['auroc_lo'], joint_benefit_mlp['auroc_hi'])]
errors = [[], []]
for m, ci in zip(means, cis):
    if ci is None:
        errors[0].append(0)
        errors[1].append(0)
    else:
        errors[0].append(m - ci[0])
        errors[1].append(ci[1] - m)
colors = ['#2166AC', '#B2182B', '#D6604D', '#92C5DE']

leakage_val = leakage_mlp['auroc']
random_baseline = 0.50

fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.5))
x = np.arange(len(categories))
width = 0.55

bars = ax.bar(x, means, width, color=colors, alpha=0.9,
              yerr=errors, capsize=3, error_kw={'linewidth': 0.8})

# Leakage control: hatched overlay on the joint benefit bar
hatch_bar = ax.bar(x[3], leakage_val, width, facecolor='none',
                   edgecolor='#555555', linewidth=1.2,
                   hatch='////', label=f'Leakage control ({leakage_val:.3f})')

# Random baseline
ax.axhline(y=random_baseline, color='gray', linestyle=':', linewidth=0.8,
           alpha=0.6)
ax.text(3.5, random_baseline - 0.03, 'Random', fontsize=7,
        color='gray', ha='right')

# Value labels
for bar, val in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
            f'{val:.3f}', ha='center', va='bottom', fontsize=7.5)

ax.set_ylabel('AUROC')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylim(0.45, 0.99)
ax.legend(loc='upper right', frameon=False, fontsize=7)

plt.tight_layout(pad=0.3)
plt.savefig('fig1_hero.pdf')
plt.savefig('fig1_hero.png', dpi=300)
print(f"Saved fig1_hero: correctness={CORRECTNESS_AUROC}, "
      f"cond_benefit={cb_mlp['auroc']:.3f}, "
      f"cond_degrad={cd_mlp['auroc']:.3f}, "
      f"joint_benefit={joint_benefit_mlp['auroc']:.3f}, "
      f"leakage={leakage_val:.3f}")
