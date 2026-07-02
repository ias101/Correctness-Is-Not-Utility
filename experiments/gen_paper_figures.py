"""Generate publication-quality figures for AAAI 2027 paper."""
import json, numpy as np, matplotlib
matplotlib.rcParams.update({
    'font.size': 9,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'text.usetex': False,
})
import matplotlib.pyplot as plt

FIGS = 'paper/figures'
COLORS = ['#2166ac', '#b2182b', '#4daf4a', '#ff7f00', '#984ea3']
GRAY = '#666666'

# ============================================================
# Fig 2: Per-stage AUROC — grouped bar chart (V1 vs V3)
# ============================================================
v1_auroc = {'Retrieval': 0.867, 'Reranking': 0.918, 'Assembly': 0.924, 'Generation': 0.924}
v3_auroc = {'Retrieval': 0.924, 'Reranking': 0.933, 'Assembly': 0.922, 'Generation': 0.941}

fig, ax = plt.subplots(figsize=(5.5, 3.2))
stages = list(v1_auroc.keys())
x = np.arange(len(stages))
w = 0.35
bars1 = ax.bar(x - w/2, [v1_auroc[s] for s in stages], w, label='V1 (BM25, fixed top-8)', color=COLORS[1], alpha=0.85)
bars2 = ax.bar(x + w/2, [v3_auroc[s] for s in stages], w, label='V3 (CE, progressive [2,4,6,8])', color=COLORS[0], alpha=0.85)

for bar, val in zip(bars1, [v1_auroc[s] for s in stages]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{val:.3f}', ha='center', va='bottom', fontsize=7)
for bar, val in zip(bars2, [v3_auroc[s] for s in stages]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{val:.3f}', ha='center', va='bottom', fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels(stages, fontsize=8)
ax.set_ylabel('AUROC', fontsize=9)
ax.set_ylim(0.82, 0.98)
ax.legend(frameon=False, fontsize=7.5, loc='lower right')
ax.axhline(y=0.65, color=GRAY, linestyle='--', linewidth=0.8, alpha=0.5)
ax.text(3.5, 0.653, 'Gate (0.65)', fontsize=7, color=GRAY, ha='right')
fig.tight_layout()
fig.savefig(f'{FIGS}/fig2_auroc_comparison.pdf')
print('Saved: fig2_auroc_comparison.pdf')

# ============================================================
# Fig 3: Stage accuracy + both-gold@k line plot
# ============================================================
context_sizes = [2, 4, 6, 8]
stage_acc = [59.6, 64.2, 67.0, 69.3]
both_gold = [45.6, 67.8, 78.1, 87.2]

fig, ax1 = plt.subplots(figsize=(5.5, 3.2))
color_acc = COLORS[0]
color_bg = COLORS[3]

ax1.plot(context_sizes, stage_acc, 'o-', color=color_acc, linewidth=2, markersize=8, label='Stage Accuracy (%)')
ax1.set_xlabel('Context Size (passages)', fontsize=9)
ax1.set_ylabel('Accuracy (%)', fontsize=9, color=color_acc)
ax1.tick_params(axis='y', labelcolor=color_acc)
ax1.set_ylim(40, 75)

ax2 = ax1.twinx()
ax2.plot(context_sizes, both_gold, 's--', color=color_bg, linewidth=2, markersize=8, label='Both-Gold Coverage (%)')
ax2.set_ylabel('Both-Gold Coverage (%)', fontsize=9, color=color_bg)
ax2.tick_params(axis='y', labelcolor=color_bg)
ax2.set_ylim(40, 95)

# Annotate
for i, (k, a, b) in enumerate(zip(context_sizes, stage_acc, both_gold)):
    ax1.annotate(f'{a:.1f}%', (k, a), textcoords="offset points", xytext=(0, 12), ha='center', fontsize=7, color=color_acc)
    ax2.annotate(f'{b:.1f}%', (k, b), textcoords="offset points", xytext=(0, -16), ha='center', fontsize=7, color=color_bg)

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=7.5, loc='center right')
fig.tight_layout()
fig.savefig(f'{FIGS}/fig3_stage_progression.pdf')
print('Saved: fig3_stage_progression.pdf')

# ============================================================
# Fig 4: Pareto frontier — regenerated for publication quality
# ============================================================
points = {
    'full\\_pipeline': (0.733, 1.080),
    'fixed\\_S3': (0.683, 0.080),
    'fixed\\_S2': (0.657, 0.070),
    'fixed\\_S1': (0.613, 0.020),
    'ours': (0.613, 0.020),
    'oracle': (0.800, 0.286),
}

fig, ax = plt.subplots(figsize=(5.5, 3.5))
# Static baselines
names_static = ['fixed\\_S1', 'fixed\\_S2', 'fixed\\_S3', 'full\\_pipeline']
xs_static = [points[n][1] for n in names_static]
ys_static = [points[n][0] for n in names_static]
ax.scatter(xs_static, ys_static, c=COLORS[1], marker='s', s=80, zorder=3, label='Static baselines')
for n, x, y in zip(names_static, xs_static, ys_static):
    ax.annotate(n.replace('\\_', '_'), (x, y), textcoords="offset points", xytext=(8, -4), fontsize=7)

# Our method
ax.scatter([points['ours'][1]], [points['ours'][0]], c=COLORS[0], marker='o', s=100, zorder=4, edgecolors='black', linewidth=1, label='Ours (= fixed\\_S1)')

# Oracle
ax.scatter([points['oracle'][1]], [points['oracle'][0]], c=COLORS[2], marker='D', s=100, zorder=4, edgecolors='black', linewidth=1, label='Oracle')

# Pareto frontier line
pareto_x = [0.020, 0.080, 0.286, 1.080]
pareto_y = [0.613, 0.683, 0.800, 0.733]
ax.plot(pareto_x, pareto_y, '--', color=GRAY, linewidth=1, alpha=0.7, label='Pareto frontier')

ax.set_xlabel('Normalized Cost', fontsize=9)
ax.set_ylabel('Accuracy', fontsize=9)
ax.set_xlim(-0.02, 1.15)
ax.set_ylim(0.55, 0.85)
ax.legend(frameon=False, fontsize=7.5, loc='lower right')
fig.tight_layout()
fig.savefig(f'{FIGS}/fig4_pareto.pdf')
print('Saved: fig4_pareto.pdf')

# ============================================================
# Table 1 LaTeX: Per-stage AUROC
# ============================================================
table1 = r"""\begin{table}[t]
\centering
\caption{Per-stage AUROC for predicting final answer correctness. V1 uses BM25 ranking with fixed top-8 context at all stages; V3 uses cross-encoder ranking with progressive context sizes [2,4,6,8]. Mean improvement from V1 to V3: +0.022 AUROC.}
\label{tab:auroc}
\begin{tabular}{lccc}
\toprule
\textbf{Stage} & \textbf{V1 (BM25)} & \textbf{V3 (CE, prog.)} & \textbf{Improvement} \\
\midrule
Retrieval (S0)   & 0.867 & 0.924 & +0.057 \\
Reranking (S1)   & 0.918 & 0.933 & +0.015 \\
Assembly (S2)    & 0.924 & 0.922 & $-$0.002 \\
Generation (S3)  & 0.924 & \textbf{0.941} & +0.017 \\
\midrule
\textbf{Mean}    & 0.908 & \textbf{0.930} & +0.022 \\
\bottomrule
\end{tabular}
\end{table}
"""
with open(f'{FIGS}/table1_auroc.tex', 'w') as f:
    f.write(table1)
print('Saved: table1_auroc.tex')

# ============================================================
# Table 2 LaTeX: Main results
# ============================================================
table2 = r"""\begin{table}[t]
\centering
\caption{Main results on V3 test set (300 queries). All routing variants achieve performance identical to fixed\_S1 within statistical noise. Oracle routing provides the theoretical upper bound. CWA computed at $\lambda=0.5$. Stage costs: S0=0.02, S1=0.05, S2=0.01, S3=1.00.}
\label{tab:main}
\begin{tabular}{lccc}
\toprule
\textbf{Method} & \textbf{Accuracy} & \textbf{Norm. Cost} & \textbf{CWA} \\
\midrule
Oracle            & 0.800 & 0.264 & \textbf{0.668} \\
fixed\_S3         & 0.683 & 0.074 & 0.646 \\
fixed\_S2         & 0.657 & 0.065 & 0.624 \\
confidence        & 0.663 & 0.289 & 0.519 \\
random            & 0.710 & 0.691 & 0.364 \\
fixed\_S1         & 0.613 & 0.019 & 0.604 \\
\midrule
Ours (threshold)  & 0.613 & 0.019 & 0.604 \\
Ours (delta)      & 0.613 & 0.023 & 0.603 \\
Ours (combined)   & 0.583 & 0.025 & 0.779$^*$ \\
\midrule
full\_pipeline     & 0.733 & 1.000 & 0.233 \\
\bottomrule
\multicolumn{4}{l}{\small $^*$Below fixed\_S1 CWA (0.781) on the test set.} \\
\end{tabular}
\end{table}
"""
with open(f'{FIGS}/table2_main.tex', 'w') as f:
    f.write(table2)
print('Saved: table2_main.tex')

# ============================================================
# LaTeX includes snippet
# ============================================================
latex_includes = r"""
% === Fig 2: Per-stage AUROC Comparison ===
\begin{figure}[t]
    \centering
    \includegraphics[width=0.95\linewidth]{figures/fig2_auroc_comparison.pdf}
    \caption{Per-stage AUROC for predicting final answer correctness, comparing V1 (BM25 ranking, fixed top-8 context) and V3 (cross-encoder ranking, progressive context sizes [2,4,6,8]). The gate threshold of 0.65 is shown for reference. All stages in both versions substantially exceed the gate.}
    \label{fig:auroc}
\end{figure}

% === Fig 3: Stage Accuracy + Both-Gold Coverage ===
\begin{figure}[t]
    \centering
    \includegraphics[width=0.95\linewidth]{figures/fig3_stage_progression.pdf}
    \caption{Stage-level accuracy (left axis, blue) and both-gold paragraph coverage (right axis, orange) as context size increases from 2 to 8 passages (V3, 2000 queries). The 9.7pp accuracy gap between S0 (59.6\%) and S3 (69.3\%) creates the cost-accuracy tradeoff that routing methods must navigate.}
    \label{fig:stage_prog}
\end{figure}

% === Fig 4: Pareto Frontier ===
\begin{figure}[t]
    \centering
    \includegraphics[width=0.85\linewidth]{figures/fig4_pareto.pdf}
    \caption{Cost-Accuracy Pareto frontier. Our method coincides with fixed\_S1 (bottom-left). Oracle routing achieves 80\% accuracy at 29\% of full pipeline cost (top-left). The gap between our method and the oracle represents the routing challenge addressed in Section~5.}
    \label{fig:pareto}
\end{figure}

% === Table 1: AUROC Comparison ===
% \input{figures/table1_auroc.tex}

% === Table 2: Main Results ===
% \input{figures/table2_main.tex}
"""
with open(f'{FIGS}/latex_includes.tex', 'w') as f:
    f.write(latex_includes)
print('Saved: latex_includes.tex')
print('\nAll figures generated successfully!')
