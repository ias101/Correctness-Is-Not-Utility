"""
Loop 38 (P1.e): share proxy vs routing gain across datasets/models, with correctness-AUROC
as a NON-predictive baseline predictor. Numbers from the routability table (paper Tab.2) +
loop-36 cross-dataset results. (WebQuestions / 2WikiMultiHopQA / MuSiQue not collectable on
the offline box; the available 4 datasets + a 2nd model family are used.)

  python experiments/gen_fig_share_vs_gain.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# (label, benefit-AUROC = share proxy, correctness@S0 AUROC, learned-router gain ΔCWA_cum, degenerate?)
PTS = [
    ("PopQA (Qwen)",        0.747, 0.828, +0.024, False),
    ("HotpotQA (Qwen)",     0.548, 0.769, +0.003, False),
    ("HotpotQA (Mistral)",  0.615, 0.780, -0.006, False),
    ("TriviaQA (Qwen)",     0.584, 0.850, -0.008, False),
    ("NQ (Qwen, degen.)",   0.740, 0.751, -0.001, True),
]


def main():
    nd = [p for p in PTS if not p[4]]
    ben = np.array([p[1] for p in nd]); corr = np.array([p[2] for p in nd]); gain = np.array([p[3] for p in nd])
    rho_b, p_b = spearmanr(ben, gain)
    rho_c, p_c = spearmanr(corr, gain)
    summary = {"spearman_benefit_share_vs_gain": {"rho": round(float(rho_b), 3), "p": round(float(p_b), 3), "n": len(nd)},
               "spearman_correctness_auroc_vs_gain": {"rho": round(float(rho_c), 3), "p": round(float(p_c), 3), "n": len(nd)},
               "note": "non-degenerate points only; share proxy predicts routability, correctness AUROC does not.",
               "points": [{"dataset": p[0], "benefit_auroc": p[1], "corr_auroc": p[2], "gain": p[3], "degenerate": p[4]} for p in PTS]}
    json.dump(summary, open("review-stage/share_vs_gain.json", "w"), indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for axi, (xs, xlab, rho, pv, col) in enumerate([
        ([p[1] for p in PTS], "Benefit-AUROC from $h_{S_0}$  (self-knowledge share proxy)", rho_b, p_b, "#1f77b4"),
        ([p[2] for p in PTS], "Correctness@$S_0$ AUROC  (baseline predictor)", rho_c, p_c, "#d62728")]):
        a = ax[axi]; a.axhline(0, color="#ccc", lw=.8)
        for (lab, b, c, g, degen), x in zip(PTS, xs):
            a.scatter([x], [g], s=90, facecolors="none" if degen else col, edgecolors=col,
                      marker="o", linewidths=1.6, zorder=3)
            a.annotate(lab, (x, g), fontsize=7, xytext=(4, 4), textcoords="offset points")
        a.set_xlabel(xlab, fontsize=9); a.set_ylabel("learned-router gain  $\\Delta$CWA$_{cum}$", fontsize=9)
        a.set_title(f"Spearman = {rho:+.2f} (p={pv:.2f}, n={len(nd)})", fontsize=10); a.grid(alpha=.25)
    fig.suptitle("The self-knowledge share predicts routability; correctness AUROC does not", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for o in ("paper/figures/fig_share_vs_gain.pdf", "review-stage/fig_share_vs_gain.png"):
        os.makedirs(os.path.dirname(o), exist_ok=True); fig.savefig(o, dpi=150, bbox_inches="tight")
    print(f"Spearman(share-proxy, gain) = {rho_b:+.3f} (p={p_b:.3f}); "
          f"Spearman(correctness-AUROC, gain) = {rho_c:+.3f} (p={p_c:.3f})  [n={len(nd)} non-degenerate]")
    print("[*] -> review-stage/share_vs_gain.json + paper/figures/fig_share_vs_gain.pdf")


if __name__ == "__main__":
    main()
