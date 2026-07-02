"""Generate the thesis figure: (A) causal share-collapse, (B) routability vs share.
Data from results/v5_experiment/routing_learned_*.json and the causal degradation table.
Single-column AAAI width (~3.3in)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 7.2, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(3.34, 1.62))

# ---- Panel A: causal retrieval degradation (interior p where draw has variance) ----
p        = [0.0, 0.25, 0.50, 0.75]
share    = [1.00, 0.21, 0.11, 0.12]
gain     = [-0.167, -0.218, -0.286, -0.325]
lA = axA.plot(p, share, "o-", color="#1f5fa6", lw=1.3, ms=3.4, label="self-know. share")[0]
axA.set_ylabel("self-knowledge share", color="#1f5fa6", fontsize=7)
axA.set_ylim(-0.05, 1.08); axA.set_xlabel("retrieval reliability $1{-}p$ degrades $\\rightarrow$", fontsize=6.6)
axA.set_xticks([0, 0.25, 0.5, 0.75]); axA.tick_params(axis="y", labelcolor="#1f5fa6")
axA.spines["top"].set_visible(False)
axt = axA.twinx()
lB = axt.plot(p, gain, "s--", color="#c0392b", lw=1.3, ms=3.2, label="router gain")[0]
axt.set_ylabel("router gain", color="#c0392b", fontsize=7)
axt.tick_params(axis="y", labelcolor="#c0392b"); axt.spines["top"].set_visible(False)
axA.set_title("(A) Causal: degrade retrieval term", fontsize=7, pad=3)
axA.legend(handles=[lA, lB], fontsize=5.6, loc="center right", frameon=False,
           handlelength=1.4, borderaxespad=0.2)

# ---- Panel B: routability vs benefit decodability (4 dataset-model cells) ----
names = ["PopQA", "HotpotQA(Q)", "HotpotQA(M)", "TriviaQA"]
auroc = [0.747, 0.548, 0.615, 0.584]
g     = [0.024, 0.003, -0.006, -0.008]
lo    = [0.008, -0.005, -0.014, -0.012]
hi    = [0.041, 0.011, 0.001, -0.005]
yerr  = [[g[i]-lo[i] for i in range(4)], [hi[i]-g[i] for i in range(4)]]
axB.axhspan(0, 0.05, color="#27ae60", alpha=0.08)
axB.axhline(0, color="0.45", lw=0.7, ls=":")
cols = ["#27ae60", "0.35", "0.35", "0.35"]
for i in range(4):
    axB.errorbar(auroc[i], g[i], yerr=[[yerr[0][i]], [yerr[1][i]]], fmt="o",
                 color=cols[i], ms=4.2 if i == 0 else 3.4, lw=1.0, capsize=1.6)
axB.annotate("PopQA\n(routes)", (auroc[0], g[0]), (0.655, 0.032), fontsize=5.7,
             color="#1d7d40", ha="left")
axB.annotate("HotpotQA,\nTriviaQA (tie/no)", (0.548, 0.003), (0.515, 0.022), fontsize=5.7,
             color="0.3", ha="left")
axB.set_xlabel("benefit AUROC from $h$", fontsize=6.8)
axB.set_ylabel("learned router $\\Delta$CWA", fontsize=7)
axB.set_xlim(0.49, 0.80); axB.set_ylim(-0.022, 0.05)
axB.set_title("(B) Routability tracks share", fontsize=7, pad=3)
axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)

fig.tight_layout(pad=0.3, w_pad=1.2)
fig.savefig("fig_thesis.pdf", bbox_inches="tight")
print("wrote fig_thesis.pdf")
