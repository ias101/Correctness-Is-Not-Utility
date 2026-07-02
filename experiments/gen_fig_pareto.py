"""
Cost-performance (Pareto) figure for the deployed two-tier router (Loop 37).

Reads review-stage/two_tier_analysis.json (produced by analyze_two_tier.py) and plots
accuracy vs measured cost for every method, with our hidden-state methods (passive
Tier-1, post-retrieval verifier, two-tier) tracing the frontier and the generative
baselines (Self-RAG, FLARE) far to the right (per-stage generation).

  python experiments/gen_fig_pareto.py
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {
    "static":                  dict(color="#888888", marker="s", ls="--", label="Static (fixed depth)"),
    "query_only":              dict(color="#9467bd", marker="D", ls="",   label="Query-only (Adaptive-RAG-style)"),
    "passive_hs":              dict(color="#1f77b4", marker="o", ls="-",  label="Passive HS router (Tier 1)"),
    "post_retrieval_verifier": dict(color="#ff7f0e", marker="P", ls="-",  label="Post-retrieval verifier (Tier 2)"),
    "flare":                   dict(color="#2ca02c", marker="^", ls="-",  label="FLARE (generative)"),
    "selfrag":                 dict(color="#d62728", marker="v", ls="-",  label="Self-RAG (generative)"),
    "two_tier":                dict(color="#000000", marker="*", ls="-",  label="Two-tier (ours)"),
}
ORDER = ["static", "query_only", "passive_hs", "post_retrieval_verifier", "flare", "selfrag", "two_tier"]
CURVE = {"passive_hs", "post_retrieval_verifier", "flare", "selfrag", "two_tier"}


def plot_dataset(ax, res, title):
    cur = res["curves"]
    for name in ORDER:
        if name not in cur:
            continue
        pts = sorted({(p["cost"], p["acc"]) for p in cur[name]})
        xs, ys = zip(*pts)
        st = STYLE[name]
        if name in CURVE:
            lw = 2.8 if name == "two_tier" else 1.5
            ms = 11 if name == "two_tier" else 5
            z = 6 if name == "two_tier" else 3
            ax.plot(xs, ys, color=st["color"], marker=st["marker"], ls=st["ls"], lw=lw, ms=ms,
                    label=st["label"], zorder=z, alpha=1.0 if name == "two_tier" else 0.85)
        else:
            ax.plot(xs, ys, color=st["color"], marker=st["marker"], ls=st["ls"], ms=8,
                    label=st["label"], zorder=2)
    # annotate the headline cost saving vs Self-RAG (matched accuracy)
    v = res.get("vs_selfrag", {}).get("matched_acc")
    sr = res.get("vs_selfrag", {})
    if v and sr:
        ax.annotate(f"matches Self-RAG acc\nat −{v['cost_saved_pct']:.0f}% cost",
                    xy=(v["tt_cost"], v["tt_acc"]), xytext=(v["tt_cost"] + 2.0, v["tt_acc"] - 0.06),
                    fontsize=8, ha="left",
                    arrowprops=dict(arrowstyle="->", color="#000", lw=1.0))
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Cost / query (measured Qwen2.5-7B units)", fontsize=9)
    ax.grid(alpha=0.25)


def main():
    path = "review-stage/two_tier_analysis.json"
    if not os.path.exists(path):
        print("run analyze_two_tier.py first"); return
    d = json.load(open(path))
    panels = [(f"{k} ({'high' if k=='PopQA' else 'low'} self-knowledge share)", d[k])
              for k in ("PopQA", "HotpotQA") if k in d]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.3), squeeze=False)
    axes = axes[0]
    for ax, (title, res) in zip(axes, panels):
        plot_dataset(ax, res, title)
    axes[0].set_ylabel("Accuracy", fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Two-tier hidden-state router vs adaptive-RAG baselines (cost-performance)", fontsize=12)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    os.makedirs("paper/figures", exist_ok=True)
    for out in ("paper/figures/fig_two_tier_pareto.pdf", "review-stage/fig_two_tier_pareto.png"):
        fig.savefig(out, dpi=160, bbox_inches="tight"); print("[*] ->", out)


if __name__ == "__main__":
    main()
