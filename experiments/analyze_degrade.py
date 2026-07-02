"""
Analyze the causal PopQA-degradation experiment (Loop 36 centerpiece).

Predicts (self-knowledge-share theory): as retrieval reliability falls (p: 0 -> 1),
  (1) the learned optimal-stop router's gain over best-static decreases MONOTONICALLY -> ~0;
  (2) the self-knowledge SHARE of benefit variance decreases;
  (3) benefit among closed-book-wrong becomes governed by the random reliability draw
      (not in h_S0), so a router built on h_S0 loses its edge.

Data: degrade_popqa_causal.jsonl. S0 record carries h_S0 (closed-book, p-independent);
S1..S4 records carry per-p, per-query correctness + realized reliability draw.

  python experiments/analyze_degrade.py
"""
import json, argparse, numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

SEED = 42
# 5-stage cumulative cost (S0 closed-book cheap, then growing). Normalized.
CUM = np.array([0.10, 0.35, 0.85, 1.60, 2.68]); NCUM = CUM / CUM[-1]
LAM = 0.5


def load(path):
    recs = [json.loads(l) for l in open(path)]
    h0 = {}; cb = {}; pop = {}
    # per (qid,p): correctness by stage 1..4, reliability
    seq = defaultdict(lambda: {"c": {}, "rel": None})
    for r in recs:
        q = r["qid"]
        if r["stage"] == 0:
            h0[q] = np.asarray(r["hs_concat"], dtype=np.float32)
            cb[q] = int(r["correct"]); pop[q] = r.get("pop")
        else:
            key = (q, r["p"])
            seq[key]["c"][r["stage"]] = int(r["correct"])
            seq[key]["rel"] = r["reliable"]
    return h0, cb, pop, seq


def oof_auroc(X, y, g):
    if len(np.unique(y)) < 2 or len(y) < 20: return float("nan")
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(min(5, len(np.unique(g)))).split(X, y, groups=g):
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=SEED)
        lr.fit(sc.transform(X[tr]), y[tr]); oof[te] = lr.predict_proba(sc.transform(X[te]))[:, 1]
    return roc_auc_score(y, oof)


def router_gain_for_p(qids, h0arr, C5):
    """5-stage learned optimal-stop-from-h0 (closed-book) router CWA gap vs best static."""
    Q = len(qids)
    acc = C5.mean(0)
    static = acc - LAM * NCUM
    best_static = static.max()
    opt = np.array([next((t for t in range(5) if C5[i, t] == 1), 4) for i in range(Q)])
    pred = np.full(Q, -1)
    for tr, te in GroupKFold(5).split(h0arr, opt, groups=qids):
        sc = StandardScaler().fit(h0arr[tr])
        lr = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=SEED)
        lr.fit(sc.transform(h0arr[tr]), opt[tr]); pred[te] = lr.predict(sc.transform(h0arr[te]))
    chosen_acc = np.mean([C5[i, pred[i]] for i in range(Q)])
    chosen_cost = np.mean([NCUM[pred[i]] for i in range(Q)])
    return (chosen_acc - LAM * chosen_cost) - best_static, best_static, acc.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="degrade_popqa_causal.jsonl")
    ap.add_argument("--out", default="review-stage/degrade_popqa_causal_analysis.json")
    args = ap.parse_args()
    h0, cb, pop, seq = load(args.data)
    qids_all = sorted(h0.keys())
    P_GRID = sorted({k[1] for k in seq.keys()})
    cb_arr = np.array([cb[q] for q in qids_all])
    print(f"n_q={len(qids_all)}  closed-book S0 acc={cb_arr.mean():.3f}  p-grid={P_GRID}", flush=True)

    rows = []
    for p in P_GRID:
        qids = [q for q in qids_all if (q, p) in seq]
        H0 = np.array([h0[q] for q in qids])
        # build 5-stage correctness matrix: S0=closed-book, S1..S4 from seq
        C5 = np.array([[cb[q]] + [seq[(q, p)]["c"].get(t, 0) for t in range(1, 5)] for q in qids])
        rel = np.array([seq[(q, p)]["rel"] for q in qids])
        # benefit | wrong@S0 (closed-book-wrong): becomes correct at any later stage
        wrong = C5[:, 0] == 0
        ben = ((C5[:, 1:].max(1) == 1) & wrong).astype(int)[wrong]
        gw = np.array(qids)[wrong]
        # self-knowledge benefit AUROC: predict benefit from h_S0 among CB-wrong
        ben_auroc_h = oof_auroc(H0[wrong], ben, gw) if wrong.sum() > 20 else float("nan")
        # how much does the RANDOM reliability draw explain benefit? (not in h)
        # share proxy: corr(reliable, benefit) among CB-wrong
        rel_w = rel[wrong]
        rel_ben_corr = float(np.corrcoef(rel_w, ben)[0, 1]) if wrong.sum() > 2 and len(np.unique(rel_w)) > 1 else float("nan")
        gain, best_static, acc = router_gain_for_p(np.array(qids), H0, C5)
        row = {"p": p, "n_q": len(qids), "stage_acc": [round(a, 3) for a in acc],
               "n_cb_wrong": int(wrong.sum()), "benefit_rate": round(float(ben.mean()), 3),
               "benefit_auroc_from_hS0": round(ben_auroc_h, 3) if ben_auroc_h == ben_auroc_h else None,
               "corr_reliabilityDraw_benefit": round(rel_ben_corr, 3) if rel_ben_corr == rel_ben_corr else None,
               "router_gain_vs_static": round(gain, 4), "best_static": round(best_static, 3),
               "frac_reliable": round(float(rel.mean()), 3)}
        rows.append(row); print(json.dumps(row), flush=True)

    # monotonicity tests
    gains = [r["router_gain_vs_static"] for r in rows]
    bens = [r["benefit_auroc_from_hS0"] for r in rows if r["benefit_auroc_from_hS0"] is not None]
    from scipy.stats import spearmanr
    rho_g, pg = spearmanr(P_GRID, gains)
    summary = {"rows": rows,
               "monotonic_gain_in_p": {"spearman_rho": rho_g, "p": pg,
                                       "note": "theory predicts NEGATIVE rho (gain falls as p rises)"},
               "cost_model": {"CUM": CUM.tolist(), "lambda": LAM}}
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"\nSpearman(p, router_gain) = {rho_g:.3f} (p={pg:.3f})  [predict < 0]", flush=True)
    print("[*] ->", args.out, flush=True)


if __name__ == "__main__":
    main()
