"""
Why is PopQA routable but HotpotQA not? (Loop 36 regime analysis.)

Hypothesis: PopQA benefit tracks PARAMETRIC KNOWLEDGE / entity popularity (the model
knows popular facts -> stop early; rare facts -> retrieve), which IS encoded in h_S0;
HotpotQA benefit depends on unseen multi-hop passage content, which is not.

Measures on PopQA (popqa_v4_500q_states.jsonl.gz; has `popularity`):
  1. correctness@S0 AUROC from h_S0 (sanity)
  2. conditional benefit | wrong@S0 AUROC from h_S0  (compare to HotpotQA 0.575)
  3. popularity vs S0-correctness and vs benefit (parametric-knowledge signature)
  4. popularity-ONLY as a benefit predictor (non-hidden-state routable signal)

  python analyze_popqa_mechanism.py --data data/popqa_v4_500q_states.jsonl.gz
"""
import argparse, json, gzip, numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
SEED = 42


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    by_q = defaultdict(list)
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            by_q[d["query_id"]].append(d)
    rows = []
    for q, st in by_q.items():
        st.sort(key=lambda x: x.get("stage", x.get("stage_idx", 0)))
        if len(st) < 4:
            continue
        st = st[:4]
        c = [int(s.get("correct", s.get("stage_correctness", 0))) for s in st]
        h0 = np.asarray(st[0].get("hs_concat", st[0].get("multi_layer_hidden_states")), dtype=np.float32).ravel()
        pop = st[0].get("popularity", None)
        rows.append({"qid": q, "c": c, "h0": h0, "pop": pop})
    return rows


def oof_auroc(X, y, g):
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, groups=g):
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        lr = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=SEED)
        lr.fit(Xtr, y[tr]); oof[te] = lr.predict_proba(Xte)[:, 1]
    return roc_auc_score(y, oof), oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/popqa_v4_500q_states.jsonl.gz")
    args = ap.parse_args()
    rows = load(args.data)
    qid = np.array([r["qid"] for r in rows])
    H0 = np.array([r["h0"] for r in rows])
    C = np.array([r["c"] for r in rows])
    pops = np.array([r["pop"] if r["pop"] is not None else np.nan for r in rows], dtype=float)
    n = len(rows)
    print(f"n={n}, has_popularity={np.isfinite(pops).mean()*100:.0f}%")
    print(f"stage acc: {C.mean(0).round(3).tolist()}")

    # 1. correctness@S0
    a_corr, _ = oof_auroc(H0, C[:, 0], qid)
    print(f"\n[1] correctness@S0 AUROC (h_S0): {a_corr:.3f}")

    # 2. conditional benefit | wrong@S0
    wmask = C[:, 0] == 0
    yb = (C[wmask, 1:].max(1) == 1).astype(int)   # correct at ANY later stage
    a_ben, _ = oof_auroc(H0[wmask], yb, qid[wmask])
    print(f"[2] conditional benefit | wrong@S0 AUROC (h_S0): {a_ben:.3f}  "
          f"(n={wmask.sum()}, benefit-rate {yb.mean()*100:.0f}%)  vs HotpotQA 0.575")

    # 3. popularity signatures
    fin = np.isfinite(pops)
    lp = np.log10(np.clip(pops, 1, None))
    s0 = C[:, 0]
    print(f"\n[3] popularity (log10) correlations:")
    print(f"    corr(log_pop, correct@S0) = {np.corrcoef(lp[fin], s0[fin])[0,1]:+.3f}")
    # benefit among wrong@S0, vs popularity
    bw = fin & wmask
    yb_all = (C[:, 1:].max(1) == 1).astype(int)
    print(f"    corr(log_pop, benefit|wrong@S0) = {np.corrcoef(lp[bw], (C[bw,1:].max(1)==1).astype(int))[0,1]:+.3f}")
    # terciles
    qs = np.nanpercentile(lp[fin], [33, 67])
    for name, m in [("low-pop", lp <= qs[0]), ("mid-pop", (lp > qs[0]) & (lp <= qs[1])), ("high-pop", lp > qs[1])]:
        m = m & fin
        print(f"    {name:<9} (n={m.sum():3d}): S0-acc={s0[m].mean():.3f}  ever-correct={yb_all[m].mean():.3f}")

    # 4. popularity-only as benefit predictor (non-hidden-state)
    if bw.sum() > 20 and len(set((C[bw,1:].max(1)==1))) > 1:
        ybw = (C[bw, 1:].max(1) == 1).astype(int)
        a_pop = roc_auc_score(ybw, lp[bw])
        print(f"\n[4] popularity-ONLY benefit|wrong@S0 AUROC: {max(a_pop,1-a_pop):.3f} "
              f"(non-hidden-state; shows the routable signal is largely entity familiarity)")


if __name__ == "__main__":
    main()
