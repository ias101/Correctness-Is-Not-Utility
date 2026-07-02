"""
Self-knowledge SHARE metric for the causal degradation experiment (R1's exact ask).

share(p) = (AUC_self - 0.5) / [ (AUC_self - 0.5) + (AUC_retr - 0.5) ]
  AUC_self = AUROC predicting benefit|CB-wrong from h_S0      (self-knowledge, IN h)
  AUC_retr = AUROC predicting benefit|CB-wrong from the realized reliability draw (NOT in h)
Theory: as retrieval noise p rises, AUC_retr rises and share collapses -> routing dies.
Endpoints p in {0,1} have no reliability variance (AUC_retr undefined) => share defined as
1.0 at p=0 (pure self-knowledge) and reported NA at p=1 (no benefit events).
"""
import json, numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
SEED = 42

recs = [json.loads(l) for l in open("degrade_popqa_causal.jsonl")]
h0 = {}; cb = {}
seq = defaultdict(lambda: {"c": {}, "rel": None})
for r in recs:
    q = r["qid"]
    if r["stage"] == 0:
        h0[q] = np.asarray(r["hs_concat"], dtype=np.float32); cb[q] = int(r["correct"])
    else:
        seq[(q, r["p"])]["c"][r["stage"]] = int(r["correct"]); seq[(q, r["p"])]["rel"] = r["reliable"]

qids_all = sorted(h0.keys())
P = sorted({k[1] for k in seq})

def oof(X, y, g):
    if len(np.unique(y)) < 2 or len(y) < 20: return float("nan")
    o = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, groups=g):
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=SEED)
        lr.fit(sc.transform(X[tr]), y[tr]); o[te] = lr.predict_proba(sc.transform(X[te]))[:, 1]
    return roc_auc_score(y, o)

out = []
for p in P:
    qids = [q for q in qids_all if (q, p) in seq]
    H0 = np.array([h0[q] for q in qids])
    cbw = np.array([cb[q] for q in qids]) == 0
    ben = np.array([(max(seq[(q, p)]["c"].values()) if seq[(q, p)]["c"] else 0) for q in qids])
    ben = ((ben == 1) & cbw).astype(int)
    rel = np.array([seq[(q, p)]["rel"] for q in qids])
    bw = ben[cbw]; gw = np.array(qids)[cbw]; Hw = H0[cbw]; rw = rel[cbw]
    auc_self = oof(Hw, bw, gw)
    # retrieval-term AUROC: single binary feature (the draw); use rank = the draw itself
    auc_retr = roc_auc_score(bw, rw) if len(np.unique(rw)) > 1 and len(np.unique(bw)) > 1 else float("nan")
    def adv(a): return max(0.0, a - 0.5) if a == a else 0.0
    s_self, s_retr = adv(auc_self), adv(auc_retr)
    share = (s_self / (s_self + s_retr)) if (s_self + s_retr) > 0 else (1.0 if p == 0.0 else float("nan"))
    out.append({"p": p, "n_cb_wrong": int(cbw.sum()), "benefit_rate": round(float(bw.mean()), 3),
                "auc_self_hS0": round(auc_self, 3) if auc_self == auc_self else None,
                "auc_retr_draw": round(auc_retr, 3) if auc_retr == auc_retr else None,
                "self_knowledge_share": round(share, 3) if share == share else None,
                "frac_reliable": round(float(rel.mean()), 3)})
    print(json.dumps(out[-1]), flush=True)

json.dump(out, open("review-stage/degrade_selfknowledge_share.json", "w"), indent=2)
# monotonicity over the NON-degenerate interior (p with reliability variance)
interior = [r for r in out if r["self_knowledge_share"] is not None and 0 < r["p"] < 1]
print("\ninterior shares (p in 0.25..0.75):", [(r["p"], r["self_knowledge_share"]) for r in interior])
print("[*] -> review-stage/degrade_selfknowledge_share.json")
