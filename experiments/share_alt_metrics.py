"""
Loop 38 (P0.3 grounding): alternative self-knowledge-share estimators on the REAL causal
degrade data, to show the AUROC-ratio is one PROXY among several that agree on the verdict.

Per reliability level p, on the closed-book-wrong support, estimate the share three ways:
  (1) AUROC-ratio  (the paper's current estimator): (AUCself-.5)/[(AUCself-.5)+(AUCretr-.5)]
  (2) variance-ratio (faithful to the DEFINITION Var(E[B|h])/Var(B)): use OOF E[b|h] and
      E[b|r] and form share = Vh/(Vh+Vr), Vh=Var(OOF E[b|h]), Vr=Var(E[b|r]).
  (3) Brier-skill ratio: skill_self/(skill_self+skill_retr), skill = 1 - Brier/Brier_base.
All OOF (GroupKFold by query). If all three collapse together as p rises, the verdict
(share governs routability) is robust to the proxy choice.

  python experiments/share_alt_metrics.py  # run on the box where degrade_popqa_causal.jsonl lives
"""
import json, numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
SEED = 42

recs = [json.loads(l) for l in open("degrade_popqa_causal.jsonl")]
h0, cb = {}, {}
seq = defaultdict(lambda: {"c": {}, "rel": None})
for r in recs:
    q = r["qid"]
    if r["stage"] == 0:
        h0[q] = np.asarray(r["hs_concat"], np.float32); cb[q] = int(r["correct"])
    else:
        seq[(q, r["p"])]["c"][r["stage"]] = int(r["correct"]); seq[(q, r["p"])]["rel"] = r["reliable"]
qids_all = sorted(h0); P = sorted({k[1] for k in seq})


def oof_proba(X, y, g):
    if len(np.unique(y)) < 2 or len(y) < 20:
        return None
    o = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, groups=g):
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=SEED)
        lr.fit(sc.transform(X[tr]), y[tr]); o[te] = lr.predict_proba(sc.transform(X[te]))[:, 1]
    return o


def adv(a):
    return max(0.0, a - 0.5) if (a is not None and a == a) else 0.0


out = []
for p in P:
    qids = [q for q in qids_all if (q, p) in seq]
    H = np.array([h0[q] for q in qids]); cbw = np.array([cb[q] for q in qids]) == 0
    ben = np.array([(max(seq[(q, p)]["c"].values()) if seq[(q, p)]["c"] else 0) for q in qids])
    ben = ((ben == 1) & cbw).astype(int)
    rel = np.array([seq[(q, p)]["rel"] for q in qids], float)
    b = ben[cbw]; g = np.array(qids)[cbw]; Hw = H[cbw]; rw = rel[cbw].reshape(-1, 1)
    if len(b) < 20 or len(np.unique(b)) < 2:
        out.append({"p": p, "note": "degenerate"}); continue
    ph = oof_proba(Hw, b, g)                       # OOF E[b|h]
    pr = oof_proba(rw, b, g)                        # OOF E[b|r]
    # (1) AUROC-ratio
    auc_s = roc_auc_score(b, ph) if ph is not None else None
    auc_r = roc_auc_score(b, rw.ravel()) if len(np.unique(rw)) > 1 else None
    sh_auroc = adv(auc_s) / (adv(auc_s) + adv(auc_r)) if (adv(auc_s) + adv(auc_r)) > 0 else None
    # (2) variance-ratio (faithful to definition)
    Vh = float(np.var(ph)) if ph is not None else 0.0
    Vr = float(np.var(pr)) if pr is not None else 0.0
    sh_var = Vh / (Vh + Vr) if (Vh + Vr) > 0 else None
    # (3) Brier-skill ratio
    base = b.mean()
    bs_base = brier_score_loss(b, np.full_like(b, base, float))
    sk_h = 1 - brier_score_loss(b, ph) / bs_base if ph is not None and bs_base > 0 else 0
    sk_r = 1 - brier_score_loss(b, pr) / bs_base if pr is not None and bs_base > 0 else 0
    sk_h, sk_r = max(0, sk_h), max(0, sk_r)
    sh_brier = sk_h / (sk_h + sk_r) if (sk_h + sk_r) > 0 else None
    row = {"p": p, "n_wrong": int(cbw.sum()), "benefit_rate": round(float(b.mean()), 3),
           "share_auroc": round(sh_auroc, 3) if sh_auroc is not None else None,
           "share_varratio": round(sh_var, 3) if sh_var is not None else None,
           "share_brierskill": round(sh_brier, 3) if sh_brier is not None else None,
           "Vh": round(Vh, 4), "Vr": round(Vr, 4)}
    out.append(row); print(json.dumps(row), flush=True)

json.dump(out, open("review-stage/share_alt_metrics.json", "w"), indent=2)
print("[*] -> review-stage/share_alt_metrics.json")
