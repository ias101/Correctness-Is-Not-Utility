"""
Canonical-data confirmation of Loop 33 + Loop 34 (multi-layer 14336-d, 4-bit).

Runs on the canonical collection (collect_canonical_v5.py output):
  A. HORIZON robustness (Loop 34): benefit/degradation at 1-step/eventual/final;
     HS probe vs stage-only baseline.
  B. INFO-ASYMMETRY (Loop 33 Control A CE-part + Control B): benefit | wrong_t from
     HS vs non-oracle incoming-passage cross-encoder RELEVANCE (stage-free, from the
     stored ce_scores) vs stage-only; degradation | correct_t HS-full vs
     correctness-confidence-only.

Reads `multi_layer_hidden_states` (14336-d) and per-stage `ce_scores` (top-k sorted
desc), so the incoming passages at t->t+1 are ce_scores[t+1][k_t:k_{t+1}].
LR, 5-fold GroupKFold by query, 2000 query-grouped bootstrap. CPU-ok.

  python analyze_canonical_v5.py --data collected_states_hotpotqa_v5_canon.jsonl \
      --out analyze_canonical_v5.json
"""
import argparse, json, os, numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

N_BOOT = 2000
SEED = 42
STAGE_SIZES = [2, 4, 6, 8]


def load(path):
    by_q = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            by_q[d["query_id"]].append(d)
    rows = []
    for qid, st in by_q.items():
        st.sort(key=lambda x: x["stage_idx"])
        if len(st) < 4:
            continue
        c = [int(s["stage_correctness"]) for s in st]
        ce = [s.get("ce_scores", []) for s in st]   # per-stage top-k scores
        for t in range(4):
            rows.append({"qid": qid, "t": t,
                         "hs": np.asarray(st[t]["multi_layer_hidden_states"], dtype=np.float32).ravel(),
                         "c": c, "ce": ce})
    return rows


def incoming_ce(ce, t):
    """CE relevance magnitudes of passages NEWLY revealed at t+1 (stage-free)."""
    if t + 1 >= len(ce) or not ce[t + 1]:
        return np.zeros(3, dtype=np.float32)
    kt = STAGE_SIZES[t]
    new = ce[t + 1][kt:STAGE_SIZES[t + 1]] if len(ce[t + 1]) > kt else ce[t + 1][-2:]
    new = np.asarray(new, dtype=np.float32)
    if new.size == 0:
        new = np.asarray(ce[t + 1][-1:], dtype=np.float32)
    return np.array([new.mean(), new.max(), new.min()], dtype=np.float32)


def cv_oof(X, y, g):
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, groups=g):
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        lr = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced", random_state=SEED)
        lr.fit(Xtr, y[tr]); oof[te] = lr.predict_proba(Xte)[:, 1]
    return oof


def boot(y, s, g, rng):
    uq = list(set(g)); q2i = defaultdict(list)
    for i, q in enumerate(g):
        q2i[q].append(i)
    aus = []
    for _ in range(N_BOOT):
        idx = []
        for q in rng.choice(uq, len(uq), replace=True):
            idx.extend(q2i[q])
        idx = np.array(idx)
        if len(set(y[idx])) < 2:
            continue
        aus.append(roc_auc_score(y[idx], s[idx]))
    a = np.array(aus)
    return {"auroc": float(a.mean()), "lo": float(np.percentile(a, 2.5)),
            "hi": float(np.percentile(a, 97.5)), "n": int(len(y)),
            "n_pos": int(y.sum()), "prev": float(y.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="collected_states_hotpotqa_v5_canon.jsonl")
    ap.add_argument("--out", default="analyze_canonical_v5.json")
    args = ap.parse_args()
    rng = np.random.RandomState(SEED)
    rows = load(args.data)
    HS = np.array([r["hs"] for r in rows]); QID = np.array([r["qid"] for r in rows])
    T = np.array([r["t"] for r in rows]); C = [r["c"] for r in rows]
    print(f"records={len(rows)}  queries={len(set(QID))}  hs_dim={HS.shape[1]}")
    res = {"_meta": {"data": args.data, "precision": "4bit-nf4", "hs_dim": int(HS.shape[1]),
                     "n_records": len(rows), "n_queries": int(len(set(QID))),
                     "probe": "LR(C=1,balanced)+StandardScaler, 5-fold GroupKFold, "
                              f"{N_BOOT} bootstrap"}}
    y_corr = np.array([C[i][T[i]] for i in range(len(rows))])
    corr_oof = cv_oof(HS, y_corr, QID)
    res["_meta"]["correctness_probe_auroc"] = float(roc_auc_score(y_corr, corr_oof))
    print(f"correctness-at-t probe AUROC: {res['_meta']['correctness_probe_auroc']:.3f}")

    def stagecol(mask):
        return np.array([[rows[i]["t"]] for i in np.where(mask)[0]], dtype=np.float32)

    def ev(mask, y, tag, out, extra_feats=None):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return
        X = HS[idx]; g = QID[idx]; yy = y[idx]
        if yy.sum() < 8 or len(set(yy)) < 2:
            out[tag] = {"skipped": True, "n_pos": int(yy.sum()), "n": int(len(yy))}
            print(f"  {tag:<28} SKIP n_pos={int(yy.sum())}"); return
        rec = {"hs": boot(yy, cv_oof(X, yy, g), g, rng),
               "stage_only": boot(yy, cv_oof(stagecol(mask), yy, g), g, rng),
               "prev": float(yy.mean()), "n": int(len(yy))}
        line = f"  {tag:<28} HS={rec['hs']['auroc']:.3f}[{rec['hs']['lo']:.3f},{rec['hs']['hi']:.3f}] | stage={rec['stage_only']['auroc']:.3f}"
        if extra_feats:
            for nm, Xe in extra_feats.items():
                rec[nm] = boot(yy, cv_oof(Xe[idx], yy, g), g, rng)
                line += f" | {nm}={rec[nm]['auroc']:.3f}"
        rec["confidence_leak"] = boot(yy, corr_oof[idx], g, rng)
        line += f" | conf={rec['confidence_leak']['auroc']:.3f}  (n={rec['n']},prev {rec['prev']*100:.0f}%)"
        print(line); out[tag] = rec

    # ---- A. HORIZON (Loop 34) ----
    print("\n[A. HORIZON] benefit | currently-wrong")
    wrong = np.array([C[i][T[i]] == 0 and T[i] < 3 for i in range(len(rows))])
    b = {}
    ev(wrong, np.array([1 if (C[i][T[i]]==0 and T[i]<3 and C[i][T[i]+1]==1) else 0 for i in range(len(rows))]), "benefit_1step", b)
    ev(wrong, np.array([1 if (C[i][T[i]]==0 and T[i]<3 and any(C[i][tt]==1 for tt in range(T[i]+1,4))) else 0 for i in range(len(rows))]), "benefit_eventual", b)
    ev(wrong, np.array([1 if (C[i][T[i]]==0 and T[i]<3 and C[i][3]==1) else 0 for i in range(len(rows))]), "benefit_final", b)
    res["horizon_benefit"] = b
    print("[A. HORIZON] degradation | currently-correct")
    correct = np.array([C[i][T[i]] == 1 and T[i] < 3 for i in range(len(rows))])
    dd = {}
    ev(correct, np.array([1 if (C[i][T[i]]==1 and T[i]<3 and C[i][T[i]+1]==0) else 0 for i in range(len(rows))]), "degrade_1step", dd)
    ev(correct, np.array([1 if (C[i][T[i]]==1 and T[i]<3 and any(C[i][tt]==0 for tt in range(T[i]+1,4))) else 0 for i in range(len(rows))]), "degrade_eventual", dd)
    res["horizon_degradation"] = dd

    # ---- B. INFO-ASYMMETRY (Loop 33 A CE-part + B) ----
    print("\n[B. INFO-ASYM] benefit | wrong_t: HS vs incoming-CE-relevance vs stage")
    INC = np.array([incoming_ce(rows[i]["ce"], rows[i]["t"]) for i in range(len(rows))])
    yb = np.array([1 if (C[i][T[i]]==0 and T[i]<3 and C[i][T[i]+1]==1) else 0 for i in range(len(rows))])
    ev(wrong, yb, "benefit_1step_vs_incomingCE", res.setdefault("info_asym", {}),
       extra_feats={"incoming_ce_relevance": INC})
    print("[B. INFO-ASYM] degradation | correct_t: HS-full vs confidence-only (already in 'conf')")
    # (degradation HS-full vs confidence captured by ev's conf field on degrade labels above)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
