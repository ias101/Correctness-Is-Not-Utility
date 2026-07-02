"""
Multi-step (eventual) utility horizon control (Loop 34, auto-review-loop-llm).

Critique: the Delta Probe labels benefit ONE step ahead (wrong_t -> correct_{t+1}).
But routing cares about EVENTUAL benefit: a query wrong at t may stay wrong at t+1
yet flip correct at t+2/t+3. So "conditional benefit near chance (0.575)" might be
a myopic-label artifact. Same for degradation over a multi-step horizon.

This relabels benefit/degradation at several horizons and re-measures probe AUROC,
on the matched proxy protocol (BF16 single-layer HS; LR, 5-fold GroupKFold by query,
2000 query-grouped bootstrap). Crucially it ALSO reports the stage-only baseline and
a correctness-confidence leakage baseline for every horizon (the Loop-33 lesson:
benefit prevalence is stage-dependent, and "how-wrong" leaks).

Benefit horizons, on the currently-wrong support (c[t]==0):
  1step     : correct_{t+1}
  multistep : correct at ANY t'>t   (= "eventual benefit"; the routing-relevant one)
  final     : correct_3             (does going all the way help?)
  two_step  : among wrong_t & wrong_{t+1}, predict correct_{t+2} (literal n+1 fail, n+2 win)
Degradation horizons, on the currently-correct support (c[t]==1):
  1step     : wrong_{t+1}
  eventual  : wrong at ANY t'>t
  final     : wrong_3

  python experiments/control_multistep_benefit.py \
      --data data/bf16_hotpotqa_progressive.jsonl \
      --out review-stage/control_multistep_benefit.json
"""
import argparse, json, os, numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

N_BOOT = 2000
SEED = 42


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
        for t in range(4):
            rows.append({"qid": qid, "t": t,
                         "hs": np.asarray(st[t]["hidden_state"], dtype=np.float32),
                         "c": c})  # full correctness vector per query
    return rows


def cv_oof(X, y, g, n_splits=5):
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups=g):
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        lr = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced", random_state=SEED)
        lr.fit(Xtr, y[tr]); oof[te] = lr.predict_proba(Xte)[:, 1]
    return oof


def boot(y, s, g, rng, n_boot=N_BOOT):
    uq = list(set(g)); q2i = defaultdict(list)
    for i, q in enumerate(g):
        q2i[q].append(i)
    aus, aps = [], []
    for _ in range(n_boot):
        idx = []
        for q in rng.choice(uq, len(uq), replace=True):
            idx.extend(q2i[q])
        idx = np.array(idx)
        if len(set(y[idx])) < 2:
            continue
        aus.append(roc_auc_score(y[idx], s[idx])); aps.append(average_precision_score(y[idx], s[idx]))
    a = np.array(aus)
    return {"auroc": float(a.mean()), "lo": float(np.percentile(a, 2.5)),
            "hi": float(np.percentile(a, 97.5)), "auprc": float(np.mean(aps)),
            "n": int(len(y)), "n_pos": int(y.sum()), "prev": float(y.mean())}


def evaluate(rows, mask, y, label, rng, out, corr_oof=None):
    """Probe (HS) + stage-only + correctness-confidence leakage, on a given support."""
    idx = np.where(mask)[0]
    X = np.array([rows[i]["hs"] for i in idx]); g = np.array([rows[i]["qid"] for i in idx])
    stage = np.array([[rows[i]["t"]] for i in idx], dtype=np.float32)
    yy = y[idx]
    rec = {"prev": float(yy.mean()), "n": int(len(yy)), "n_pos": int(yy.sum())}
    if len(set(yy)) < 2 or yy.sum() < 8:
        rec["skipped"] = True; out[label] = rec; print(f"  {label:<26} SKIP (n_pos={int(yy.sum())})"); return
    rec["hs"] = boot(yy, cv_oof(X, yy, g), g, rng)
    rec["stage_only"] = boot(yy, cv_oof(stage, yy, g), g, rng)
    if corr_oof is not None:
        rec["confidence_leak"] = boot(yy, corr_oof[idx], g, rng)  # P(correct_t) as predictor
    out[label] = rec
    extra = f" | stage={rec['stage_only']['auroc']:.3f}" + (
        f" | conf={rec['confidence_leak']['auroc']:.3f}" if corr_oof is not None else "")
    print(f"  {label:<26} HS={rec['hs']['auroc']:.3f}[{rec['hs']['lo']:.3f},{rec['hs']['hi']:.3f}]"
          f"{extra}  (n={rec['n']}, prev {rec['prev']*100:.0f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bf16_hotpotqa_progressive.jsonl")
    ap.add_argument("--out", default="review-stage/control_multistep_benefit.json")
    args = ap.parse_args()
    rng = np.random.RandomState(SEED)
    rows = load(args.data)
    HS = np.array([r["hs"] for r in rows]); QID = np.array([r["qid"] for r in rows])
    T = np.array([r["t"] for r in rows]); C = [r["c"] for r in rows]
    print(f"transitions(all stages)={len(rows)}  hs_dim={HS.shape[1]}")

    # correctness-at-t OOF score (for the confidence-leakage baseline among wrong_t/correct_t)
    y_corr = np.array([r["c"][r["t"]] for r in rows])
    corr_oof = cv_oof(HS, y_corr, QID)
    print(f"correctness-at-t probe AUROC (sanity): "
          f"{roc_auc_score(y_corr, corr_oof):.3f}")

    res = {"_meta": {"data": args.data, "precision": "bf16", "hs_layers": 1,
                     "probe": "LR(C=1,balanced)+StandardScaler, 5-fold GroupKFold, "
                              f"{N_BOOT} query-grouped bootstrap",
                     "note": "BF16 single-layer proxy; canonical 4-bit/4-layer/2000q pending. "
                             "Relative horizon comparison is the signal, not absolute level."}}

    # ===== BENEFIT horizons (support: currently wrong, t<3) =====
    print("\n[BENEFIT] support = currently-wrong (c[t]==0), t<3")
    wrong = np.array([C[i][T[i]] == 0 and T[i] < 3 for i in range(len(rows))])
    y_1 = np.array([1 if (C[i][T[i]] == 0 and T[i] < 3 and C[i][T[i]+1] == 1) else 0 for i in range(len(rows))])
    y_multi = np.array([1 if (C[i][T[i]] == 0 and T[i] < 3 and any(C[i][tt] == 1 for tt in range(T[i]+1, 4))) else 0 for i in range(len(rows))])
    y_final = np.array([1 if (C[i][T[i]] == 0 and T[i] < 3 and C[i][3] == 1) else 0 for i in range(len(rows))])
    b = {}
    evaluate(rows, wrong, y_1, "benefit_1step", rng, b, corr_oof)
    evaluate(rows, wrong, y_multi, "benefit_multistep(eventual)", rng, b, corr_oof)
    evaluate(rows, wrong, y_final, "benefit_final(reach_S3)", rng, b, corr_oof)
    # literal two-step: support = wrong_t & wrong_{t+1} (t<2); label = correct_{t+2}
    twostep_mask = np.array([C[i][T[i]] == 0 and T[i] < 2 and C[i][T[i]+1] == 0 for i in range(len(rows))])
    y_2 = np.array([1 if (twostep_mask[i] and C[i][T[i]+2] == 1) else 0 for i in range(len(rows))])
    evaluate(rows, twostep_mask, y_2, "benefit_2step(n+1fail,n+2win)", rng, b, corr_oof)
    res["benefit"] = b

    # ===== DEGRADATION horizons (support: currently correct, t<3) =====
    print("\n[DEGRADATION] support = currently-correct (c[t]==1), t<3")
    correct = np.array([C[i][T[i]] == 1 and T[i] < 3 for i in range(len(rows))])
    d_1 = np.array([1 if (C[i][T[i]] == 1 and T[i] < 3 and C[i][T[i]+1] == 0) else 0 for i in range(len(rows))])
    d_evt = np.array([1 if (C[i][T[i]] == 1 and T[i] < 3 and any(C[i][tt] == 0 for tt in range(T[i]+1, 4))) else 0 for i in range(len(rows))])
    d_fin = np.array([1 if (C[i][T[i]] == 1 and T[i] < 3 and C[i][3] == 0) else 0 for i in range(len(rows))])
    dd = {}
    evaluate(rows, correct, d_1, "degrade_1step", rng, dd, corr_oof)
    evaluate(rows, correct, d_evt, "degrade_eventual", rng, dd, corr_oof)
    evaluate(rows, correct, d_fin, "degrade_final(wrong_S3)", rng, dd, corr_oof)
    res["degradation"] = dd

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
