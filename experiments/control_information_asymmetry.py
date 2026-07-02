"""
Information-availability controls (Loop 33, auto-review-loop-llm Round 1).

Reviewer critique (advisor): the generation-side hidden state is captured BEFORE
the candidate passage is added, so it structurally cannot encode the incoming
passage's content. Therefore "hidden states can't predict benefit" may be a
trivial information-availability artifact, not a representational property; and
the retrieval-side benefit signal is trivial because it has the passage. Two
decisive controls:

  CONTROL A (non-oracle passage concatenation).
    Concatenate the ACTUAL incoming-passage representation -- here the NON-ORACLE
    cross-encoder relevance features of the passages revealed at t+1 (ce_mean/max/
    min/n at t+1 and their delta vs t); NO gold-passage labels -- to the
    generation-side hidden state, and re-test conditional benefit (among wrong_t).
    If benefit AUROC recovers above HS-only, the advisor's information-availability
    reading is CONFIRMED (-> reframe). If it stays near chance, F2 is vindicated.
    Comparable to paper Table 6 (HS-only 0.523 vs oracle 0.780) but with a clean,
    deployable, NON-oracle incoming-evidence descriptor.

  CONTROL B (degradation leakage baseline).
    The paper claims conditional degradation (0.693) is a "present-state property"
    while benefit (0.575) is not. Test whether degradation predictability is just
    correctness/answer-fragility leakage: use the correctness-direction probe's
    OOF confidence (P(wrong_t), i.e. fragility) as the ONLY predictor of
    degradation among correct_t. If confidence-only ~ full-HS degradation, the
    asymmetry is a leakage artifact (advisor's stronger point). If full-HS clearly
    beats confidence-only, the present-state-property defense holds.

Data: data/bf16_hotpotqa_progressive.jsonl (Qwen2.5-7B, BF16, 500q x 4 stages,
single-layer 3584-d hidden_state). This is the LOCAL BF16 precision-control dump;
the canonical 4-bit multi-layer V5 re-run is a pending follow-up (server offline).
Protocol matches oracle_conditional_benefit.py: 5-fold GroupKFold by query, LR
(class_weight balanced, StandardScaler per fold), OOF predictions, query-grouped
bootstrap CIs. CPU-only.

  python experiments/control_information_asymmetry.py \
      --data data/bf16_hotpotqa_progressive.jsonl \
      --out review-stage/control_information_asymmetry.json
"""
import argparse, json, os, numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

N_BOOT = 2000
SEED = 42


def load_transitions(path):
    """Return per-transition records with HS at t, incoming-passage (non-oracle)
    CE features for t+1, and cur/next correctness."""
    by_q = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            by_q[d["query_id"]].append(d)

    def ce_vec(rec):
        ce = rec.get("ce_scores", []) or []
        n = float(len(ce))
        return np.array([
            float(rec.get("ce_mean", np.mean(ce) if ce else 0.0)),
            float(rec.get("ce_max", np.max(ce) if ce else 0.0)),
            float(rec.get("ce_min", np.min(ce) if ce else 0.0)),
            n,
        ], dtype=np.float32)

    rows = []
    for qid, stages in by_q.items():
        stages.sort(key=lambda x: x["stage_idx"])
        if len(stages) < 4:
            continue
        for t in range(len(stages) - 1):
            cur, nxt = stages[t], stages[t + 1]
            hs = np.asarray(cur["hidden_state"], dtype=np.float32).ravel()
            ce_t, ce_t1 = ce_vec(cur), ce_vec(nxt)
            # incoming-evidence descriptor: next-stage CE absolute + delta (non-oracle)
            incoming = np.concatenate([ce_t1, ce_t1 - ce_t]).astype(np.float32)  # 8-d
            rows.append({
                "qid": qid, "t": t, "hs": hs, "incoming": incoming,
                "cur": int(cur["stage_correctness"]),
                "next": int(nxt["stage_correctness"]),
            })
    return rows


def cv_oof(X, y, groups, n_splits=5):
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups=groups):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        lr = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced",
                                random_state=SEED)
        lr.fit(Xtr, y[tr])
        oof[te] = lr.predict_proba(Xte)[:, 1]
    return oof


def boot(y, score, groups, rng, n_boot=N_BOOT):
    uq = list(set(groups)); nq = len(uq)
    q2i = defaultdict(list)
    for i, q in enumerate(groups):
        q2i[q].append(i)
    aus, aps = [], []
    for _ in range(n_boot):
        sq = rng.choice(uq, nq, replace=True)
        idx = []
        for q in sq:
            idx.extend(q2i[q])
        idx = np.array(idx)
        if len(set(y[idx])) < 2:
            continue
        aus.append(roc_auc_score(y[idx], score[idx]))
        aps.append(average_precision_score(y[idx], score[idx]))
    a, p = np.array(aus), np.array(aps)
    return {"auroc": float(a.mean()), "auroc_lo": float(np.percentile(a, 2.5)),
            "auroc_hi": float(np.percentile(a, 97.5)),
            "auprc": float(p.mean()),
            "n": int(len(y)), "n_pos": int(int(y.sum())), "prev": float(y.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bf16_hotpotqa_progressive.jsonl")
    ap.add_argument("--out", default="review-stage/control_information_asymmetry.json")
    args = ap.parse_args()

    rng = np.random.RandomState(SEED)
    print(f"Loading {args.data} ...")
    rows = load_transitions(args.data)
    qids = np.array([r["qid"] for r in rows])
    cur = np.array([r["cur"] for r in rows])
    nxt = np.array([r["next"] for r in rows])
    HS = np.array([r["hs"] for r in rows])
    INC = np.array([r["incoming"] for r in rows])
    print(f"  transitions={len(rows)}  hs_dim={HS.shape[1]}  incoming_dim={INC.shape[1]}")
    print(f"  wrong_t={int((cur==0).sum())}  correct_t={int((cur==1).sum())}")

    res = {"_meta": {"data": args.data, "precision": "bf16", "hs_layers": 1,
                     "hs_dim": int(HS.shape[1]), "n_transitions": len(rows),
                     "probe": "LR(C=1,balanced)+StandardScaler, 5-fold GroupKFold, "
                              f"{N_BOOT} query-grouped bootstrap",
                     "note": "BF16 single-layer local replication; canonical 4-bit "
                             "multi-layer V5 re-run pending (server offline)"}}

    # ===== CONTROL A: non-oracle incoming-passage concatenation, benefit | wrong_t =====
    print("\n[CONTROL A] conditional benefit (wrong_t), non-oracle incoming-passage features")
    wmask = cur == 0
    yb = nxt[wmask]                         # benefit: wrong_t -> correct_{t+1}
    gb = qids[wmask]
    feats_A = {"hs_only": HS[wmask],
               "incoming_only": INC[wmask],
               "hs_plus_incoming": np.concatenate([HS[wmask], INC[wmask]], axis=1)}
    A = {"_support": "wrong_t", "benefit_prev": float(yb.mean()), "n": int(wmask.sum())}
    for name, X in feats_A.items():
        A[name] = boot(yb, cv_oof(X, yb, gb), gb, rng)
        m = A[name]
        print(f"  {name:<18} AUROC={m['auroc']:.3f}[{m['auroc_lo']:.3f},{m['auroc_hi']:.3f}] "
              f"AUPRC={m['auprc']:.3f}  n={m['n']} pos={m['n_pos']} ({m['prev']*100:.1f}%)")
    A["delta_incoming_minus_hs"] = A["hs_plus_incoming"]["auroc"] - A["hs_only"]["auroc"]
    A["delta_incomingonly_minus_hs"] = A["incoming_only"]["auroc"] - A["hs_only"]["auroc"]
    print(f"  Delta(HS+incoming - HS) = {A['delta_incoming_minus_hs']:+.3f} | "
          f"(incoming_only - HS) = {A['delta_incomingonly_minus_hs']:+.3f}")
    res["control_A_benefit_incoming"] = A

    # ===== CONTROL A2: stage-confound check (reviewer R2 weakness #1) =====
    # Is incoming_only's signal just a stage-prevalence proxy? Test relevance
    # MAGNITUDES only (ce_mean/max/min at t+1; drop n_passages and deltas) and a
    # stage-index-only baseline; plus per-stage benefit AUROC of incoming_only.
    print("\n[CONTROL A2] stage-confound check (benefit | wrong_t)")
    relmag = INC[wmask][:, :3]                                   # ce_mean,max,min at t+1
    stage_w = np.array([[r["t"]] for r in rows], dtype=np.float32)[wmask]
    A2 = {}
    A2["incoming_relmag_only"] = boot(yb, cv_oof(relmag, yb, gb), gb, rng)   # 3-d, stage-free magnitudes
    A2["stage_only"] = boot(yb, cv_oof(stage_w, yb, gb), gb, rng)            # benefit-from-stage baseline
    inc_oof = cv_oof(feats_A["incoming_only"], yb, gb)
    tw = stage_w.ravel().astype(int)
    per_stage = {}
    for t in sorted(set(tw)):
        m = tw == t
        if len(set(yb[m])) > 1:
            per_stage[f"S{t}->S{t+1}"] = {"n": int(m.sum()), "n_pos": int(yb[m].sum()),
                                          "prev": float(yb[m].mean()),
                                          "auroc": float(roc_auc_score(yb[m], inc_oof[m]))}
    A2["incoming_only_per_stage"] = per_stage
    A2["delta_relmag_minus_stage"] = A2["incoming_relmag_only"]["auroc"] - A2["stage_only"]["auroc"]
    for name in ["incoming_relmag_only", "stage_only"]:
        m = A2[name]
        print(f"  {name:<20} AUROC={m['auroc']:.3f}[{m['auroc_lo']:.3f},{m['auroc_hi']:.3f}]")
    print(f"  Delta(relmag_only - stage_only) = {A2['delta_relmag_minus_stage']:+.3f}")
    print(f"  incoming_only per-stage benefit AUROC: " +
          " | ".join(f"{k}:{v['auroc']:.3f}(prev {v['prev']*100:.0f}%)" for k, v in per_stage.items()))
    res["control_A2_stage_check"] = A2

    # ===== CONTROL B: degradation confidence-leakage baseline, degradation | correct_t =====
    print("\n[CONTROL B] conditional degradation (correct_t): full-HS vs confidence-only")
    # correctness-direction probe: predict wrong_t from HS over ALL transitions (OOF)
    y_wrong = (cur == 0).astype(int)
    p_wrong = cv_oof(HS, y_wrong, qids)     # OOF P(wrong_t) = fragility/confidence
    cmask = cur == 1
    yd = (nxt[cmask] == 0).astype(int)      # degradation: correct_t -> wrong_{t+1}
    gd = qids[cmask]
    B = {"_support": "correct_t", "degradation_prev": float(yd.mean()), "n": int(cmask.sum())}
    # full-HS conditional degradation probe (LR on HS, correct_t support)
    B["hs_full"] = boot(yd, cv_oof(HS[cmask], yd, gd), gd, rng)
    # confidence-only: fragility = P(wrong_t) restricted to correct_t, as sole predictor
    B["confidence_only"] = boot(yd, p_wrong[cmask], gd, rng)
    # sanity: correctness-direction probe quality
    B["correctness_direction_auroc"] = boot(y_wrong, p_wrong, qids, rng)
    for name in ["hs_full", "confidence_only"]:
        m = B[name]
        print(f"  {name:<18} AUROC={m['auroc']:.3f}[{m['auroc_lo']:.3f},{m['auroc_hi']:.3f}] "
              f"AUPRC={m['auprc']:.3f}  n={m['n']} pos={m['n_pos']} ({m['prev']*100:.1f}%)")
    B["delta_hsfull_minus_confidence"] = B["hs_full"]["auroc"] - B["confidence_only"]["auroc"]
    print(f"  Delta(HS_full - confidence_only) = {B['delta_hsfull_minus_confidence']:+.3f}  "
          f"(corr-dir probe AUROC={B['correctness_direction_auroc']['auroc']:.3f})")
    res["control_B_degradation_leakage"] = B

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
