"""
Oracle-relevance on the CONDITIONAL benefit support (Loop 27 fix).

Round-1 reviewer (gemini-3.1-pro-preview): the paper's oracle-relevance +0.10
claim uses the JOINT benefit label, which the paper itself debunks as inflated by
correctness leakage. Rerun on the routing-relevant CONDITIONAL support:
  among currently-WRONG transitions (wrong_t), does adding oracle retriever-side
  relevance features predict whether the NEXT stage fixes the answer
  (P(correct_{t+1} | wrong_t))?  This is the routing-saving question.

We report HS-only, oracle-only, and HS+oracle for BOTH the joint benefit (all
transitions) and the conditional benefit (wrong_t only), under a matched protocol:
5-fold GroupKFold by query, OOF predictions, 5000 query-grouped bootstraps, LR
(class-weight balanced). Reuses the oracle feature computation from
oracle_relevance_experiment.py. Run CPU-only (CUDA_VISIBLE_DEVICES="") so it does
not contend with a GPU collection job.

  python oracle_conditional_benefit.py --hs_data <jsonl> --num_queries 2000 \
      --out results/oracle_conditional_benefit.json
"""
import argparse, json, os, sys, numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oracle_relevance_experiment as ore  # reuse load/rank/feature functions

N_BOOT = 5000


def build(hs_data, oracle_tuples):
    """Join HS + oracle features per transition; keep cur/next correctness."""
    oracle_by_key = {(str(o["query_id"]), int(o["stage_idx"])): o for o in oracle_tuples}
    rows = []
    for (qid, sid), entries in hs_data.items():
        if sid >= ore.NUM_STAGES - 1 or sid < 0:
            continue
        cur = entries[0]
        nxts = hs_data.get((qid, sid + 1), [])
        if not nxts:
            continue
        nxt = nxts[0]
        key = (str(qid), int(sid))
        if key not in oracle_by_key:
            continue
        hs = np.array(cur.get("multi_layer_hidden_states", cur.get("hidden_state", [])),
                      dtype=np.float32).flatten()
        if hs.size == 0:
            continue
        rows.append({
            "qid": qid, "sid": sid,
            "hs": hs,
            "oracle": np.asarray(oracle_by_key[key]["oracle_features"], dtype=np.float32),
            "cur_correct": int(cur.get("stage_correctness", 0)),
            "next_correct": int(nxt.get("stage_correctness", 0)),
        })
    return rows


def cv_oof(X, y, groups):
    """5-fold GroupKFold LR, return OOF probabilities."""
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, y, groups=groups):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        lr = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                                random_state=42)
        lr.fit(Xtr, y[tr])
        oof[te] = lr.predict_proba(Xte)[:, 1]
    return oof


def boot(y, score, groups, rng):
    uq = list(set(groups)); nq = len(uq)
    q2i = defaultdict(list)
    for i, q in enumerate(groups):
        q2i[q].append(i)
    aus, aps = [], []
    for _ in range(N_BOOT):
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
            "auprc": float(p.mean()), "auprc_lo": float(np.percentile(p, 2.5)),
            "auprc_hi": float(np.percentile(p, 97.5)),
            "n": int(len(y)), "n_pos": int(y.sum()), "prev": float(y.mean())}


def eval_support(rows, support, rng):
    """support: 'joint' (all) or 'wrong_t' (conditional benefit)."""
    if support == "wrong_t":
        rows = [r for r in rows if r["cur_correct"] == 0]
    y = np.array([r["next_correct"] if support == "wrong_t"
                  else int(r["cur_correct"] == 0 and r["next_correct"] == 1)
                  for r in rows])
    groups = np.array([r["qid"] for r in rows])
    Xhs = np.array([r["hs"] for r in rows])
    Xor = np.array([r["oracle"] for r in rows])
    Xcb = np.concatenate([Xhs, Xor], axis=1)
    out = {}
    for name, X in [("hs", Xhs), ("oracle", Xor), ("hs_plus_oracle", Xcb)]:
        oof = cv_oof(X, y, groups)
        out[name] = boot(y, oof, groups, rng)
    out["_support"] = support
    out["delta_auroc_combined_minus_hs"] = out["hs_plus_oracle"]["auroc"] - out["hs"]["auroc"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hs_data", default="data/collected_states_hotpotqa_v5_2000.jsonl")
    ap.add_argument("--num_queries", type=int, default=2000)
    ap.add_argument("--out", default="results/oracle_conditional_benefit.json")
    args = ap.parse_args()

    print("[1/4] Loading HotpotQA raw (gold facts)...")
    queries = ore.load_hotpotqa_raw(args.num_queries)
    print("[2/4] Cross-encoder ranking (CPU)...")
    queries = ore.rank_passages(queries)
    oracle_tuples = ore.compute_oracle_features(queries)
    print(f"  {len(oracle_tuples)} oracle transition tuples")
    print("[3/4] Loading collected hidden states...")
    hs_data = ore.load_collected_states(args.hs_data)
    rows = build(hs_data, oracle_tuples)
    print(f"  joined transitions: {len(rows)}")

    rng = np.random.RandomState(42)
    print("[4/4] Evaluating joint + conditional supports...")
    res = {"_meta": {"hs_data": args.hs_data, "n_transitions": len(rows),
                     "oracle_dim": int(len(rows[0]["oracle"])) if rows else 0,
                     "hs_dim": int(len(rows[0]["hs"])) if rows else 0}}
    res["joint_benefit"] = eval_support(rows, "joint", rng)
    res["conditional_benefit_wrong_t"] = eval_support(rows, "wrong_t", rng)

    print("\n" + "=" * 70)
    print("ORACLE RELEVANCE — JOINT vs CONDITIONAL BENEFIT (LR, HotpotQA V5)")
    print("=" * 70)
    for sup in ["joint_benefit", "conditional_benefit_wrong_t"]:
        r = res[sup]
        print(f"\n[{sup}]  n={r['hs']['n']} pos={r['hs']['n_pos']} ({r['hs']['prev']*100:.1f}%)")
        for name in ["hs", "oracle", "hs_plus_oracle"]:
            m = r[name]
            print(f"  {name:<16} AUROC={m['auroc']:.3f}[{m['auroc_lo']:.3f},{m['auroc_hi']:.3f}] "
                  f"AUPRC={m['auprc']:.3f}[{m['auprc_lo']:.3f},{m['auprc_hi']:.3f}]")
        print(f"  Δ(HS+oracle − HS) AUROC = {r['delta_auroc_combined_minus_hs']:+.3f}")

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
