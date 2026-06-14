"""
Compute cumulative CWA for PopQA routing from per-query stop stages.

The original eval_routing_popqa.py computed per-query stop stages but only
saved aggregate means with per-stage (non-cumulative) cost accounting.
This script re-runs the routing policies and produces BOTH:
  - Per-stage CWA (denominator max=1.08, original)
  - Cumulative CWA (denominator sum=2.58, correct sequential semantics)

The bimodality finding is cost-agnostic: threshold routing fails regardless.
Only "which static baseline wins" and "oracle headroom" are cost-dependent.
"""
import json, gzip, sys, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

STAGE_COSTS = np.array([0.25, 0.50, 0.75, 1.08])
MAX_PER_STAGE = max(STAGE_COSTS)        # 1.08 — primary denominator
MAX_CUMULATIVE = STAGE_COSTS.sum()      # 2.58 — appendix sensitivity

def load_data(path):
    """Load PopQA state JSONL (gzipped or plain)."""
    if path.endswith('.gz'):
        f = gzip.open(path, 'rt')
    else:
        f = open(path)
    data = [json.loads(line) for line in f]
    f.close()

    by_qid = {}
    for d in data:
        qid = d.get("query_id", d.get("qid"))
        by_qid.setdefault(qid, []).append(d)
    for qid in by_qid:
        by_qid[qid].sort(key=lambda x: x.get("stage", x.get("stage_idx", 0)))
    return data, by_qid

def train_probe(by_qid):
    """Train LR correctness probe on PopQA hidden states."""
    Xl, yl = [], []
    for qid, tups in by_qid.items():
        for t in tups:
            hs = t.get("hs_concat") or t.get("multi_layer_hidden_state") or t.get("hidden_state")
            if hs is None or (isinstance(hs, list) and len(hs) == 0):
                continue
            Xl.append(np.array(hs, dtype=np.float32).flatten())
            yl.append(t.get("correct", t.get("stage_correctness", 0)))
    X, y = np.array(Xl), np.array(yl)
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xs, y)
    def predict(hs):
        hs_f = np.array(hs, dtype=np.float32).flatten().reshape(1, -1)
        return clf.predict_proba(sc.transform(hs_f))[0, 1]
    return predict

def self_report_confidence(answer):
    uncertainty = ["unknown", "not sure", "cannot", "don't know", "unclear",
                   "not mentioned", "no information", "not provided", "not stated"]
    return 0.15 if any(m in str(answer).lower() for m in uncertainty) else 0.85

def eval_policy_with_stages(by_qid, policy_fn, n_queries):
    """Evaluate policy, returning per-query stop stages and both cost models."""
    per_query_stops = []  # list of (stop_stage, is_correct)
    for qid, tups in by_qid.items():
        stopped = False
        for s, t in enumerate(tups):
            score = policy_fn(t, s, tups)
            if s == 3 or score >= 0.5:
                per_query_stops.append((s, t.get("correct", t.get("stage_correctness", 0))))
                stopped = True
                break
        if not stopped:
            t = tups[-1]
            per_query_stops.append((3, t.get("correct", t.get("stage_correctness", 0))))

    stops = np.array([s for s, _ in per_query_stops])
    corrects = np.array([c for _, c in per_query_stops])

    acc = corrects.mean()

    # Per-stage costs (original)
    per_stage_costs = STAGE_COSTS[stops]
    cost_ps = per_stage_costs.mean()
    cwa_ps = acc - 0.5 * cost_ps / MAX_PER_STAGE

    # Cumulative costs (proper sequential)
    cumulative_costs = np.array([STAGE_COSTS[:s+1].sum() for s in stops])
    cost_cum = cumulative_costs.mean()
    cwa_cum = acc - 0.5 * cost_cum / MAX_CUMULATIVE

    return {
        "accuracy": acc, "n": len(stops),
        "per_stage": {"cost": cost_ps, "cwa_0.5": cwa_ps, "denom": MAX_PER_STAGE},
        "cumulative": {"cost": cost_cum, "cwa_0.5": cwa_cum, "denom": MAX_CUMULATIVE},
        "stop_distribution": {int(s): int((stops == s).sum()) for s in range(4)},
    }

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/popqa/popqa_v4_500q_states.jsonl.gz"
    print(f"Loading: {path}")
    data, by_qid = load_data(path)
    n_q = len(by_qid)
    print(f"  {n_q} queries, {len(data)} tuples")

    # Train probe
    print("Training correctness probe (LR)...")
    probe_fn = train_probe(by_qid)

    results = {}

    # Fixed-stage baselines
    for s in range(4):
        labels = [tups[min(s, len(tups)-1)].get("correct", tups[min(s, len(tups)-1)].get("stage_correctness", 0))
                  for tups in by_qid.values()]
        acc = np.mean(labels)
        ps_cost = STAGE_COSTS[s]
        cum_cost = STAGE_COSTS[:s+1].sum()
        results[f"fixed_S{s}"] = {
            "accuracy": acc, "n": n_q,
            "per_stage": {"cost": ps_cost, "cwa_0.5": acc - 0.5 * ps_cost / MAX_PER_STAGE, "denom": MAX_PER_STAGE},
            "cumulative": {"cost": cum_cost, "cwa_0.5": acc - 0.5 * cum_cost / MAX_CUMULATIVE, "denom": MAX_CUMULATIVE},
            "stop_distribution": {s: n_q},
        }

    # Full pipeline
    labels = [tups[-1].get("correct", tups[-1].get("stage_correctness", 0)) for tups in by_qid.values()]
    acc = np.mean(labels)
    results["full_pipeline"] = {
        "accuracy": acc, "n": n_q,
        "per_stage": {"cost": STAGE_COSTS[3], "cwa_0.5": acc - 0.5 * STAGE_COSTS[3] / MAX_PER_STAGE, "denom": MAX_PER_STAGE},
        "cumulative": {"cost": MAX_CUMULATIVE, "cwa_0.5": acc - 0.5 * MAX_CUMULATIVE / MAX_CUMULATIVE, "denom": MAX_CUMULATIVE},
        "stop_distribution": {3: n_q},
    }

    # Oracle
    stops, corrects_o = [], []
    for qid, tups in by_qid.items():
        best = max(tups, key=lambda t: t.get("correct", t.get("stage_correctness", 0)))
        stops.append(best.get("stage", best.get("stage_idx", 0)))
        corrects_o.append(best.get("correct", best.get("stage_correctness", 0)))
    stops = np.array(stops)
    corrects_o = np.array(corrects_o)
    acc_o = corrects_o.mean()
    ps_c = STAGE_COSTS[stops].mean()
    cum_c = np.array([STAGE_COSTS[:s+1].sum() for s in stops]).mean()
    results["oracle"] = {
        "accuracy": acc_o, "n": n_q,
        "per_stage": {"cost": ps_c, "cwa_0.5": acc_o - 0.5 * ps_c / MAX_PER_STAGE, "denom": MAX_PER_STAGE},
        "cumulative": {"cost": cum_c, "cwa_0.5": acc_o - 0.5 * cum_c / MAX_CUMULATIVE, "denom": MAX_CUMULATIVE},
        "stop_distribution": {int(s): int((stops == s).sum()) for s in range(4)},
    }

    # Probe routing
    for tau in [0.3, 0.5, 0.7]:
        def make_fn(t):
            return lambda tu, s, tups: probe_fn(tu.get("hs_concat") or tu.get("multi_layer_hidden_state") or tu.get("hidden_state"))
        results[f"probe_tau{tau}"] = eval_policy_with_stages(by_qid, make_fn(tau), n_q)

    # Self-report routing
    for tau in [0.3, 0.5]:
        results[f"self_report_tau{tau}"] = eval_policy_with_stages(
            by_qid, lambda t, s, tups: self_report_confidence(t.get("answer", "")), n_q)

    # Random baseline
    np.random.seed(42)
    all_random_cwa_ps, all_random_cwa_cum = [], []
    for _ in range(100):
        stops_r = np.random.randint(0, 4, n_q)
        corrects_r = np.array([list(by_qid.values())[i][min(stops_r[i], 3)].get("correct", 0)
                               for i in range(n_q)])
        acc_r = corrects_r.mean()
        ps_c_r = STAGE_COSTS[stops_r].mean()
        cum_c_r = np.array([STAGE_COSTS[:s+1].sum() for s in stops_r]).mean()
        all_random_cwa_ps.append(acc_r - 0.5 * ps_c_r / MAX_PER_STAGE)
        all_random_cwa_cum.append(acc_r - 0.5 * cum_c_r / MAX_CUMULATIVE)
    results["random"] = {
        "accuracy": None, "n": n_q,
        "per_stage": {"cwa_0.5_mean": np.mean(all_random_cwa_ps), "cwa_0.5_std": np.std(all_random_cwa_ps)},
        "cumulative": {"cwa_0.5_mean": np.mean(all_random_cwa_cum), "cwa_0.5_std": np.std(all_random_cwa_cum)},
    }

    # --- OUTPUT ---
    print(f"\n{'='*70}")
    print(f"PopQA Routing — Per-Stage vs Cumulative CWA")
    print(f"{'Policy':<22} {'Acc':>6} {'CWA_ps':>8} {'CWA_cum':>8} {'Δ':>8}")
    print("-" * 70)
    for name, r in results.items():
        if name == "random":
            cwa_ps = r["per_stage"]["cwa_0.5_mean"]
            cwa_cum = r["cumulative"]["cwa_0.5_mean"]
            print(f"{name:<22} {'--':>6} {cwa_ps:>8.4f} {cwa_cum:>8.4f} {'--':>8}")
        else:
            cwa_ps = r["per_stage"]["cwa_0.5"]
            cwa_cum = r["cumulative"]["cwa_0.5"]
            delta = cwa_cum - cwa_ps
            print(f"{name:<22} {r['accuracy']:>6.4f} {cwa_ps:>8.4f} {cwa_cum:>8.4f} {delta:>+8.4f}")

    # Best static under each model
    static_names = ["fixed_S0", "fixed_S1", "fixed_S2", "fixed_S3"]
    best_ps = max(static_names, key=lambda n: results[n]["per_stage"]["cwa_0.5"])
    best_cum = max(static_names, key=lambda n: results[n]["cumulative"]["cwa_0.5"])
    print(f"\nBest static (per-stage): {best_ps} CWA={results[best_ps]['per_stage']['cwa_0.5']:.4f}")
    print(f"Best static (cumulative): {best_cum} CWA={results[best_cum]['cumulative']['cwa_0.5']:.4f}")

    # Does any policy beat the best static?
    print(f"\nKey question: Does any policy beat best static baseline?")
    for name, r in results.items():
        if name in static_names or name in ("random", "full_pipeline"):
            continue
        beats_ps = r["per_stage"]["cwa_0.5"] > results[best_ps]["per_stage"]["cwa_0.5"]
        beats_cum = r["cumulative"]["cwa_0.5"] > results[best_cum]["cumulative"]["cwa_0.5"]
        print(f"  {name}: >best_static? per-stage={beats_ps}, cumulative={beats_cum}")

    # Save
    out_path = "results/popqa/cumulative_cwa_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "stop_distribution"}
                   for k, v in results.items()}, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
