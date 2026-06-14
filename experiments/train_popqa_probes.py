"""
Train correctness probe + delta probe on PopQA hidden states.
Evaluate routing (CWA vs fixed-stage) with bootstrap CIs.

Uses pre-extracted hidden states from collect_popqa_v3.py.
Runs on CPU (MLP training is fast with 14336-dim features).
"""
import json, os, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

POPQA_DATA = "/workspace/popqa_v3_full/popqa_states.jsonl"
STAGE_COSTS = [0.25, 0.50, 0.75, 1.08]  # FULL model
STAGE_SIZES = [150, 400, 800, 1500]
N_BOOTSTRAP = 5000
N_FOLDS = 5
SEEDS = [42, 123, 456]

def load_data(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def organize(data):
    """Organize by query_id, sort by stage."""
    by_qid = {}
    for d in data:
        qid = d["query_id"]
        by_qid.setdefault(qid, []).append(d)
    for qid in by_qid:
        by_qid[qid].sort(key=lambda x: x["stage"])
    return by_qid

def compute_delta_labels(by_qid):
    """Compute benefit and degradation labels from stage transitions."""
    all_features = []
    all_benefit = []
    all_degradation = []
    all_correctness = []
    all_stages = []
    all_qids = []

    for qid, tuples in by_qid.items():
        for t in range(len(tuples) - 1):
            cur = tuples[t]
            nxt = tuples[t + 1]
            cur_correct = cur["correct"]
            nxt_correct = nxt["correct"]

            # Benefit: wrong -> correct
            benefit = int(cur_correct == 0 and nxt_correct == 1)
            # Degradation: correct -> wrong
            degradation = int(cur_correct == 1 and nxt_correct == 0)
            # Current correctness
            correctness = cur_correct

            # Feature: hidden state (can be large)
            if "hs_concat" in cur and isinstance(cur["hs_concat"], list):
                feat = np.array(cur["hs_concat"], dtype=np.float32)
            elif "hs_dim" in cur and cur.get("hs_dim", 0) > 0:
                # Need to find the actual vector
                feat = None
                continue
            else:
                continue

            all_features.append(feat)
            all_benefit.append(benefit)
            all_degradation.append(degradation)
            all_correctness.append(correctness)
            all_stages.append(t)
            all_qids.append(qid)

    return (np.array(all_features), np.array(all_benefit),
            np.array(all_degradation), np.array(all_correctness),
            np.array(all_stages), all_qids)

def train_eval(X, y, n_folds=5, seed=42, n_bootstrap=5000):
    """Cross-validated training + bootstrap CIs."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    all_probs = np.zeros(len(y))
    all_y = np.zeros(len(y))

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        # Logistic regression for clean statistical testing
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X_train_s, y_train)
        probs = clf.predict_proba(X_val_s)[:, 1]

        all_probs[val_idx] = probs
        all_y[val_idx] = y_val

    auroc = roc_auc_score(all_y, all_probs)
    auprc = average_precision_score(all_y, all_probs)

    # Bootstrap CIs
    np.random.seed(seed)
    aurocs, auprcs = [], []
    n = len(all_y)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        yb, pb = all_y[idx], all_probs[idx]
        try:
            aurocs.append(roc_auc_score(yb, pb))
            auprcs.append(average_precision_score(yb, pb))
        except ValueError:
            pass

    auroc_ci = (np.percentile(aurocs, 2.5), np.percentile(aurocs, 97.5))
    auprc_ci = (np.percentile(auprcs, 2.5), np.percentile(auprcs, 97.5))
    pos_rate = y.mean()

    return {"auroc": auroc, "auprc": auprc,
            "auroc_ci": auroc_ci, "auprc_ci": auprc_ci,
            "pos_rate": pos_rate, "n_samples": len(y)}

def evaluate_routing(by_qid, scores_dict, policy_name):
    """Evaluate CWA for a routing policy."""
    accs, costs = [], []
    for qid, tuples in by_qid.items():
        scores = scores_dict.get(qid, [0.5] * len(tuples))
        stopped = False
        for s, t in enumerate(tuples):
            if s == 3 or (scores[s] >= 0.5):
                accs.append(t["correct"])
                costs.append(STAGE_COSTS[s])
                stopped = True
                break
        if not stopped:
            # Full pipeline
            t = tuples[-1]
            accs.append(t["correct"])
            costs.append(STAGE_COSTS[-1])

    acc = np.mean(accs)
    cost = np.mean(costs)
    cwa = acc - 0.5 * cost / max(STAGE_COSTS)
    return {"policy": policy_name, "accuracy": acc, "cost": cost, "cwa_0.5": cwa}

def main():
    print("Loading PopQA data...")
    data = load_data(POPQA_DATA)
    print(f"  {len(data)} tuples, {len(set(d['query_id'] for d in data))} queries")

    by_qid = organize(data)

    # Stage accuracies (sanity check)
    print("\n=== Stage Accuracies ===")
    for s in range(4):
        rs = [r for qid, tuples in by_qid.items() for r in tuples if r["stage"] == s]
        acc = np.mean([r["correct"] for r in rs])
        print(f"  S{s}: {acc:.3f}")

    # Prepare features + labels
    X, y_benefit, y_degradation, y_correctness, stages, qids = compute_delta_labels(by_qid)
    print(f"\nFeature matrix: {X.shape}")
    print(f"  Benefit rate: {y_benefit.mean():.3f} ({y_benefit.sum():.0f} events)")
    print(f"  Degradation rate: {y_degradation.mean():.3f} ({y_degradation.sum():.0f} events)")
    print(f"  Correctness rate: {y_correctness.mean():.3f}")

    # Train probes
    results = {}

    print("\n=== Correctness Probe ===")
    r = train_eval(X, y_correctness)
    results["correctness"] = r
    print(f"  AUROC: {r['auroc']:.4f} [{r['auroc_ci'][0]:.4f}, {r['auroc_ci'][1]:.4f}]")
    print(f"  AUPRC: {r['auprc']:.4f} [{r['auprc_ci'][0]:.4f}, {r['auprc_ci'][1]:.4f}]")
    print(f"  Pos rate: {r['pos_rate']:.3f}")

    print("\n=== Benefit Probe (Delta+) ===")
    r = train_eval(X, y_benefit)
    results["benefit"] = r
    print(f"  AUROC: {r['auroc']:.4f} [{r['auroc_ci'][0]:.4f}, {r['auroc_ci'][1]:.4f}]")
    print(f"  AUPRC: {r['auprc']:.4f} [{r['auprc_ci'][0]:.4f}, {r['auprc_ci'][1]:.4f}]")
    print(f"  Pos rate: {r['pos_rate']:.3f}")

    print("\n=== Degradation Probe (Delta-) ===")
    r = train_eval(X, y_degradation)
    results["degradation"] = r
    print(f"  AUROC: {r['auroc']:.4f} [{r['auroc_ci'][0]:.4f}, {r['auroc_ci'][1]:.4f}]")
    print(f"  AUPRC: {r['auprc']:.4f} [{r['auprc_ci'][0]:.4f}, {r['auprc_ci'][1]:.4f}]")
    print(f"  Pos rate: {r['pos_rate']:.3f}")

    # Asymmetry check
    asym = results["degradation"]["auroc"] - results["benefit"]["auroc"]
    print(f"\n  Degradation - Benefit AUROC delta: {asym:+.4f}")

    # Routing evaluation
    print("\n=== Routing Evaluation ===")
    routing = []

    # Fixed-stage
    for s in range(4):
        rs = [tuples[min(s, len(tuples)-1)] for tuples in by_qid.values()]
        acc = np.mean([r["correct"] for r in rs])
        cost = STAGE_COSTS[s]
        cwa = acc - 0.5 * cost / max(STAGE_COSTS)
        routing.append({"policy": f"fixed_S{s}", "accuracy": acc, "cost": cost, "cwa_0.5": cwa})

    # Full pipeline
    rs = [tuples[-1] for tuples in by_qid.values()]
    acc = np.mean([r["correct"] for r in rs])
    cost = STAGE_COSTS[-1]
    cwa = acc - 0.5 * cost / max(STAGE_COSTS)
    routing.append({"policy": "full_pipeline", "accuracy": acc, "cost": cost, "cwa_0.5": cwa})

    # Oracle (best possible per-query)
    accs, costs_ = [], []
    for qid, tuples in by_qid.items():
        best = max(tuples, key=lambda t: t["correct"])
        accs.append(best["correct"])
        costs_.append(STAGE_COSTS[best["stage"]])
    acc = np.mean(accs)
    cost = np.mean(costs_)
    cwa = acc - 0.5 * cost / max(STAGE_COSTS)
    routing.append({"policy": "oracle", "accuracy": acc, "cost": cost, "cwa_0.5": cwa})

    print(f"{'Policy':<20} {'Accuracy':>8} {'Cost':>8} {'CWA(0.5)':>8}")
    for r in routing:
        print(f"{r['policy']:<20} {r['accuracy']:8.4f} {r['cost']:8.3f} {r['cwa_0.5']:8.4f}")

    # Save
    out = {"probes": {}, "routing": routing}
    for k, v in results.items():
        out["probes"][k] = {kk: (vv.tolist() if hasattr(vv, 'tolist') else vv)
                           for kk, vv in v.items()}

    save_path = "/workspace/popqa_v3_full/probe_results.json"
    with open(save_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
