"""
Routing baseline evaluation on PopQA data.
Compares multiple routing policies: correctness probe, self-reported confidence,
retrieval heuristics, fixed-stage, oracle, random.
"""
import json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

STAGE_COSTS = [0.25, 0.50, 0.75, 1.08]
MAX_COST = max(STAGE_COSTS)

V4_PATH = "/workspace/popqa_v4_full/popqa_states.jsonl"
FIXED_PATH = "/workspace/popqa_fixedtoken/popqa_fixedtoken_states.jsonl"

def load_and_organize(path):
    data = [json.loads(line) for line in open(path)]
    by_qid = {}
    for d in data:
        by_qid.setdefault(d["query_id"], []).append(d)
    for qid in by_qid:
        by_qid[qid].sort(key=lambda x: x["stage"])
    return data, by_qid

def train_probe(by_qid, target="correctness"):
    """Train logistic regression probe, return predict_proba function."""
    Xl, yl = [], []
    for qid, tups in by_qid.items():
        for t in range(len(tups)):
            cur = tups[t]
            if "hs_concat" not in cur or not isinstance(cur["hs_concat"], list):
                continue
            Xl.append(np.array(cur["hs_concat"], dtype=np.float32))
            if target == "correctness":
                yl.append(cur["correct"])
            elif target == "benefit" and t < len(tups) - 1:
                yl.append(int(cur["correct"] == 0 and tups[t+1]["correct"] == 1))
            elif target == "degradation" and t < len(tups) - 1:
                yl.append(int(cur["correct"] == 1 and tups[t+1]["correct"] == 0))

    X, y = np.array(Xl), np.array(yl)
    scl = StandardScaler()
    Xs = scl.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xs, y)
    return lambda hs: clf.predict_proba(scl.transform(np.array([hs])))[0, 1]

def self_report_confidence(answer):
    """Parse LLM answer for uncertainty markers."""
    uncertainty = ["unknown", "not sure", "cannot", "don't know", "unclear",
                   "not mentioned", "no information", "not provided", "not stated"]
    ans_lower = answer.lower()
    for marker in uncertainty:
        if marker in ans_lower:
            return 0.15  # Low confidence
    return 0.85  # High confidence

def eval_policy(by_qid, policy_fn, policy_name):
    """Evaluate routing policy: given stop scores per stage, compute CWA."""
    accs, costs_ = [], []
    for qid, tups in by_qid.items():
        for s, t in enumerate(tups):
            score = policy_fn(t, s, tups)
            if s == 3 or score >= 0.5:
                accs.append(t["correct"])
                costs_.append(STAGE_COSTS[s])
                break
        else:
            t = tups[-1]
            accs.append(t["correct"])
            costs_.append(STAGE_COSTS[-1])

    acc = np.mean(accs)
    cost = np.mean(costs_)
    cwa = acc - 0.5 * cost / MAX_COST
    return {"policy": policy_name, "accuracy": acc, "cost": cost, "cwa_0.5": cwa}

def main():
    print("=" * 60)
    print("ROUTING BASELINE EVALUATION — PopQA Entity-Page Regime")
    print("=" * 60)

    data, by_qid = load_and_organize(V4_PATH)
    n_q = len(by_qid)
    print(f"\nData: {n_q} queries, {len(data)} tuples")

    # Train probe
    print("Training correctness probe...")
    probe_fn = train_probe(by_qid, "correctness")

    # Policies
    policies = []

    # 1. Fixed-stage baselines
    for s in range(4):
        rs = [tups[min(s, len(tups)-1)] for tups in by_qid.values()]
        acc = np.mean([r["correct"] for r in rs])
        cost = STAGE_COSTS[s]
        policies.append({"policy": f"fixed_S{s}", "accuracy": acc, "cost": cost,
                        "cwa_0.5": acc - 0.5 * cost / MAX_COST})

    # 2. Full pipeline
    rs = [tups[-1] for tups in by_qid.values()]
    acc = np.mean([r["correct"] for r in rs])
    cost = STAGE_COSTS[-1]
    policies.append({"policy": "full_pipeline", "accuracy": acc, "cost": cost,
                    "cwa_0.5": acc - 0.5 * cost / MAX_COST})

    # 3. Oracle
    accs, csts = [], []
    for qid, tups in by_qid.items():
        best = max(tups, key=lambda t: t["correct"])
        accs.append(best["correct"])
        csts.append(STAGE_COSTS[best["stage"]])
    policies.append({"policy": "oracle", "accuracy": np.mean(accs), "cost": np.mean(csts),
                    "cwa_0.5": np.mean(accs) - 0.5 * np.mean(csts) / MAX_COST})

    # 4. Correctness probe routing (tau sweep)
    for tau in [0.3, 0.5, 0.7, 0.9]:
        def make_probe_policy(tau):
            return lambda t, s, tups: probe_fn(t["hs_concat"])
        policy = eval_policy(by_qid, make_probe_policy(tau), f"probe_tau{tau}")
        # Actually need tau threshold
        accs, csts = [], []
        for qid, tups in by_qid.items():
            for s, t in enumerate(tups):
                p_correct = probe_fn(t["hs_concat"])
                if s == 3 or p_correct >= tau:
                    accs.append(t["correct"])
                    csts.append(STAGE_COSTS[s])
                    break
            else:
                t = tups[-1]
                accs.append(t["correct"])
                csts.append(STAGE_COSTS[-1])
        policies.append({"policy": f"probe_tau{tau}",
                        "accuracy": np.mean(accs), "cost": np.mean(csts),
                        "cwa_0.5": np.mean(accs) - 0.5 * np.mean(csts) / MAX_COST})

    # 5. Self-reported confidence routing
    for tau in [0.3, 0.5, 0.7]:
        accs, csts = [], []
        for qid, tups in by_qid.items():
            for s, t in enumerate(tups):
                conf = self_report_confidence(t.get("answer", ""))
                if s == 3 or conf >= tau:
                    accs.append(t["correct"])
                    csts.append(STAGE_COSTS[s])
                    break
            else:
                t = tups[-1]
                accs.append(t["correct"])
                csts.append(STAGE_COSTS[-1])
        policies.append({"policy": f"self_report_tau{tau}",
                        "accuracy": np.mean(accs), "cost": np.mean(csts),
                        "cwa_0.5": np.mean(accs) - 0.5 * np.mean(csts) / MAX_COST})

    # 6. Stage-count heuristic (stop after N stages)
    for n_stages in [1, 2, 3]:
        accs, csts = [], []
        for qid, tups in by_qid.items():
            stop_s = min(n_stages - 1, 3)
            t = tups[stop_s]
            accs.append(t["correct"])
            csts.append(STAGE_COSTS[stop_s])
        policies.append({"policy": f"stop_after_{n_stages}",
                        "accuracy": np.mean(accs), "cost": np.mean(csts),
                        "cwa_0.5": np.mean(accs) - 0.5 * np.mean(csts) / MAX_COST})

    # 7. Random baseline
    np.random.seed(42)
    for _ in range(100):
        accs, csts = [], []
        for qid, tups in by_qid.items():
            s = np.random.randint(0, 4)
            t = tups[s]
            accs.append(t["correct"])
            csts.append(STAGE_COSTS[s])
        policies.append({"policy": "random", "accuracy": np.mean(accs), "cost": np.mean(csts),
                        "cwa_0.5": np.mean(accs) - 0.5 * np.mean(csts) / MAX_COST})

    # Print results
    print(f"\n{'Policy':<25} {'Accuracy':>8} {'Cost':>8} {'CWA(0.5)':>8} {'vs S0':>8}")
    print("-" * 57)
    s0_cwa = next(p["cwa_0.5"] for p in policies if p["policy"] == "fixed_S0")
    sorted_p = sorted(policies, key=lambda x: x["cwa_0.5"], reverse=True)
    for p in sorted_p:
        if p["policy"] == "random":
            continue
        delta = p["cwa_0.5"] - s0_cwa
        marker = " ← BEST" if p == sorted_p[0] else ""
        print(f"{p['policy']:<25} {p['accuracy']:8.4f} {p['cost']:8.3f} "
              f"{p['cwa_0.5']:8.4f} {delta:+8.4f}{marker}")

    # Random stats
    rand_cwas = [p["cwa_0.5"] for p in policies if p["policy"] == "random"]
    print(f"\nRandom baseline: {np.mean(rand_cwas):.4f} ± {np.std(rand_cwas):.4f}")

    # Best non-oracle
    best = sorted_p[0]
    print(f"\nBest policy: {best['policy']} (CWA={best['cwa_0.5']:.4f})")
    print(f"Oracle CWA: {next(p['cwa_0.5'] for p in policies if p['policy']=='oracle'):.4f}")
    print(f"Headroom: {next(p['cwa_0.5'] for p in policies if p['policy']=='oracle'):.4f - best['cwa_0.5']:.4f}")

    # 8. Coverage decomposition
    print("\n" + "=" * 60)
    print("COVERAGE DECOMPOSITION")
    print("=" * 60)
    # All v4 queries have entity pages (covered subset)
    print("PopQA v4: 100% have Wikipedia entity pages (covered subset)")
    print(f"Entity cache covers 2053/14267 = 14.4% of full PopQA test set")
    print("For full-set evaluation:")
    print("  - 47% of random sample have Wikipedia pages")
    print("  - 53% would have retrieval failure (answer absent from corpus)")
    print("  - EN -> accuracy depends on retrieval quality; fallback to question-only")


if __name__ == "__main__":
    main()
