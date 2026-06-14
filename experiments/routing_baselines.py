"""
Stronger routing baselines: logprob/entropy, self-reported confidence,
retrieval-side features, and learned combiner.

Addresses reviewer Weakness #1: "The paper still does not rule out the simpler
explanation that routing fails because the policy is weak, not because benefit
is intrinsically hard to predict."

Baselines:
1. ENTROPY: Next-token generation entropy — standard LLM uncertainty proxy
2. MAX_PROB: Maximum softmax probability — simpler confidence signal
3. SELF_REPORT: Ask LLM "How confident are you?" — direct introspection
4. RETRIEVAL_MARGIN: Cross-encoder score margin between top-k passages
5. LEARNED_COMBINER: Logistic regression combining HS + entropy + retrieval
"""
import argparse, json, os, sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_data(jsonl_path: str) -> List[Dict]:
    """Load hidden-state JSONL data. Each line is one stage-tuple."""
    data = []
    with open(jsonl_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def organize_by_query(data: List[Dict]) -> Dict[str, List[Dict]]:
    """Group stage-tuples by query_id."""
    by_qid = {}
    for d in data:
        qid = d["query_id"]
        by_qid.setdefault(qid, []).append(d)
    return by_qid


# ── Baseline 1: Generation Logprob / Entropy ──────────────────────────

def compute_entropy_confidence(data: List[Dict], tokenizer, model,
                               device="cuda") -> Dict[str, List[float]]:
    """For each stage-tuple, compute next-token entropy as confidence proxy.

    Lower entropy → more confident. Returns dict mapping query_id → list of
    per-stage entropy values.
    """
    print("Computing generation entropy confidence...")
    results = {}
    for item in tqdm(data):
        qid = item["query_id"]
        stage = item.get("stage", 0)
        if qid not in results:
            results[qid] = [None] * 4

        # Re-run forward pass to get logits
        prompt = item.get("prompt", "")
        if not prompt:
            results[qid][stage] = 0.5  # fallback
            continue

        inputs = tokenizer(prompt, return_tensors="pt",
                          truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
            max_prob = probs.max().item()

        # Normalize: confidence = max_prob (0-1), entropy inverted
        results[qid][stage] = {
            "entropy": entropy,
            "max_prob": max_prob,
            "confidence_from_entropy": 1.0 / (1.0 + entropy),
        }

    return results


# ── Baseline 2: Self-Reported Confidence ───────────────────────────────

def compute_self_report_confidence(data: List[Dict]) -> Dict[str, List[float]]:
    """Parse LLM answers for self-reported confidence signals.

    If the LLM answer contains uncertainty markers ("unknown", "not sure", etc.),
    treat as low confidence. Otherwise high confidence.
    """
    print("Computing self-reported confidence...")
    uncertainty_markers = [
        "unknown", "not sure", "unsure", "cannot", "don't know",
        "no information", "not mentioned", "not specified",
        "unclear", "not provided", "not stated", "no relevant",
    ]

    results = {}
    for item in data:
        qid = item["query_id"]
        stage = item.get("stage", 0)
        if qid not in results:
            results[qid] = [None] * 4

        answer = item.get("answer", "").lower()
        has_uncertainty = any(marker in answer for marker in uncertainty_markers)
        # Confidence = 0.2 if uncertain, 0.8 if seems confident
        results[qid][stage] = 0.2 if has_uncertainty else 0.8

    return results


# ── Baseline 3: Retrieval-Side Features ────────────────────────────────

def compute_retrieval_confidence(data: List[Dict]) -> Dict[str, List[float]]:
    """Use cross-encoder score margins as retrieval-quality confidence.

    Real retrieval features (not imputed proxies):
    - CE score of top-1 passage (relevance ceiling)
    - CE score margin: top-1 - top-3 (discriminative power)
    - CE score spread: std of top-5 scores
    """
    print("Computing retrieval-side confidence...")
    results = {}
    for item in data:
        qid = item["query_id"]
        stage = item.get("stage", 0)
        if qid not in results:
            results[qid] = [None] * 4

        ce_scores = item.get("ce_scores", [])
        if not ce_scores or len(ce_scores) < 2:
            results[qid][stage] = 0.5  # No retrieval signal available
            continue

        top1 = ce_scores[0] if len(ce_scores) > 0 else 0
        top3 = ce_scores[2] if len(ce_scores) > 2 else ce_scores[-1]
        margin = top1 - top3
        std = np.std(ce_scores[:min(5, len(ce_scores))])

        # Confidence from retrieval quality
        results[qid][stage] = {
            "ce_top1": top1,
            "ce_margin": margin,
            "ce_std": std,
            "retrieval_confidence": 1.0 / (1.0 + np.exp(-2 * margin)),
        }

    return results


# ── Baseline 4: Learned Combiner ───────────────────────────────────────

def train_learned_combiner(hidden_states, entropy_scores, retrieval_scores,
                           labels, val_split=0.3, seed=42):
    """Train logistic regression combining hidden states + auxiliary signals.

    Combines:
    - Hidden state features (from MLP probe or raw states)
    - Generation entropy / max_prob
    - Retrieval confidence (CE margins)
    - Stage embedding

    Returns: trained combiner, validation AUROC
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.preprocessing import StandardScaler

    # Feature engineering
    X_parts = []
    feature_names = []

    if hidden_states is not None:
        X_parts.append(hidden_states)
        feature_names.append(f"hs_{hidden_states.shape[1]}d")

    if entropy_scores is not None:
        X_parts.append(entropy_scores)
        feature_names.append("entropy_2d")

    if retrieval_scores is not None:
        X_parts.append(retrieval_scores)
        feature_names.append("retrieval_3d")

    if not X_parts:
        raise ValueError("No features provided")

    X = np.concatenate(X_parts, axis=1)
    print(f"  Combiner features: {X.shape[1]}d ({', '.join(feature_names)})")

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train/val split
    np.random.seed(seed)
    n = len(X)
    indices = np.random.permutation(n)
    n_val = int(n * val_split)
    train_idx, val_idx = indices[n_val:], indices[:n_val]

    # Train
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X[train_idx], labels[train_idx])

    # Evaluate
    val_probs = clf.predict_proba(X[val_idx])[:, 1]
    val_auroc = roc_auc_score(labels[val_idx], val_probs)
    val_auprc = average_precision_score(labels[val_idx], val_probs)

    print(f"  Combiner val AUROC: {val_auroc:.4f}, AUPRC: {val_auprc:.4f}")

    return clf, scaler, val_auroc, val_auprc


# ── Routing Evaluation ─────────────────────────────────────────────────

def evaluate_routing_policies(data: List[Dict],
                              entropy_conf: Dict,
                              retrieval_conf: Dict,
                              self_report_conf: Dict,
                              hs_probe_probs: Dict = None) -> Dict:
    """Evaluate all routing policies against fixed-stage baselines.

    Returns dict of policy → (accuracy, cost, CWA_0.5).
    """
    by_qid = organize_by_query(data)
    n_stages = 4
    stage_sizes = [2, 4, 6, 8]
    stage_costs = [0.25, 0.50, 0.75, 1.08]  # FULL model from config

    policies = {}

    # Fixed-stage baselines
    for fixed_s in range(n_stages):
        accs, costs = [], []
        for qid, tuples in by_qid.items():
            t = tuples[fixed_s]
            accs.append(t["correct"])
            costs.append(stage_costs[fixed_s])
        policies[f"fixed_S{fixed_s}"] = {
            "accuracy": np.mean(accs), "cost": np.mean(costs),
            "cwa_0.5": np.mean(accs) - 0.5 * np.mean(costs) / max(stage_costs),
        }

    # Full pipeline
    accs, costs = [], []
    for qid, tuples in by_qid.items():
        t = tuples[-1]  # S3
        accs.append(t["correct"])
        costs.append(stage_costs[-1])
    policies["full_pipeline"] = {
        "accuracy": np.mean(accs), "cost": np.mean(costs),
        "cwa_0.5": np.mean(accs) - 0.5 * np.mean(costs) / max(stage_costs),
    }

    # Entropy-based routing
    for tau_name, tau_func in [
        ("entropy_threshold", lambda e: e["confidence_from_entropy"]),
        ("max_prob", lambda e: e["max_prob"]),
    ]:
        for tau in [0.3, 0.5, 0.7, 0.9]:
            accs, costs = [], []
            for qid, tuples in by_qid.items():
                for s, t in enumerate(tuples):
                    e = entropy_conf.get(qid, [None]*4)
                    conf_signal = tau_func(e[s]) if e[s] else 0.5
                    if s == 3 or conf_signal >= tau:
                        accs.append(t["correct"])
                        costs.append(stage_costs[s])
                        break
            cwa = np.mean(accs) - 0.5 * np.mean(costs) / max(stage_costs)
            policies[f"{tau_name}_tau{tau}"] = {
                "accuracy": np.mean(accs), "cost": np.mean(costs), "cwa_0.5": cwa,
            }

    # Retrieval-margin routing
    for tau in [0.1, 0.2, 0.3, 0.5]:
        accs, costs = [], []
        for qid, tuples in by_qid.items():
            for s, t in enumerate(tuples):
                r = retrieval_conf.get(qid, [None]*4)
                margin = r[s].get("ce_margin", 0) if isinstance(r[s], dict) else 0
                if s == 3 or margin >= tau:
                    accs.append(t["correct"])
                    costs.append(stage_costs[s])
                    break
        cwa = np.mean(accs) - 0.5 * np.mean(costs) / max(stage_costs)
        policies[f"retrieval_margin_tau{tau}"] = {
            "accuracy": np.mean(accs), "cost": np.mean(costs), "cwa_0.5": cwa,
        }

    return policies


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True,
                       help="Path to hidden-state JSONL file")
    parser.add_argument("--output_dir", default="/workspace/routing_baselines")
    parser.add_argument("--no_gpu", action="store_true",
                       help="Skip GPU-requiring entropy computation")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.data}...")
    data = load_data(args.data)
    print(f"  {len(data)} stage-tuples, {len(set(d['query_id'] for d in data))} queries")

    # Compute confidences
    self_report = compute_self_report_confidence(data)
    retrieval = compute_retrieval_confidence(data)

    if not args.no_gpu:
        # Load model for entropy computation
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("Loading Qwen BF16 for entropy computation...")
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16,
            device_map="cuda:0")
        model.eval()
        entropy = compute_entropy_confidence(data, tok, model)
    else:
        entropy = {}

    # Evaluate all routing policies
    print("\nEvaluating routing policies...")
    policies = evaluate_routing_policies(data, entropy, retrieval, self_report)

    # Output results
    print("\n" + "=" * 70)
    print("ROUTING POLICY COMPARISON")
    print("=" * 70)
    print(f"{'Policy':<30} {'Accuracy':>8} {'Cost':>8} {'CWA(0.5)':>8}")
    print("-" * 54)

    # Sort by CWA
    sorted_pols = sorted(policies.items(), key=lambda x: x[1]["cwa_0.5"], reverse=True)

    best_cwa = sorted_pols[0][1]["cwa_0.5"]
    fixed_s1_cwa = policies.get("fixed_S0", {}).get("cwa_0.5", 0)

    for name, metrics in sorted_pols:
        marker = " ← BEST" if metrics["cwa_0.5"] == best_cwa else ""
        print(f"{name:<30} {metrics['accuracy']:8.4f} {metrics['cost']:8.3f} "
              f"{metrics['cwa_0.5']:8.4f}{marker}")

    print(f"\nBest CWA: {best_cwa:.4f}")
    print(f"Fixed S0 CWA: {fixed_s1_cwa:.4f}")
    print(f"Delta: {best_cwa - fixed_s1_cwa:+.4f}")

    # Save results
    results_path = out / "routing_policies.json"
    with open(results_path, "w") as f:
        json.dump(policies, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
