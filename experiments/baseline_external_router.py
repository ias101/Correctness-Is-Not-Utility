"""
Naive External Baseline Router — requested by reviewer Round 3.

Question: "Do LLM internal states provide uniquely useful signals beyond
what's trivially available from the query text?"

This baseline embeds the input query using a lightweight sentence transformer
(all-MiniLM-L6-v2, 384-dim) and trains a logistic regression classifier to
predict routing decisions. If this external router matches or beats the
internal-state MLP, the premise of using LLM hidden states is weakened.

Comparison:
  - Internal-state router: hidden_state (3584-dim from Qwen2.5-7B) + MLP
  - External router:     query embedding (384-dim from MiniLM) + Logistic Regression
  - Random baseline:     random routing
  - Fixed baselines:     fixed_S1, fixed_S2, fixed_S3

Usage:
  python baseline_external_router.py \
      --data data/collected_states_hotpotqa_v3.jsonl \
      --output_dir results/external_baseline
"""

import argparse
import json
import os
import sys
import numpy as np
from collections import defaultdict
from typing import Dict, List

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experiments'))
from config import NUM_STAGES

# Use the FULL cost model from evaluate.py
STAGE_COSTS = np.array([0.25, 0.50, 0.75, 1.08])
MAX_COST = STAGE_COSTS.sum()

def cost_of_pipeline(stop_stage):
    if stop_stage is None:
        return MAX_COST
    return STAGE_COSTS[:stop_stage + 1].sum()


def embed_queries(queries: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Embed queries using lightweight sentence transformer."""
    print(f"[*] Loading sentence transformer: {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(queries, show_progress_bar=True)
    print(f"[*] Embedded {len(queries)} queries → {embeddings.shape[1]}-dim")
    return embeddings


def build_external_labels(data: List[Dict]) -> Dict[str, int]:
    """
    Build routing labels for each query:
      0 = stop at S0 (cheapest, answerable at S0)
      1 = continue to S3 (need more context)

    Label = 0 if S0 is correct (would stop early)
    Label = 1 if S0 is wrong but S3 is correct (would benefit from continuing)
    """
    by_query = defaultdict(list)
    for d in data:
        by_query[d['query_id']].append(d)

    labels = {}
    for qid, stages in by_query.items():
        stages_sorted = sorted(stages, key=lambda s: s['stage_idx'])
        s0_correct = stages_sorted[0].get('stage_correctness', 0)
        s3_correct = stages_sorted[-1].get('final_correctness', 0)

        if s0_correct == 1:
            labels[qid] = 0  # stop early (correct at S0)
        elif s3_correct == 1:
            labels[qid] = 1  # continue (need S3)
        else:
            labels[qid] = 0  # neither correct → stop (waste of compute)

    return labels


def evaluate_external_router(
    data_path: str,
    output_dir: str,
    embed_model: str = "all-MiniLM-L6-v2",
    seed: int = 42,
):
    """Train and evaluate external query-embedding router."""
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)

    # Load data
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))

    # Get unique queries
    by_query = defaultdict(list)
    for d in data:
        by_query[d['query_id']].append(d)
    query_ids = sorted(by_query.keys())

    # Get question texts
    qid_to_question = {}
    for qid in query_ids:
        qid_to_question[qid] = by_query[qid][0]['question']

    questions = [qid_to_question[qid] for qid in query_ids]

    # Build labels
    labels = build_external_labels(data)

    # Embed queries
    embeddings = embed_queries(questions, embed_model)

    # Split by query
    n = len(query_ids)
    np.random.shuffle(query_ids)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    train_qids = set(query_ids[:n_train])
    val_qids = set(query_ids[n_train:n_train + n_val])
    test_qids = set(query_ids[n_train + n_val:])

    # Prepare train/val/test
    X_train = np.array([embeddings[query_ids.index(q)] for q in query_ids if q in train_qids])
    y_train = np.array([labels[q] for q in query_ids if q in train_qids])
    X_val = np.array([embeddings[query_ids.index(q)] for q in query_ids if q in val_qids])
    y_val = np.array([labels[q] for q in query_ids if q in val_qids])
    X_test = np.array([embeddings[query_ids.index(q)] for q in query_ids if q in test_qids])
    y_test = np.array([labels[q] for q in query_ids if q in test_qids])

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    pos_rate = y_train.mean()
    print(f"[*] Positive rate (continue to S3): {pos_rate:.3f}")
    print(f"[*] Train: {len(train_qids)}q, Val: {len(val_qids)}q, Test: {len(test_qids)}q")

    # Train Logistic Regression
    clf = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        C=1.0,
        solver='lbfgs',
        random_state=seed,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred_proba = clf.predict_proba(X_val)[:, 1]
    val_auroc = roc_auc_score(y_val, y_pred_proba)
    val_acc = accuracy_score(y_val, clf.predict(X_val))

    y_test_proba = clf.predict_proba(X_test)[:, 1]
    test_auroc = roc_auc_score(y_test, y_test_proba)
    test_acc = accuracy_score(y_test, clf.predict(X_test))

    print(f"\n[*] External Router Results:")
    print(f"  Val AUROC: {val_auroc:.4f}, Val Acc: {val_acc:.4f}")
    print(f"  Test AUROC: {test_auroc:.4f}, Test Acc: {test_acc:.4f}")

    # ── Routing Evaluation ──
    test_qid_list = [q for q in query_ids if q in test_qids]
    test_emb = np.array([embeddings[query_ids.index(q)] for q in test_qid_list])
    test_emb = scaler.transform(test_emb)
    test_probs = clf.predict_proba(test_emb)[:, 1]

    # CWA evaluation
    total_correct = 0
    total_cost = 0
    n_test = len(test_qid_list)

    for qid, prob in zip(test_qid_list, test_probs):
        stages = sorted(by_query[qid], key=lambda d: d['stage_idx'])

        # Decision: prob > 0.5 → continue to S3, else stop at S0
        stop_stage = 3 if prob > 0.5 else 0

        is_correct = stages[stop_stage].get('stage_correctness', 0) if stop_stage < 3 else stages[-1].get('final_correctness', 0)
        total_correct += is_correct
        total_cost += cost_of_pipeline(stop_stage)

    acc = total_correct / n_test
    avg_cost = total_cost / n_test
    norm_cost = avg_cost / MAX_COST
    cwa_05 = acc - 0.5 * norm_cost
    cwa_08 = acc - 0.8 * norm_cost

    # Fixed baselines
    fixed_results = {}
    for stop_s, name in [(0, "fixed_S1"), (1, "fixed_S2"), (2, "fixed_S3"), (3, "full_pipeline")]:
        correct = sum(
            1 for qid in test_qid_list
            if sorted(by_query[qid], key=lambda d: d['stage_idx'])[min(stop_s, 3)].get(
                'stage_correctness' if stop_s < 3 else 'final_correctness', 0)
        )
        acc_b = correct / n_test
        cost_b = cost_of_pipeline(stop_s)
        norm_b = cost_b / MAX_COST
        cwa_05_b = acc_b - 0.5 * norm_b
        cwa_08_b = acc_b - 0.8 * norm_b
        fixed_results[name] = {
            "accuracy": acc_b, "avg_cost": cost_b,
            "cwa_05": cwa_05_b, "cwa_08": cwa_08_b,
        }

    # Print comparison
    print(f"\n{'='*60}")
    print(f"ROUTING COMPARISON — External (MiniLM) vs Internal (LLM HS)")
    print(f"{'='*60}")
    print(f"{'Method':>25} {'Acc':>6} {'Cost':>7} {'CWA(0.5)':>9} {'CWA(0.8)':>9}")
    print(f"{'-'*60}")
    for name, res in fixed_results.items():
        print(f"{name:>25}: {res['accuracy']:.4f} {res['avg_cost']:.4f}  {res['cwa_05']:.4f}     {res['cwa_08']:.4f}")
    print(f"{'external (MiniLM)':>25}: {acc:.4f} {avg_cost:.4f}  {cwa_05:.4f}     {cwa_08:.4f}")
    print(f"{'-'*60}")

    # Compare with internal-state router (if available)
    internal_cwa = fixed_results.get("fixed_S1", {}).get("cwa_05", 0)  # placeholder
    print(f"\n  External CWA(0.5) = {cwa_05:.4f}")
    print(f"  Internal (LLM HS) mean CWA(0.5) ≈ 0.5474 (from multi-seed)")
    print(f"  External advantage: {cwa_05 - 0.5474:+.4f}")

    # Save
    results = {
        "model": embed_model,
        "val_auroc": float(val_auroc),
        "test_auroc": float(test_auroc),
        "test_accuracy": float(test_acc),
        "external_router": {
            "accuracy": acc, "avg_cost": avg_cost,
            "cwa_05": cwa_05, "cwa_08": cwa_08,
        },
        "fixed_baselines": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in fixed_results.items()},
        "comparison": {
            "external_cwa_05": cwa_05,
            "internal_mean_cwa_05": 0.5474,
            "delta": cwa_05 - 0.5474,
            "interpretation": "external > internal → LLM hidden states NOT uniquely useful" if cwa_05 > 0.5474 else "internal > external → LLM hidden states provide unique signal",
        },
    }
    with open(os.path.join(output_dir, "external_baseline_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[*] Results saved to {output_dir}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/collected_states_hotpotqa_v3.jsonl")
    ap.add_argument("--output_dir", type=str, default="results/external_baseline")
    ap.add_argument("--embed_model", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    evaluate_external_router(args.data, args.output_dir, args.embed_model, args.seed)


if __name__ == "__main__":
    main()
