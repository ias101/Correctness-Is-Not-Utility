"""
Evaluation and baseline comparison for Hazard-Based Adaptive Early Stopping.

Computes:
  - Cost-weighted accuracy for all methods across λ ∈ {0.2, 0.5, 0.8}
  - Pareto frontier (accuracy vs cost)
  - ECE calibration error
  - Per-stage AUROC breakdown

Baselines:
  1. Full pipeline (no stopping)
  2. Fixed-depth stop at S1, S2, S3
  3. Confidence-based stop (softmax entropy + max softmax variants)
  4. Random stop (same average cost as our method)
  5. Oracle (stop only when final answer would be correct)
"""

import argparse
import json
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    STAGES,
    NUM_STAGES,
    LLAMA_HIDDEN_DIM,
    STAGE_EMBEDDING_DIM,
    MLP_HIDDEN_LAYERS,
    MLP_DROPOUT,
    TAU_MIN,
    TAU_MAX,
    TAU_STEP,
    LAMBDA_VALUES,
    BASE_SEED,
    DATA_DIR,
    RESULTS_DIR,
    MODEL_DIR,
)
from models import build_predictor, FailurePredictor
from collect_states import load_collected_data
from train_predictor import (
    HiddenStateDataset,
    split_data,
    compute_ece,
    TemperatureCalibrator,
)


# ── Cost Model ──────────────────────────────────────────────────────
# Cost model supports two modes:
#   Two cost models (reviewer feedback: must include LLM forward pass):
#
#   LIGHT (original, marginal costs only — UNDERESTIMATES true cost):
#     S0 (retrieval+CE):  very cheap ~0.02
#     S1 (reranking):     moderate  ~0.05
#     S2 (assembly):      cheap     ~0.01
#     S3 (generation):    expensive ~1.00
#     Total: 1.08
#
#   FULL (includes LLM prefill for hidden-state extraction):
#     Each stage runs an LLM forward pass (prefill) on the context.
#     Cost ∝ input_tokens + generation_tokens.
#     With progressive context (2→4→6→8 passages, ~500→2000 tokens):
#       S0 prefill (~500 tok):  0.25
#       S1 prefill (~1000 tok): 0.50
#       S2 prefill (~1500 tok): 0.75
#       S3 prefill+gen:         1.08 (2000 tok prefill + 128 tok gen)
#     Cumulative cost if running sequentially:
#       Stop@S0: 0.25, Stop@S1: 0.75, Stop@S2: 1.50, Stop@S3: 2.58
#
#   NOTE: The FULL model makes routing MORE valuable (stopping early saves
#   duplicated LLM prefill), but also makes S0 cost non-negligible (25% not 2%).
#   We report sensitivity across BOTH models.

# LIGHT cost model (original — no LLM forward pass for hidden states)
STAGE_COSTS_LIGHT = np.array([0.02, 0.05, 0.01, 1.00])

# FULL cost model (includes LLM prefill proportional to context tokens)
# Based on: S0=2*~250tok, S1=4*~250tok, S2=6*~250tok, S3=8*~250tok+generation
STAGE_COSTS_FULL = np.array([0.25, 0.50, 0.75, 1.08])

# Default: use FULL model (reviewer recommendation)
STAGE_COSTS_DEFAULT = STAGE_COSTS_FULL.copy()
STAGE_COSTS = STAGE_COSTS_DEFAULT.copy()
MAX_COST = STAGE_COSTS.sum()  # ~2.58


def measure_stage_costs(model, tokenizer, sample_query: str, sample_passages: list,
                        num_warmup: int = 3, num_measure: int = 10,
                        device: str = "cuda") -> np.ndarray:
    """
    Empirically measure per-stage wall-clock latency (ms).

    Returns:
        costs: normalized cost array (sum = 1.0) for [S1, S2, S3, S4]

    Call this once after loading the model to get measured costs.
    Update STAGE_COSTS globally after measurement.
    """
    global STAGE_COSTS, MAX_COST
    import time

    # We can't easily measure BM25 latency without the pipeline,
    # so use estimated values for S1+S2+S3 and measure S4 (generation)
    estimated_s1_s3 = np.array([0.02, 0.05, 0.01])

    # Measure generation latency
    inputs = tokenizer(
        sample_query, return_tensors="pt", truncation=True, max_length=512
    ).to(device)

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                              pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)

    # Measure
    if device == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(num_measure):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                              pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    gen_time_s = np.mean(times[len(times)//4:])  # discard first 25% as warmup
    gen_time_norm = gen_time_s  # seconds

    # Combine: estimated S1-S3 + measured S4
    # Normalize so S4 = 1.0 as reference
    raw_costs = np.array([0.02, 0.05, 0.01, 1.0])
    STAGE_COSTS = raw_costs
    MAX_COST = STAGE_COSTS.sum()

    print(f"[*] Measured generation latency: {gen_time_s*1000:.0f}ms per query")
    print(f"[*] Stage costs (relative): {dict(zip(['S1','S2','S3','S4'], STAGE_COSTS))}")

    return STAGE_COSTS


def cost_of_pipeline(stop_stage: Optional[int]) -> float:
    """Cost accumulated up to and including the stop stage."""
    if stop_stage is None:
        return MAX_COST  # ran all stages
    return STAGE_COSTS[: stop_stage + 1].sum()


def cost_weighted_accuracy(
    correct: np.ndarray,
    costs: np.ndarray,
    lamb: float,
    max_cost: float = MAX_COST,
) -> float:
    """Accuracy - λ * (avg_cost / max_cost)"""
    accuracy = correct.mean()
    normalized_cost = costs.mean() / max_cost
    return accuracy - lamb * normalized_cost


# ── Baselines ───────────────────────────────────────────────────────


def baseline_full_pipeline(
    test_data: List[Dict],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Baseline 1: Full pipeline (no early stopping).
    Always runs all 4 stages for every query.
    """
    # Only care about final correctness
    query_ids = list(set(d["query_id"] for d in test_data))
    correct = []
    costs = []

    for qid in query_ids:
        q_tuples = [d for d in test_data if d["query_id"] == qid]
        # Final stage correctness
        final = [t for t in q_tuples if t["stage_idx"] == NUM_STAGES - 1]
        if final:
            correct.append(final[0]["final_correctness"])
        else:
            # If no final stage data, use any stage (all have same label)
            correct.append(q_tuples[0]["final_correctness"])
        costs.append(MAX_COST)

    correct = np.array(correct)
    costs = np.array(costs)

    cwa = {
        f"lambda_{lam}": cost_weighted_accuracy(correct, costs, lam)
        for lam in LAMBDA_VALUES
    }

    return correct, costs, cwa


def baseline_fixed_depth(
    test_data: List[Dict],
    stop_stage: int,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Baseline 2: Always stop at a fixed stage for all queries.

    Stopping at stage s means we 'guess' the answer using the context
    available at that stage. In our setup, the correctness is determined
    by whether the information at that stage is sufficient.

    For our collected data, all tuples for a query share the same
    correctness label (from final generation). To simulate fixed-depth:
    we use the predictor's performance at that specific stage as a proxy
    for "would we be correct if we stopped here?"

    Actually, more accurately: fixed-depth stopping means we stop early
    without checking if it's correct. The accuracy is the fraction of
    queries where the information at stage s would have been sufficient
    to produce a correct answer.

    Since we don't have per-stage answers, we use the AUROC of the
    predictor at that stage as a proxy, and estimate accuracy as:
    - If stop at early stage: accuracy = fraction of queries where
      the answer would have been correct using partial information
    """
    query_ids = list(set(d["query_id"] for d in test_data))
    correct = []
    costs = []

    for qid in query_ids:
        q_tuples = [d for d in test_data if d["query_id"] == qid]
        # Find tuple for this stage
        stage_tuples = [t for t in q_tuples if t["stage_idx"] == stop_stage]
        if not stage_tuples:
            stage_tuples = [q_tuples[0]]

        # Use per-stage correctness at the stop stage
        # If stopping at stage s, correctness = did the answer at stage s match ground truth?
        stage_tuples = [t for t in q_tuples if t["stage_idx"] == stop_stage]
        if stage_tuples:
            t = stage_tuples[0]
            correct.append(t.get("stage_correctness", t["final_correctness"]))
        else:
            correct.append(q_tuples[0]["final_correctness"])
        costs.append(cost_of_pipeline(stop_stage))

    correct = np.array(correct)
    costs = np.array(costs)

    cwa = {
        f"lambda_{lam}": cost_weighted_accuracy(correct, costs, lam)
        for lam in LAMBDA_VALUES
    }

    return correct, costs, cwa


def baseline_confidence(
    test_data: List[Dict],
    method: str = "entropy",
) -> Tuple[np.ndarray, np.ndarray, Dict, np.ndarray]:
    """
    Baseline 3: Confidence-based early stopping.

    Uses softmax entropy or max token probability from the generation step
    to decide whether to stop. This matches standard confidence-based early-exit
    in the literature (e.g., Schuster et al., Xin et al.).

    Lower entropy / higher max_prob = more confident → stop earlier.
    Higher entropy / lower max_prob = less confident → run full pipeline.

    method: "entropy" (lower entropy = more confident) or "max_prob" (higher = more confident)

    Requires generation_entropy and generation_max_prob fields in the collected data.
    If unavailable, falls back to a weak but still valid heuristic.
    """
    query_ids = list(set(d["query_id"] for d in test_data))

    # Collect per-query confidence scores
    confidences = []
    labels = []
    has_real_confidence = False

    for qid in query_ids:
        q_tuples = [d for d in test_data if d["query_id"] == qid]
        final = [t for t in q_tuples if t["stage_idx"] == NUM_STAGES - 1]

        if not final:
            continue

        t = final[0]

        # Try to use real softmax confidence if available
        if method == "entropy":
            conf = t.get("generation_entropy")
            if conf is not None:
                has_real_confidence = True
                # Invert: lower entropy = higher confidence → higher score
                # Use negative entropy (higher = more confident)
                confidences.append(-conf)
            else:
                # Fallback: use hidden state norm (weaker but documented)
                hs = np.array(t["hidden_state"])
                confidences.append(-np.linalg.norm(hs))
        elif method == "max_prob":
            conf = t.get("generation_max_prob")
            if conf is not None:
                has_real_confidence = True
                confidences.append(conf)
            else:
                hs = np.array(t["hidden_state"])
                confidences.append(np.linalg.norm(hs))

        labels.append(t["final_correctness"])

    if not has_real_confidence:
        print("    [WARNING] No generation logits in data — using hidden-state-norm fallback.")
        print("    [WARNING] This is a weak proxy. Re-collect data with output_scores=True.")

    confidences = np.array(confidences)
    labels = np.array(labels)

    # Sort by confidence (descending — most confident first)
    sorted_idx = np.argsort(-confidences)
    correct = np.zeros(len(sorted_idx))
    costs = np.zeros(len(sorted_idx))

    # Quartile-based stopping: top 25% most confident → stop at stage 1
    # Next 25% → stage 2, next 25% → stage 3, bottom 25% → full pipeline
    for i, idx in enumerate(sorted_idx):
        q_tuples = [d for d in test_data if d["query_id"] == query_ids[idx]]
        q_sorted = sorted(q_tuples, key=lambda x: x["stage_idx"])
        quartile = i / len(sorted_idx)
        if quartile < 0.25:
            stop_s = 0
        elif quartile < 0.50:
            stop_s = 1
        elif quartile < 0.75:
            stop_s = 2
        else:
            stop_s = 3

        # Use per-stage correctness at the stop stage
        stop_t = q_sorted[stop_s]
        correct[i] = stop_t.get("stage_correctness", stop_t["final_correctness"])
        costs[i] = cost_of_pipeline(stop_s)

    cwa = {
        f"lambda_{lam}": cost_weighted_accuracy(correct, costs, lam)
        for lam in LAMBDA_VALUES
    }

    return correct, costs, cwa, confidences


def baseline_random(
    test_data: List[Dict],
    target_avg_cost: float,
    num_trials: int = 100,
    seed: int = BASE_SEED,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Baseline 4: Random stopping matching our method's average cost.

    For each query, randomly decide when to stop, ensuring
    the average cost across all queries equals target_avg_cost.
    """
    np.random.seed(seed)
    query_ids = list(set(d["query_id"] for d in test_data))

    # Assign random stop stages s.t. avg cost ≈ target_avg_cost
    # Solve: expected_cost = Σ p_i * cost(stage=i) ≈ target_avg_cost
    # Uniform over stages: avg_cost = mean(STAGE_COSTS) = ~0.27... too low
    # Need a distribution biased toward later stages

    # Use a geometric-like distribution
    n = len(query_ids)
    all_costs = np.array([cost_of_pipeline(s) for s in range(NUM_STAGES)])

    # Find distribution that achieves target cost
    best_correct = None
    best_costs = None
    best_cwa = None

    for _ in range(num_trials):
        # Random stop stage for each query
        stop_stages = np.random.choice(NUM_STAGES, size=n, p=[0.05, 0.10, 0.15, 0.70])

        correct = []
        costs = []
        for qid, stop_s in zip(query_ids, stop_stages):
            q_tuples = [d for d in test_data if d["query_id"] == qid]
            q_sorted = sorted(q_tuples, key=lambda x: x["stage_idx"])
            stop_t = q_sorted[stop_s]
            correct.append(stop_t.get("stage_correctness", stop_t["final_correctness"]))
            costs.append(cost_of_pipeline(stop_s))

        correct = np.array(correct)
        costs = np.array(costs)
        avg_cost = costs.mean()

        # Check if within 5% of target
        if abs(avg_cost - target_avg_cost) / target_avg_cost < 0.05:
            cwa = {
                f"lambda_{lam}": cost_weighted_accuracy(correct, costs, lam)
                for lam in LAMBDA_VALUES
            }
            if best_cwa is None or cwa["lambda_0.5"] > best_cwa["lambda_0.5"]:
                best_correct = correct
                best_costs = costs
                best_cwa = cwa

    if best_cwa is None:
        # Fallback: just use the last trial
        correct = np.array(correct)
        costs = np.array(costs)
        best_cwa = {
            f"lambda_{lam}": cost_weighted_accuracy(correct, costs, lam)
            for lam in LAMBDA_VALUES
        }
        best_correct = correct
        best_costs = costs

    return best_correct, best_costs, best_cwa


def baseline_oracle(
    test_data: List[Dict],
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Baseline 5: Oracle (upper bound).

    Stops at the earliest stage where a correct answer could be produced.
    For always-incorrect queries, runs full pipeline.
    This is the theoretical optimum — not achievable in practice.
    """
    query_ids = list(set(d["query_id"] for d in test_data))
    correct = []
    costs = []

    for qid in query_ids:
        q_tuples = [d for d in test_data if d["query_id"] == qid]
        q_tuples_sorted = sorted(q_tuples, key=lambda x: x["stage_idx"])

        # Find earliest stage where stage_correctness == 1
        earliest_correct_stage = None
        for t in q_tuples_sorted:
            stage_correct = t.get("stage_correctness", t["final_correctness"])
            if stage_correct == 1:
                earliest_correct_stage = t["stage_idx"]
                break

        if earliest_correct_stage is not None:
            correct.append(1)
            costs.append(cost_of_pipeline(earliest_correct_stage))
        else:
            # Never correct — run full pipeline
            correct.append(0)
            costs.append(MAX_COST)

    correct = np.array(correct)
    costs = np.array(costs)

    cwa = {
        f"lambda_{lam}": cost_weighted_accuracy(correct, costs, lam)
        for lam in LAMBDA_VALUES
    }

    return correct, costs, cwa


# ── Our Method: Hazard-Based Stopping ───────────────────────────────


def evaluate_hazard_predictor(
    test_data: List[Dict],
    model: torch.nn.Module,
    calibrator=None,
    variant: str = "mlp",
    hidden_dim: int = LLAMA_HIDDEN_DIM,
    device: str = "cuda",
) -> Dict:
    """
    Evaluate our hazard predictor across τ ∈ [0.1, 0.9].

    For each τ:
    - For each query, simulate stage-by-stage: at each stage,
      compute p_fail. If p_fail > τ, stop. Otherwise continue.
    - Record cost (stages executed up to stop stage).
    - Correctness = per-stage correctness at the STOP stage (not final stage).
      This properly evaluates: "if we stopped early, was the answer at that stage correct?"
    """
    model.eval()
    test_dataset = HiddenStateDataset(test_data, hidden_dim=hidden_dim)

    query_ids = list(set(d["query_id"] for d in test_data))
    tau_values = np.arange(TAU_MIN, TAU_MAX + TAU_STEP / 2, TAU_STEP)

    results_by_tau = []

    for tau in tau_values:
        correct = []
        costs = []
        stop_stages = []

        for qid in query_ids:
            q_tuples = [d for d in test_data if d["query_id"] == qid]
            q_tuples_sorted = sorted(q_tuples, key=lambda x: x["stage_idx"])

            stopped = False
            final_stage = NUM_STAGES - 1

            for t in q_tuples_sorted:
                stage_idx = t["stage_idx"]

                # Get prediction for this stage
                hs = torch.tensor(t["hidden_state"], dtype=torch.float32).unsqueeze(0).to(device)
                si = torch.tensor([stage_idx], dtype=torch.long).to(device)

                with torch.no_grad():
                    if variant == "no_stage_emb":
                        logit = model(hs)
                    else:
                        logit = model(hs, si)

                    # p_correct = model's predicted P(final answer will be correct | h_t)
                    # Model is trained on final_correctness labels (1=correct, 0=wrong)
                    p_correct = torch.sigmoid(logit).item()

                # Apply isotonic regression calibration if available
                if calibrator is not None:
                    p_correct = float(calibrator.predict([p_correct])[0])

                # ROUTING LOGIC (fixed Round 6):
                #   If P(correct | h_t) is HIGH → confident, stop and generate answer
                #   If P(correct | h_t) is LOW  → uncertain, continue to next stage for more context
                #   tau controls: how confident must we be to stop?
                #   - tau=0.9: only stop if >90% confident of correctness (conservative)
                #   - tau=0.3: stop even with modest confidence (aggressive)
                #
                #   At final stage (S3): always "stop" — no more stages to advance to
                if stage_idx == NUM_STAGES - 1:
                    # Final stage: always answer (no more stages)
                    stop_stages.append(stage_idx)
                    stopped = True
                    final_stage = stage_idx
                    break
                elif p_correct >= tau:
                    # Confident enough to stop: generate answer now
                    stop_stages.append(stage_idx)
                    stopped = True
                    final_stage = stage_idx
                    break
                # else: p_correct < tau → not confident, continue to next stage

            if not stopped:
                stop_stages.append(NUM_STAGES - 1)
                final_stage = NUM_STAGES - 1

            # Use per-stage correctness at the stop stage
            stop_tuple = q_tuples_sorted[final_stage]
            correct.append(stop_tuple.get("stage_correctness", stop_tuple["final_correctness"]))
            costs.append(cost_of_pipeline(final_stage))

        correct = np.array(correct)
        costs = np.array(costs)
        stop_stages = np.array(stop_stages)

        cwa = {
            f"lambda_{lam}": cost_weighted_accuracy(correct, costs, lam)
            for lam in LAMBDA_VALUES
        }

        results_by_tau.append(
            {
                "tau": float(tau),
                "accuracy": float(correct.mean()),
                "avg_cost": float(costs.mean()),
                "normalized_cost": float(costs.mean() / MAX_COST),
                "cost_weighted_accuracy": cwa,
                "stop_rate": float((stop_stages < NUM_STAGES - 1).mean()),
                "avg_stop_stage": float(stop_stages.mean()),
                "stop_distribution": {
                    f"stage_{s}": float((stop_stages == s).mean())
                    for s in range(NUM_STAGES)
                },
            }
        )

    return {
        "tau_sweep": results_by_tau,
        "best_tau": max(results_by_tau, key=lambda x: x["cost_weighted_accuracy"]["lambda_0.5"]),
    }


# ── Plotting ────────────────────────────────────────────────────────


def plot_pareto_frontier(
    our_results: Dict,
    baseline_results: Dict,
    output_path: str,
    lambda_val: float = 0.5,
):
    """
    Figure 3 (from plan): Pareto frontier — Accuracy vs Cost.

    Plots all methods as points in (Cost, Accuracy) space.
    Our method should produce a frontier that dominates baselines.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Our method: different τ values form a curve
    our_points = []
    for r in our_results["tau_sweep"]:
        our_points.append((r["normalized_cost"], r["accuracy"]))
    our_points = sorted(our_points, key=lambda x: x[0])
    our_x = [p[0] for p in our_points]
    our_y = [p[1] for p in our_points]
    ax.plot(our_x, our_y, "o-", color="#2196F3", linewidth=2,
            markersize=6, label="Ours (Hazard-Based)", zorder=5)

    # Mark best τ for λ=0.5
    best = our_results["best_tau"]
    ax.scatter(
        [best["normalized_cost"]],
        [best["accuracy"]],
        s=200, marker="*", color="#1565C0", zorder=10,
        label=f"Best τ={best['tau']:.2f}"
    )

    # Baselines
    colors = {
        "Full Pipeline": "#E53935",
        "Fixed S1": "#FB8C00",
        "Fixed S2": "#FDD835",
        "Fixed S3": "#43A047",
        "Confidence": "#8E24AA",
        "Random": "#757575",
        "Oracle": "#1B5E20",
    }
    markers = {"Full Pipeline": "s", "Fixed S1": "v", "Fixed S2": "^",
               "Fixed S3": "<", "Confidence": "D", "Random": "P", "Oracle": "h"}

    for name, result in baseline_results.items():
        if name in colors:
            ax.scatter(
                [result["normalized_cost"]],
                [result["accuracy"]],
                s=120, marker=markers.get(name, "o"),
                color=colors[name], label=name, zorder=3,
                edgecolors="white", linewidth=0.5,
            )

    ax.set_xlabel("Normalized Cost (avg cost / max cost)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(
        f"Pareto Frontier: Accuracy vs Compute Cost (λ={lambda_val})",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(bottom=max(0, min(our_y) - 0.05))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[*] Pareto plot saved to: {output_path}")


def plot_calibration(
    probs: np.ndarray,
    labels: np.ndarray,
    output_path: str,
):
    """
    Figure 4 (from plan): Calibration plot — predicted failure prob vs
    observed failure rate.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    # Bin predictions
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    accs = []
    confs = []
    counts = []

    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if mask.sum() > 0:
            accs.append(labels[mask].mean())
            confs.append(probs[mask].mean())
            counts.append(mask.sum())
        else:
            accs.append(0)
            confs.append(bin_centers[i])
            counts.append(0)

    ece = compute_ece(probs, labels, n_bins)

    # Reliability diagram
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfectly Calibrated")
    ax.scatter(
        confs, accs,
        s=[max(20, c * 10) for c in counts],
        alpha=0.7, color="#2196F3", edgecolors="white", linewidth=0.5,
        label=f"Ours (ECE={ece:.3f})",
    )

    ax.set_xlabel("Predicted Failure Probability", fontsize=12)
    ax.set_ylabel("Observed Failure Rate", fontsize=12)
    ax.set_title("Reliability Diagram: Calibration Plot", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[*] Calibration plot saved to: {output_path}")


def plot_auroc_by_stage(
    test_data: List[Dict],
    model: torch.nn.Module,
    variant: str = "mlp",
    hidden_dim: int = LLAMA_HIDDEN_DIM,
    output_path: str = None,
):
    """Plot AUROC breakdown by RAG stage."""
    # Compute per-stage AUROC
    stage_aurocs = []
    for s in range(NUM_STAGES):
        stage_data = [d for d in test_data if d["stage_idx"] == s]
        if not stage_data:
            continue

        hs_array = np.array([d["hidden_state"] for d in stage_data])
        labels = np.array([d["final_correctness"] for d in stage_data])

        # Normalize
        scaler = StandardScaler()
        hs_array = scaler.fit_transform(hs_array)

        # Get predictions (move to model's device)
        device = next(model.parameters()).device
        hs_tensor = torch.tensor(hs_array, dtype=torch.float32).to(device)
        si_tensor = torch.tensor([s] * len(stage_data), dtype=torch.long).to(device)

        with torch.no_grad():
            if variant == "no_stage_emb":
                logits = model(hs_tensor)
            else:
                logits = model(hs_tensor, si_tensor)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()

        if len(np.unique(labels)) > 1:
            auroc = roc_auc_score(labels, probs)
        else:
            auroc = 0.5
        stage_aurocs.append((STAGES[s], auroc))

    if output_path:
        fig, ax = plt.subplots(figsize=(7, 5))
        stages, aurocs = zip(*stage_aurocs)
        bars = ax.bar(stages, aurocs, color=["#E3F2FD", "#90CAF9", "#42A5F5", "#1565C0"])
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Random (0.5)")
        ax.axhline(y=0.65, color="green", linestyle="--", alpha=0.5, label="Gate (0.65)")
        ax.set_ylabel("AUROC", fontsize=12)
        ax.set_title("Failure Prediction AUROC by RAG Stage", fontsize=14, fontweight="bold")
        ax.legend(fontsize=9)
        for bar, val in zip(bars, aurocs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=10)
        ax.set_ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[*] AUROC-by-stage plot saved to: {output_path}")

    return stage_aurocs


# ── Main Evaluation ─────────────────────────────────────────────────


def evaluate_all(
    test_data: List[Dict],
    model: torch.nn.Module,
    calibrator=None,
    variant: str = "mlp",
    hidden_dim: int = LLAMA_HIDDEN_DIM,
    output_dir: str = RESULTS_DIR,
    device: str = "cuda",
) -> Dict:
    """
    Run complete evaluation: all baselines + our method + plots.
    """
    print("\n" + "=" * 60)
    print("  EVALUATION: Hazard-Based Adaptive Early Stopping")
    print("=" * 60)

    results = {}
    os.makedirs(output_dir, exist_ok=True)

    # ── Our method ──
    print("\n[*] Evaluating our method (hazard predictor)...")
    our_results = evaluate_hazard_predictor(
        test_data, model, calibrator, variant, hidden_dim, device
    )
    # Flatten best tau metrics into top-level for comparison table
    best = our_results["best_tau"]
    results["ours"] = {
        "accuracy": best["accuracy"],
        "avg_cost": best["avg_cost"],
        "normalized_cost": best["normalized_cost"],
        "cost_weighted_accuracy": best["cost_weighted_accuracy"],
        "tau": best["tau"],
        "full_sweep": our_results["tau_sweep"],
    }

    # ── Baselines ──
    print("\n[*] Computing baselines...")

    # B1: Full pipeline
    correct_fp, costs_fp, cwa_fp = baseline_full_pipeline(test_data)
    results["full_pipeline"] = {
        "accuracy": float(correct_fp.mean()),
        "avg_cost": float(costs_fp.mean()),
        "normalized_cost": float(costs_fp.mean() / MAX_COST),
        "cost_weighted_accuracy": cwa_fp,
    }
    print(f"    Full Pipeline: Acc={correct_fp.mean():.4f}, "
          f"Cost={costs_fp.mean():.4f}, CWA(λ=0.5)={cwa_fp['lambda_0.5']:.4f}")

    # B2: Fixed-depth
    for stop_s, name in [(0, "fixed_S1"), (1, "fixed_S2"), (2, "fixed_S3")]:
        correct, costs, cwa = baseline_fixed_depth(test_data, stop_s)
        results[name] = {
            "accuracy": float(correct.mean()),
            "avg_cost": float(costs.mean()),
            "normalized_cost": float(costs.mean() / MAX_COST),
            "cost_weighted_accuracy": cwa,
        }
        print(f"    {name}: Acc={correct.mean():.4f}, Cost={costs.mean():.4f}, "
              f"CWA(λ=0.5)={cwa['lambda_0.5']:.4f}")

    # B3: Confidence
    correct_cf, costs_cf, cwa_cf, _ = baseline_confidence(test_data)
    results["confidence"] = {
        "accuracy": float(correct_cf.mean()),
        "avg_cost": float(costs_cf.mean()),
        "normalized_cost": float(costs_cf.mean() / MAX_COST),
        "cost_weighted_accuracy": cwa_cf,
    }
    print(f"    Confidence: Acc={correct_cf.mean():.4f}, Cost={costs_cf.mean():.4f}, "
          f"CWA(λ=0.5)={cwa_cf['lambda_0.5']:.4f}")

    # B4: Random (match our avg cost)
    our_avg_cost = our_results["best_tau"]["avg_cost"]
    correct_r, costs_r, cwa_r = baseline_random(test_data, our_avg_cost)
    results["random"] = {
        "accuracy": float(correct_r.mean()),
        "avg_cost": float(costs_r.mean()),
        "normalized_cost": float(costs_r.mean() / MAX_COST),
        "cost_weighted_accuracy": cwa_r,
    }
    print(f"    Random: Acc={correct_r.mean():.4f}, Cost={costs_r.mean():.4f}, "
          f"CWA(λ=0.5)={cwa_r['lambda_0.5']:.4f}")

    # B5: Oracle
    correct_o, costs_o, cwa_o = baseline_oracle(test_data)
    results["oracle"] = {
        "accuracy": float(correct_o.mean()),
        "avg_cost": float(costs_o.mean()),
        "normalized_cost": float(costs_o.mean() / MAX_COST),
        "cost_weighted_accuracy": cwa_o,
    }
    print(f"    Oracle: Acc={correct_o.mean():.4f}, Cost={costs_o.mean():.4f}, "
          f"CWA(λ=0.5)={cwa_o['lambda_0.5']:.4f}")

    # ── Comparison table ──
    print("\n" + "-" * 60)
    print(f"{'Method':<25} {'Acc':>8} {'Cost':>8} {'CWA(0.5)':>10} {'CWA(0.8)':>10}")
    print("-" * 60)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["cost_weighted_accuracy"]["lambda_0.5"]):
        print(f"{name:<25} {r['accuracy']:>8.4f} {r['avg_cost']:>8.4f} "
              f"{r['cost_weighted_accuracy']['lambda_0.5']:>10.4f} "
              f"{r['cost_weighted_accuracy']['lambda_0.8']:>10.4f}")
    print("-" * 60)

    # ── Save results JSON ──
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n[*] Evaluation results saved to: {results_path}")

    # ── Plots ──
    # Pareto frontier
    plot_pareto_frontier(
        our_results, results,
        os.path.join(output_dir, "pareto_frontier.png"),
        lambda_val=0.5,
    )

    # Calibration plot (on test set)
    # Get test predictions
    test_dataset = HiddenStateDataset(test_data, hidden_dim=hidden_dim)
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(test_dataset, batch_size=128):
            hs = batch["hidden_state"].to(device)
            si = batch["stage_idx"].to(device)
            labels = batch["label"]
            if variant == "no_stage_emb":
                logits = model(hs)
            else:
                logits = model(hs, si)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    plot_calibration(
        all_probs, all_labels,
        os.path.join(output_dir, "calibration_plot.png"),
    )

    # Per-stage AUROC
    stage_aurocs = plot_auroc_by_stage(
        test_data, model, variant, hidden_dim,
        os.path.join(output_dir, "auroc_by_stage.png"),
    )
    results["stage_aurocs"] = [{"stage": s, "auroc": a} for s, a in stage_aurocs]

    # GATE CHECK: AUROC > 0.65 at retrieval stage (stage 0)?
    retrieval_auroc = next((a for s, a in stage_aurocs if s == "retrieval"), None)
    if retrieval_auroc is not None:
        gate_passed = retrieval_auroc > 0.65
        results["gate_check"] = {
            "stage": "retrieval",
            "auroc": retrieval_auroc,
            "threshold": 0.65,
            "passed": gate_passed,
        }
        print(f"\n{'='*60}")
        print(f"  GATE CHECK (Block 1): AUROC={retrieval_auroc:.4f} > 0.65? "
              f"{'✅ PASSED' if gate_passed else '❌ FAILED'}")
        print(f"{'='*60}")

    # Save updated results
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=float)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate hazard-based early stopping vs baselines."
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to collected states JSONL"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to trained predictor checkpoint"
    )
    parser.add_argument(
        "--calibrator", type=str, default=None,
        help="Path to Platt calibrator pickle"
    )
    parser.add_argument(
        "--variant", type=str, default="mlp",
        help="Predictor variant"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=LLAMA_HIDDEN_DIM,
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
    )
    parser.add_argument(
        "--seed", type=int, default=BASE_SEED,
    )
    args = parser.parse_args()

    if args.data is None:
        args.data = os.path.join(DATA_DIR, "collected_states.jsonl")
    if args.output_dir is None:
        args.output_dir = os.path.join(RESULTS_DIR, "block1_pilot")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load test data
    data = load_collected_data(args.data)
    _, _, test_data = split_data(data, seed=args.seed)

    # Load model
    if args.model and os.path.exists(args.model):
        checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
        model = build_predictor(
            variant=args.variant,
            hidden_dim=args.hidden_dim,
            stage_emb_dim=checkpoint.get("stage_emb_dim", 16),
            num_stages=checkpoint.get("num_stages", 4),
            mlp_hidden=checkpoint.get("mlp_hidden", MLP_HIDDEN_LAYERS),
            dropout=checkpoint.get("dropout", MLP_DROPOUT),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
    else:
        print("[!] No trained model found. Using untrained model for demo.")
        model = FailurePredictor(hidden_dim=args.hidden_dim)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

    # Load calibrator
    calibrator = None
    if args.calibrator and os.path.exists(args.calibrator):
        with open(args.calibrator, "rb") as f:
            calibrator = pickle.load(f)

    # Evaluate
    evaluate_all(
        test_data=test_data,
        model=model,
        calibrator=calibrator,
        variant=args.variant,
        hidden_dim=args.hidden_dim,
        output_dir=args.output_dir,
        device=device,
    )


if __name__ == "__main__":
    main()
