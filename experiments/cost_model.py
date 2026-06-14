"""
Unified cost model for CWA (Cost-Weighted Accuracy) computation.

This is the SINGLE SOURCE OF TRUTH for all routing cost calculations.
All scripts that compute CWA MUST import from this module.

Primary semantics (used in main text):
  Per-stage normalization: CWA(λ) = Accuracy - λ × (StopStageCost / MAX_STAGE_COST)
  where MAX_STAGE_COST = max(STAGE_COSTS) = 1.08.
  The cost of stopping at stage k is STAGE_COSTS[k] (NOT cumulative).

Cumulative variant (Appendix sensitivity only):
  CWA_cumul(λ) = Accuracy - λ × (CumulativeCost / MAX_CUMUL_COST)
  where MAX_CUMUL_COST = sum(STAGE_COSTS) = 2.58.
  Requires raw per-query stop-stage distributions for exact computation.

S0 = retrieval (2 passages), S1 = reranking (4 passages),
S2 = assembly (6 passages), S3 = generation (8 passages).
Costs include LLM prefill proportional to context length.
"""

import numpy as np

# ── Primary Cost Model (Prefill-Aware, Per-Stage Normalization) ──
STAGE_COSTS = np.array([0.25, 0.50, 0.75, 1.08])
MAX_STAGE_COST = max(STAGE_COSTS)  # 1.08 — denominator for primary CWA

# ── Cumulative Variant (Appendix sensitivity only) ──
MAX_CUMUL_COST = STAGE_COSTS.sum()  # 2.58

# ── Legacy LIGHT Model (for sensitivity appendix only) ──
STAGE_COSTS_LIGHT = np.array([0.02, 0.05, 0.01, 1.00])
MAX_STAGE_COST_LIGHT = max(STAGE_COSTS_LIGHT)  # 1.00


def stop_cost(stop_stage: int, costs: np.ndarray = STAGE_COSTS) -> float:
    """Per-stage cost of stopping at a given stage (primary semantics)."""
    return costs[stop_stage]


def cumulative_cost(stop_stage: int, costs: np.ndarray = STAGE_COSTS) -> float:
    """Cumulative cost through stop_stage (appendix sensitivity)."""
    return costs[: stop_stage + 1].sum()


def cost_weighted_accuracy(
    accuracy: float,
    avg_stop_cost: float,
    lam: float = 0.5,
    max_cost: float = MAX_STAGE_COST,
) -> float:
    """CWA(λ) = Accuracy - λ × (AvgStopCost / MaxCost).

    Args:
        accuracy: Mean accuracy over queries.
        avg_stop_cost: Mean per-stage stop cost over queries.
        lam: Cost-accuracy tradeoff weight.
        max_cost: Normalization denominator (default: MAX_STAGE_COST = 1.08).

    Returns:
        CWA value (higher = better).
    """
    return accuracy - lam * (avg_stop_cost / max_cost)


# ── Stage Name Mapping ──
STAGE_NAMES = {0: "S0 (2 passages)", 1: "S1 (4 passages)",
               2: "S2 (6 passages)", 3: "S3 (8 passages, generation)"}

# ── Validation ──
if __name__ == "__main__":
    print(f"Primary cost model (per-stage normalization):")
    print(f"  STAGE_COSTS      = {STAGE_COSTS}")
    print(f"  MAX_STAGE_COST   = {MAX_STAGE_COST}")
    print(f"  S0 stop cost     = {stop_cost(0):.2f}")
    print(f"  S3 stop cost     = {stop_cost(3):.2f}")
    print()
    print(f"Cumulative variant (appendix):")
    print(f"  MAX_CUMUL_COST   = {MAX_CUMUL_COST}")
    print(f"  S0 cumulative    = {cumulative_cost(0):.2f}")
    print(f"  S3 cumulative    = {cumulative_cost(3):.2f}")
    print()
    print("CWA examples (λ=0.5, per-stage normalization):")
    for name, acc, cost in [
        ("Fixed-S0", 0.354, stop_cost(0)),
        ("Fixed-S1", 0.422, stop_cost(1)),
        ("Full pipeline", 0.452, stop_cost(3)),
        ("Oracle", 0.508, 0.311),
    ]:
        cwa = cost_weighted_accuracy(acc, cost)
        print(f"  {name}: Acc={acc:.3f}, Cost={cost:.3f}, CWA(0.5)={cwa:.4f}")
