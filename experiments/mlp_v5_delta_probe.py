"""
MLP-on-V5 Delta Probe — Confound Disentanglement Experiment
============================================================

Isolates model capacity (MLP vs LR) from the other confounded factors in
the V4→V5 transition (sample size, label scope, evaluation methodology).

Design:
  - Same V5 data as the LR bootstrap (2000 queries, 6000 transitions)
  - Same MLP architecture as V4 ([256,128] hidden, ReLU, stage embedding)
  - Same evaluation protocol as V5/LR bootstrap (5-fold CV grouped by query,
    out-of-fold predictions, 5000 bootstrap resamples)
  - Same benefit/degradation labels as V5/LR bootstrap

Interpretation:
  - If MLP-on-V5 shows degradation > benefit (Δ ≥ 0.07):
    → Model capacity matters; LR was too weak to extract the asymmetry.
  - If MLP-on-V5 shows degradation ≈ benefit (Δ ≈ 0):
    → V4 asymmetry was sampling noise; the signal doesn't exist at scale.
  - Intermediate → partial capacity effect, signal genuinely weaker than V4.

Matches the critical missing row:
  | Protocol | Model | N    | Eval          | Ben AUROC | Deg AUROC |
  |----------|-------|------|---------------|-----------|-----------|
  | V4       | MLP   | 450  | single split  | 0.76      | 0.85      |
  | V5       | LR    | 2000 | 5-fold CV+bs  | 0.658     | 0.659     |
  | V5       | MLP   | 2000 | 5-fold CV+bs  | ← THIS    | ← ROW     |

Requirements:
  - V5 data: data/collected_states_hotpotqa_v5_2000.jsonl (or similar)
  - ~5-10 min on RTX 3080 for training across 5 folds
  - No GPU needed for LR baseline (just sklearn)
"""

import argparse
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

# ── Path setup ────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "mlp_v5_delta")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── MLP with sklearn (matching V4 architecture) ────────────────────────
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class MLPProbe(nn.Module):
    """MLP matching V4 architecture: [256, 128] hidden, ReLU, dropout 0.2,
    learned 16-dim stage embedding, ~955K params."""
    def __init__(self, input_dim: int, num_stages: int = 4,
                 hidden: List[int] = None, dropout: float = 0.2):
        super().__init__()
        if hidden is None:
            hidden = [256, 128]
        self.stage_emb = nn.Embedding(num_stages, 16)
        layers = []
        in_dim = input_dim + 16
        for h in hidden:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, stage_idx):
        s = self.stage_emb(stage_idx)
        return self.net(torch.cat([x, s], dim=-1)).squeeze(-1)


def load_v5_data(data_path: str) -> Dict:
    """Load V5 JSONL data. Returns dict with features, labels, query_ids."""
    print(f"Loading V5 data from: {data_path}")
    data = []
    with open(data_path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    print(f"  Loaded {len(data)} records")

    # Extract features and labels
    X, y_benefit, y_degradation = [], [], []
    query_ids, stage_idxs = [], []

    for d in data:
        # Hidden state: multi_layer (4×3584=14336) or final_token (3584)
        # Try multi_layer first, fall back to hidden_state
        if "multi_layer_hidden_state" in d:
            hs = np.array(d["multi_layer_hidden_state"], dtype=np.float32)
        elif "mean_pool_hidden_state" in d:
            hs = np.array(d["mean_pool_hidden_state"], dtype=np.float32)
        else:
            hs = np.array(d["hidden_state"], dtype=np.float32)

        hs = hs.flatten()  # ensure 1D

        stage = d.get("stage_idx", d.get("stage", 0))
        qid = d.get("query_id", d.get("query_idx", 0))

        # Compute benefit label: was wrong at stage t, becomes correct at t+1
        # We need stage_correctness and next_stage_correctness
        sc = d.get("stage_correctness", d.get("correct", None))
        nsc = d.get("next_stage_correctness", None)

        # If next_stage_correctness not directly available, try to infer
        if nsc is None and "stage_correctness" in d:
            # This record alone doesn't have next stage info
            # The data format should have pairs or sequences
            pass

        # For now, use the benefit/degradation labels if already computed
        benefit = d.get("benefit_label", d.get("delta_benefit", None))
        degradation = d.get("degradation_label", d.get("delta_degradation", None))

        X.append(hs)
        query_ids.append(qid)
        stage_idxs.append(stage)

        if benefit is not None:
            y_benefit.append(int(benefit))
        if degradation is not None:
            y_degradation.append(int(degradation))

    X = np.array(X, dtype=np.float32)
    query_ids = np.array(query_ids)
    stage_idxs = np.array(stage_idxs, dtype=np.int64)

    result = {"X": X, "query_ids": query_ids, "stage_idxs": stage_idxs}
    if y_benefit:
        result["y_benefit"] = np.array(y_benefit)
    if y_degradation:
        result["y_degradation"] = np.array(y_degradation)

    # Report prevalence
    for name, y in [("benefit", result.get("y_benefit")),
                     ("degradation", result.get("y_degradation"))]:
        if y is not None:
            print(f"  {name}: {y.sum()} positive / {len(y)} total "
                  f"({y.mean()*100:.1f}%)")

    return result


def bootstrap_auroc_auprc(y_true, y_pred, n_bootstrap=5000, seed=42):
    """Paired bootstrap CIs for AUROC and AUPRC."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aurocs, auprcs = [], []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        aurocs.append(roc_auc_score(y_true[idx], y_pred[idx]))
        auprcs.append(average_precision_score(y_true[idx], y_pred[idx]))
    aurocs, auprcs = np.array(aurocs), np.array(auprcs)
    return {
        "auroc_mean": aurocs.mean(),
        "auroc_ci": (np.percentile(aurocs, 2.5), np.percentile(aurocs, 97.5)),
        "auprc_mean": auprcs.mean(),
        "auprc_ci": (np.percentile(auprcs, 2.5), np.percentile(auprcs, 97.5)),
        "auroc_std": aurocs.std(),
        "auprc_std": auprcs.std(),
    }


def evaluate_lr_cv(data: Dict, target: str, n_folds: int = 5,
                   n_bootstrap: int = 5000) -> Dict:
    """Logistic regression with 5-fold CV (grouped by query) + bootstrap CIs.
    This matches the V5/LR bootstrap protocol exactly."""
    X, query_ids = data["X"], data["query_ids"]
    y = data[f"y_{target}"]

    print(f"\n{'='*60}")
    print(f"LR Baseline: {target} prediction")
    print(f"  Samples: {len(y)}, Positive: {y.sum()} ({y.mean()*100:.1f}%)")

    gkf = GroupKFold(n_splits=n_folds)
    oof_preds = np.full(len(y), np.nan)

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(X, y, groups=query_ids)
    ):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        lr = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced",
                                random_state=42)
        lr.fit(X_tr_s, y_tr)
        oof_preds[test_idx] = lr.predict_proba(X_te_s)[:, 1]

        fold_auroc = roc_auc_score(y_te, oof_preds[test_idx])
        print(f"  Fold {fold+1}: AUROC={fold_auroc:.4f}, "
              f"pos rate={y_te.mean():.3f}")

    bs = bootstrap_auroc_auprc(y, oof_preds, n_bootstrap)
    print(f"  OOF AUROC: {bs['auroc_mean']:.4f} [{bs['auroc_ci'][0]:.4f}, "
          f"{bs['auroc_ci'][1]:.4f}]")
    print(f"  OOF AUPRC: {bs['auprc_mean']:.4f} [{bs['auprc_ci'][0]:.4f}, "
          f"{bs['auprc_ci'][1]:.4f}]")
    print(f"  Random AUPRC baseline: {y.mean():.4f}")

    return {
        "target": target,
        "model": "LR",
        "n_samples": len(y),
        "n_positive": int(y.sum()),
        "prevalence": float(y.mean()),
        "auroc_mean": float(bs["auroc_mean"]),
        "auroc_ci": [float(x) for x in bs["auroc_ci"]],
        "auprc_mean": float(bs["auprc_mean"]),
        "auprc_ci": [float(x) for x in bs["auprc_ci"]],
        "auroc_std": float(bs["auroc_std"]),
        "random_auprc": float(y.mean()),
    }


def evaluate_mlp_cv(data: Dict, target: str, n_folds: int = 5,
                    n_bootstrap: int = 5000, n_epochs: int = 50,
                    batch_size: int = 128, lr: float = 1e-4,
                    patience: int = 10, seed: int = 42) -> Dict:
    """MLP with 5-fold CV + bootstrap CIs.
    Same architecture as V4, matched CV protocol as V5/LR."""
    if not HAS_TORCH:
        print("ERROR: PyTorch not available. Cannot run MLP.")
        return None

    X, query_ids = data["X"], data["query_ids"]
    stage_idxs = data["stage_idxs"]
    y = data[f"y_{target}"]

    print(f"\n{'='*60}")
    print(f"MLP on V5: {target} prediction")
    print(f"  Input dim: {X.shape[1]}, Samples: {len(y)}, "
          f"Positive: {y.sum()} ({y.mean()*100:.1f}%)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    gkf = GroupKFold(n_splits=n_folds)
    oof_preds = np.full(len(y), np.nan)

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(X, y, groups=query_ids)
    ):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        s_tr = torch.tensor(stage_idxs[train_idx], dtype=torch.long, device=device)
        s_te = torch.tensor(stage_idxs[test_idx], dtype=torch.long, device=device)

        # Standardize
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # To torch
        X_tr_t = torch.tensor(X_tr_s, dtype=torch.float32, device=device)
        X_te_t = torch.tensor(X_te_s, dtype=torch.float32, device=device)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)

        # Build model (matching V4: [256, 128], ReLU, dropout 0.2)
        model = MLPProbe(X.shape[1], num_stages=4).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([(len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)],
                                    device=device)
        )

        # Train with early stopping
        best_val_auroc, best_state = 0, None
        patience_counter = 0

        # Use 15% of training as validation
        np.random.seed(seed + fold)
        val_mask = np.random.choice(len(X_tr_s), int(0.15 * len(X_tr_s)),
                                     replace=False)
        train_mask = np.ones(len(X_tr_s), dtype=bool)
        train_mask[val_mask] = False

        for epoch in range(n_epochs):
            model.train()
            # Mini-batch training
            perm = torch.randperm(train_mask.sum(), device=device)
            for i in range(0, train_mask.sum(), batch_size):
                idx = perm[i:i+batch_size]
                opt.zero_grad()
                logits = model(X_tr_t[train_mask][idx],
                               s_tr[train_mask][idx])
                loss = loss_fn(logits, y_tr_t[train_mask][idx])
                loss.backward()
                opt.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_logits = model(X_tr_t[val_mask], s_tr[val_mask])
                val_preds = torch.sigmoid(val_logits).cpu().numpy()
                val_auroc = roc_auc_score(y_tr[val_mask], val_preds)
                if val_auroc > best_val_auroc:
                    best_val_auroc = val_auroc
                    best_state = {k: v.cpu().clone()
                                  for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

            if patience_counter >= patience:
                break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            te_logits = model(X_te_t, s_te)
            oof_preds[test_idx] = torch.sigmoid(te_logits).cpu().numpy()

        print(f"  Fold {fold+1}: AUROC={roc_auc_score(y_te, oof_preds[test_idx]):.4f}, "
              f"best_val_auroc={best_val_auroc:.4f}, epochs={epoch+1}")

    bs = bootstrap_auroc_auprc(y, oof_preds, n_bootstrap)
    print(f"  OOF AUROC: {bs['auroc_mean']:.4f} [{bs['auroc_ci'][0]:.4f}, "
          f"{bs['auroc_ci'][1]:.4f}]")
    print(f"  OOF AUPRC: {bs['auprc_mean']:.4f} [{bs['auprc_ci'][0]:.4f}, "
          f"{bs['auprc_ci'][1]:.4f}]")
    print(f"  Random AUPRC baseline: {y.mean():.4f}")

    return {
        "target": target,
        "model": "MLP (V4 architecture on V5 data)",
        "n_samples": len(y),
        "n_positive": int(y.sum()),
        "prevalence": float(y.mean()),
        "auroc_mean": float(bs["auroc_mean"]),
        "auroc_ci": [float(x) for x in bs["auroc_ci"]],
        "auprc_mean": float(bs["auprc_mean"]),
        "auprc_ci": [float(x) for x in bs["auprc_ci"]],
        "auroc_std": float(bs["auroc_std"]),
        "random_auprc": float(y.mean()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="MLP-on-V5 Delta Probe — Confound Disentanglement")
    parser.add_argument("--data", type=str,
                        default=os.path.join(DATA_DIR,
                            "collected_states_hotpotqa_v5_2000.jsonl"),
                        help="Path to V5 JSONL data")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--lr-only", action="store_true",
                        help="Run only LR baseline (no GPU needed)")
    parser.add_argument("--mlp-only", action="store_true",
                        help="Run only MLP (skip LR)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--output", type=str,
                        default=os.path.join(RESULTS_DIR,
                            "mlp_v5_delta_results.json"))
    args = parser.parse_args()

    if not os.path.exists(args.data):
        # Try alternate paths
        alternates = [
            os.path.join(DATA_DIR, "collected_states_hotpotqa_v5.jsonl"),
            os.path.join(DATA_DIR, "hotpotqa_v5_states.jsonl"),
            os.path.join(PROJECT_DIR, "results", "hotpotqa_v5",
                         "collected_states.jsonl"),
            "/home/shenjikun/experiments/hazard-early-stopping/data/"
            "collected_states_hotpotqa_v5_2000.jsonl",
        ]
        found = False
        for alt in alternates:
            if os.path.exists(alt):
                args.data = alt
                found = True
                break
        if not found:
            print(f"ERROR: V5 data not found at {args.data}")
            print("Tried alternates:", alternates)
            print("\nPlease collect V5 data first with collect_v5_multilayer.py")
            sys.exit(1)

    # Load data
    data = load_v5_data(args.data)

    # Check if benefit/degradation labels exist
    has_benefit = "y_benefit" in data
    has_degrad = "y_degradation" in data

    if not has_benefit and not has_degrad:
        print("\nERROR: Data does not contain benefit or degradation labels.")
        print("Expected fields: benefit_label/delta_benefit or "
              "degradation_label/delta_degradation")
        print("\nAvailable fields:", list(data.keys()))
        print("\nNeed to compute Delta labels from stage_correctness. "
              "Run bootstrap_analysis.py first to generate V5 transition data.")
        sys.exit(1)

    results = []

    # LR baseline (matching V5 bootstrap)
    if not args.mlp_only:
        for target in (["benefit"] if has_benefit else []):
            r = evaluate_lr_cv(data, target, args.n_folds, args.n_bootstrap)
            if r:
                results.append(r)
        for target in (["degradation"] if has_degrad else []):
            r = evaluate_lr_cv(data, target, args.n_folds, args.n_bootstrap)
            if r:
                results.append(r)

    # MLP on V5 (the critical experiment)
    if not args.lr_only:
        if not HAS_TORCH:
            print("\nWARNING: PyTorch not available, skipping MLP. "
                  "Install with: pip install torch")
        else:
            for target in (["benefit"] if has_benefit else []):
                r = evaluate_mlp_cv(data, target, args.n_folds,
                                    args.n_bootstrap, args.epochs)
                if r:
                    results.append(r)
            for target in (["degradation"] if has_degrad else []):
                r = evaluate_mlp_cv(data, target, args.n_folds,
                                    args.n_bootstrap, args.epochs)
                if r:
                    results.append(r)

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY: V5 Delta Probe — MLP vs LR")
    print(f"{'Target':<15} {'Model':<5} {'AUROC':>8} {'95% CI':>22} "
          f"{'AUPRC':>8} {'n_pos':>6}")
    print("-" * 70)
    for r in results:
        print(f"{r['target']:<15} {r['model']:<5} "
              f"{r['auroc_mean']:>8.4f} "
              f"[{r['auroc_ci'][0]:.4f}, {r['auroc_ci'][1]:.4f}] "
              f"{r['auprc_mean']:>8.4f} "
              f"{r['n_positive']:>6d}")

    # ── Key comparison ─────────────────────────────────────────────────
    lr_results = {r["target"]: r for r in results if r["model"] == "LR"}
    mlp_results = {r["target"]: r for r in results if "MLP" in r["model"]}

    print(f"\n{'='*60}")
    print("DISENTANGLEMENT: Does MLP capacity restore the V4 asymmetry?")
    for target in ["benefit", "degradation"]:
        if target in lr_results and target in mlp_results:
            lr_auroc = lr_results[target]["auroc_mean"]
            mlp_auroc = mlp_results[target]["auroc_mean"]
            delta = mlp_auroc - lr_auroc
            print(f"  {target}: LR={lr_auroc:.4f}, MLP={mlp_auroc:.4f}, "
                  f"Δ={delta:+.4f}")

    if "benefit" in mlp_results and "degradation" in mlp_results:
        mlp_ben = mlp_results["benefit"]["auroc_mean"]
        mlp_deg = mlp_results["degradation"]["auroc_mean"]
        asymmetry = mlp_deg - mlp_ben
        print(f"\n  MLP asymmetry (deg - ben): {asymmetry:+.4f}")
        if asymmetry >= 0.07:
            print("  → HYPOTHESIS (ii): Capacity matters. "
                  "LR was too weak; asymmetry IS real.")
        elif asymmetry <= 0.02:
            print("  → HYPOTHESIS (i): V4 noise. "
                  "The 0.85 was an underpowered artifact; asymmetry does "
                  "NOT exist at scale.")
        else:
            print("  → INTERMEDIATE: Partial capacity effect. "
                  "Asymmetry is weaker than V4 suggested but not entirely "
                  "absent.")

    # Save results
    with open(args.output, "w") as f:
        json.dump({
            "results": results,
            "config": {
                "data": args.data,
                "n_folds": args.n_folds,
                "n_bootstrap": args.n_bootstrap,
                "v4_reference": {
                    "benefit_auroc_mlp": 0.76,
                    "degradation_auroc_mlp": 0.85,
                    "n_queries": 450,
                    "n_transitions": 1800,
                },
                "v5_lr_reference": {
                    "benefit_auroc_lr": 0.658,
                    "degradation_auroc_lr": 0.659,
                    "n_queries": 2000,
                    "n_transitions": 6000,
                },
            },
        }, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
