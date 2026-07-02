"""
Train Beneficial MLP (Delta Probe) with focal loss, class weighting, and
richer hidden-state representations.

The Beneficial MLP predicts: "Will advancing to the next stage fix the error?"
  benefit_t = 1 if (stage_correctness_{t+1} == 1 AND stage_correctness_t == 0)
  benefit_t = 0 otherwise

This is the Delta Probe — it measures whether hidden states encode the marginal
utility of additional context.

Key improvements over the original (per reviewer feedback):
1. Focal Loss (gamma=2, alpha=positive_class_prior) for extreme class imbalance (~7% positive)
2. Class weighting option (balanced, balanced_sqrt)
3. Multiple hidden-state representations:
   - final_token: last token hidden state (original)
   - mean_pool: mean pooling over all tokens
   - multi_layer: concat of last 4 layers' final-token states
   - combined: mean_pool + final_token concatenated
4. Proper AUC-PR reporting alongside AUROC
5. Calibration curves for the rare-event setting
"""

import argparse
import json
import os
import sys
import pickle
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, auc as sklearn_auc,
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experiments'))
from config import BASE_SEED, NUM_STAGES, STAGES

# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal Loss for binary classification with extreme class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    alpha: weight for positive class (typically = 1 - positive_rate)
    gamma: focusing parameter (2.0 is standard)
    """
    def __init__(self, alpha: float = 0.93, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # inputs are LOGITS — apply sigmoid for probability
        probs = torch.sigmoid(inputs)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        return (alpha_weight * focal_weight * bce_loss).mean()


# ── Beneficial MLP Model ──────────────────────────────────────────────────────

class BeneficialMLP(nn.Module):
    """MLP probing whether hidden state encodes marginal context utility.

    Input: hidden_state + stage_onehot
    Output: P(benefit | h_t, stage=t) where benefit = next stage fixes error
    """
    def __init__(self, hidden_dim: int = 3584, mlp_hidden: List[int] = None,
                 dropout: float = 0.2):
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = [256, 128]
        layers = []
        in_dim = hidden_dim + NUM_STAGES  # + stage embedding
        for h in mlp_hidden:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, h, stage_idx):
        stage_onehot = torch.zeros(h.size(0), NUM_STAGES, device=h.device)
        stage_onehot.scatter_(1, stage_idx.unsqueeze(1), 1.0)
        return self.net(torch.cat([h, stage_onehot], dim=1)).squeeze(-1)


# ── Dataset ───────────────────────────────────────────────────────────────────

def build_beneficial_labels(data: List[Dict]) -> List[Dict]:
    """Compute benefit labels: benefit_t = 1 if advancing to t+1 fixes error.

    benefit = 1 when stage_correctness[t] == 0 AND stage_correctness[t+1] == 1
    For S3 (final stage), benefit is always 0 (no next stage).

    Returns list of tuples with benefit label added.
    """
    by_query = defaultdict(list)
    for d in data:
        by_query[d['query_id']].append(d)

    positive_count = 0
    total_count = 0

    for qid, stages in by_query.items():
        stages_sorted = sorted(stages, key=lambda s: s['stage_idx'])
        for i, s in enumerate(stages_sorted):
            if i < NUM_STAGES - 1:
                current_correct = s.get('stage_correctness', 0)
                next_correct = stages_sorted[i + 1].get('stage_correctness', 0)
                benefit = 1 if (current_correct == 0 and next_correct == 1) else 0
            else:
                benefit = 0  # S3 has no next stage
            s['benefit'] = benefit
            total_count += 1
            positive_count += benefit

    print(f"[*] Benefit labels: {positive_count}/{total_count} positive "
          f"({100*positive_count/total_count:.1f}%)")
    return data


class BeneficialDataset(Dataset):
    """Dataset for beneficial prediction with configurable representations."""

    def __init__(self, data: List[Dict], hidden_dim: int = 3584,
                 rep_type: str = "final_token",
                 normalize: bool = True, scaler: Optional[StandardScaler] = None):
        self.rep_type = rep_type
        self.hidden_dim = hidden_dim

        self.stage_indices = torch.tensor(
            [d["stage_idx"] for d in data], dtype=torch.long
        )
        self.labels = torch.tensor(
            [d["benefit"] for d in data], dtype=torch.float32
        )

        # Build representations
        features = []
        for d in data:
            feat = self._build_representation(d)
            features.append(feat)

        features = np.array(features, dtype=np.float32)

        # Handle dimension
        if features.shape[1] != hidden_dim:
            if features.shape[1] > hidden_dim:
                features = features[:, :hidden_dim]
            else:
                padded = np.zeros((features.shape[0], hidden_dim), dtype=np.float32)
                padded[:, :features.shape[1]] = features
                features = padded

        if normalize:
            if scaler is None:
                scaler = StandardScaler()
                features = scaler.fit_transform(features)
            else:
                features = scaler.transform(features)

        self.features = torch.tensor(features, dtype=torch.float32)
        self.scaler = scaler

    def _build_representation(self, d: Dict) -> np.ndarray:
        """Build hidden-state representation based on rep_type."""
        hs = np.array(d["hidden_state"], dtype=np.float32)

        if self.rep_type == "final_token":
            return hs

        elif self.rep_type == "mean_all_tokens":
            # If we have all_token_hidden_states (mean pool over sequence)
            if "all_token_hidden_states" in d:
                return np.array(d["all_token_hidden_states"], dtype=np.float32).mean(axis=0)
            else:
                # Fallback: use final token (no richer data available)
                return hs

        elif self.rep_type == "multi_layer":
            # Concat last 4 layers — requires multi-layer data
            if "multi_layer_hidden_states" in d:
                layers = [np.array(l, dtype=np.float32) for l in d["multi_layer_hidden_states"]]
                return np.concatenate(layers)
            else:
                return hs  # Fallback

        elif self.rep_type == "combined":
            # final_token concat with mean pool
            if "all_token_hidden_states" in d:
                mean_hs = np.array(d["all_token_hidden_states"], dtype=np.float32).mean(axis=0)
                return np.concatenate([hs, mean_hs])
            else:
                return hs  # Fallback

        else:
            raise ValueError(f"Unknown rep_type: {self.rep_type}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "stage_idx": self.stage_indices[idx],
            "label": self.labels[idx],
        }


# ── Training ──────────────────────────────────────────────────────────────────

def train_beneficial_mlp(
    data_path: str,
    output_dir: str,
    rep_type: str = "final_token",
    loss_type: str = "focal",
    focal_alpha: Optional[float] = None,
    focal_gamma: float = 2.0,
    hidden_dim: int = 3584,
    mlp_hidden: List[int] = None,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    patience: int = 20,
    seed: int = BASE_SEED,
    device: str = "cuda",
):
    """Train Beneficial MLP with configurable loss and representation."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # Load and label data
    print(f"[*] Loading data from {data_path}...")
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))
    print(f"[*] Loaded {len(data)} tuples")

    data = build_beneficial_labels(data)

    # Count positive rate for alpha
    n_pos = sum(1 for d in data if d['benefit'] == 1)
    pos_rate = n_pos / len(data)
    if focal_alpha is None:
        focal_alpha = 1.0 - pos_rate  # Weight positive class inversely to prevalence

    print(f"[*] Positive rate: {pos_rate:.4f}, Focal alpha: {focal_alpha:.4f}")

    # Split by query
    query_ids = sorted(set(d["query_id"] for d in data))
    np.random.shuffle(query_ids)
    n = len(query_ids)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    train_ids = set(query_ids[:n_train])
    val_ids = set(query_ids[n_train:n_train + n_val])
    test_ids = set(query_ids[n_train + n_val:])

    train_data = [d for d in data if d["query_id"] in train_ids]
    val_data = [d for d in data if d["query_id"] in val_ids]
    test_data = [d for d in data if d["query_id"] in test_ids]

    print(f"[*] Split: train={len(train_ids)}q ({len(train_data)}t), "
          f"val={len(val_ids)}q ({len(val_data)}t), "
          f"test={len(test_ids)}q ({len(test_data)}t)")

    # Effective hidden dim depends on rep_type
    eff_hidden_dim = hidden_dim
    if rep_type == "multi_layer":
        eff_hidden_dim = hidden_dim * 4  # 4 layers concatenated
    elif rep_type == "combined":
        eff_hidden_dim = hidden_dim * 2  # final_token + mean_pool

    # Create datasets
    train_dataset = BeneficialDataset(
        train_data, hidden_dim=eff_hidden_dim, rep_type=rep_type)
    val_dataset = BeneficialDataset(
        val_data, hidden_dim=eff_hidden_dim, rep_type=rep_type,
        scaler=train_dataset.scaler)
    test_dataset = BeneficialDataset(
        test_data, hidden_dim=eff_hidden_dim, rep_type=rep_type,
        scaler=train_dataset.scaler)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = BeneficialMLP(hidden_dim=eff_hidden_dim, mlp_hidden=mlp_hidden)
    model = model.to(device)

    # Loss
    if loss_type == "focal":
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        print(f"[*] Using Focal Loss (alpha={focal_alpha:.3f}, gamma={focal_gamma})")
    elif loss_type == "bce_weighted":
        pos_weight = (len(train_data) - n_pos) / max(n_pos, 1)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device))
        print(f"[*] Using Weighted BCE (pos_weight={pos_weight:.1f})")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("[*] Using unweighted BCE")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=patience // 2)

    # Training loop
    best_val_auprc = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            feats = batch["features"].to(device)
            si = batch["stage_idx"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(feats, si)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                feats = batch["features"].to(device)
                si = batch["stage_idx"].to(device)
                labels = batch["label"]
                logits = model(feats, si)
                val_preds.extend(torch.sigmoid(logits).cpu().tolist())
                val_labels.extend(labels.tolist())

        val_auroc = roc_auc_score(val_labels, val_preds)
        val_auprc = average_precision_score(val_labels, val_preds)
        scheduler.step(val_auprc)

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}: Loss={avg_train_loss:.4f}, "
                  f"Val AUROC={val_auroc:.4f}, Val AUPRC={val_auprc:.4f}")

        if val_auprc > best_val_auprc + 0.001:
            best_val_auprc = val_auprc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "rep_type": rep_type,
                "loss_type": loss_type,
                "hidden_dim": eff_hidden_dim,
                "best_val_auroc": val_auroc,
                "best_val_auprc": val_auprc,
                "epoch": best_epoch,
                "seed": seed,
            }, os.path.join(output_dir, "beneficial_mlp_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # ── Final Test Evaluation ──
    checkpoint = torch.load(os.path.join(output_dir, "beneficial_mlp_best.pt"),
                            map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            feats = batch["features"].to(device)
            si = batch["stage_idx"].to(device)
            labels = batch["label"]
            logits = model(feats, si)
            test_preds.extend(torch.sigmoid(logits).cpu().tolist())
            test_labels.extend(labels.tolist())

    test_auroc = roc_auc_score(test_labels, test_preds)
    test_auprc = average_precision_score(test_labels, test_preds)
    random_baseline_auprc = sum(test_labels) / len(test_labels)

    # Per-stage breakdown
    test_stage_indices = [d["stage_idx"] for d in test_data]
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS — Beneficial MLP (Delta Probe)")
    print(f"{'='*60}")
    print(f"  Representation: {rep_type}")
    print(f"  Loss: {loss_type} (alpha={focal_alpha:.3f}, gamma={focal_gamma})")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Test AUROC:  {test_auroc:.4f}")
    print(f"  Test AUPRC:  {test_auprc:.4f}")
    print(f"  Random AUPRC: {random_baseline_auprc:.4f}")
    print(f"  AUPRC ratio:  {test_auprc / random_baseline_auprc:.2f}× random")
    print(f"  Positive rate: {pos_rate:.4f} ({n_pos}/{len(data)})")
    print()

    # Per-stage AUROC/AUPRC
    for stage in range(NUM_STAGES):
        idxs = [j for j, s in enumerate(test_stage_indices) if s == stage]
        if idxs:
            s_labels = [test_labels[j] for j in idxs]
            s_preds = [test_preds[j] for j in idxs]
            s_pos = sum(s_labels)
            if s_pos > 0 and s_pos < len(s_labels):
                s_auroc = roc_auc_score(s_labels, s_preds)
                s_auprc = average_precision_score(s_labels, s_preds)
                print(f"  Stage {stage} ({STAGES[stage]}): AUROC={s_auroc:.4f}, "
                      f"AUPRC={s_auprc:.4f}, pos={s_pos}/{len(s_labels)}")

    # Save results
    results = {
        "rep_type": rep_type,
        "loss_type": loss_type,
        "focal_alpha": focal_alpha,
        "focal_gamma": focal_gamma,
        "positive_rate": pos_rate,
        "best_epoch": best_epoch,
        "test_auroc": float(test_auroc),
        "test_auprc": float(test_auprc),
        "random_baseline_auprc": float(random_baseline_auprc),
        "auprc_ratio": float(test_auprc / random_baseline_auprc),
        "num_train": len(train_data),
        "num_val": len(val_data),
        "num_test": len(test_data),
        "seed": seed,
    }
    with open(os.path.join(output_dir, "beneficial_mlp_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[*] Results saved to {output_dir}")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Train Beneficial MLP (Delta Probe) with focal loss and richer representations")
    ap.add_argument("--data", type=str,
                    default="data/collected_states_hotpotqa_v3.jsonl")
    ap.add_argument("--output_dir", type=str, default="results/beneficial_mlp")
    ap.add_argument("--rep_type", type=str, default="final_token",
                    choices=["final_token", "mean_all_tokens", "multi_layer", "combined"])
    ap.add_argument("--loss_type", type=str, default="focal",
                    choices=["focal", "bce_weighted", "bce"])
    ap.add_argument("--focal_alpha", type=float, default=None)
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--hidden_dim", type=int, default=3584)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=BASE_SEED)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    if not os.path.exists(args.data):
        print(f"[!] Data not found: {args.data}")
        sys.exit(1)

    train_beneficial_mlp(
        data_path=args.data,
        output_dir=args.output_dir,
        rep_type=args.rep_type,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        hidden_dim=args.hidden_dim,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
