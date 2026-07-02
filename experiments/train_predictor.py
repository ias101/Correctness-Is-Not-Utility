"""
Train failure predictor MLP on collected hidden states.

Supports:
  - Base MLP (Block 1.3, 2.2)
  - Logistic regression baseline (Block 1.2)
  - All ablation variants (Block 3):
    - Stage-specific classifiers (3.1)
    - LSTM recurrence (3.2)
    - No stage embedding (3.3)
    - +Retrieval features (3.4)

Training procedure (from FINAL_PROPOSAL.md):
  1. Class-balanced sampling (equal pos/neg per batch)
  2. Binary cross-entropy loss
  3. Early stopping on validation AUROC
  4. Platt scaling calibration on validation set
  5. Save model checkpoint + results JSON
"""

import argparse
import json
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
import pickle


class TemperatureCalibrator:
    """Temperature scaling calibrator — preserves probability variance for routing."""
    def __init__(self, T: float = 1.0):
        self.T = T

    def predict(self, probs_array):
        """Apply temperature scaling: p → sigmoid(logit(p) / T)."""
        import numpy as _np
        p = _np.asarray(probs_array, dtype=_np.float32)
        p = _np.clip(p, 1e-7, 1 - 1e-7)
        logits = _np.log(p / (1 - p))
        return 1.0 / (1.0 + _np.exp(-logits / self.T))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    LLAMA_HIDDEN_DIM,
    STAGE_EMBEDDING_DIM,
    NUM_STAGES,
    MLP_HIDDEN_LAYERS,
    MLP_DROPOUT,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    BATCH_SIZE_TRAIN,
    CLASS_BALANCED_SAMPLING,
    SEEDS,
    BASE_SEED,
    DATA_DIR,
    RESULTS_DIR,
    MODEL_DIR,
)
from models import build_predictor
from collect_states import load_collected_data


def set_seed(seed: int = BASE_SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class HiddenStateDataset(Dataset):
    """Dataset for (hidden_state, stage, label) tuples.

    label_type="final": Predict FINAL (S3) correctness from any stage.
        Measures whether early signals predict eventual outcome.
        Used for routing: "should I continue to get more context?"
    label_type="stage": Predict stage-specific correctness from that stage.
        Cleaner probe design — predicts "can I answer with current context?"
        Used for failure detection: "is my current answer wrong?"
    """

    def __init__(
        self,
        data: List[Dict],
        hidden_dim: int = LLAMA_HIDDEN_DIM,
        normalize: bool = True,
        scaler: Optional[StandardScaler] = None,
        label_type: str = "final",
    ):
        self.hidden_dim = hidden_dim
        self.label_type = label_type
        self.stage_indices = torch.tensor(
            [d["stage_idx"] for d in data], dtype=torch.long
        )

        # Use stage_correctness or final_correctness based on label_type
        label_key = "stage_correctness" if label_type == "stage" else "final_correctness"
        self.labels = torch.tensor(
            [d[label_key] for d in data], dtype=torch.float32
        )

        # Convert hidden states to tensor
        hidden_states = np.array(
            [d["hidden_state"] for d in data], dtype=np.float32
        )

        # Handle dimension mismatch
        if hidden_states.shape[1] != hidden_dim:
            if hidden_states.shape[1] > hidden_dim:
                hidden_states = hidden_states[:, :hidden_dim]
            else:
                padded = np.zeros((hidden_states.shape[0], hidden_dim), dtype=np.float32)
                padded[:, :hidden_states.shape[1]] = hidden_states
                hidden_states = padded

        # Normalize
        if normalize:
            if scaler is None:
                scaler = StandardScaler()
                hidden_states = scaler.fit_transform(hidden_states)
            else:
                hidden_states = scaler.transform(hidden_states)

        self.hidden_states = torch.tensor(hidden_states, dtype=torch.float32)
        self.scaler = scaler

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "hidden_state": self.hidden_states[idx],
            "stage_idx": self.stage_indices[idx],
            "label": self.labels[idx],
        }


def split_data(
    data: List[Dict],
    train_frac: float = TRAIN_SPLIT,
    val_frac: float = VAL_SPLIT,
    test_frac: float = TEST_SPLIT,
    seed: int = BASE_SEED,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split data by QUERY (not tuple) to avoid data leakage.
    All tuples from the same query go to the same split.
    """
    # Group by query_id
    query_ids = list(set(d["query_id"] for d in data))
    np.random.seed(seed)
    np.random.shuffle(query_ids)

    n_total = len(query_ids)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)

    train_ids = set(query_ids[:n_train])
    val_ids = set(query_ids[n_train : n_train + n_val])
    test_ids = set(query_ids[n_train + n_val :])

    train_data = [d for d in data if d["query_id"] in train_ids]
    val_data = [d for d in data if d["query_id"] in val_ids]
    test_data = [d for d in data if d["query_id"] in test_ids]

    print(f"[*] Data split by query: train={len(train_ids)} queries "
          f"({len(train_data)} tuples), val={len(val_ids)} queries "
          f"({len(val_data)} tuples), test={len(test_ids)} queries "
          f"({len(test_data)} tuples)")

    return train_data, val_data, test_data


def train_logistic_regression(
    train_data: List[Dict],
    val_data: List[Dict],
    results_path: str,
) -> Dict:
    """
    Block 1.2: Train logistic regression baseline.

    Uses sklearn LogisticRegression with class balancing.
    """
    print("\n[*] Training Logistic Regression baseline...")

    # Prepare data
    X_train = np.array([d["hidden_state"] for d in train_data], dtype=np.float32)
    y_train = np.array([d["final_correctness"] for d in train_data])

    X_val = np.array([d["hidden_state"] for d in val_data], dtype=np.float32)
    y_val = np.array([d["final_correctness"] for d in val_data])

    # Handle dimension
    hidden_dim = LLAMA_HIDDEN_DIM
    if X_train.shape[1] > hidden_dim:
        X_train = X_train[:, :hidden_dim]
        X_val = X_val[:, :hidden_dim]
    elif X_train.shape[1] < hidden_dim:
        padded = np.zeros((X_train.shape[0], hidden_dim), dtype=np.float32)
        padded[:, :X_train.shape[1]] = X_train
        X_train = padded
        padded = np.zeros((X_val.shape[0], hidden_dim), dtype=np.float32)
        padded[:, :X_val.shape[1]] = X_val
        X_val = padded

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Train
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=BASE_SEED,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred_proba = clf.predict_proba(X_val)[:, 1]
    y_pred = clf.predict(X_val)

    val_auroc = roc_auc_score(y_val, y_pred_proba)
    val_acc = accuracy_score(y_val, y_pred)

    print(f"    Logistic Regression: AUROC={val_auroc:.4f}, Acc={val_acc:.4f}")

    # Save model
    model_path = os.path.join(MODEL_DIR, "logistic_regression.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": clf, "scaler": scaler}, f)

    results = {
        "model": "logistic_regression",
        "val_auroc": float(val_auroc),
        "val_accuracy": float(val_acc),
        "model_path": model_path,
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def train_mlp(
    train_data: List[Dict],
    val_data: List[Dict],
    results_path: str,
    model_path: str,
    variant: str = "mlp",
    hidden_dim: int = LLAMA_HIDDEN_DIM,
    stage_emb_dim: int = STAGE_EMBEDDING_DIM,
    num_stages: int = NUM_STAGES,
    mlp_hidden: List[int] = None,
    dropout: float = MLP_DROPOUT,
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    patience: int = EARLY_STOPPING_PATIENCE,
    batch_size: int = BATCH_SIZE_TRAIN,
    class_balanced: bool = CLASS_BALANCED_SAMPLING,
    seed: int = BASE_SEED,
    device: str = "cuda",
    label_type: str = "final",
) -> Dict:
    """
    Train MLP failure predictor (any variant).

    Training includes:
    - Class-balanced sampling
    - Early stopping on validation AUROC
    - Best model checkpoint saving
    """
    if mlp_hidden is None:
        mlp_hidden = MLP_HIDDEN_LAYERS

    print(f"\n[*] Training {variant.upper()} predictor (seed={seed})...")

    set_seed(seed)

    # Create datasets
    train_dataset = HiddenStateDataset(train_data, hidden_dim=hidden_dim, label_type=label_type)
    val_dataset = HiddenStateDataset(
        val_data,
        hidden_dim=hidden_dim,
        scaler=train_dataset.scaler,
        label_type=label_type,
    )

    # Class-balanced sampling (use EITHER sampler OR pos_weight, not both)
    if class_balanced:
        labels = train_dataset.labels.numpy()
        n_pos = labels.sum()
        n_neg = len(labels) - n_pos
        sample_weights = np.where(labels == 1, len(labels) / max(n_pos, 1), len(labels) / max(n_neg, 1))
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        pos_weight_tensor = None  # Sampler handles balancing
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler
        )
    else:
        sampler = None
        n_pos = train_dataset.labels.sum().item()
        n_neg = len(train_dataset.labels) - n_pos
        pos_weight_tensor = torch.tensor([n_neg / max(n_pos, 1)], device=device)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Build model
    model = build_predictor(
        variant=variant,
        hidden_dim=hidden_dim,
        stage_emb_dim=stage_emb_dim,
        num_stages=num_stages,
        mlp_hidden=mlp_hidden,
        dropout=dropout,
    )
    model = model.to(device)

    # Loss and optimizer
    # pos_weight_tensor is set above based on class_balanced flag
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=patience // 2
    )

    # Training loop
    best_val_auroc = 0.0
    best_epoch = 0
    patience_counter = 0
    train_history = []

    for epoch in range(num_epochs):
        # ── Train ──
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for batch in train_loader:
            hs = batch["hidden_state"].to(device)
            si = batch["stage_idx"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            if variant == "no_stage_emb":
                logits = model(hs)
            else:
                logits = model(hs, si)

            loss = criterion(logits.squeeze(-1), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(
                torch.sigmoid(logits.squeeze(-1)).detach().cpu().tolist()
            )
            train_labels.extend(labels.cpu().tolist())

        avg_train_loss = train_loss / len(train_loader)
        train_auroc = roc_auc_score(train_labels, train_preds)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                hs = batch["hidden_state"].to(device)
                si = batch["stage_idx"].to(device)
                labels = batch["label"].to(device)

                if variant == "no_stage_emb":
                    logits = model(hs)
                else:
                    logits = model(hs, si)

                loss = criterion(logits.squeeze(-1), labels)
                val_loss += loss.item()
                val_preds.extend(
                    torch.sigmoid(logits.squeeze(-1)).cpu().tolist()
                )
                val_labels.extend(labels.cpu().tolist())

        avg_val_loss = val_loss / len(val_loader)
        val_auroc = roc_auc_score(val_labels, val_preds)

        # LR scheduling
        scheduler.step(val_auroc)

        # Logging
        train_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_auroc": train_auroc,
                "val_loss": avg_val_loss,
                "val_auroc": val_auroc,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        if (epoch + 1) % 10 == 0:
            print(
                f"    Epoch {epoch+1:3d}/{num_epochs}: "
                f"Train Loss={avg_train_loss:.4f}, Train AUROC={train_auroc:.4f}, "
                f"Val Loss={avg_val_loss:.4f}, Val AUROC={val_auroc:.4f}"
            )

        # Early stopping
        if val_auroc > best_val_auroc + 0.001:
            best_val_auroc = val_auroc
            best_epoch = epoch + 1
            patience_counter = 0
            # Save best model
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "variant": variant,
                    "hidden_dim": hidden_dim,
                    "stage_emb_dim": stage_emb_dim,
                    "num_stages": num_stages,
                    "mlp_hidden": mlp_hidden,
                    "dropout": dropout,
                    "best_epoch": best_epoch,
                    "best_val_auroc": best_val_auroc,
                    "seed": seed,
                },
                model_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1} "
                      f"(best: epoch {best_epoch}, AUROC={best_val_auroc:.4f})")
                break

    # ── Results ──
    results = {
        "variant": variant,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_auroc": float(best_val_auroc),
        "final_train_loss": avg_train_loss,
        "final_val_loss": avg_val_loss,
        "model_path": model_path,
        "train_history": train_history,
        "num_params": sum(p.numel() for p in model.parameters()),
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"    {variant.upper()} best Val AUROC={best_val_auroc:.4f} "
          f"({results['num_params']:,} params)")
    return results


def calibrate_platt(
    model: nn.Module,
    val_data: List[Dict],
    variant: str = "mlp",
    hidden_dim: int = LLAMA_HIDDEN_DIM,
    device: str = "cuda",
    label_type: str = "final",
) -> Tuple:
    """
    Apply temperature scaling (Platt scaling) on validation logits.

    Temperature scaling uses a single scalar T to calibrate:
        p_calibrated = sigmoid(logit / T)

    Unlike isotonic regression, this preserves the relative ordering and
    variance of predictions while shifting the mean toward the empirical prior.
    This is CRITICAL for routing: routing needs probability variance to make
    threshold decisions; isotonic regression collapses variance.

    Also saves a secondary isotonic calibrator for ECE reporting only.
    """
    print("\n[*] Applying calibration...")

    model.eval()
    val_dataset = HiddenStateDataset(val_data, hidden_dim=hidden_dim, label_type=label_type)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    all_logits = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            hs = batch["hidden_state"].to(device)
            si = batch["stage_idx"].to(device)
            labels = batch["label"]

            if variant == "no_stage_emb":
                logits = model(hs)
            else:
                logits = model(hs, si)

            all_logits.extend(logits.squeeze(-1).cpu().tolist())
            all_probs.extend(torch.sigmoid(logits.squeeze(-1)).cpu().tolist())
            all_labels.extend(labels.tolist())

    all_logits = np.array(all_logits, dtype=np.float32)
    all_probs = np.array(all_probs, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.float32)

    # --- Method 1: Temperature Scaling (preserves variance, good for routing) ---
    # Optimize T via grid search to minimize ECE on validation set
    best_T = 1.0
    best_loss = float("inf")
    for T in np.arange(0.2, 5.1, 0.1):
        cal_probs = 1.0 / (1.0 + np.exp(-all_logits / T))
        ece = compute_ece(cal_probs, all_labels, n_bins=10)
        if ece < best_loss:
            best_loss = ece
            best_T = float(T)

    temp_calibrated = 1.0 / (1.0 + np.exp(-all_logits / best_T))
    temp_ece = compute_ece(temp_calibrated, all_labels, n_bins=10)

    # --- Method 2: Isotonic (for ECE reporting, destructive to variance) ---
    iso_cal = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso_cal.fit(all_probs, all_labels)
    iso_calibrated = iso_cal.predict(all_probs)
    iso_ece = compute_ece(iso_calibrated, all_labels, n_bins=10)

    calibrator = TemperatureCalibrator(best_T)

    raw_ece = compute_ece(all_probs, all_labels, n_bins=10)

    calibration_results = {
        "method": "temperature_scaling",
        "temperature": best_T,
        "raw_ece": float(raw_ece),
        "temperature_ece": float(temp_ece),
        "isotonic_ece": float(iso_ece),
        "num_calibration_samples": len(all_labels),
        "raw_prob_mean": float(all_probs.mean()),
        "raw_prob_std": float(all_probs.std()),
        "calibrated_prob_mean": float(temp_calibrated.mean()),
        "calibrated_prob_std": float(temp_calibrated.std()),
        "label_mean": float(all_labels.mean()),
    }

    print(f"    Raw ECE: {raw_ece:.4f} → Temp-scaled ECE: {temp_ece:.4f} (T={best_T:.3f})")
    print(f"    Iso ECE: {iso_ece:.4f} (for reference, variance-destructive)")
    print(f"    Raw prob: mean={all_probs.mean():.4f}, std={all_probs.std():.4f}")
    print(f"    Temp-scaled: mean={temp_calibrated.mean():.4f}, std={temp_calibrated.std():.4f}")
    print(f"    Label mean: {all_labels.mean():.4f}")

    return calibrator, calibration_results


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error.

    ECE = Σ (|B_m|/n) * |acc(B_m) - conf(B_m)|
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        if in_bin.sum() == 0:
            continue
        bin_acc = labels[in_bin].mean()
        bin_conf = probs[in_bin].mean()
        ece += (in_bin.sum() / len(labels)) * abs(bin_acc - bin_conf)

    return ece


def main():
    parser = argparse.ArgumentParser(
        description="Train failure predictor on collected hidden states."
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to collected states JSONL (default: data/collected_states.jsonl)"
    )
    parser.add_argument(
        "--variant", type=str, default="mlp",
        choices=["mlp", "logistic", "stage_specific", "lstm",
                 "no_stage_emb", "retrieval_features"],
        help="Predictor variant to train"
    )
    parser.add_argument(
        "--seed", type=int, default=BASE_SEED,
        help="Random seed"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for results (default: results/)"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=LLAMA_HIDDEN_DIM,
        help="LLM hidden dimension"
    )
    parser.add_argument(
        "--epochs", type=int, default=NUM_EPOCHS,
        help="Max training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=LEARNING_RATE,
        help="Learning rate"
    )
    parser.add_argument(
        "--no_calibrate", action="store_true",
        help="Skip Platt scaling calibration"
    )
    parser.add_argument(
        "--label_type", type=str, default="stage",
        choices=["final", "stage"],
        help="Label to predict: 'final' = final (S3) correctness from any stage, "
             "'stage' = stage-specific correctness (cleaner probe design)"
    )
    args = parser.parse_args()

    set_seed(args.seed)

    # Default paths
    if args.data is None:
        args.data = os.path.join(DATA_DIR, "collected_states.jsonl")
    if args.output_dir is None:
        args.output_dir = RESULTS_DIR

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    if not os.path.exists(args.data):
        print(f"[!] Data file not found: {args.data}")
        print("[!] Run collect_states.py first.")
        sys.exit(1)

    data = load_collected_data(args.data)
    print(f"[*] Loaded {len(data)} tuples from {args.data}")

    # Split by query
    train_data, val_data, test_data = split_data(data, seed=args.seed)

    # ── Train ──
    if args.variant == "logistic":
        results = train_logistic_regression(
            train_data, val_data,
            results_path=os.path.join(
                args.output_dir, "logistic_regression_results.json"
            ),
        )
    else:
        model_path = os.path.join(
            MODEL_DIR, f"predictor_{args.variant}_seed{args.seed}.pt"
        )
        results_path = os.path.join(
            args.output_dir, f"predictor_{args.variant}_seed{args.seed}_results.json"
        )

        results = train_mlp(
            train_data=train_data,
            val_data=val_data,
            results_path=results_path,
            model_path=model_path,
            variant=args.variant,
            hidden_dim=args.hidden_dim,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            seed=args.seed,
            label_type=args.label_type,
        )

        # Calibration
        if not args.no_calibrate:
            # Load best model
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            model = build_predictor(
                variant=args.variant,
                hidden_dim=args.hidden_dim,
                stage_emb_dim=checkpoint.get("stage_emb_dim", 16),
                num_stages=checkpoint.get("num_stages", 4),
                mlp_hidden=checkpoint.get("mlp_hidden", MLP_HIDDEN_LAYERS),
                dropout=checkpoint.get("dropout", MLP_DROPOUT),
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to("cuda" if torch.cuda.is_available() else "cpu")

            calibrator, cal_results = calibrate_platt(
                model, val_data, variant=args.variant, hidden_dim=args.hidden_dim,
                label_type=args.label_type,
            )
            results["calibration"] = cal_results

            # Save calibration
            cal_path = model_path.replace(".pt", "_calibrator.pkl")
            with open(cal_path, "wb") as f:
                pickle.dump(calibrator, f)

    print(f"\n[*] Training complete. Results saved to {args.output_dir}")
    print(f"[*] Best Val AUROC: {results.get('best_val_auroc', results.get('val_auroc', 'N/A'))}")


if __name__ == "__main__":
    main()
