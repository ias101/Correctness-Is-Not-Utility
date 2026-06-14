"""
Critical Ablation Runner: Delta Probe (#8), LSTM (#2), Layer-wise (#6), Retrieval (#4).

Runs all 5 critical ablations identified in Round 1 (6.5/10 review).
Uses existing V4 multi-layer HotpotQA data — no new data collection needed.

Ablations:
  #8  Delta Probe Architecture:  linear | shallow_mlp | standard_mlp | deep_mlp
                                 × bce | weighted_bce | focal_loss
  #2  LSTM Recurrence:          StaticMLP vs LSTM on correctness + delta
  #6  Layer-wise Contribution:  Per-layer AUROC analysis
  #4  Retrieval Features:       hidden_states_only vs +retrieval_scores

Output: `results/ablations/` — JSON results + summary table
"""

import argparse
import json
import os
import sys
import pickle
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Add both experiments/ dir and parent project root to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
sys.path.insert(0, _script_dir)
sys.path.insert(0, _parent_dir)
from config import (
    BASE_SEED, SEEDS, NUM_STAGES, STAGES,
    LLAMA_HIDDEN_DIM, STAGE_EMBEDDING_DIM, MLP_HIDDEN_LAYERS, MLP_DROPOUT,
    BATCH_SIZE_TRAIN, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = BASE_SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def compute_ece(probs, labels, n_bins=10):
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


# ═══════════════════════════════════════════════════════════════════════════════
# Focal Loss
# ═══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.93, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        return (alpha_weight * focal_weight * bce_loss).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# Data Preparation
# ═══════════════════════════════════════════════════════════════════════════════

def build_benefit_labels(data: List[Dict]) -> List[Dict]:
    """benefit_t = 1 if stage_correctness[t]==0 AND stage_correctness[t+1]==1."""
    by_query = defaultdict(list)
    for d in data:
        by_query[d['query_id']].append(d)

    for qid, stages in by_query.items():
        stages_sorted = sorted(stages, key=lambda s: s['stage_idx'])
        for i, s in enumerate(stages_sorted):
            if i < NUM_STAGES - 1:
                cur = s.get('stage_correctness', 0)
                nxt = stages_sorted[i + 1].get('stage_correctness', 0)
                s['benefit'] = 1 if (cur == 0 and nxt == 1) else 0
            else:
                s['benefit'] = 0
    return data


def extract_multi_layer_features(data: List[Dict]) -> List[Dict]:
    """Extract individual layer features from multi_layer_hidden_states."""
    for d in data:
        mls = d.get('multi_layer_hidden_states', None)
        if mls is not None and len(mls) > 0:
            for i, layer_state in enumerate(mls):
                d[f'layer_{i}_state'] = layer_state
        # Fallback: use final_token hidden_state as single layer
        if 'layer_0_state' not in d:
            d['layer_0_state'] = d['hidden_state']
    return data


def split_by_query(data: List[Dict], train_frac=0.70, val_frac=0.15, seed=BASE_SEED):
    """Split by query_id to avoid leakage."""
    query_ids = sorted(set(d['query_id'] for d in data))
    np.random.seed(seed)
    np.random.shuffle(query_ids)
    n = len(query_ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_ids = set(query_ids[:n_train])
    val_ids = set(query_ids[n_train:n_train + n_val])
    test_ids = set(query_ids[n_train + n_val:])
    train = [d for d in data if d['query_id'] in train_ids]
    val = [d for d in data if d['query_id'] in val_ids]
    test = [d for d in data if d['query_id'] in test_ids]
    return train, val, test


# ═══════════════════════════════════════════════════════════════════════════════
# Models for Ablations
# ═══════════════════════════════════════════════════════════════════════════════

class LinearProbe(nn.Module):
    """Simple logistic probe for Ablation #8."""
    def __init__(self, input_dim=3584):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class ShallowMLP(nn.Module):
    """Single hidden layer [128] for Ablation #8."""
    def __init__(self, input_dim=3584, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class StandardMLP(nn.Module):
    """Two hidden layers [256, 128] for Ablation #8."""
    def __init__(self, input_dim=3584, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DeepMLP(nn.Module):
    """Four hidden layers [512, 256, 128, 64] for Ablation #8."""
    def __init__(self, input_dim=3584, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SeqMLP(nn.Module):
    """MLP over concatenated stage sequence for Ablation #2."""
    def __init__(self, input_dim=3584, num_stages=4):
        super().__init__()
        full_dim = input_dim * num_stages
        self.net = nn.Sequential(
            nn.Linear(full_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):  # x: (B, 4*hidden_dim)
        return self.net(x).squeeze(-1)


class SeqLSTM(nn.Module):
    """LSTM over stage sequence for Ablation #2."""
    def __init__(self, input_dim=3584, lstm_hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, lstm_hidden, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=0.2)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):  # x: (B, 4, hidden_dim)
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :]).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Training Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def make_dataloader(X, y, batch_size=128, shuffle=True, class_balanced=False):
    """Create DataLoader from numpy arrays."""
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )
    if class_balanced:
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        weights = np.where(y == 1, len(y) / max(n_pos, 1), len(y) / max(n_neg, 1))
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights, num_samples=len(y), replacement=True
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(model, train_loader, val_loader, loss_fn, device='cuda',
                lr=1e-3, wd=1e-4, epochs=100, patience=15):
    """Generic training loop with early stopping."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=patience // 3
    )
    best_auroc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                logits = model(x)
                val_probs.extend(torch.sigmoid(logits).cpu().tolist())
                val_labels.extend(y.cpu().tolist())

        val_auroc = roc_auc_score(val_labels, val_probs)
        val_auprc = average_precision_score(val_labels, val_probs)
        scheduler.step(val_auroc)

        if val_auroc > best_auroc + 0.001:
            best_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load best state
    if best_state:
        model.load_state_dict(best_state)

    # Full evaluation
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            logits = model(x)
            all_probs.extend(torch.sigmoid(logits).cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    return {
        'auroc': float(roc_auc_score(all_labels, all_probs)),
        'auprc': float(average_precision_score(all_labels, all_probs)),
        'ece': float(compute_ece(all_probs, all_labels)),
        'best_epoch': epoch - patience_counter,
        'num_params': sum(p.numel() for p in model.parameters()),
        'val_probs': all_probs.tolist(),
        'val_labels': all_labels.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Ablation #8: Delta Probe Architecture Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def run_ablation_8_delta_probe(data: List[Dict], device='cuda'):
    """Compare different probe architectures for Delta prediction."""
    print("\n" + "="*70)
    print("  ABLATION #8: Delta Probe Architecture Comparison")
    print("="*70)

    data = build_benefit_labels(data)
    train_data, val_data, test_data = split_by_query(data)

    # Extract features (use final_token + stage one-hot)
    def prepare_data(data_split):
        X = np.array([d['hidden_state'] for d in data_split], dtype=np.float32)
        y = np.array([d['benefit'] for d in data_split], dtype=np.float32)
        # Add stage one-hot
        stage = np.zeros((len(data_split), NUM_STAGES), dtype=np.float32)
        for i, d in enumerate(data_split):
            stage[i, d['stage_idx']] = 1.0
        X = np.concatenate([X, stage], axis=1)
        return X, y

    X_train, y_train = prepare_data(train_data)
    X_val, y_val = prepare_data(val_data)
    X_test, y_test = prepare_data(test_data)

    # Normalize (fit on train only)
    scaler = StandardScaler()
    hs_dim = 3584
    X_train[:, :hs_dim] = scaler.fit_transform(X_train[:, :hs_dim])
    X_val[:, :hs_dim] = scaler.transform(X_val[:, :hs_dim])
    X_test[:, :hs_dim] = scaler.transform(X_test[:, :hs_dim])

    pos_rate = y_train.mean()
    print(f"  Train: {len(X_train)} tuples, pos_rate={pos_rate:.4f}")
    print(f"  Val:   {len(X_val)} tuples, pos_rate={y_val.mean():.4f}")
    print(f"  Test:  {len(X_test)} tuples, pos_rate={y_test.mean():.4f}")

    input_dim = X_train.shape[1]  # hidden_dim + num_stages

    architectures = {
        'linear': lambda: LinearProbe(input_dim),
        'shallow_mlp': lambda: ShallowMLP(input_dim),
        'standard_mlp': lambda: StandardMLP(input_dim),
        'deep_mlp': lambda: DeepMLP(input_dim),
    }

    loss_fns = {
        'bce': ('BCE', nn.BCEWithLogitsLoss()),
        'weighted_bce': ('WeightedBCE', nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([(1 - pos_rate) / max(pos_rate, 0.01)]))),
        'focal': ('Focal(γ=2)', FocalLoss(alpha=1 - pos_rate, gamma=2.0)),
    }

    results = []
    for arch_name, arch_fn in architectures.items():
        for loss_name, (loss_label, loss_fn) in loss_fns.items():
            label = f"{arch_name}_{loss_name}"
            print(f"\n  [{label}] Training...")
            model = arch_fn()
            train_loader = make_dataloader(X_train, y_train,
                                           class_balanced=(loss_name != 'weighted_bce'))
            val_loader = make_dataloader(X_val, y_val, shuffle=False)
            r = train_model(model, train_loader, val_loader,
                           loss_fn.to(device), device=device, epochs=100)
            r['architecture'] = arch_name
            r['loss'] = loss_name
            r['loss_label'] = loss_label
            print(f"    AUROC={r['auroc']:.4f}  AUPRC={r['auprc']:.4f}  "
                  f"ECE={r['ece']:.4f}  params={r['num_params']:,}")
            results.append(r)

    # Summary table
    print("\n  Delta Probe Architecture Comparison:")
    print(f"  {'Architecture':<18} {'Loss':<14} {'AUROC':>8} {'AUPRC':>8} {'ECE':>8}")
    print("  " + "-"*58)
    best = max(results, key=lambda r: r['auprc'])
    for r in sorted(results, key=lambda r: r['auprc'], reverse=True):
        marker = " ← BEST" if r is best else ""
        print(f"  {r['architecture']:<18} {r['loss_label']:<14} "
              f"{r['auroc']:>8.4f} {r['auprc']:>8.4f} {r['ece']:>8.4f}{marker}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Ablation #2: LSTM Sequential Modeling
# ═══════════════════════════════════════════════════════════════════════════════

def build_sequential_data(data: List[Dict]):
    """Group hidden states by query → create sequences of (4, hidden_dim)."""
    data = build_benefit_labels(data)
    by_query = defaultdict(list)
    for d in data:
        by_query[d['query_id']].append(d)

    X_seq, y_correct, y_benefit = [], [], []
    for qid, stages in by_query.items():
        stages_sorted = sorted(stages, key=lambda s: s['stage_idx'])
        if len(stages_sorted) != NUM_STAGES:
            continue
        seq = np.array([s['hidden_state'] for s in stages_sorted], dtype=np.float32)
        X_seq.append(seq.flatten())  # (4*hidden_dim,)
        y_correct.append(stages_sorted[-1]['final_correctness'])
        # Benefit: is there ANY stage where advancing helps?
        y_benefit.append(1 if any(s.get('benefit', 0) for s in stages_sorted) else 0)

    return (np.array(X_seq, dtype=np.float32),
            np.array(y_correct, dtype=np.float32),
            np.array(y_benefit, dtype=np.float32))


def run_ablation_2_lstm(data: List[Dict], device='cuda'):
    """Compare Static MLP vs LSTM on sequential data."""
    print("\n" + "="*70)
    print("  ABLATION #2: LSTM Sequential Modeling")
    print("="*70)

    train_data, val_data, test_data = split_by_query(data)
    X_train, yc_train, yb_train = build_sequential_data(train_data)
    X_val, yc_val, yb_val = build_sequential_data(val_data)
    X_test, yc_test, yb_test = build_sequential_data(test_data)

    print(f"  Sequential data: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
    print(f"  Correctness pos_rate: {yc_train.mean():.3f}")
    print(f"  Benefit pos_rate: {yb_train.mean():.3f}")

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    results = []

    for target_name, y_train, y_val, y_test in [
        ('correctness', yc_train, yc_val, yc_test),
        ('benefit', yb_train, yb_val, yb_test),
    ]:
        pos_rate = y_train.mean()
        print(f"\n  --- Target: {target_name} (pos_rate={pos_rate:.3f}) ---")

        for model_name, model_fn in [
            ('static_mlp', lambda: SeqMLP(3584, NUM_STAGES)),
            ('lstm', lambda: SeqLSTM(3584)),
        ]:
            print(f"  [{model_name}/{target_name}] Training...")
            model = model_fn()

            # For LSTM, reshape to (B, 4, hidden_dim)
            if model_name == 'lstm':
                Xt = X_train.reshape(-1, NUM_STAGES, 3584)
                Xv = X_val.reshape(-1, NUM_STAGES, 3584)
            else:
                Xt = X_train
                Xv = X_val

            train_loader = make_dataloader(Xt, y_train)
            val_loader = make_dataloader(Xv, y_val, shuffle=False)

            loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([(1 - pos_rate) / max(pos_rate, 0.01)]))
            r = train_model(model, train_loader, val_loader,
                           loss_fn.to(device), device=device, epochs=80)
            r['target'] = target_name
            r['model_type'] = model_name
            print(f"    AUROC={r['auroc']:.4f}  AUPRC={r['auprc']:.4f}  "
                  f"ECE={r['ece']:.4f}")
            results.append(r)

    # Summary
    print(f"\n  Sequential Modeling Comparison:")
    print(f"  {'Model':<14} {'Target':<14} {'AUROC':>8} {'AUPRC':>8} {'ECE':>8}")
    print("  " + "-"*46)
    for r in sorted(results, key=lambda r: r['auprc'], reverse=True):
        print(f"  {r['model_type']:<14} {r['target']:<14} "
              f"{r['auroc']:>8.4f} {r['auprc']:>8.4f} {r['ece']:>8.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Ablation #6: Layer-wise Contribution
# ═══════════════════════════════════════════════════════════════════════════════

def run_ablation_6_layers(data: List[Dict], device='cuda'):
    """Analyze per-layer contribution to correctness prediction."""
    print("\n" + "="*70)
    print("  ABLATION #6: Layer-wise Contribution Analysis")
    print("="*70)

    train_data, val_data, test_data = split_by_query(data)

    # We need multi_layer data
    has_multi = all('multi_layer_hidden_states' in d for d in data[:10])
    if not has_multi:
        print("  [!] No multi_layer data available — skipping")
        return []

    NUM_LAYERS_IN_DATA = len(data[0].get('multi_layer_hidden_states', []))
    print(f"  Layers available in data: {NUM_LAYERS_IN_DATA}")

    def prepare_layer_data(data_split, layer_idx=None):
        """Extract features from a specific layer or concatenated."""
        X, y = [], []
        for d in data_split:
            mls = d.get('multi_layer_hidden_states', None)
            if mls is None:
                continue
            if layer_idx is not None:
                X.append(mls[layer_idx])
            else:
                X.append(np.concatenate(mls))  # all layers
            y.append(d['final_correctness'])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    results = []

    # Individual layer analysis
    for layer_idx in range(NUM_LAYERS_IN_DATA):
        X_train, y_train = prepare_layer_data(train_data, layer_idx)
        X_val, y_val = prepare_layer_data(val_data, layer_idx)

        # Normalize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # Quick logistic regression
        clf = LogisticRegression(max_iter=500, class_weight='balanced', C=1.0)
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_val)[:, 1]

        auroc = roc_auc_score(y_val, y_prob)
        auprc = average_precision_score(y_val, y_prob)
        results.append({
            'layer': f'layer_{layer_idx}',
            'auroc': float(auroc),
            'auprc': float(auprc),
        })
        print(f"  Layer {layer_idx}: AUROC={auroc:.4f}, AUPRC={auprc:.4f}")

    # All layers concatenated
    X_train, y_train = prepare_layer_data(train_data, None)
    X_val, y_val = prepare_layer_data(val_data, None)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    clf = LogisticRegression(max_iter=500, class_weight='balanced', C=1.0)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_val)[:, 1]
    auroc = roc_auc_score(y_val, y_prob)
    auprc = average_precision_score(y_val, y_prob)
    results.append({
        'layer': 'all_concatenated',
        'auroc': float(auroc),
        'auprc': float(auprc),
    })
    print(f"  All concatenated: AUROC={auroc:.4f}, AUPRC={auprc:.4f}")

    # Also: first token only (no pooling), final token only, mean of all layers
    # Use original hidden_state (final_token) for comparison
    X_train_ft = np.array([d['hidden_state'] for d in train_data], dtype=np.float32)
    y_train_ft = np.array([d['final_correctness'] for d in train_data])
    X_val_ft = np.array([d['hidden_state'] for d in val_data], dtype=np.float32)
    y_val_ft = np.array([d['final_correctness'] for d in val_data])
    scaler_ft = StandardScaler()
    X_train_ft = scaler_ft.fit_transform(X_train_ft)
    X_val_ft = scaler_ft.transform(X_val_ft)
    clf.fit(X_train_ft, y_train_ft)
    y_prob_ft = clf.predict_proba(X_val_ft)[:, 1]
    results.append({
        'layer': 'final_token_only',
        'auroc': float(roc_auc_score(y_val_ft, y_prob_ft)),
        'auprc': float(average_precision_score(y_val_ft, y_prob_ft)),
    })
    print(f"  Final token only: AUROC={results[-1]['auroc']:.4f}, "
          f"AUPRC={results[-1]['auprc']:.4f}")

    # Summary
    print(f"\n  Layer-wise AUROC Summary:")
    for r in sorted(results, key=lambda r: r['auroc'], reverse=True):
        marker = " ← BEST" if r is results[0] else ""
        print(f"  {r['layer']:<22} AUROC={r['auroc']:.4f}  AUPRC={r['auprc']:.4f}{marker}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Ablation #4: Retrieval Features
# ═══════════════════════════════════════════════════════════════════════════════

def run_ablation_4_retrieval(data: List[Dict], device='cuda'):
    """Compare hidden_states_only vs hidden_states + retrieval features."""
    print("\n" + "="*70)
    print("  ABLATION #4: Retrieval Features Augmentation")
    print("="*70)

    # Check if retrieval features exist in data
    sample = data[0]
    has_bm25 = 'bm25_score' in sample
    has_ce = 'cross_encoder_score' in sample
    has_passage_count = 'passage_count' in sample

    print(f"  Retrieval features in data: bm25={has_bm25}, "
          f"ce_score={has_ce}, passage_count={has_passage_count}")

    if not any([has_bm25, has_ce, has_passage_count]):
        print("  [!] No retrieval features in data.")
        print("  [*] Using IMPUTED features: stage_idx as proxy for passage_count")
        print("  [*] stage_idx→passage_count mapping: S0→2, S1→4, S2→6, S3→8")

    train_data, val_data, test_data = split_by_query(data)

    def prepare_retrieval_data(data_split):
        h = np.array([d['hidden_state'] for d in data_split], dtype=np.float32)
        y = np.array([d['final_correctness'] for d in data_split], dtype=np.float32)
        stage = np.zeros((len(data_split), NUM_STAGES), dtype=np.float32)
        for i, d in enumerate(data_split):
            stage[i, d['stage_idx']] = 1.0

        # Build retrieval features
        ret_feats = np.zeros((len(data_split), 3), dtype=np.float32)
        for i, d in enumerate(data_split):
            si = d['stage_idx']
            # Impute: passage_count from stage (2,4,6,8)
            ret_feats[i, 0] = float(d.get('bm25_score', 0.0))
            ret_feats[i, 1] = float(d.get('passage_count', 2 + si * 2))
            ret_feats[i, 2] = float(d.get('cross_encoder_score', 0.0))

        X_hs_only = np.concatenate([h, stage], axis=1)
        X_with_ret = np.concatenate([h, stage, ret_feats], axis=1)
        return X_hs_only, X_with_ret, y

    X_train_hs, X_train_ret, y_train = prepare_retrieval_data(train_data)
    X_val_hs, X_val_ret, y_val = prepare_retrieval_data(val_data)

    # Normalize
    hs_dim = 3584
    scaler_hs = StandardScaler()
    X_train_hs[:, :hs_dim] = scaler_hs.fit_transform(X_train_hs[:, :hs_dim])
    X_val_hs[:, :hs_dim] = scaler_hs.transform(X_val_hs[:, :hs_dim])

    scaler_ret = StandardScaler()
    X_train_ret[:, :hs_dim + NUM_STAGES] = scaler_ret.fit_transform(
        X_train_ret[:, :hs_dim + NUM_STAGES])
    X_val_ret[:, :hs_dim + NUM_STAGES] = scaler_ret.transform(
        X_val_ret[:, :hs_dim + NUM_STAGES])

    results = []
    for label, X_train, X_val in [
        ('hidden_states_only', X_train_hs, X_val_hs),
        ('hs_plus_retrieval', X_train_ret, X_val_ret),
    ]:
        for target_name, target_key in [
            ('correctness', 'stage_correctness'),
            ('benefit', 'benefit'),
        ]:
            if target_key == 'benefit':
                data_with_benefit = build_benefit_labels(data)
                # Re-split for benefit labels
                tr, va, _ = split_by_query(data_with_benefit)
                y_tr = np.array([d['benefit'] for d in tr], dtype=np.float32)
                y_va = np.array([d['benefit'] for d in va], dtype=np.float32)
                # Re-prepare features
                Xt_hs, Xt_ret, _ = prepare_retrieval_data(tr)
                Xv_hs, Xv_ret, _ = prepare_retrieval_data(va)
                if label == 'hidden_states_only':
                    Xt, Xv = Xt_hs, Xv_hs
                else:
                    Xt, Xv = Xt_ret, Xv_ret
            else:
                Xt, Xv = X_train, X_val
                y_tr, y_va = y_train, y_val

            print(f"\n  [{label}/{target_name}] Training...")
            model = StandardMLP(Xt.shape[1])
            train_loader = make_dataloader(Xt, y_tr)
            val_loader = make_dataloader(Xv, y_va, shuffle=False)
            pos_rate = y_tr.mean()
            loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([(1 - pos_rate) / max(pos_rate, 0.01)]))
            r = train_model(model, train_loader, val_loader,
                           loss_fn.to(device), device=device, epochs=80)
            r['features'] = label
            r['target'] = target_name
            print(f"    AUROC={r['auroc']:.4f}  AUPRC={r['auprc']:.4f}  "
                  f"ECE={r['ece']:.4f}")
            results.append(r)

    # Summary
    print(f"\n  Retrieval Features Comparison:")
    print(f"  {'Features':<22} {'Target':<14} {'AUROC':>8} {'AUPRC':>8} {'ECE':>8}")
    print("  " + "-"*54)
    for r in sorted(results, key=lambda r: r['auprc'], reverse=True):
        print(f"  {r['features']:<22} {r['target']:<14} "
              f"{r['auroc']:>8.4f} {r['auprc']:>8.4f} {r['ece']:>8.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Run critical ablation experiments')
    parser.add_argument('--data', type=str,
                        default='data/collected_states_hotpotqa_v4.jsonl',
                        help='Path to V4 JSONL data')
    parser.add_argument('--ablations', type=str, default='all',
                        choices=['all', '8', '2', '6', '4'],
                        help='Which ablation(s) to run')
    parser.add_argument('--output_dir', type=str, default='results/ablations',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"[*] Device: {device}")

    # Load data
    print(f"[*] Loading data from {args.data}...")
    t0 = time.time()
    data = load_jsonl(args.data)
    print(f"[*] Loaded {len(data)} tuples in {time.time()-t0:.1f}s")

    all_results = {}

    if args.ablations in ('all', '8'):
        r8 = run_ablation_8_delta_probe(data, device)
        all_results['ablation_8_delta_probe'] = r8
        with open(os.path.join(args.output_dir, 'ablation_8_delta_probe.json'), 'w') as f:
            json.dump(r8, f, indent=2)

    if args.ablations in ('all', '2'):
        r2 = run_ablation_2_lstm(data, device)
        all_results['ablation_2_lstm'] = r2
        with open(os.path.join(args.output_dir, 'ablation_2_lstm.json'), 'w') as f:
            json.dump(r2, f, indent=2)

    if args.ablations in ('all', '6'):
        r6 = run_ablation_6_layers(data, device)
        all_results['ablation_6_layers'] = r6
        with open(os.path.join(args.output_dir, 'ablation_6_layers.json'), 'w') as f:
            json.dump(r6, f, indent=2)

    if args.ablations in ('all', '4'):
        r4 = run_ablation_4_retrieval(data, device)
        all_results['ablation_4_retrieval'] = r4
        with open(os.path.join(args.output_dir, 'ablation_4_retrieval.json'), 'w') as f:
            json.dump(r4, f, indent=2)

    # Save summary
    summary_path = os.path.join(args.output_dir, 'ablation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\n[*] All results saved to {args.output_dir}/")
    print(f"[*] Summary: {summary_path}")


if __name__ == '__main__':
    main()
