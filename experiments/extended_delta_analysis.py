"""
Extended Delta Label Analysis: Multi-step, Degradation, and Cost-Sensitive Utility.

Computes five Delta label variants from existing collected states (no re-collection needed):
  1. delta_1step  = 1[wrong@t & correct@t+1]   (existing, one-step benefit)
  2. delta_multistep = 1[wrong@t & correct@any t'>t] (any-step benefit, includes multi-step)
  3. delta_2step  = 1[wrong@t & wrong@t+1 & correct@t+2] (two-step only)
  4. delta_degrade = 1[correct@t & wrong@t+1]   (degradation risk, NOT captured before)
  5. delta_net     = benefit - degradation        (net utility change)

Also computes per-query transition matrices and cost-sensitive expected CWA.

Author: Loop 11 — Supplementary Delta experiments
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experiments'))
from config import BASE_SEED, NUM_STAGES, STAGES

# ── Label Construction ──────────────────────────────────────────────────────────

def compute_extended_delta_labels(data: List[Dict]) -> List[Dict]:
    """Compute all five Delta label variants from per-stage correctness.

    Requires: stage_correctness per tuple, query_id, stage_idx.

    Label definitions:
      delta_1step[t]    = 1 if wrong@t AND correct@t+1  (original)
      delta_multistep[t] = 1 if wrong@t AND correct@any t'>t
      delta_2step[t]    = 1 if wrong@t AND wrong@t+1 AND correct@t+2
      delta_degrade[t]  = 1 if correct@t AND wrong@t+1  (NEW)
      delta_net[t]      = delta_multistep[t] - delta_degrade[t]  (net utility)
    """
    by_query = defaultdict(list)
    for d in data:
        by_query[d['query_id']].append(d)

    stats = {
        'delta_1step_pos': 0, 'delta_multistep_pos': 0,
        'delta_2step_pos': 0, 'delta_degrade_pos': 0,
        'total_transitions': 0, 'total_queries': len(by_query),
        'multi_step_queries': 0,    # queries that need >1 step to become correct
        'degrade_queries': 0,       # queries that degrade at any stage
        'never_correct': 0,         # queries never correct at any stage
        'always_correct': 0,        # queries correct at all stages
    }

    for qid, stages in by_query.items():
        stages_sorted = sorted(stages, key=lambda s: s['stage_idx'])
        correctness = [s.get('stage_correctness', 0) for s in stages_sorted]

        # Track query-level patterns
        if all(c == 0 for c in correctness):
            stats['never_correct'] += 1
        if all(c == 1 for c in correctness):
            stats['always_correct'] += 1

        # Check for multi-step: wrong at some t, still wrong at t+1, correct later
        has_multistep = False
        has_degrade = False
        for t in range(NUM_STAGES - 1):
            if correctness[t] == 0 and correctness[t+1] == 0:
                if any(correctness[tt] == 1 for tt in range(t+2, NUM_STAGES)):
                    has_multistep = True
            if correctness[t] == 1 and correctness[t+1] == 0:
                has_degrade = True

        if has_multistep:
            stats['multi_step_queries'] += 1
        if has_degrade:
            stats['degrade_queries'] += 1

        # Per-transition labels
        for i, s in enumerate(stages_sorted):
            t = s['stage_idx']
            stats['total_transitions'] += 1

            if t < NUM_STAGES - 1:
                # --- delta_1step ---
                if correctness[t] == 0 and correctness[t+1] == 1:
                    s['delta_1step'] = 1
                    stats['delta_1step_pos'] += 1
                else:
                    s['delta_1step'] = 0

                # --- delta_multistep (any future correction) ---
                if correctness[t] == 0 and any(correctness[tt] == 1 for tt in range(t+1, NUM_STAGES)):
                    s['delta_multistep'] = 1
                    stats['delta_multistep_pos'] += 1
                else:
                    s['delta_multistep'] = 0

                # --- delta_2step (exactly two-step, wrong at t+1) ---
                if (t < NUM_STAGES - 2 and
                    correctness[t] == 0 and correctness[t+1] == 0 and correctness[t+2] == 1):
                    s['delta_2step'] = 1
                    stats['delta_2step_pos'] += 1
                else:
                    s['delta_2step'] = 0

                # --- delta_degrade (correct -> wrong) ---
                if correctness[t] == 1 and correctness[t+1] == 0:
                    s['delta_degrade'] = 1
                    stats['delta_degrade_pos'] += 1
                else:
                    s['delta_degrade'] = 0

                # --- delta_net ---
                s['delta_net'] = s['delta_multistep'] - s['delta_degrade']  # {-1, 0, 1}
            else:
                # S3: no next stage
                for key in ['delta_1step', 'delta_multistep', 'delta_2step',
                           'delta_degrade', 'delta_net']:
                    s[key] = 0

    # Compute transition matrix
    trans_matrix = np.zeros((NUM_STAGES, NUM_STAGES), dtype=int)
    for qid, stages in by_query.items():
        stages_sorted = sorted(stages, key=lambda s: s['stage_idx'])
        correctness = [s.get('stage_correctness', 0) for s in stages_sorted]
        for t in range(NUM_STAGES - 1):
            trans_matrix[correctness[t], correctness[t+1]] += 1

    stats['transition_matrix'] = trans_matrix.tolist()
    stats['transition_labels'] = {
        '00': 'wrong→wrong', '01': 'wrong→correct (Δ⁺ one-step)',
        '10': 'correct→wrong (Δ⁻ degradation)', '11': 'correct→correct'
    }

    # Print summary
    tot = stats['total_transitions']
    print(f"\n{'='*60}")
    print(f"EXTENDED DELTA LABEL STATISTICS")
    print(f"{'='*60}")
    print(f"  Total queries:        {stats['total_queries']}")
    print(f"  Total transitions:    {tot}")
    print(f"  Never correct:        {stats['never_correct']} ({100*stats['never_correct']/stats['total_queries']:.1f}%)")
    print(f"  Always correct:       {stats['always_correct']} ({100*stats['always_correct']/stats['total_queries']:.1f}%)")
    print(f"  Multi-step queries:   {stats['multi_step_queries']} ({100*stats['multi_step_queries']/stats['total_queries']:.1f}%)")
    print(f"  Degrade queries:      {stats['degrade_queries']} ({100*stats['degrade_queries']/stats['total_queries']:.1f}%)")
    print()
    for name, key in [('Δ⁺ 1-step (existing)', 'delta_1step_pos'),
                       ('Δ⁺ multi-step (any)', 'delta_multistep_pos'),
                       ('Δ⁺ 2-step (exact)', 'delta_2step_pos'),
                       ('Δ⁻ degrade (NEW)', 'delta_degrade_pos')]:
        pos = stats[key]
        print(f"  {name:<25}: {pos:>5} / {tot} ({100*pos/tot:.1f}%)")
    print(f"\n  Transition matrix (t→t+1):")
    print(f"    00 (wrong→wrong):               {trans_matrix[0,0]:>5}")
    print(f"    01 (wrong→correct, Δ⁺ one-step): {trans_matrix[0,1]:>5}")
    print(f"    10 (correct→wrong, Δ⁻ degrade):  {trans_matrix[1,0]:>5}")
    print(f"    11 (correct→correct):           {trans_matrix[1,1]:>5}")
    print(f"{'='*60}\n")

    return data, stats


# ── MLP Model for Extended Delta Probes ─────────────────────────────────────────

class DeltaProbeMLP(nn.Module):
    def __init__(self, hidden_dim=3584, mlp_hidden=None, dropout=0.2):
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = [256, 128]
        layers = []
        in_dim = hidden_dim + NUM_STAGES
        for h in mlp_hidden:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, h, stage_idx):
        stage_onehot = torch.zeros(h.size(0), NUM_STAGES, device=h.device)
        stage_onehot.scatter_(1, stage_idx.unsqueeze(1), 1.0)
        return self.net(torch.cat([h, stage_onehot], dim=1)).squeeze(-1)


class DeltaDataset(Dataset):
    def __init__(self, data, label_key, hidden_dim=3584, scaler=None):
        self.hidden_dim = hidden_dim
        self.stage_indices = torch.tensor([d['stage_idx'] for d in data], dtype=torch.long)
        self.labels = torch.tensor([d[label_key] for d in data], dtype=torch.float32)

        features = np.array([d['hidden_state'] for d in data], dtype=np.float32)
        if features.shape[1] != hidden_dim:
            if features.shape[1] > hidden_dim:
                features = features[:, :hidden_dim]
            else:
                p = np.zeros((features.shape[0], hidden_dim), dtype=np.float32)
                p[:, :features.shape[1]] = features
                features = p

        if scaler is None:
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
        else:
            features = scaler.transform(features)

        self.features = torch.tensor(features, dtype=torch.float32)
        self.scaler = scaler

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {'features': self.features[idx], 'stage_idx': self.stage_indices[idx],
                'label': self.labels[idx]}


# ── Training ────────────────────────────────────────────────────────────────────

def train_delta_probe(data, label_key, output_dir, label_name,
                      hidden_dim=3584, num_epochs=100, lr=1e-3,
                      batch_size=64, patience=20, seed=BASE_SEED, device='cuda'):
    """Train a Delta Probe for a specific label type."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Compute positive rate
    n_pos = sum(1 for d in data if d[label_key] == 1)
    pos_rate = n_pos / len(data)
    n_pos_s3 = sum(1 for d in data if d[label_key] == 1 and d['stage_idx'] < 3)
    pos_rate_no_s3 = n_pos_s3 / max(1, sum(1 for d in data if d['stage_idx'] < 3))

    # Split by query
    query_ids = sorted(set(d['query_id'] for d in data))
    np.random.seed(seed)
    np.random.shuffle(query_ids)
    n = len(query_ids)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    train_ids = set(query_ids[:n_train])
    val_ids = set(query_ids[n_train:n_train + n_val])
    test_ids = set(query_ids[n_train + n_val:])

    train_data = [d for d in data if d['query_id'] in train_ids]
    val_data = [d for d in data if d['query_id'] in val_ids]
    test_data = [d for d in data if d['query_id'] in test_ids]

    train_ds = DeltaDataset(train_data, label_key, hidden_dim)
    val_ds = DeltaDataset(val_data, label_key, hidden_dim, scaler=train_ds.scaler)
    test_ds = DeltaDataset(test_data, label_key, hidden_dim, scaler=train_ds.scaler)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = DeltaProbeMLP(hidden_dim=hidden_dim).to(device)

    # Weighted BCE
    n_neg = len(train_data) - sum(1 for d in train_data if d[label_key] == 1)
    pos_weight = n_neg / max(sum(1 for d in train_data if d[label_key] == 1), 1)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=patience//2)

    best_val_auprc = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            feats = batch['features'].to(device)
            si = batch['stage_idx'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            loss = criterion(model(feats, si), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_preds, val_labels_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                feats = batch['features'].to(device)
                si = batch['stage_idx'].to(device)
                logits = model(feats, si)
                val_preds.extend(torch.sigmoid(logits).cpu().tolist())
                val_labels_list.extend(batch['label'].tolist())

        val_auprc = average_precision_score(val_labels_list, val_preds)
        val_auroc = roc_auc_score(val_labels_list, val_preds)
        scheduler.step(val_auprc)

        if val_auprc > best_val_auprc + 0.001:
            best_val_auprc = val_auprc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({'model_state_dict': model.state_dict(),
                        'best_val_auroc': val_auroc, 'best_val_auprc': val_auprc,
                        'epoch': best_epoch, 'seed': seed},
                       os.path.join(output_dir, f'delta_{label_key}_best.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Test evaluation
    ckpt = torch.load(os.path.join(output_dir, f'delta_{label_key}_best.pt'),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    test_preds, test_labels_list = [], []
    test_stage_idxs = []
    with torch.no_grad():
        for batch in test_loader:
            feats = batch['features'].to(device)
            si = batch['stage_idx'].to(device)
            logits = model(feats, si)
            test_preds.extend(torch.sigmoid(logits).cpu().tolist())
            test_labels_list.extend(batch['label'].tolist())
    test_stage_idxs = [d['stage_idx'] for d in test_data]

    test_auroc = roc_auc_score(test_labels_list, test_preds)
    test_auprc = average_precision_score(test_labels_list, test_preds)
    random_auprc = sum(test_labels_list) / max(len(test_labels_list), 1)

    # Per-stage breakdown
    stage_results = {}
    for s in range(NUM_STAGES):
        idxs = [j for j, si in enumerate(test_stage_idxs) if si == s]
        if idxs:
            s_labels = [test_labels_list[j] for j in idxs]
            s_preds = [test_preds[j] for j in idxs]
            s_pos = sum(s_labels)
            if s_pos > 0 and s_pos < len(s_labels):
                stage_results[f'stage_{s}'] = {
                    'stage': STAGES[s], 'n': len(s_labels),
                    'n_pos': s_pos,
                    'auroc': float(roc_auc_score(s_labels, s_preds)),
                    'auprc': float(average_precision_score(s_labels, s_preds)),
                }

    results = {
        'label_key': label_key, 'label_name': label_name,
        'positive_rate_all': float(pos_rate),
        'positive_rate_excl_s3': float(pos_rate_no_s3),
        'n_pos': n_pos, 'n_total': len(data),
        'test_auroc': float(test_auroc),
        'test_auprc': float(test_auprc),
        'random_auprc': float(random_auprc),
        'auprc_ratio': float(test_auprc / max(random_auprc, 1e-8)),
        'best_epoch': best_epoch,
        'best_val_auroc': float(ckpt['best_val_auroc']),
        'best_val_auprc': float(ckpt['best_val_auprc']),
        'per_stage': stage_results,
        'n_train': len(train_data), 'n_val': len(val_data), 'n_test': len(test_data),
    }

    with open(os.path.join(output_dir, f'delta_{label_key}_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  [{label_name}] Test AUROC={test_auroc:.4f}, AUPRC={test_auprc:.4f} "
          f"({test_auprc/max(random_auprc,1e-8):.1f}× random), pos_rate={pos_rate:.3f}")
    return results


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Extended Delta Label Analysis')
    ap.add_argument('--data', type=str,
                    default='data/collected_states_hotpotqa_v3.jsonl',
                    help='Path to collected states JSONL')
    ap.add_argument('--output_dir', type=str, default='results/extended_delta',
                    help='Output directory')
    ap.add_argument('--hidden_dim', type=int, default=3584)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--seed', type=int, default=BASE_SEED)
    ap.add_argument('--stats_only', action='store_true',
                    help='Only compute label statistics, skip probe training')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print(f"[*] Loading data from {args.data}...")
    data = []
    with open(args.data) as f:
        for line in f:
            data.append(json.loads(line))
    print(f"[*] Loaded {len(data)} tuples")

    # Compute extended labels
    data, stats = compute_extended_delta_labels(data)

    # Save statistics
    stats_path = os.path.join(args.output_dir, 'extended_delta_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"[*] Statistics saved to {stats_path}")

    if args.stats_only:
        print("[*] Stats-only mode — skipping probe training.")
        return

    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[*] Using device: {device}")

    # Train probes for each Delta variant
    label_configs = [
        ('delta_1step',     'Δ⁺ 1-step (existing)'),
        ('delta_multistep', 'Δ⁺ Multi-step (any future correction)'),
        ('delta_2step',     'Δ⁺ 2-step (exact two-step)'),
        ('delta_degrade',   'Δ⁻ Degradation (correct→wrong)'),
    ]

    all_results = {}
    for label_key, label_name in label_configs:
        n_pos = sum(1 for d in data if d[label_key] == 1)
        if n_pos < 10:
            print(f"\n[!] Skipping {label_name}: only {n_pos} positive events (need ≥10)")
            continue

        print(f"\n{'='*60}")
        print(f"Training probe: {label_name}")
        print(f"{'='*60}")

        results = train_delta_probe(
            data, label_key, args.output_dir, label_name,
            hidden_dim=args.hidden_dim, num_epochs=args.epochs,
            seed=args.seed, device=device)
        all_results[label_key] = results

    # Summary table
    print(f"\n{'='*70}")
    print(f"EXTENDED DELTA PROBE RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Label':<25} {'AUROC':>8} {'AUPRC':>8} {'×Rand':>6} {'Pos%':>6} {'N_pos':>6}")
    print(f"{'-'*70}")
    for label_key in ['delta_1step', 'delta_multistep', 'delta_2step', 'delta_degrade']:
        if label_key in all_results:
            r = all_results[label_key]
            print(f"{r['label_name']:<25} {r['test_auroc']:>8.4f} {r['test_auprc']:>8.4f} "
                  f"{r['auprc_ratio']:>6.1f} {r['positive_rate_all']:>6.3f} {r['n_pos']:>6}")
    print(f"{'='*70}")

    # Save all results
    all_path = os.path.join(args.output_dir, 'extended_delta_all_results.json')
    with open(all_path, 'w') as f:
        json.dump({'statistics': stats, 'probe_results': all_results}, f, indent=2)
    print(f"\n[*] All results saved to {all_path}")


if __name__ == '__main__':
    main()
