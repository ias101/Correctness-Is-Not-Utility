"""
Enhanced Routing Baselines: Direct Policy Learning, Cost-Sensitive Optimization,
and Oracle-Informed Routing.

Implements stronger routing baselines that go beyond correctness-probability
thresholding, addressing the concern that routing failure may be a policy-design
problem rather than a hidden-state signal problem.

Baselines:
  1. Direct Stop/Continue Classifier — binary decision at each stage
  2. Multi-Class Stage Selector — predict optimal stop stage directly
  3. Cost-Sensitive CWA-Optimized Policy — loss weighted by CWA impact
  4. Degradation-Aware Router — use degradation probe (AUROC 0.85) to avoid harm
  5. Oracle Policy Upper Bound — best possible performance of any policy

All baselines use existing collected hidden states (no re-collection needed).

Author: Loop 12 — Enhanced Routing Baselines
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
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experiments'))
from config import BASE_SEED, NUM_STAGES, STAGES

# ── Cost Model ─────────────────────────────────────────────────────────────────
STAGE_COSTS = np.array([0.02, 0.05, 0.01, 1.00])
MAX_COST = STAGE_COSTS.sum()  # 1.08


def cost_of_pipeline(stop_stage):
    return STAGE_COSTS[:stop_stage + 1].sum()


def compute_cwa(acc, avg_cost, lam=0.5):
    return acc - lam * (avg_cost / MAX_COST)


# ── Label Construction ─────────────────────────────────────────────────────────

def build_routing_labels(data: List[Dict], label_type: str = 'oracle_stop_stage'):
    """Build labels for different routing policies.

    label_type:
      - 'oracle_stop_stage': optimal stop stage (0-3), earliest correct stage
      - 'oracle_continue': binary, 1 if continuing could improve answer
      - 'cwa_optimal_stop': stop stage that maximizes CWA for this query
      - 'degradation_avoid': label 1 if continuing would degrade answer
    """
    by_query = defaultdict(list)
    for d in data:
        by_query[d['query_id']].append(d)

    for qid, stages in by_query.items():
        stages_sorted = sorted(stages, key=lambda s: s['stage_idx'])
        correctness = [s.get('stage_correctness', 0) for s in stages_sorted]

        # Oracle: earliest correct stage
        earliest_correct = None
        for t, c in enumerate(correctness):
            if c == 1:
                earliest_correct = t
                break

        for i, s in enumerate(stages_sorted):
            t = s['stage_idx']

            if label_type == 'oracle_stop_stage':
                s['oracle_stop_stage'] = earliest_correct if earliest_correct is not None else 3

            elif label_type == 'oracle_continue':
                # Continue if: currently wrong AND will become correct at some later stage
                should_continue = 0
                if t < 3 and correctness[t] == 0:
                    if any(correctness[tt] == 1 for tt in range(t + 1, 4)):
                        should_continue = 1
                s['oracle_continue'] = should_continue

            elif label_type == 'cwa_optimal_stop':
                # Compute CWA for each possible stop stage
                best_stage = t  # default: stop now
                best_cwa = -float('inf')
                for stop_t in range(t, 4):
                    acc = correctness[stop_t]
                    cost = cost_of_pipeline(stop_t)
                    cwa = compute_cwa(acc, cost)
                    if cwa > best_cwa:
                        best_cwa = cwa
                        best_stage = stop_t
                s['cwa_optimal_stop'] = best_stage

            elif label_type == 'degradation_avoid':
                # 1 if continuing to t+1 would degrade (correct->wrong)
                s['degradation_avoid'] = 0
                if t < 3 and correctness[t] == 1 and correctness[t + 1] == 0:
                    s['degradation_avoid'] = 1

    return data


# ── Models ─────────────────────────────────────────────────────────────────────

class RoutingPolicyMLP(nn.Module):
    """MLP for routing decisions: stop/continue or multi-class stage selection."""
    def __init__(self, hidden_dim=3584, num_classes=2, mlp_hidden=None, dropout=0.2):
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = [256, 128]
        layers = []
        in_dim = hidden_dim + NUM_STAGES  # hidden state + stage onehot
        for h in mlp_hidden:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)
        self.num_classes = num_classes

    def forward(self, h, stage_idx):
        stage_onehot = torch.zeros(h.size(0), NUM_STAGES, device=h.device)
        stage_onehot.scatter_(1, stage_idx.unsqueeze(1), 1.0)
        return self.net(torch.cat([h, stage_onehot], dim=1))


class CostSensitiveLoss(nn.Module):
    """BCE loss weighted by the CWA impact of false positive vs false negative.

    For stop/continue decisions:
      - False positive (stop when should continue): lose potential accuracy gain
      - False negative (continue when should stop): waste compute cost
    """
    def __init__(self, fp_cost=0.05, fn_cost=0.02):
        super().__init__()
        self.fp_cost = fp_cost
        self.fn_cost = fn_cost

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        # Weight FP (predict continue but oracle says stop) more heavily
        fp_weight = torch.where((probs > 0.5) & (targets == 0),
                                torch.tensor(self.fp_cost, device=logits.device),
                                torch.tensor(1.0, device=logits.device))
        fn_weight = torch.where((probs < 0.5) & (targets == 1),
                                torch.tensor(self.fn_cost, device=logits.device),
                                torch.tensor(1.0, device=logits.device))
        return (bce * fp_weight * fn_weight).mean()


# ── Dataset ────────────────────────────────────────────────────────────────────

class RoutingDataset(Dataset):
    def __init__(self, data, label_key, hidden_dim=3584, scaler=None):
        self.stage_indices = torch.tensor([d['stage_idx'] for d in data], dtype=torch.long)
        self.labels = torch.tensor([d[label_key] for d in data], dtype=torch.long)

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

def train_routing_policy(data, label_key, output_dir, label_name,
                         num_classes=2, use_cost_sensitive=False,
                         hidden_dim=3584, num_epochs=100, lr=1e-3,
                         batch_size=64, patience=20, seed=BASE_SEED, device='cuda'):
    """Train a routing policy model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Label distribution
    labels = [d[label_key] for d in data]
    label_counts = {i: labels.count(i) for i in range(num_classes)}

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

    train_ds = RoutingDataset(train_data, label_key, hidden_dim)
    val_ds = RoutingDataset(val_data, label_key, hidden_dim, scaler=train_ds.scaler)
    test_ds = RoutingDataset(test_data, label_key, hidden_dim, scaler=train_ds.scaler)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = RoutingPolicyMLP(hidden_dim=hidden_dim, num_classes=num_classes).to(device)

    if num_classes == 2:
        pos_count = sum(1 for d in train_data if d[label_key] == 1)
        neg_count = len(train_data) - pos_count
        pos_weight = neg_count / max(pos_count, 1)
        if use_cost_sensitive:
            criterion = CostSensitiveLoss(fp_cost=0.05, fn_cost=0.02)
        else:
            criterion = nn.CrossEntropyLoss(
                weight=torch.tensor([1.0, pos_weight], device=device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=patience // 2)

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            feats = batch['features'].to(device)
            si = batch['stage_idx'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            logits = model(feats, si)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                feats = batch['features'].to(device)
                si = batch['stage_idx'].to(device)
                logits = model(feats, si)
                preds = logits.argmax(dim=1)
                val_correct += (preds == batch['label'].to(device)).sum().item()
                val_total += len(preds)
        val_acc = val_correct / val_total
        scheduler.step(val_acc)

        if val_acc > best_val_acc + 0.001:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({'model_state_dict': model.state_dict(), 'val_acc': val_acc,
                        'epoch': best_epoch, 'num_classes': num_classes},
                       os.path.join(output_dir, f'routing_{label_key}_best.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Test evaluation
    ckpt = torch.load(os.path.join(output_dir, f'routing_{label_key}_best.pt'),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            feats = batch['features'].to(device)
            si = batch['stage_idx'].to(device)
            logits = model(feats, si)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(batch['label'].tolist())

    test_acc = accuracy_score(all_labels, all_preds)
    per_class_acc = {}
    for c in range(num_classes):
        idxs = [j for j, l in enumerate(all_labels) if l == c]
        if idxs:
            per_class_acc[f'class_{c}'] = sum(1 for j in idxs if all_preds[j] == c) / len(idxs)

    # Simulate routing performance
    routing_results = simulate_routing(test_data, model, label_key, device)

    results = {
        'label_key': label_key, 'label_name': label_name,
        'num_classes': num_classes,
        'label_distribution': label_counts,
        'test_accuracy': float(test_acc),
        'per_class_accuracy': per_class_acc,
        'best_epoch': best_epoch,
        'routing_results': routing_results,
        'n_train': len(train_data), 'n_val': len(val_data), 'n_test': len(test_data),
    }

    with open(os.path.join(output_dir, f'routing_{label_key}_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  [{label_name}] Test Acc={test_acc:.4f}, "
          f"Routing CWA(0.5)={routing_results.get('cwa_0.5', 'N/A')}")
    return results


def simulate_routing(test_data, model, label_key, device):
    """Simulate stage-by-stage routing using the learned policy.

    For direct stop/continue classifier: at each stage, decide stop or continue.
    For stage selector: directly predict which stage to stop at.
    """
    by_query = defaultdict(list)
    for d in test_data:
        by_query[d['query_id']].append(d)

    correct = []
    costs = []

    model.eval()
    with torch.no_grad():
        for qid, stages in by_query.items():
            stages_sorted = sorted(stages, key=lambda s: s['stage_idx'])

            if 'oracle_continue' in label_key or 'degrade' in label_key:
                # Binary stop/continue: iterate through stages
                stop_stage = 3  # default: run full pipeline
                for s in stages_sorted:
                    t = s['stage_idx']
                    if t == 3:
                        stop_stage = 3
                        break
                    hs = torch.tensor(s['hidden_state'], dtype=torch.float32).unsqueeze(0).to(device)
                    si = torch.tensor([t], dtype=torch.long).to(device)
                    logits = model(hs, si)
                    decision = logits.argmax(dim=1).item()
                    # decision=0: stop, decision=1: continue
                    if decision == 0:  # STOP
                        stop_stage = t
                        break
                    # else CONTINUE to next stage
            elif 'oracle_stop' in label_key or 'cwa_optimal' in label_key:
                # Direct stage prediction: use hidden state at S0
                s0 = stages_sorted[0]
                hs = torch.tensor(s0['hidden_state'], dtype=torch.float32).unsqueeze(0).to(device)
                si = torch.tensor([0], dtype=torch.long).to(device)
                logits = model(hs, si)
                stop_stage = logits.argmax(dim=1).item()
            else:
                stop_stage = 3

            # Record outcome at stop stage
            stop_s = stages_sorted[stop_stage]
            correct.append(stop_s.get('stage_correctness', 0))
            costs.append(cost_of_pipeline(stop_stage))

    correct = np.array(correct)
    costs = np.array(costs)

    return {
        'accuracy': float(correct.mean()),
        'avg_cost': float(costs.mean()),
        'cwa_0.2': float(compute_cwa(correct.mean(), costs.mean(), 0.2)),
        'cwa_0.5': float(compute_cwa(correct.mean(), costs.mean(), 0.5)),
        'cwa_0.8': float(compute_cwa(correct.mean(), costs.mean(), 0.8)),
    }


def compute_oracle_policy_upper_bound(data):
    """Compute the best possible routing performance any policy could achieve.

    Oracle: at each stage, knows whether stopping now is optimal.
    This is the theoretical upper bound for any routing policy.
    """
    by_query = defaultdict(list)
    for d in data:
        by_query[d['query_id']].append(d)

    correct = []
    costs = []

    for qid, stages in by_query.items():
        stages_sorted = sorted(stages, key=lambda s: s['stage_idx'])
        correctness = [s.get('stage_correctness', 0) for s in stages_sorted]

        # Find stop stage that maximizes CWA (true oracle)
        best_stage = 0
        best_cwa = -float('inf')
        for stop_t in range(4):
            acc = correctness[stop_t]
            cost = cost_of_pipeline(stop_t)
            cwa = compute_cwa(acc, cost)
            if cwa > best_cwa:
                best_cwa = cwa
                best_stage = stop_t

        correct.append(correctness[best_stage])
        costs.append(cost_of_pipeline(best_stage))

    correct = np.array(correct)
    costs = np.array(costs)

    return {
        'policy_type': 'Oracle CWA-Optimal Policy',
        'accuracy': float(correct.mean()),
        'avg_cost': float(costs.mean()),
        'cwa_0.2': float(compute_cwa(correct.mean(), costs.mean(), 0.2)),
        'cwa_0.5': float(compute_cwa(correct.mean(), costs.mean(), 0.5)),
        'cwa_0.8': float(compute_cwa(correct.mean(), costs.mean(), 0.8)),
    }


def compute_fixed_baselines(data):
    """Compute fixed-Sk baselines for comparison."""
    results = {}
    for stop_k, name in [(0, 'fixed_S0'), (1, 'fixed_S1'), (2, 'fixed_S2'), (3, 'full_pipeline')]:
        by_query = defaultdict(list)
        for d in data:
            by_query[d['query_id']].append(d)

        correct = []
        costs = []
        for qid, stages in by_query.items():
            stages_sorted = sorted(stages, key=lambda s: s['stage_idx'])
            if stop_k < len(stages_sorted):
                s = stages_sorted[stop_k]
                correct.append(s.get('stage_correctness', 0))
            else:
                s = stages_sorted[-1]
                correct.append(s.get('stage_correctness', 0))
            costs.append(cost_of_pipeline(stop_k))

        correct = np.array(correct)
        costs = np.array(costs)
        results[name] = {
            'accuracy': float(correct.mean()),
            'avg_cost': float(costs.mean()),
            'cwa_0.5': float(compute_cwa(correct.mean(), costs.mean(), 0.5)),
        }

    return results


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Enhanced Routing Baselines')
    ap.add_argument('--data', type=str,
                    default='data/collected_states_hotpotqa_v4.jsonl')
    ap.add_argument('--output_dir', type=str, default='results/enhanced_routing')
    ap.add_argument('--hidden_dim', type=int, default=3584)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--seed', type=int, default=BASE_SEED)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[*] Using device: {device}")

    # Load data
    print(f"[*] Loading data from {args.data}...")
    data = []
    with open(args.data) as f:
        for line in f:
            data.append(json.loads(line))
    print(f"[*] Loaded {len(data)} tuples from {len(set(d['query_id'] for d in data))} queries")

    # ── Fixed baselines ──
    print(f"\n{'='*60}")
    print("FIXED BASELINES")
    print(f"{'='*60}")
    fixed = compute_fixed_baselines(data)
    for name, r in fixed.items():
        print(f"  {name:<15}: Acc={r['accuracy']:.4f}, Cost={r['avg_cost']:.4f}, CWA(0.5)={r['cwa_0.5']:.4f}")

    # ── Oracle policy upper bound ──
    print(f"\n{'='*60}")
    print("ORACLE POLICY UPPER BOUND")
    print(f"{'='*60}")
    oracle_policy = compute_oracle_policy_upper_bound(data)
    for k, v in oracle_policy.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
    with open(os.path.join(args.output_dir, 'oracle_policy_bound.json'), 'w') as f:
        json.dump(oracle_policy, f, indent=2)

    # ── Train enhanced routing policies ──
    configs = [
        ('oracle_continue', 'Direct Stop/Continue (Oracle-Informed)', 2, False),
        ('oracle_continue', 'Cost-Sensitive Stop/Continue', 2, True),
        ('oracle_stop_stage', 'Multi-Class Stage Selector', 4, False),
        ('cwa_optimal_stop', 'CWA-Optimal Stage Selector', 4, False),
        ('degradation_avoid', 'Degradation-Avoid Router', 2, False),
    ]

    all_results = {'fixed_baselines': fixed, 'oracle_policy_bound': oracle_policy}

    for label_key, label_name, num_classes, cost_sensitive in configs:
        data = build_routing_labels(data, label_key)

        n_classes_actual = len(set(d[label_key] for d in data))
        print(f"\n{'='*60}")
        print(f"Training: {label_name} ({n_classes_actual} classes)")
        print(f"  Label distribution: { {i: sum(1 for d in data if d[label_key]==i) for i in range(n_classes_actual)} }")
        print(f"{'='*60}")

        results = train_routing_policy(
            data, label_key, args.output_dir, label_name,
            num_classes=num_classes, use_cost_sensitive=cost_sensitive,
            hidden_dim=args.hidden_dim, num_epochs=args.epochs,
            seed=args.seed, device=device)
        all_results[label_key] = results

    # ── Summary ──
    print(f"\n{'='*75}")
    print("ENHANCED ROUTING BASELINES — FINAL SUMMARY")
    print(f"{'='*75}")
    print(f"{'Method':<35} {'Acc':>7} {'Cost':>7} {'CWA(0.5)':>9}")
    print(f"{'-'*75}")

    # Fixed baselines
    for name in ['fixed_S0', 'fixed_S1', 'fixed_S2', 'full_pipeline']:
        if name in fixed:
            r = fixed[name]
            print(f"{name:<35} {r['accuracy']:>7.4f} {r['avg_cost']:>7.4f} {r['cwa_0.5']:>9.4f}")

    # Oracle
    print(f"{'Oracle Policy Upper Bound':<35} {oracle_policy['accuracy']:>7.4f} "
          f"{oracle_policy['avg_cost']:>7.4f} {oracle_policy['cwa_0.5']:>9.4f}")

    # Learned policies
    for label_key in ['oracle_continue', 'oracle_stop_stage', 'cwa_optimal_stop', 'degradation_avoid']:
        if label_key in all_results and 'routing_results' in all_results[label_key]:
            r = all_results[label_key]['routing_results']
            name = all_results[label_key]['label_name']
            print(f"{name:<35} {r['accuracy']:>7.4f} {r['avg_cost']:>7.4f} {r['cwa_0.5']:>9.4f}")

    print(f"{'='*75}")

    # Save
    with open(os.path.join(args.output_dir, 'enhanced_routing_all.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[*] Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
