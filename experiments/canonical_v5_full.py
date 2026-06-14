"""
Canonical V5 full recompute — ONE self-consistent dataset.
Computes: LR+MLP AUROC/AUPRC with bootstrap CIs, asymmetry test,
per-stage transition breakdown — all from the same WSL2 V5 collection.
"""
import json, numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import torch, torch.nn as nn

DATA = "/home/shenjikun/experiments/hazard-early-stopping/data/collected_states_hotpotqa_v5_2000.jsonl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading V5 data...")
records = [json.loads(line) for line in open(DATA)]
print(f"  {len(records)} records")

# Group by query, compute Delta labels + per-stage breakdown
queries = defaultdict(list)
for r in records:
    queries[r["query_id"]].append(r)

stage_names = {0: "S0->S1", 1: "S1->S2", 2: "S2->S3"}
stage_trans = {0: defaultdict(int), 1: defaultdict(int), 2: defaultdict(int)}
overall_trans = defaultdict(int)

# Build feature matrix + labels for transitions (stages 0,1,2)
X, qids, stage_idx_list = [], [], []
y_ben, y_deg = [], []

for qid, stages in queries.items():
    stages.sort(key=lambda x: x["stage_idx"])
    if len(stages) < 4:
        continue
    for t in range(3):
        cur = stages[t]["stage_correctness"]
        nxt = stages[t+1]["stage_correctness"]
        # Per-stage transition matrix
        if cur == 0 and nxt == 0:   key = "stable_wrong"
        elif cur == 0 and nxt == 1: key = "benefit"
        elif cur == 1 and nxt == 0: key = "degrad"
        else:                        key = "stable_correct"
        stage_trans[t][key] += 1
        overall_trans[key] += 1
        # Feature = concat of 4 layers (multi_layer_hidden_states)
        ml = stages[t]["multi_layer_hidden_states"]
        feat = np.array(ml, dtype=np.float32).flatten()  # 4*3584 = 14336
        X.append(feat)
        qids.append(qid)
        stage_idx_list.append(t)
        y_ben.append(int(cur == 0 and nxt == 1))
        y_deg.append(int(cur == 1 and nxt == 0))

X = np.array(X); qids = np.array(qids)
y_ben = np.array(y_ben); y_deg = np.array(y_deg)
stage_idx_arr = np.array(stage_idx_list)

print(f"Transitions: {len(X)}, feat dim: {X.shape[1]}")
print(f"Benefit: {y_ben.sum()} ({y_ben.mean()*100:.1f}%)")
print(f"Degradation: {y_deg.sum()} ({y_deg.mean()*100:.1f}%)")

# ---- 5-fold CV: LR + MLP ----
gkf = GroupKFold(n_splits=5)
results = {}

for name, y in [("benefit", y_ben), ("degradation", y_deg)]:
    oof_lr = np.full(len(y), np.nan)
    oof_mlp = np.full(len(y), np.nan)
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=qids)):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        # LR
        lr = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=42)
        lr.fit(Xtr, y[tr]); oof_lr[te] = lr.predict_proba(Xte)[:, 1]
        # MLP
        Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
        Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
        ytr_t = torch.tensor(y[tr], dtype=torch.float32, device=device)
        model = nn.Sequential(
            nn.Linear(X.shape[1], 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1)).to(device)
        pos_w = (len(y[tr])-y[tr].sum())/max(y[tr].sum(),1)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        np.random.seed(42+fold)
        n_val = int(0.15*len(Xtr))
        val_idx = np.random.choice(len(Xtr), n_val, replace=False)
        tr_mask = np.ones(len(Xtr), dtype=bool); tr_mask[val_idx] = False
        best_auroc, best_state, patience = 0, None, 0
        for epoch in range(100):
            model.train()
            perm = torch.randperm(tr_mask.sum(), device=device)
            for i in range(0, tr_mask.sum(), 128):
                idx = perm[i:i+128]
                opt.zero_grad()
                loss = loss_fn(model(Xtr_t[tr_mask][idx]).squeeze(-1), ytr_t[tr_mask][idx])
                loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                vp = torch.sigmoid(model(Xtr_t[val_idx]).squeeze(-1)).cpu().numpy()
                va = roc_auc_score(y[tr][val_idx], vp)
                if va > best_auroc:
                    best_auroc = va
                    best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
                    patience = 0
                else: patience += 1
            if patience >= 15: break
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            oof_mlp[te] = torch.sigmoid(model(Xte_t).squeeze(-1)).cpu().numpy()
    results[name] = {"y": y, "oof_lr": oof_lr, "oof_mlp": oof_mlp}

# ---- Bootstrap CIs (grouped by query) ----
rng = np.random.RandomState(42)
unique_qids = list(set(qids))
nq = len(unique_qids)
qid_to_idx = defaultdict(list)
for i, q in enumerate(qids): qid_to_idx[q].append(i)

out = {}
for name in ["benefit", "degradation"]:
    y = results[name]["y"]
    for model_name, oof in [("lr", results[name]["oof_lr"]), ("mlp", results[name]["oof_mlp"])]:
        aurocs, auprcs = [], []
        for _ in range(5000):
            sq = rng.choice(unique_qids, nq, replace=True)
            idx = []
            for q in sq: idx.extend(qid_to_idx[q])
            idx = np.array(idx)
            aurocs.append(roc_auc_score(y[idx], oof[idx]))
            auprcs.append(average_precision_score(y[idx], oof[idx]))
        aurocs = np.array(aurocs); auprcs = np.array(auprcs)
        out[f"{name}_{model_name}"] = {
            "auroc": float(aurocs.mean()), "auroc_lo": float(np.percentile(aurocs,2.5)),
            "auroc_hi": float(np.percentile(aurocs,97.5)),
            "auprc": float(auprcs.mean()), "auprc_lo": float(np.percentile(auprcs,2.5)),
            "auprc_hi": float(np.percentile(auprcs,97.5)),
            "n_pos": int(y.sum()), "prev": float(y.mean())}

# Asymmetry test
ben_mlp = results["benefit"]["oof_mlp"]; deg_mlp = results["degradation"]["oof_mlp"]
y_b = results["benefit"]["y"]; y_d = results["degradation"]["y"]
asym = []
for _ in range(5000):
    sq = rng.choice(unique_qids, nq, replace=True)
    idx = []
    for q in sq: idx.extend(qid_to_idx[q])
    idx = np.array(idx)
    asym.append(roc_auc_score(y_d[idx], deg_mlp[idx]) - roc_auc_score(y_b[idx], ben_mlp[idx]))
asym = np.array(asym)
out["asymmetry"] = {"mean": float(asym.mean()), "lo": float(np.percentile(asym,2.5)),
                    "hi": float(np.percentile(asym,97.5)), "p_neg": float((asym<0).mean())}

# ---- Per-stage AUPRC/AUROC (groupby on OOF predictions) ----
per_stage = {}
for t in range(3):
    smask = stage_idx_arr == t
    per_stage[stage_names[t]] = {}
    for name in ["benefit", "degradation"]:
        y = results[name]["y"][smask]
        if y.sum() == 0:
            per_stage[stage_names[t]][name] = {"n_pos": 0, "auroc": None}
            continue
        mlp = results[name]["oof_mlp"][smask]
        per_stage[stage_names[t]][name] = {
            "n_pos": int(y.sum()), "n": int(smask.sum()), "prev": float(y.mean()),
            "auroc": float(roc_auc_score(y, mlp)) if len(set(y))>1 else None}

# ---- Output ----
print("\n" + "="*70)
print("CANONICAL V5 — SELF-CONSISTENT (WSL2 collection)")
print("="*70)
print("\nDelta Probe bootstrap CIs:")
for name in ["benefit", "degradation"]:
    for m in ["lr", "mlp"]:
        r = out[f"{name}_{m}"]
        print(f"  {name:<12} {m.upper():<4} AUROC={r['auroc']:.3f}[{r['auroc_lo']:.3f},{r['auroc_hi']:.3f}] "
              f"AUPRC={r['auprc']:.3f}[{r['auprc_lo']:.3f},{r['auprc_hi']:.3f}] n_pos={r['n_pos']} ({r['prev']*100:.1f}%)")
print(f"\nAsymmetry (deg-ben MLP): {out['asymmetry']['mean']:.3f} "
      f"[{out['asymmetry']['lo']:.3f},{out['asymmetry']['hi']:.3f}] p={1-out['asymmetry']['p_neg']:.4f}")

print("\nPer-stage transition matrix:")
print(f"{'Stage':<10} {'stable_wrong':>14} {'benefit':>12} {'degrad':>12} {'stable_correct':>16} {'Total':>7}")
for t in range(3):
    d = stage_trans[t]; tot = sum(d.values())
    print(f"{stage_names[t]:<10} {d['stable_wrong']:>5}({100*d['stable_wrong']/tot:.1f}%) {d['benefit']:>4}({100*d['benefit']/tot:.1f}%) "
          f"{d['degrad']:>4}({100*d['degrad']/tot:.1f}%) {d['stable_correct']:>7}({100*d['stable_correct']/tot:.1f}%) {tot:>7}")
tot_all = sum(overall_trans.values())
print(f"{'Overall':<10} {overall_trans['stable_wrong']:>5}({100*overall_trans['stable_wrong']/tot_all:.1f}%) "
      f"{overall_trans['benefit']:>4}({100*overall_trans['benefit']/tot_all:.1f}%) "
      f"{overall_trans['degrad']:>4}({100*overall_trans['degrad']/tot_all:.1f}%) "
      f"{overall_trans['stable_correct']:>7}({100*overall_trans['stable_correct']/tot_all:.1f}%) {tot_all:>7}")

print("\nPer-stage Delta AUROC (MLP):")
for t in range(3):
    s = stage_names[t]
    b = per_stage[s]["benefit"]; d = per_stage[s]["degradation"]
    print(f"  {s}: benefit n={b['n_pos']} ({b['prev']*100:.1f}%) AUROC={b['auroc']:.3f if b['auroc'] else 0}  "
          f"degrad n={d['n_pos']} ({d['prev']*100:.1f}%) AUROC={d['auroc'] if d['auroc'] else 'NA'}")

# Save
result_out = {"bootstrap": out, "per_stage_transition": {stage_names[t]: dict(stage_trans[t]) for t in range(3)},
              "overall_transition": dict(overall_trans), "per_stage_auroc": per_stage,
              "total_transitions": tot_all}
with open("/home/shenjikun/experiments/hazard-early-stopping/results/canonical_v5_full.json", "w") as f:
    json.dump(result_out, f, indent=2)
print("\nSaved: results/canonical_v5_full.json")
