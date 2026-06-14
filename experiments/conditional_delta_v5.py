"""
Conditional Delta Probe analysis (W1 fix, Loop 26) — canonical V5 data.

The canonical Delta Probe evaluates the JOINT event 1[wrong_t & correct_{t+1}]
over all transitions. Routing requires the CONDITIONAL estimand
P(correct_{t+1} | wrong_t). This script reports:

  1. cond_benefit:  probe trained+evaluated ONLY on wrong_t transitions
                    (positives = flip to correct, negatives = stay wrong)
  2. cond_degradation: probe trained+evaluated ONLY on correct_t transitions
  3. conditional asymmetry (deg - ben, MLP) with query-grouped bootstrap
  4. leakage control: correctness-direction classifier (predict wrong_t)
     scored as a benefit/degradation classifier on the full transition set —
     quantifies how much of the joint AUROC is pure correctness decoding
  5. cross-eval: joint-trained probes evaluated within the conditional subsets

Protocol matched to canonical_v5_full.py: same features (4-layer concat,
14336-d), same MLP ([256,128], ReLU, dropout 0.2, pos-weighted BCE, early
stopping), 5-fold GroupKFold by query, 5000 query-grouped bootstrap resamples.
"""
import json, numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import torch, torch.nn as nn

DATA = "/home/shenjikun/experiments/hazard-early-stopping/data/collected_states_hotpotqa_v5_2000.jsonl"
OUT = "/home/shenjikun/experiments/hazard-early-stopping/results/conditional_delta_v5.json"
N_BOOT = 5000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading V5 data...")
records = [json.loads(line) for line in open(DATA)]
print(f"  {len(records)} records")

queries = defaultdict(list)
for r in records:
    queries[r["query_id"]].append(r)

X, qids, stage_idx_list, cur_list, nxt_list = [], [], [], [], []
for qid, stages in queries.items():
    stages.sort(key=lambda x: x["stage_idx"])
    if len(stages) < 4:
        continue
    for t in range(3):
        cur = stages[t]["stage_correctness"]
        nxt = stages[t + 1]["stage_correctness"]
        ml = stages[t]["multi_layer_hidden_states"]
        X.append(np.array(ml, dtype=np.float32).flatten())
        qids.append(qid)
        stage_idx_list.append(t)
        cur_list.append(int(cur))
        nxt_list.append(int(nxt))

X = np.array(X)
qids = np.array(qids)
stage_idx_arr = np.array(stage_idx_list)
cur_arr = np.array(cur_list)
nxt_arr = np.array(nxt_list)
y_ben = ((cur_arr == 0) & (nxt_arr == 1)).astype(int)
y_deg = ((cur_arr == 1) & (nxt_arr == 0)).astype(int)
y_wrong = (cur_arr == 0).astype(int)  # correctness-direction target

print(f"Transitions: {len(X)}, feat dim: {X.shape[1]}")
print(f"wrong_t: {y_wrong.sum()} ({y_wrong.mean()*100:.1f}%)")
print(f"benefit (joint): {y_ben.sum()} ({y_ben.mean()*100:.1f}%)")
print(f"degradation (joint): {y_deg.sum()} ({y_deg.mean()*100:.1f}%)")


def train_cv(X_, y_, qids_, tag):
    """5-fold GroupKFold LR + MLP, returns OOF predictions. Matches canonical."""
    gkf = GroupKFold(n_splits=5)
    oof_lr = np.full(len(y_), np.nan)
    oof_mlp = np.full(len(y_), np.nan)
    for fold, (tr, te) in enumerate(gkf.split(X_, y_, groups=qids_)):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_[tr]); Xte = sc.transform(X_[te])
        lr = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=42)
        lr.fit(Xtr, y_[tr]); oof_lr[te] = lr.predict_proba(Xte)[:, 1]

        Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
        Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
        ytr_t = torch.tensor(y_[tr], dtype=torch.float32, device=device)
        model = nn.Sequential(
            nn.Linear(X_.shape[1], 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1)).to(device)
        pos_w = (len(y_[tr]) - y_[tr].sum()) / max(y_[tr].sum(), 1)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        np.random.seed(42 + fold)
        n_val = int(0.15 * len(Xtr))
        val_idx = np.random.choice(len(Xtr), n_val, replace=False)
        tr_mask = np.ones(len(Xtr), dtype=bool); tr_mask[val_idx] = False
        best_auroc, best_state, patience = 0, None, 0
        for epoch in range(100):
            model.train()
            perm = torch.randperm(int(tr_mask.sum()), device=device)
            for i in range(0, int(tr_mask.sum()), 128):
                idx = perm[i:i + 128]
                opt.zero_grad()
                loss = loss_fn(model(Xtr_t[tr_mask][idx]).squeeze(-1), ytr_t[tr_mask][idx])
                loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                vp = torch.sigmoid(model(Xtr_t[val_idx]).squeeze(-1)).cpu().numpy()
                if len(set(y_[tr][val_idx])) > 1:
                    va = roc_auc_score(y_[tr][val_idx], vp)
                else:
                    va = 0.5
                if va > best_auroc:
                    best_auroc = va
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
            if patience >= 15:
                break
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            oof_mlp[te] = torch.sigmoid(model(Xte_t).squeeze(-1)).cpu().numpy()
        print(f"  [{tag}] fold {fold} done (val AUROC {best_auroc:.3f})")
    return oof_lr, oof_mlp


def boot_ci(y_, score_, qids_, rng, n_boot=N_BOOT):
    """Query-grouped bootstrap of AUROC/AUPRC. Matches canonical."""
    uq = list(set(qids_)); nq = len(uq)
    q2i = defaultdict(list)
    for i, q in enumerate(qids_):
        q2i[q].append(i)
    aurocs, auprcs = [], []
    for _ in range(n_boot):
        sq = rng.choice(uq, nq, replace=True)
        idx = []
        for q in sq:
            idx.extend(q2i[q])
        idx = np.array(idx)
        if len(set(y_[idx])) < 2:
            continue
        aurocs.append(roc_auc_score(y_[idx], score_[idx]))
        auprcs.append(average_precision_score(y_[idx], score_[idx]))
    a, p = np.array(aurocs), np.array(auprcs)
    return {"auroc": float(a.mean()), "auroc_lo": float(np.percentile(a, 2.5)),
            "auroc_hi": float(np.percentile(a, 97.5)),
            "auprc": float(p.mean()), "auprc_lo": float(np.percentile(p, 2.5)),
            "auprc_hi": float(np.percentile(p, 97.5)),
            "n": int(len(y_)), "n_pos": int(y_.sum()), "prev": float(y_.mean())}


out = {}
rng = np.random.RandomState(42)

# ---- 1+2: conditional probes ----
wrong_mask = cur_arr == 0
correct_mask = cur_arr == 1
Xw, qw, yw = X[wrong_mask], qids[wrong_mask], nxt_arr[wrong_mask]          # benefit | wrong_t
Xc, qc, yc = X[correct_mask], qids[correct_mask], (1 - nxt_arr)[correct_mask]  # degradation | correct_t
print(f"\ncond_benefit subset: n={len(yw)}, pos={yw.sum()} ({yw.mean()*100:.1f}%)")
print(f"cond_degradation subset: n={len(yc)}, pos={yc.sum()} ({yc.mean()*100:.1f}%)")

print("\nTraining conditional benefit probe (wrong_t only)...")
lr_w, mlp_w = train_cv(Xw, yw, qw, "cond_ben")
print("Training conditional degradation probe (correct_t only)...")
lr_c, mlp_c = train_cv(Xc, yc, qc, "cond_deg")

out["cond_benefit_lr"] = boot_ci(yw, lr_w, qw, rng)
out["cond_benefit_mlp"] = boot_ci(yw, mlp_w, qw, rng)
out["cond_degradation_lr"] = boot_ci(yc, lr_c, qc, rng)
out["cond_degradation_mlp"] = boot_ci(yc, mlp_c, qc, rng)

# ---- 3: conditional asymmetry (deg - ben, MLP), joint query bootstrap ----
uq_all = list(set(qids)); nq_all = len(uq_all)
q2i_w = defaultdict(list); q2i_c = defaultdict(list)
for i, q in enumerate(qw): q2i_w[q].append(i)
for i, q in enumerate(qc): q2i_c[q].append(i)
asym = []
for _ in range(N_BOOT):
    sq = rng.choice(uq_all, nq_all, replace=True)
    iw, ic = [], []
    for q in sq:
        iw.extend(q2i_w.get(q, [])); ic.extend(q2i_c.get(q, []))
    iw, ic = np.array(iw), np.array(ic)
    if len(set(yw[iw])) < 2 or len(set(yc[ic])) < 2:
        continue
    asym.append(roc_auc_score(yc[ic], mlp_c[ic]) - roc_auc_score(yw[iw], mlp_w[iw]))
asym = np.array(asym)
out["cond_asymmetry_deg_minus_ben_mlp"] = {
    "mean": float(asym.mean()), "lo": float(np.percentile(asym, 2.5)),
    "hi": float(np.percentile(asym, 97.5)),
    "p_deg_ge_ben": float((asym >= 0).mean())}

# ---- 4: correctness-leakage control ----
print("\nTraining correctness-direction classifier (predict wrong_t) on full set...")
lr_dir, mlp_dir = train_cv(X, y_wrong, qids, "corr_dir")
out["leakage_wrongness_as_benefit_lr"] = boot_ci(y_ben, lr_dir, qids, rng)
out["leakage_wrongness_as_benefit_mlp"] = boot_ci(y_ben, mlp_dir, qids, rng)
out["leakage_correctness_as_degradation_lr"] = boot_ci(y_deg, 1 - lr_dir, qids, rng)
out["leakage_correctness_as_degradation_mlp"] = boot_ci(y_deg, 1 - mlp_dir, qids, rng)
out["correctness_direction_auroc_mlp"] = boot_ci(y_wrong, mlp_dir, qids, rng)

# ---- 5: cross-eval — joint-trained probes scored within conditional subsets ----
print("\nTraining joint (canonical-style) probes for cross-eval...")
lr_jb, mlp_jb = train_cv(X, y_ben, qids, "joint_ben")
lr_jd, mlp_jd = train_cv(X, y_deg, qids, "joint_deg")
out["joint_benefit_mlp_full"] = boot_ci(y_ben, mlp_jb, qids, rng)        # canonical replication
out["joint_degradation_mlp_full"] = boot_ci(y_deg, mlp_jd, qids, rng)
out["joint_benefit_mlp_within_wrong"] = boot_ci(yw, mlp_jb[wrong_mask], qw, rng)
out["joint_degradation_mlp_within_correct"] = boot_ci(yc, mlp_jd[correct_mask], qc, rng)

# ---- per-stage conditional breakdown (point estimates) ----
stage_names = {0: "S0->S1", 1: "S1->S2", 2: "S2->S3"}
per_stage = {}
sw = stage_idx_arr[wrong_mask]; sc_ = stage_idx_arr[correct_mask]
for t in range(3):
    e = {}
    mw = sw == t
    if yw[mw].sum() > 0 and len(set(yw[mw])) > 1:
        e["cond_benefit"] = {"n": int(mw.sum()), "n_pos": int(yw[mw].sum()),
                             "prev": float(yw[mw].mean()),
                             "auroc_mlp": float(roc_auc_score(yw[mw], mlp_w[mw]))}
    mc = sc_ == t
    if yc[mc].sum() > 0 and len(set(yc[mc])) > 1:
        e["cond_degradation"] = {"n": int(mc.sum()), "n_pos": int(yc[mc].sum()),
                                 "prev": float(yc[mc].mean()),
                                 "auroc_mlp": float(roc_auc_score(yc[mc], mlp_c[mc]))}
    per_stage[stage_names[t]] = e
out["per_stage_conditional"] = per_stage

# ---- report ----
print("\n" + "=" * 70)
print("CONDITIONAL DELTA PROBE — V5 (W1 fix)")
print("=" * 70)
for k in ["cond_benefit_lr", "cond_benefit_mlp", "cond_degradation_lr", "cond_degradation_mlp",
          "leakage_wrongness_as_benefit_mlp", "leakage_correctness_as_degradation_mlp",
          "correctness_direction_auroc_mlp",
          "joint_benefit_mlp_full", "joint_degradation_mlp_full",
          "joint_benefit_mlp_within_wrong", "joint_degradation_mlp_within_correct"]:
    r = out[k]
    print(f"  {k:<42} AUROC={r['auroc']:.3f}[{r['auroc_lo']:.3f},{r['auroc_hi']:.3f}] "
          f"AUPRC={r['auprc']:.3f}[{r['auprc_lo']:.3f},{r['auprc_hi']:.3f}] "
          f"n={r['n']} pos={r['n_pos']} ({r['prev']*100:.1f}%)")
a = out["cond_asymmetry_deg_minus_ben_mlp"]
print(f"\nConditional asymmetry (deg-ben, MLP): {a['mean']:.3f} [{a['lo']:.3f},{a['hi']:.3f}] "
      f"P(deg>=ben)={a['p_deg_ge_ben']:.4f}")
print("\nPer-stage conditional:")
for s, e in per_stage.items():
    print(f"  {s}: {e}")

with open(OUT, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved -> {OUT}")
