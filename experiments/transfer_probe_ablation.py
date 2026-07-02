#!/usr/bin/env python3
"""
HotpotQA <-> PopQA zero-shot correctness-probe transfer ablation.

Tests whether a correctness probe trained on ONE dataset's Qwen2.5-7B hidden
states detects answer correctness on the OTHER dataset WITHOUT retraining.

LAYER-ORDER ALIGNMENT (critical):
  HotpotQA V5 stores `multi_layer_hidden_states` as [final, L-2, L-3, L-4].
  PopQA v4 stores `hs_concat` (14336-d) in the REVERSED order.
  Verified via 4x4 cross-dataset cosine of per-block mean activations:
      H0->P3 (0.81), H1->P2 (0.95), H2->P1 (0.95), H3->P0 (0.94)  => clean reversal.
  We reorder PopQA blocks to the canonical [final, L-2, L-3, L-4] order so that
  dimension i of the probe sees the SAME physical layer in both datasets.
  We also report the MISALIGNED (naive concat) transfer as a control, to
  quantify how much the layer-order bug would have cost.

Probe = MLP[256,128] (ReLU, dropout 0.2) + LR, StandardScaler. Same pipeline
that produced the paper's correctness AUROC (canonical_v5_full.py /
routing_hotpotqa_v5.py).

Protocol:
  - Zero-shot transfer: fit scaler+probe on 100% source, eval on 100% target
    (target scaler = SOURCE scaler; no target statistics are used -> true zero-shot).
  - In-domain reference: GroupKFold(5) OOF by query (matches paper).
  - Metrics: pooled AUROC + per-stage AUROC, query-grouped bootstrap 95% CI.

NOTE ON CONFOUND: HotpotQA hidden states are 4-bit (NF4); PopQA are BF16. The
paper's 4-bit/8-bit control (cosine >0.99) suggests precision is a minor driver,
but the final-layer cross-dataset cosine (0.81) shows genuine representational
shift. Transfer numbers therefore mix dataset shift + precision shift; reported
as-is and discussed.
"""
import json, gzip, argparse, os, time
import numpy as np
from collections import defaultdict
import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

N_STAGES = 4
SEED = 42
N_LAYERS = 4
LAYER_DIM = 3584


def _reverse_blocks(v):
    """Reverse the 4 layer-blocks of a 14336-d concat (3584 each)."""
    return np.concatenate(np.asarray(v, dtype=np.float32).reshape(N_LAYERS, LAYER_DIM)[::-1])


def load_dataset(path, align_popqa=True):
    """Return X[n,14336], y, stage, group(query_id) in canonical [final, ...] order.
    PopQA hs_concat is reversed relative to HotpotQA; reorder to align when align_popqa."""
    opener = gzip.open if path.endswith(".gz") else open
    by_q = defaultdict(dict)
    for line in opener(path, "rt"):
        r = json.loads(line)
        if "multi_layer_hidden_states" in r:            # HotpotQA V5: already [final, ...]
            t = int(r["stage_idx"])
            feat = np.concatenate([np.asarray(v, dtype=np.float32)
                                   for v in r["multi_layer_hidden_states"]])
            correct = int(r["stage_correctness"])
        else:                                           # PopQA v4: hs_concat (reversed)
            t = int(r["stage"])
            feat = np.asarray(r["hs_concat"], dtype=np.float32)
            if align_popqa:
                feat = _reverse_blocks(feat)
            correct = int(r["correct"])
        by_q[r["query_id"]][t] = (feat, correct)
    qids = sorted(q for q, d in by_q.items() if all(t in d for t in range(N_STAGES)))
    X, y, st, g = [], [], [], []
    for q in qids:
        for t in range(N_STAGES):
            f, c = by_q[q][t]
            X.append(f); y.append(c); st.append(t); g.append(q)
    return np.asarray(X, dtype=np.float32), np.asarray(y), np.asarray(st), np.asarray(g)


def make_mlp(d, device):
    return nn.Sequential(
        nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 1)).to(device)


def train_mlp(Xtr, ytr, device, epochs=120):
    torch.manual_seed(SEED); np.random.seed(SEED)
    m = make_mlp(Xtr.shape[1], device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    pw = torch.tensor([(ytr == 0).sum() / max(1, (ytr == 1).sum())],
                      dtype=torch.float32, device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.float32, device=device)
    m.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(Xtr), 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            loss = lossf(m(Xt[idx]).squeeze(-1), yt[idx])
            loss.backward(); opt.step()
    m.eval()
    return m


def predict_mlp(m, X, device):
    with torch.no_grad():
        return torch.sigmoid(
            m(torch.tensor(X, dtype=torch.float32, device=device)).squeeze(-1)).cpu().numpy()


def auroc_safe(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def per_stage_auroc(y, s, st):
    return {int(t): auroc_safe(y[st == t], s[st == t]) for t in range(N_STAGES)}


def boot_ci(y, s, g, n=2000):
    """Query-grouped bootstrap 95% CI for AUROC."""
    rng = np.random.default_rng(SEED)
    uq = np.unique(g)
    idxq = {q: np.where(g == q)[0] for q in uq}
    vals = []
    for _ in range(n):
        samp = rng.choice(uq, size=len(uq), replace=True)
        ii = np.concatenate([idxq[q] for q in samp])
        if len(np.unique(y[ii])) < 2:
            continue
        vals.append(roc_auc_score(y[ii], s[ii]))
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def _summary(y, s, st, g):
    lo, hi = boot_ci(y, s, g)
    return {"pooled_auroc": auroc_safe(y, s), "ci95": [lo, hi],
            "per_stage": per_stage_auroc(y, s, st), "n": int(len(y)),
            "pos_rate": float(np.mean(y))}


def transfer(Xs, ys, Xt, yt, stt, gt, device):
    sc = StandardScaler().fit(Xs)                 # SOURCE scaler only -> zero-shot
    Xs2, Xt2 = sc.transform(Xs), sc.transform(Xt)
    m = train_mlp(Xs2, ys, device)
    s_mlp = predict_mlp(m, Xt2, device)
    lr = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                            random_state=SEED).fit(Xs2, ys)
    s_lr = lr.predict_proba(Xt2)[:, 1]
    return {"mlp": _summary(yt, s_mlp, stt, gt), "lr": _summary(yt, s_lr, stt, gt)}


def indomain_oof(X, y, st, g, device):
    oof_mlp = np.zeros(len(y)); oof_lr = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        m = train_mlp(Xtr, y[tr], device)
        oof_mlp[te] = predict_mlp(m, Xte, device)
        lr = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                                random_state=SEED).fit(Xtr, y[tr])
        oof_lr[te] = lr.predict_proba(Xte)[:, 1]
    return {"mlp": _summary(y, oof_mlp, st, g), "lr": _summary(y, oof_lr, st, g)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hotpot", default="data/collected_states_hotpotqa_v5_2000.jsonl")
    ap.add_argument("--popqa", default="data/popqa_v4_500q_states.jsonl.gz")
    ap.add_argument("--out", default="results/transfer_ablation/transfer_results.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    print(f"[*] device={device}")

    Xh, yh, sth, gh = load_dataset(args.hotpot)
    Xp, yp, stp, gp = load_dataset(args.popqa, align_popqa=True)
    Xpm, ypm, stpm, gpm = load_dataset(args.popqa, align_popqa=False)
    print(f"[*] HotpotQA {Xh.shape} pos_rate={yh.mean():.3f} | PopQA {Xp.shape} pos_rate={yp.mean():.3f}")

    res = {"meta": {"hotpot_n": int(len(yh)), "popqa_n": int(len(yp)),
                    "hotpot_pos_rate": float(yh.mean()), "popqa_pos_rate": float(yp.mean()),
                    "probe": "MLP[256,128]+LR, StandardScaler(source), bootstrap=2000 query-grouped",
                    "alignment": "PopQA hs_concat block-reversed to match HotpotQA [final,...]"}}

    print("[*] in-domain HotpotQA (OOF)..."); res["indomain_hotpot"] = indomain_oof(Xh, yh, sth, gh, device)
    print("[*] in-domain PopQA (OOF)...");    res["indomain_popqa"] = indomain_oof(Xp, yp, stp, gp, device)
    print("[*] transfer HotpotQA->PopQA (aligned)..."); res["transfer_hotpot_to_popqa"] = transfer(Xh, yh, Xp, yp, stp, gp, device)
    print("[*] transfer PopQA->HotpotQA (aligned)..."); res["transfer_popqa_to_hotpot"] = transfer(Xp, yp, Xh, yh, sth, gh, device)
    print("[*] transfer HotpotQA->PopQA (MISALIGNED control)..."); res["transfer_hotpot_to_popqa_MISALIGNED"] = transfer(Xh, yh, Xpm, ypm, stpm, gpm, device)

    res["meta"]["runtime_sec"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)

    def fmt(d):
        ps = ",".join(f"{d['per_stage'][t]:.3f}" for t in range(N_STAGES))
        return f"{d['pooled_auroc']:.3f} [{d['ci95'][0]:.3f},{d['ci95'][1]:.3f}]  per-stage[S0-S3]={ps}  (n={d['n']}, pos={d['pos_rate']:.3f})"

    print("\n========== RESULTS (correctness AUROC) ==========")
    for k in ["indomain_hotpot", "indomain_popqa", "transfer_hotpot_to_popqa",
              "transfer_popqa_to_hotpot", "transfer_hotpot_to_popqa_MISALIGNED"]:
        print(f"\n[{k}]")
        for mdl in ["mlp", "lr"]:
            print(f"  {mdl.upper()}: {fmt(res[k][mdl])}")
    print(f"\n[*] saved {args.out}  (runtime {res['meta']['runtime_sec']}s)")


if __name__ == "__main__":
    main()
