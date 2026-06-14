"""
HotpotQA threshold-routing table (Loop 30) — close the narrative loop.

Builds the SAME routing analysis the paper reports on PopQA (tab:routing), but on the
HotpotQA-distractor V5 data that the conditional-utility experiment uses (4-bit, 2000
queries, progressive top-k 2->4->6->8). Goal: show that threshold-based hidden-state
routing also fails to beat the best static baseline on HotpotQA, where conditional
benefit prediction is near chance (AUROC 0.575).

Protocol (matched to the paper):
  - Correctness probe = MLP [256,128] (ReLU, dropout 0.2) on concat of last-4-layer hidden
    states (14336-d), GroupKFold(5) by query, StandardScaler per fold, OOF P(correct).
    (Identical pipeline to canonical_v5_full.py, which produced the paper's correctness AUROC.)
    LR counterpart reported as a grouping-pure control.
  - Cost model (paper): per-stage cost of stopping at stage t = [0.25,0.50,0.75,1.08]
    (per-stage denom 1.08); cumulative cost = running sum [0.25,0.75,1.50,2.58] (denom 2.58).
    CWA(lambda) = acc - lambda * cost / denom.
  - Policies: Oracle (earliest-correct stop), Fixed-S0..S3, Probe(tau) [stop when P(correct)>=tau],
    Self-report(tau) [stop when generation_max_prob>=tau], Random(100 trials).
  - Reported honestly: full tau sweep, lambda sweep + crossover, both cost accountings,
    query-grouped bootstrap CIs for the headline "probe does not beat best static" comparison.

Run on WSL2 (conda research). NO LLM re-run; uses collected hidden states only.
"""
import json, sys, argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from collections import defaultdict

# ---- Paper cost model (NOT the old enhanced_routing_baselines.py [0.02,0.05,0.01,1.00]) ----
PERSTAGE_COST = np.array([0.25, 0.50, 0.75, 1.08])          # cost of STOPPING at stage t
CUMUL_COST    = np.cumsum(PERSTAGE_COST)                     # [0.25,0.75,1.50,2.58]
PERSTAGE_DENOM = PERSTAGE_COST.max()                        # 1.08
CUMUL_DENOM    = CUMUL_COST[-1]                             # 2.58
N_STAGES = 4
SEED = 42


def load_data(path):
    """Format-agnostic loader. HotpotQA V5: multi_layer_hidden_states[4]/stage_idx/stage_correctness.
    PopQA v4: hs_concat[14336]/stage/correct (already concatenated). Handles .gz."""
    import gzip
    opener = gzip.open if path.endswith(".gz") else open
    by_q = defaultdict(dict)
    for line in opener(path, "rt"):
        r = json.loads(line)
        if "multi_layer_hidden_states" in r:                 # HotpotQA V5 format
            t = int(r["stage_idx"])
            feat = np.concatenate([np.asarray(v, dtype=np.float32) for v in r["multi_layer_hidden_states"]])
            correct = int(r["stage_correctness"])
            mp = float(r.get("generation_max_prob", np.nan))
        else:                                                # PopQA v4 format
            t = int(r["stage"])
            feat = np.asarray(r["hs_concat"], dtype=np.float32)
            correct = int(r["correct"])
            mp = float(r.get("generation_max_prob", np.nan))
        by_q[r["query_id"]][t] = {"feat": feat, "correct": correct,
                                  "max_prob": mp, "entropy": np.nan}
    qids = [q for q, d in by_q.items() if all(t in d for t in range(N_STAGES))]
    qids.sort()
    return by_q, qids


def build_arrays(by_q, qids):
    X, y, g, st = [], [], [], []
    for q in qids:
        for t in range(N_STAGES):
            rec = by_q[q][t]
            X.append(rec["feat"]); y.append(rec["correct"]); g.append(q); st.append(t)
    return np.asarray(X), np.asarray(y), np.asarray(g), np.asarray(st)


def oof_probe(X, y, groups, device):
    """OOF P(correct) from MLP and LR, GroupKFold(5) by query. Matches canonical_v5_full.py."""
    torch.manual_seed(SEED); np.random.seed(SEED)
    oof_mlp = np.zeros(len(y)); oof_lr = np.zeros(len(y))
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        # LR
        lr = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=SEED)
        lr.fit(Xtr, y[tr]); oof_lr[te] = lr.predict_proba(Xte)[:, 1]
        # MLP
        Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
        Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
        ytr_t = torch.tensor(y[tr], dtype=torch.float32, device=device)
        model = nn.Sequential(
            nn.Linear(X.shape[1], 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        pos_w = torch.tensor([(y[tr] == 0).sum() / max(1, (y[tr] == 1).sum())], device=device)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        model.train()
        for ep in range(120):
            perm = torch.randperm(len(tr), device=device)
            for i in range(0, len(tr), 128):
                idx = perm[i:i + 128]
                opt.zero_grad()
                out = model(Xtr_t[idx]).squeeze(-1)
                loss = lossf(out, ytr_t[idx]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            oof_mlp[te] = torch.sigmoid(model(Xte_t).squeeze(-1)).cpu().numpy()
    return oof_mlp, oof_lr


def to_grid(arr, y, groups, stages, qids):
    """Return [n_q, 4] matrices for a per-record array, aligned to qids order."""
    idx = {(q, t): i for i, (q, t) in enumerate(zip(groups, stages))}
    n = len(qids)
    M = np.zeros((n, N_STAGES))
    for qi, q in enumerate(qids):
        for t in range(N_STAGES):
            M[qi, t] = arr[idx[(q, t)]]
    return M


def cwa(acc, cost, denom, lam):
    return acc - lam * (cost / denom)


def eval_policy(stop_stages, correct_grid):
    """Given per-query stop stage, return accuracy, per-stage cost, cumulative cost (means)."""
    n = len(stop_stages)
    acc = np.mean([correct_grid[i, stop_stages[i]] for i in range(n)])
    ps_cost = np.mean([PERSTAGE_COST[stop_stages[i]] for i in range(n)])
    cu_cost = np.mean([CUMUL_COST[stop_stages[i]] for i in range(n)])
    return acc, ps_cost, cu_cost


def threshold_stops(score_grid, tau):
    """Stop at first stage where score>=tau (confident-correct -> stop); else last stage. Vectorized."""
    above = score_grid >= tau                      # [n, 4] bool
    has = above.any(axis=1)
    first = np.argmax(above, axis=1)               # first True index per row (0 if none)
    return np.where(has, first, N_STAGES - 1).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/collected_states_hotpotqa_v5_2000.jsonl")
    ap.add_argument("--out", default="results/v5_experiment/routing_hotpotqa_v5.json")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--s0_cache", default=None,
                    help="also save the S0 OOF probe scores to this path (figure cache for gen_fig_sharpness.py)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    by_q, qids = load_data(args.data)
    X, y, g, st = build_arrays(by_q, qids)
    print(f"[data] {len(qids)} queries x {N_STAGES} stages = {len(y)} records; "
          f"per-stage acc = {[round(float(y[st==t].mean()),3) for t in range(N_STAGES)]}", flush=True)

    oof_mlp, oof_lr = oof_probe(X, y, g, device)
    # per-stage correctness AUROC (sanity vs paper 0.78-0.96)
    auroc = {f"stage{t}_mlp": float(roc_auc_score(y[st == t], oof_mlp[st == t])) for t in range(N_STAGES)}
    auroc["pooled_mlp"] = float(roc_auc_score(y, oof_mlp))
    print(f"[probe] correctness AUROC per stage (MLP): "
          f"{[round(auroc[f'stage{t}_mlp'],3) for t in range(N_STAGES)]} pooled={auroc['pooled_mlp']:.3f}", flush=True)

    correct_grid = to_grid(y.astype(float), y, g, st, qids)
    pmlp_grid = to_grid(oof_mlp, y, g, st, qids)
    maxprob_grid = to_grid(np.array([by_q[q][t]["max_prob"] for q in qids for t in range(N_STAGES)]),
                           y, g, st, qids) if not np.isnan(by_q[qids[0]][0]["max_prob"]) else None
    n = len(qids)

    # ---- bimodality of S0 probe scores ----
    import os as _os
    s0 = pmlp_grid[:, 0]
    bimod = {
        "frac_below_0.2": float((s0 < 0.2).mean()),
        "frac_above_0.8": float((s0 >= 0.8).mean()),
        "frac_extremes": float(((s0 < 0.2) | (s0 >= 0.8)).mean()),
        "frac_above_0.9": float((s0 >= 0.9).mean()),
        "bins_0_03_07_09_10": [
            float((s0 < 0.3).mean()),
            float(((s0 >= 0.3) & (s0 < 0.7)).mean()),
            float(((s0 >= 0.7) & (s0 < 0.9)).mean()),
            float((s0 >= 0.9).mean()),
        ],
        "s0_mean": float(s0.mean()), "s0_std": float(s0.std()),
    }
    _os.makedirs(_os.path.dirname(args.out), exist_ok=True)
    np.save(args.out.replace('.json', '_s0_scores.npy'), s0.astype(np.float32))
    if args.s0_cache:
        _os.makedirs(_os.path.dirname(_os.path.abspath(args.s0_cache)), exist_ok=True)
        np.save(args.s0_cache, s0.astype(np.float32))
        print(f"[s0_cache] saved {len(s0)} S0 OOF scores to {args.s0_cache}", flush=True)
    print(f"[bimodality S0] extremes={bimod['frac_extremes']:.3f} "
          f"(<0.2={bimod['frac_below_0.2']:.3f}, >=0.8={bimod['frac_above_0.8']:.3f})", flush=True)

    # ---- static + oracle + random policies ----
    policies = {}
    # Oracle (cost-aware, matched to PopQA table): earliest correct stage, else S0 (cheapest).
    oracle_stops = np.array([next((t for t in range(N_STAGES) if correct_grid[i, t] == 1), 0)
                             for i in range(n)])
    policies["Oracle"] = eval_policy(oracle_stops, correct_grid)
    for t in range(N_STAGES):
        policies[f"Fixed-S{t}"] = eval_policy(np.full(n, t), correct_grid)
    rng = np.random.default_rng(SEED)
    rand_accs, rand_ps, rand_cu, rand_cwa_ps, rand_cwa_cu = [], [], [], [], []
    for _ in range(100):
        rs = rng.integers(0, N_STAGES, size=n)
        a, p, c = eval_policy(rs, correct_grid); rand_accs.append(a); rand_ps.append(p); rand_cu.append(c)
        rand_cwa_ps.append(cwa(a, p, PERSTAGE_DENOM, 0.5)); rand_cwa_cu.append(cwa(a, c, CUMUL_DENOM, 0.5))
    policies["Random"] = (float(np.mean(rand_accs)), float(np.mean(rand_ps)), float(np.mean(rand_cu)))
    random_interval = {
        "CWA_0.5_per_stage_mean_sd": [float(np.mean(rand_cwa_ps)), float(np.std(rand_cwa_ps))],
        "CWA_0.5_cumulative_mean_sd": [float(np.mean(rand_cwa_cu)), float(np.std(rand_cwa_cu))]}

    # ---- threshold sweeps ----
    taus = np.round(np.arange(0.05, 1.0, 0.05), 2)
    def sweep(score_grid, name):
        rows = {}
        for tau in taus:
            stops = threshold_stops(score_grid, tau)
            a, p, c = eval_policy(stops, correct_grid)
            rows[float(tau)] = {"acc": float(a), "ps_cost": float(p), "cu_cost": float(c)}
        return rows
    probe_sweep = sweep(pmlp_grid, "Probe")
    selfreport_sweep = sweep(maxprob_grid, "Self-report") if maxprob_grid is not None else None

    # ---- lambda sweep: best static vs best probe CWA (per-stage & cumulative) ----
    static_names = [f"Fixed-S{t}" for t in range(N_STAGES)]
    lam_grid = np.round(np.arange(0.0, 1.01, 0.05), 2)
    lam_analysis = {"per_stage": {}, "cumulative": {}}
    for acct, denom, costidx in [("per_stage", PERSTAGE_DENOM, 1), ("cumulative", CUMUL_DENOM, 2)]:
        for lam in lam_grid:
            best_static = max(cwa(policies[nm][0], policies[nm][costidx], denom, lam) for nm in static_names)
            best_probe = max(cwa(r["acc"], (r["ps_cost"] if acct == "per_stage" else r["cu_cost"]), denom, lam)
                             for r in probe_sweep.values())
            lam_analysis[acct][float(lam)] = {
                "best_static_cwa": float(best_static), "best_probe_cwa": float(best_probe),
                "probe_beats_static": bool(best_probe > best_static + 1e-9)}
    def crossover(acct):
        # smallest lambda at which probe stops beating static (moving up from 0)
        items = sorted(lam_analysis[acct].items())
        cx = None
        for lam, d in items:
            if not d["probe_beats_static"]:
                cx = lam; break
        return cx
    cx_ps, cx_cu = crossover("per_stage"), crossover("cumulative")
    print(f"[lambda] probe stops beating best static at lambda>= {cx_ps} (per-stage), {cx_cu} (cumulative)", flush=True)

    # ---- headline table at lambda=0.5 (pick representative probe tau = best CWA at 0.5 per-stage) ----
    def best_tau_at(lam, acct):
        costkey = "ps_cost" if acct == "per_stage" else "cu_cost"
        denom = PERSTAGE_DENOM if acct == "per_stage" else CUMUL_DENOM
        return max(probe_sweep.items(),
                   key=lambda kv: cwa(kv[1]["acc"], kv[1][costkey], denom, lam))
    btau, brow = best_tau_at(0.5, "per_stage")
    probe_best = (brow["acc"], brow["ps_cost"], brow["cu_cost"])
    policies[f"Probe(tau={btau})"] = probe_best
    if selfreport_sweep is not None:
        st_tau, st_row = max(selfreport_sweep.items(),
                             key=lambda kv: cwa(kv[1]["acc"], kv[1]["ps_cost"], PERSTAGE_DENOM, 0.5))
        policies[f"Self-report(tau={st_tau})"] = (st_row["acc"], st_row["ps_cost"], st_row["cu_cost"])

    # ---- bootstrap CI (query-grouped) for CWA(0.5) gap: best-probe(POST-HOC tau) minus BEST-STATIC ----
    # Note: probe tau is swept post-hoc on OOF scores (optimistic). If even the optimistic probe
    # does not beat the best static baseline, the failure is robust.
    gaps_ps, gaps_cu = [], []
    def static_cwas(cg, denom, costarr):
        return [cg[:, t].mean() - 0.5 * costarr[t] / denom for t in range(N_STAGES)]
    for _ in range(args.boot):
        bi = rng.integers(0, n, size=n)
        cg = correct_grid[bi]; pg = pmlp_grid[bi]
        def policy_cwa(stops, denom, costarr):
            acc = cg[np.arange(len(bi)), stops].mean(); cost = costarr[stops].mean()
            return acc - 0.5 * cost / denom
        bestp_ps = max(policy_cwa(threshold_stops(pg, tau), PERSTAGE_DENOM, PERSTAGE_COST) for tau in taus)
        bestp_cu = max(policy_cwa(threshold_stops(pg, tau), CUMUL_DENOM, CUMUL_COST) for tau in taus)
        bests_ps = max(static_cwas(cg, PERSTAGE_DENOM, PERSTAGE_COST))
        bests_cu = max(static_cwas(cg, CUMUL_DENOM, CUMUL_COST))
        gaps_ps.append(bestp_ps - bests_ps); gaps_cu.append(bestp_cu - bests_cu)
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)), float(np.mean(a))]

    # ---- stop-stage distributions (probe at best post-hoc tau, oracle) ----
    probe_stops_best = threshold_stops(pmlp_grid, btau)
    stop_dist = {
        "probe_besttau": [int((probe_stops_best == t).sum()) for t in range(N_STAGES)],
        "oracle": [int((oracle_stops == t).sum()) for t in range(N_STAGES)],
        "best_static_stage_per_stage": int(np.argmax(static_cwas(correct_grid, PERSTAGE_DENOM, PERSTAGE_COST))),
        "best_static_stage_cumulative": int(np.argmax(static_cwas(correct_grid, CUMUL_DENOM, CUMUL_COST))),
    }

    out = {
        "n_queries": n,
        "per_stage_accuracy": [float(y[st == t].mean()) for t in range(N_STAGES)],
        "correctness_auroc": auroc,
        "cost_model": {"per_stage": PERSTAGE_COST.tolist(), "cumulative": CUMUL_COST.tolist(),
                       "denom_per_stage": float(PERSTAGE_DENOM), "denom_cumulative": float(CUMUL_DENOM)},
        "bimodality_S0": bimod,
        "policies_at_lambda0.5": {
            nm: {"accuracy": round(v[0], 4), "cost_per_stage": round(v[1], 4),
                 "cost_cumulative": round(v[2], 4),
                 "CWA_0.5_per_stage": round(cwa(v[0], v[1], PERSTAGE_DENOM, 0.5), 4),
                 "CWA_0.5_cumulative": round(cwa(v[0], v[2], CUMUL_DENOM, 0.5), 4)}
            for nm, v in policies.items()},
        "best_probe_tau_at_0.5_POSTHOC_optimistic": btau,
        "probe_tau_sweep": probe_sweep,
        "selfreport_tau_sweep": selfreport_sweep,
        "random_interval": random_interval,
        "stop_distributions": stop_dist,
        "lambda_crossover": {"per_stage": cx_ps, "cumulative": cx_cu},
        "lambda_sweep": lam_analysis,
        "delta_CWA0.5_bestprobe_POSTHOC_minus_BESTSTATIC": {
            "per_stage_point": round(cwa(probe_best[0], probe_best[1], PERSTAGE_DENOM, 0.5)
                                     - max(cwa(policies[f"Fixed-S{t}"][0], policies[f"Fixed-S{t}"][1], PERSTAGE_DENOM, 0.5) for t in range(N_STAGES)), 4),
            "cumulative_point": round(cwa(probe_best[0], probe_best[2], CUMUL_DENOM, 0.5)
                                      - max(cwa(policies[f"Fixed-S{t}"][0], policies[f"Fixed-S{t}"][2], CUMUL_DENOM, 0.5) for t in range(N_STAGES)), 4),
            "per_stage_[lo,hi,mean]": ci(gaps_ps),
            "cumulative_[lo,hi,mean]": ci(gaps_cu),
            "n_boot": args.boot,
            "note": "tau swept post-hoc on OOF scores (optimistic for the probe); CI<=0 means even the optimistic probe does not beat the best static baseline"},
    }
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("\n===== HotpotQA routing table (lambda=0.5) =====", flush=True)
    print(f"{'Policy':<22}{'Acc':>7}{'Cost':>8}{'CWA(.5)':>9}{'CWA_cum':>9}", flush=True)
    order = (["Oracle"] + ([f"Self-report(tau={st_tau})"] if selfreport_sweep is not None else [])
             + ["Fixed-S0", "Fixed-S1", "Fixed-S2", f"Probe(tau={btau})", "Random", "Fixed-S3"])
    for nm in order:
        v = policies[nm]
        print(f"{nm:<22}{v[0]:>7.3f}{v[1]:>8.3f}"
              f"{cwa(v[0],v[1],PERSTAGE_DENOM,0.5):>9.3f}{cwa(v[0],v[2],CUMUL_DENOM,0.5):>9.3f}", flush=True)
    print(f"\nbest-probe minus Fixed-S0 CWA(0.5): per-stage {ci(gaps_ps)} ; cumulative {ci(gaps_cu)}", flush=True)
    print(f"probe beats best static only at lambda < {cx_ps} (per-stage) / {cx_cu} (cumulative)", flush=True)
    print(f"\n[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
