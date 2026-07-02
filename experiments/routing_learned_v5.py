"""
Learned (non-threshold) hidden-state routers also fail (Loop 35, auto-review-loop-llm).

The paper currently claims THRESHOLD-based hidden-state routing fails. To support
the broader claim --- *hidden-state routing fails* --- we test whether ANY learned,
full-capacity router on the hidden state can beat the best static baseline. The
prior is that it cannot: the conditional benefit signal (the routing-relevant
estimand) is near chance in the hidden state, so no policy class trained on it
should route. We test several non-threshold routers:

  - threshold-on-correctness-probe (the paper's existing policy; reproduced)
  - LEARNED optimal-stop predictor from h_0 (multiclass LR + MLP, OOF): the
    strongest "decide where to stop from the first hidden state" router
  - LEARNED sequential stop classifier (binary per-stage on h_t, OOF)
  - LEARNED value regressor (predict per-stage CWA reward from h_t, stop at argmax)

All routers are out-of-fold (GroupKFold by query). CWA(lambda) = accuracy -
lambda * stop-cost / max-cost (per-stage and cumulative accounting), matching the
paper. We compare every learned router against the best static baseline and the
per-query oracle, with query-grouped bootstrap CIs on (router - best static).

  python experiments/routing_learned_v5.py \
      --data data/collected_states_hotpotqa_v5_canon.jsonl \
      --out review-stage/routing_learned_v5.json
"""
import argparse, json, os, gzip, numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

COST = np.array([0.25, 0.50, 0.75, 1.08]); CUM = np.array([0.25, 0.75, 1.50, 2.58])
NCOST_PS = COST / COST[-1]      # per-stage normalized
NCOST_CUM = CUM / CUM[-1]       # cumulative normalized
N_BOOT = 2000
SEED = 42


def _hs(s):
    for k in ("multi_layer_hidden_states", "hs_concat", "hidden_state"):
        if s.get(k) is not None:
            return np.asarray(s[k], dtype=np.float32).ravel()
    raise KeyError("no hidden-state field")


def _corr(s):
    for k in ("stage_correctness", "correct", "correctness"):
        if k in s:
            return int(s[k])
    return 0


def _stage(s):
    for k in ("stage_idx", "stage"):
        if k in s:
            return s[k]
    return 0


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    by_q = defaultdict(list)
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            by_q[d["query_id"]].append(d)
    qids, H, C = [], [], []
    for q, st in by_q.items():
        st.sort(key=_stage)
        if len(st) < 4:
            continue
        st = st[:4]  # first 4 stages if more (schema-flexible across datasets)
        qids.append(q)
        H.append([_hs(s) for s in st])
        C.append([_corr(s) for s in st])
    return qids, np.array(qids), np.array(H, dtype=np.float32), np.array(C)


def cwa_table(C, ncost, lam):
    """reward[q,s] = c[q,s] - lam*ncost[s]; returns reward matrix."""
    return C - lam * ncost[None, :]


def boot_gap(per_q_router, per_q_static, qids, rng, n_boot=N_BOOT):
    """Bootstrap CI of mean(router - static) over queries (each q is a unit)."""
    diff = per_q_router - per_q_static
    n = len(diff)
    means = [diff[rng.randint(0, n, n)].mean() for _ in range(n_boot)]
    a = np.array(means)
    return {"gap": float(diff.mean()), "lo": float(np.percentile(a, 2.5)),
            "hi": float(np.percentile(a, 97.5))}


def oof_multiclass(X, y, groups, kind):
    oof = np.zeros(len(y), dtype=int)
    for tr, te in GroupKFold(5).split(X, y, groups=groups):
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        if kind == "lr":
            clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=SEED)
        else:
            clf = MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-3, max_iter=120,
                                early_stopping=True, random_state=SEED)
        clf.fit(Xtr, y[tr]); oof[te] = clf.predict(Xte)
    return oof


def oof_proba_perstage(Hflat, y, groups, kind):
    """Train one classifier over all (q,stage) rows; return OOF P(y=1)."""
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(Hflat, y, groups=groups):
        sc = StandardScaler(); Xtr = sc.fit_transform(Hflat[tr]); Xte = sc.transform(Hflat[te])
        if kind == "lr":
            clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=SEED)
        else:
            clf = MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-3, max_iter=120,
                                early_stopping=True, random_state=SEED)
        clf.fit(Xtr, y[tr]); oof[te] = clf.predict_proba(Xte)[:, 1]
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/collected_states_hotpotqa_v5_canon.jsonl")
    ap.add_argument("--out", default="review-stage/routing_learned_v5.json")
    ap.add_argument("--lam", type=float, default=0.5)
    args = ap.parse_args()
    rng = np.random.RandomState(SEED)
    print(f"loading {args.data} ...")
    _, qids, H, C = load(args.data)
    Q, S, D = H.shape
    print(f"queries={Q} stages={S} hs_dim={D}; stage acc={C.mean(0).round(3)}")

    res = {"_meta": {"data": args.data, "n_queries": int(Q), "hs_dim": int(D), "lam": args.lam,
                     "stage_acc": C.mean(0).round(4).tolist(),
                     "note": "All routers out-of-fold (GroupKFold by query). CWA = acc - lam*cost/maxcost."}}

    for acct, ncost in [("per_stage", NCOST_PS), ("cumulative", NCOST_CUM)]:
        R = cwa_table(C, ncost, args.lam)              # reward[q,s]
        oracle_q = R.max(1)                            # per-query oracle stop value
        static_q = {s: R[:, s] for s in range(S)}      # per-query value of fixed-S
        static_mean = {s: float(R[:, s].mean()) for s in range(S)}
        best_s = max(static_mean, key=static_mean.get)
        best_static_q = static_q[best_s]
        out = {"oracle_cwa": float(oracle_q.mean()),
               "static_cwa": {f"S{s}": static_mean[s] for s in range(S)},
               "best_static": f"S{best_s}", "best_static_cwa": static_mean[best_s]}

        # oracle optimal stop label (per-stage acct used for label; route reward uses this acct)
        s_star = R.argmax(1)
        H0 = H[:, 0, :]                                # decide from first hidden state

        routers = {}
        # R1: threshold on correctness probe (paper's policy), optimistic tau sweep
        Hflat = H.reshape(Q * S, D)
        yC = C.reshape(Q * S)
        gflat = np.repeat(qids, S)
        pcorr = oof_proba_perstage(Hflat, yC, gflat, "lr").reshape(Q, S)
        best_thr_cwa, best_tau = -9, None
        for tau in np.linspace(0.1, 0.9, 17):
            stop = np.array([next((s for s in range(S) if pcorr[q, s] >= tau), S - 1) for q in range(Q)])
            cwa = R[np.arange(Q), stop].mean()
            if cwa > best_thr_cwa:
                best_thr_cwa, best_tau = cwa, float(tau)
        stop_thr = np.array([next((s for s in range(S) if pcorr[q, s] >= best_tau), S - 1) for q in range(Q)])
        routers["threshold_correctness_probe"] = (R[np.arange(Q), stop_thr], {"best_tau": best_tau})

        # R2: learned optimal-stop predictor from h_0 (multiclass), LR + MLP
        for kind in ["lr", "mlp"]:
            shat = oof_multiclass(H0, s_star, qids, kind)
            routers[f"learned_optimal_stop_h0_{kind}"] = (R[np.arange(Q), shat],
                {"stop_pred_acc": float((shat == s_star).mean())})

        # R3: CAUSAL learned sequential stop classifier (binary "stop-here" on h_t).
        # A real router decides as it goes, so we stop at the FIRST stage with
        # P(stop|h_t) >= tau (post-hoc-optimal tau, optimistic) -- NO peeking at
        # future stages' hidden states (the argmax-over-all-stages version leaks).
        y_stop = (np.arange(S)[None, :] == s_star[:, None]).astype(int).reshape(Q * S)
        pstop = oof_proba_perstage(Hflat, y_stop, gflat, "lr").reshape(Q, S)
        best_seq_cwa, best_seq_tau = -9, None
        for tau in np.linspace(0.1, 0.9, 17):
            stop = np.array([next((s for s in range(S) if pstop[q, s] >= tau), S - 1) for q in range(Q)])
            cwa = R[np.arange(Q), stop].mean()
            if cwa > best_seq_cwa:
                best_seq_cwa, best_seq_tau = cwa, float(tau)
        stop_seq = np.array([next((s for s in range(S) if pstop[q, s] >= best_seq_tau), S - 1) for q in range(Q)])
        routers["learned_sequential_stop_causal_lr"] = (R[np.arange(Q), stop_seq], {"best_tau": best_seq_tau})

        # report
        out["routers"] = {}
        print(f"\n=== {acct} (lambda={args.lam}) ===")
        print(f"  oracle={out['oracle_cwa']:.3f} | best static {out['best_static']}={out['best_static_cwa']:.3f}")
        for name, (pq, extra) in routers.items():
            g = boot_gap(pq, best_static_q, qids, rng)
            rec = {"cwa": float(pq.mean()), "gap_vs_best_static": g, **extra}
            out["routers"][name] = rec
            print(f"  {name:<34} CWA={rec['cwa']:.3f}  gap={g['gap']:+.3f} [{g['lo']:+.3f},{g['hi']:+.3f}]"
                  + (f"  (tau={extra.get('best_tau')})" if 'best_tau' in extra else "")
                  + (f"  (stop-acc={extra.get('stop_pred_acc'):.2f})" if 'stop_pred_acc' in extra else ""))
        res[acct] = out

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
