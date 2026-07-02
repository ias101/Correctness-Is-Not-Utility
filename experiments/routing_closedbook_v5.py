"""
Does a closed-book (0-passage) stage make hidden-state routing work? (Loop 36).

Prepends a CHEAPEST closed-book stage (no retrieval; collect_closedbook.py) to the
4 retrieval stages, joined by query_id. A router can now stop at closed-book and
save all retrieval cost for questions the model already knows. We test whether
adding this cheap option lets any hidden-state router beat the best static
baseline -- i.e. whether the *retrieve-or-not* decision (different from
how-many-passages) is routable from the hidden state.

5 stages: [closed-book, S0(2p), S1(4p), S2(6p), S3(8p)]; costs [cb, 0.25,0.50,0.75,1.08].
Routers (OOF, GroupKFold by query): threshold-on-correctness-probe; learned
optimal-stop from the closed-book state h_cb (LR+MLP); causal sequential stop.
CWA(lambda) both accountings; query-grouped bootstrap CI on gap vs best static.
Also reports closed-book accuracy and how often the oracle / routers use it.

  python routing_closedbook_v5.py --cb closedbook_hotpotqa_qwen.jsonl \
      --ret data/collected_states_hotpotqa_v5_2000.jsonl --out review-stage/routing_closedbook_qwen.json
"""
import argparse, json, os, gzip, numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
N_BOOT = 2000; SEED = 42


def _hs(s):
    for k in ("multi_layer_hidden_states", "hs_concat", "hidden_state"):
        if s.get(k) is not None:
            return np.asarray(s[k], dtype=np.float32).ravel()
    raise KeyError("no hs")


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


def load_ret(path):
    opener = gzip.open if path.endswith(".gz") else open
    by_q = defaultdict(list)
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            by_q[d["query_id"]].append(d)
    out = {}
    for q, st in by_q.items():
        st.sort(key=_stage)
        if len(st) < 4:
            continue
        st = st[:4]
        # key on normalized question text (robust to qid scheme/split differences)
        key = _qkey(st[0])
        out[key] = ([_corr(s) for s in st], [_hs(s) for s in st])
    return out


def _qkey(d):
    """Join key: normalized question text (fallback to query_id)."""
    q = d.get("question")
    if q:
        return " ".join(q.lower().split())
    return str(d.get("query_id", ""))


def load_cb(path):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            out[_qkey(d)] = (_corr(d), _hs(d))
    return out


def cv_oof_multi(X, y, g, kind):
    oof = np.zeros(len(y), dtype=int)
    for tr, te in GroupKFold(5).split(X, y, groups=g):
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        clf = (LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=SEED)
               if kind == "lr" else
               MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-3, max_iter=120,
                             early_stopping=True, random_state=SEED))
        clf.fit(Xtr, y[tr]); oof[te] = clf.predict(Xte)
    return oof


def cv_oof_proba(X, y, g):
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, groups=g):
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        lr = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=SEED)
        lr.fit(Xtr, y[tr]); oof[te] = lr.predict_proba(Xte)[:, 1]
    return oof


def boot_gap(rq, sq, rng):
    diff = rq - sq; n = len(diff)
    a = np.array([diff[rng.randint(0, n, n)].mean() for _ in range(N_BOOT)])
    return {"gap": float(diff.mean()), "lo": float(np.percentile(a, 2.5)), "hi": float(np.percentile(a, 97.5))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cb", default="closedbook_hotpotqa_qwen.jsonl")
    ap.add_argument("--ret", default="data/collected_states_hotpotqa_v5_2000.jsonl")
    ap.add_argument("--out", default="review-stage/routing_closedbook_qwen.json")
    ap.add_argument("--cb_cost", type=float, default=0.10)
    ap.add_argument("--lam", type=float, default=0.5)
    args = ap.parse_args()
    rng = np.random.RandomState(SEED)
    print("loading...")
    ret = load_ret(args.ret); cb = load_cb(args.cb)
    qids = [q for q in ret if q in cb]
    print(f"ret={len(ret)} cb={len(cb)} joined={len(qids)}")
    Q = len(qids)
    H = np.zeros((Q, 5, _hs_dim := len(cb[qids[0]][1])), dtype=np.float32)
    C = np.zeros((Q, 5), dtype=int)
    for i, q in enumerate(qids):
        cc, ch = cb[q]; rc, rh = ret[q]
        C[i] = [cc] + rc; H[i, 0] = ch
        for s in range(4):
            H[i, s + 1] = rh[s]
    qids = np.array(qids)
    print(f"stage acc (cb,S0..S3): {C.mean(0).round(3).tolist()}")

    COST = np.array([args.cb_cost, 0.25, 0.50, 0.75, 1.08])
    res = {"_meta": {"cb": args.cb, "ret": args.ret, "n_queries": int(Q), "hs_dim": int(_hs_dim),
                     "cb_cost": args.cb_cost, "lam": args.lam,
                     "stage_acc": C.mean(0).round(4).tolist(),
                     "closed_book_accuracy": float(C[:, 0].mean())}}

    for acct, ncost in [("per_stage", COST / COST.max()), ("cumulative", np.cumsum(COST) / np.cumsum(COST)[-1])]:
        R = C - args.lam * ncost[None, :]
        oracle_q = R.max(1); s_star = R.argmax(1)
        static_mean = {s: float(R[:, s].mean()) for s in range(5)}
        best_s = max(static_mean, key=static_mean.get); best_static_q = R[:, best_s]
        # static WITHOUT closed-book (retrieval-only), for comparison
        best_s_noCB = max(range(1, 5), key=lambda s: static_mean[s])
        out = {"oracle_cwa": float(oracle_q.mean()),
               "static_cwa": {("CB" if s == 0 else f"S{s-1}"): static_mean[s] for s in range(5)},
               "best_static": ("CB" if best_s == 0 else f"S{best_s-1}"), "best_static_cwa": static_mean[best_s],
               "best_static_noCB": f"S{best_s_noCB-1}", "best_static_noCB_cwa": static_mean[best_s_noCB],
               "oracle_uses_cb_frac": float((s_star == 0).mean())}

        routers = {}
        Hflat = H.reshape(Q * 5, _hs_dim); yC = C.reshape(Q * 5); gflat = np.repeat(qids, 5)
        pcorr = cv_oof_proba(Hflat, yC, gflat).reshape(Q, 5)
        bt, btau = -9, None
        for tau in np.linspace(0.1, 0.9, 17):
            stop = np.array([next((s for s in range(5) if pcorr[q, s] >= tau), 4) for q in range(Q)])
            c = R[np.arange(Q), stop].mean()
            if c > bt:
                bt, btau = c, float(tau)
        st = np.array([next((s for s in range(5) if pcorr[q, s] >= btau), 4) for q in range(Q)])
        routers["threshold_correctness_probe"] = (R[np.arange(Q), st], {"tau": btau, "routed_to_cb": float((st == 0).mean())})

        Hcb = H[:, 0, :]
        for kind in ["lr", "mlp"]:
            shat = cv_oof_multi(Hcb, s_star, qids, kind)
            routers[f"learned_optimal_stop_cb_{kind}"] = (R[np.arange(Q), shat],
                {"stop_acc": float((shat == s_star).mean()), "routed_to_cb": float((shat == 0).mean())})

        ystop = (np.arange(5)[None, :] == s_star[:, None]).astype(int).reshape(Q * 5)
        pstop = cv_oof_proba(Hflat, ystop, gflat).reshape(Q, 5)
        bs, bstau = -9, None
        for tau in np.linspace(0.1, 0.9, 17):
            stop = np.array([next((s for s in range(5) if pstop[q, s] >= tau), 4) for q in range(Q)])
            c = R[np.arange(Q), stop].mean()
            if c > bs:
                bs, bstau = c, float(tau)
        sq = np.array([next((s for s in range(5) if pstop[q, s] >= bstau), 4) for q in range(Q)])
        routers["causal_sequential_stop"] = (R[np.arange(Q), sq], {"tau": bstau, "routed_to_cb": float((sq == 0).mean())})

        out["routers"] = {}
        print(f"\n=== {acct} (lam={args.lam}, cb_cost={args.cb_cost}) ===")
        print(f"  cb_acc={C[:,0].mean():.3f} | oracle={out['oracle_cwa']:.3f} uses_cb={out['oracle_uses_cb_frac']:.2f}"
              f" | best static {out['best_static']}={out['best_static_cwa']:.3f}"
              f" | best static (no CB) {out['best_static_noCB']}={out['best_static_noCB_cwa']:.3f}")
        for name, (pq, extra) in routers.items():
            g = boot_gap(pq, best_static_q, rng)
            out["routers"][name] = {"cwa": float(pq.mean()), "gap_vs_best_static": g, **extra}
            print(f"  {name:<30} CWA={pq.mean():.3f} gap={g['gap']:+.3f} [{g['lo']:+.3f},{g['hi']:+.3f}]"
                  f"  routed_to_cb={extra.get('routed_to_cb',0):.2f}"
                  + (f" stop_acc={extra['stop_acc']:.2f}" if 'stop_acc' in extra else ""))
        res[acct] = out

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
