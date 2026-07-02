"""
Cross-dataset self-knowledge-share analysis (Loop 36, W1 breadth).

Operationalizes the universal theory: routability is governed by how much of the
conditional-benefit signal is captured by the model's SELF-KNOWLEDGE (decodable
from h), vs by not-yet-read passage content (not in h).

Per dataset, on the routing-relevant currently-wrong support, compute:
  - corr@S0 AUROC from h_S0      (self-knowledge magnitude / probe sanity)
  - conditional benefit|wrong AUROC from h_S0   (THE self-knowledge benefit signal)
  - learned optimal-stop router gain vs best static (CWA, OOF) -- the routability outcome
Then report Spearman(benefit-AUROC, routing-gain) across datasets.

  python experiments/selfknowledge_share.py
"""
import json, gzip, argparse, numpy as np
from collections import defaultdict
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

SEED = 42
COST = np.array([0.25, 0.50, 0.75, 1.08]); CUM = np.array([0.25, 0.75, 1.50, 2.58])
NCOST_CUM = CUM / CUM[-1]


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


def load(path, limit=None):
    op = gzip.open if path.endswith(".gz") else open
    by_q = defaultdict(list)
    with op(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            try: d = json.loads(line)
            except Exception: continue
            by_q[d["query_id"]].append(d)
    qids, H, C = [], [], []
    for q, st in by_q.items():
        st.sort(key=_stage)
        if len(st) < 4: continue
        st = st[:4]
        qids.append(q); H.append([_hs(s) for s in st]); C.append([_corr(s) for s in st])
        if limit and len(qids) >= limit: break
    return np.array(qids), np.array(H, dtype=np.float32), np.array(C)


def oof_auroc(X, y, g):
    if len(np.unique(y)) < 2: return float("nan")
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, groups=g):
        sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        lr = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=SEED)
        lr.fit(Xtr, y[tr]); oof[te] = lr.predict_proba(Xte)[:, 1]
    return roc_auc_score(y, oof)


def routing_gain(qids, H, C):
    """Learned optimal-stop-from-h0 router CWA gap vs best static (cumulative)."""
    Q = len(qids)
    h0 = H[:, 0, :]
    acc = C.mean(0)                       # per-stage accuracy
    # static CWA (cumulative cost accounting)
    static = acc - 0.5 * NCOST_CUM
    best_static = static.max()
    # oracle stop per query = first stage that is correct, else last
    opt = np.array([next((t for t in range(4) if C[i, t] == 1), 3) for i in range(Q)])
    # OOF multiclass router predict stop-stage from h0
    pred = np.full(Q, -1)
    for tr, te in GroupKFold(5).split(h0, opt, groups=qids):
        sc = StandardScaler().fit(h0[tr]); Xtr, Xte = sc.transform(h0[tr]), sc.transform(h0[te])
        lr = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                                multi_class="multinomial", random_state=SEED)
        lr.fit(Xtr, opt[tr]); pred[te] = lr.predict(Xte)
    # realized CWA: accuracy at chosen stop minus cumulative cost
    chosen_acc = np.mean([C[i, pred[i]] for i in range(Q)])
    chosen_cost = np.mean([NCOST_CUM[pred[i]] for i in range(Q)])
    router_cwa = chosen_acc - 0.5 * chosen_cost
    stop_acc = np.mean(pred == opt)
    return router_cwa - best_static, best_static, stop_acc, acc.tolist()


def analyze(name, path, limit=None):
    qids, H, C = load(path, limit)
    Q = len(qids)
    h0 = H[:, 0, :]
    y_corr0 = C[:, 0]
    # benefit | wrong@S0 : among wrong@S0, becomes correct at any later stage
    wrong = y_corr0 == 0
    ben = ((C[:, 1:].max(1) == 1) & wrong).astype(int)[wrong]
    g_w = qids[wrong]
    corr_auroc = oof_auroc(h0, y_corr0, qids)
    ben_auroc = oof_auroc(h0[wrong], ben, g_w) if wrong.sum() > 20 else float("nan")
    gain, best_static, stop_acc, acc = routing_gain(qids, H, C)
    return {"dataset": name, "n_q": int(Q), "stage_acc": [round(a, 3) for a in acc],
            "corr_S0_auroc": round(corr_auroc, 3),
            "benefit_wrong_auroc": round(ben_auroc, 3),
            "n_wrong": int(wrong.sum()), "benefit_rate": round(float(ben.mean()), 3),
            "routing_gain_cum": round(gain, 4), "best_static": round(best_static, 3),
            "stop_acc": round(stop_acc, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="review-stage/selfknowledge_share.json")
    args = ap.parse_args()
    DATASETS = [
        ("PopQA",          "data/popqa_v4_500q_states.jsonl.gz", None),
        ("HotpotQA-Qwen",  "data/collected_states_hotpotqa_v5_2000.jsonl", 2000),
        ("TriviaQA-open",  "data/collected_triviaqa_open_2000.jsonl", 2000),
        ("NQ",             "data/collected_states_nq_v3.jsonl", None),
    ]
    rows = []
    for name, path, lim in DATASETS:
        import os
        if not os.path.exists(path):
            print(f"[skip] {name}: {path} missing", flush=True); continue
        try:
            r = analyze(name, path, lim); rows.append(r)
            print(json.dumps(r), flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
    # correlation across datasets
    if len(rows) >= 3:
        ba = [r["benefit_wrong_auroc"] for r in rows]
        rg = [r["routing_gain_cum"] for r in rows]
        rho, p = spearmanr(ba, rg)
        print(f"\nSpearman(benefit-AUROC, routing-gain) = {rho:.3f} (p={p:.3f}) across {len(rows)} datasets")
        summary = {"rows": rows, "spearman_benefitauroc_routinggain": {"rho": rho, "p": p, "n": len(rows)}}
    else:
        summary = {"rows": rows}
    json.dump(summary, open(args.out, "w"), indent=2)
    print("[*] ->", args.out)


if __name__ == "__main__":
    main()
