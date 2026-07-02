"""
Two-tier adaptive-RAG router (Loop 37): DEPLOY the design principle end-to-end.

This script (the BOX/GPU side) fits all OOF models on the 14336-dim hidden states
and dumps per-query predictions to an .npz, plus a small diagnostics JSON. The cheap
policy / cost / Pareto / bootstrap-CI / figure analysis is done by the companion
experiments/analyze_two_tier.py (local, no GPU) so the gate and cost model can be
iterated WITHOUT re-fitting the heavy classifiers.

Design (validated by the dumped diagnostics):
  Tier 1 (cheap, PRE-retrieval): passive router on h_{S0}; one generation at stop.
  Tier 2 (expensive, POST-retrieval): a Self-RAG-style verifier -- a learned per-stage
    stop classifier on the POST-retrieval state h_{S_t} (a function of the retrieved
    passages R) + realized generation confidence. Escapes h_{S0} ⟂ R_{>0} | q.
  Gate (per query): escalate to Tier 2 the queries the cheap tier is predicted to get
    WRONG -- gate_score = OOF P(Tier-1 answer correct | h_{S0}). Where h_{S0} cannot
    vouch for the cheap answer, pay for post-retrieval verification. (R1 fix: the old
    "Tier-1 router confidence" gate was overconfident everywhere and never escalated.)

HONEST SCOPE: "selfrag"/Tier-2 uses the SAME Qwen backbone and SAME collected data as
every other method; it is NOT the released SelfRAG-Llama2. The comparison isolates the
INFORMATION SET (pre- vs post-retrieval) and its COST -- the mechanism the theory
predicts -- not a claim to beat released weights.

  python experiments/two_tier_router.py --data data/popqa_v4_500q_states.jsonl.gz \
      --out review-stage/two_tier_popqa --dataset PopQA
"""
import argparse, json, os, gzip
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

SEED = 42
NSTAGE = 4


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


def _conf(s):
    mp = s.get("generation_max_prob")
    ent = s.get("generation_entropy")
    if mp is not None:
        return float(mp)
    if ent is not None:
        return float(1.0 / (1.0 + ent))
    return float("nan")


def load(path, limit=None):
    opener = gzip.open if path.endswith(".gz") else open
    by_q = defaultdict(list)
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            by_q[d["query_id"]].append(d)
    qids, H, C, CONF, QTEXT = [], [], [], [], []
    for q, st in by_q.items():
        st.sort(key=_stage)
        if len(st) < NSTAGE:
            continue
        st = st[:NSTAGE]
        qids.append(q)
        H.append([_hs(s) for s in st])
        C.append([_corr(s) for s in st])
        CONF.append([_conf(s) for s in st])
        QTEXT.append(st[0].get("question", str(q)))
        if limit and len(qids) >= limit:
            break
    return (np.array(qids), np.array(H, dtype=np.float32),
            np.array(C), np.array(CONF, dtype=np.float32), QTEXT)


def oof_multiclass_proba(X, y, groups, n_classes):
    P = np.zeros((len(y), n_classes), dtype=float)
    for tr, te in GroupKFold(5).split(X, y, groups=groups):
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=SEED)
        clf.fit(Xtr, y[tr])
        proba = clf.predict_proba(Xte)
        for j, c in enumerate(clf.classes_):
            P[te, c] = proba[:, j]
    return P


def oof_proba(X, y, groups):
    """OOF P(y=1) for a single classifier over rows X (len n)."""
    oof = np.full(len(y), np.nan)
    if len(np.unique(y)) < 2:
        return np.full(len(y), float(y.mean()))
    for tr, te in GroupKFold(5).split(X, y, groups=groups):
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=SEED)
        clf.fit(Xtr, y[tr]); oof[te] = clf.predict_proba(Xte)[:, 1]
    return oof


def oof_auroc(X, y, g):
    if len(np.unique(y)) < 2:
        return float("nan")
    oof = oof_proba(X, y, g)
    return roc_auc_score(y, oof)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True, help="output prefix (writes .npz and .json)")
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    print(f"loading {args.data} ...", flush=True)
    qids, H, C, CONF, QTEXT = load(args.data, args.limit)
    Q, S, D = H.shape
    have_conf = bool(np.isfinite(CONF).any())
    print(f"queries={Q} stages={S} hs_dim={D} | stage_acc={C.mean(0).round(3).tolist()} "
          f"| gen_conf={'yes' if have_conf else 'no'}", flush=True)

    H0 = H[:, 0, :]
    Hflat = H.reshape(Q * S, D)
    gflat = np.repeat(qids, S)
    yC = C.reshape(Q * S)
    s_star = np.array([next((s for s in range(S) if C[q, s] == 1), S - 1) for q in range(Q)])

    # ── DIAGNOSTIC: post-retrieval state decodes benefit/correctness better (escapes bound) ──
    diag = {}
    wrong0 = C[:, 0] == 0
    ben = ((C[:, 1:].max(1) == 1) & wrong0).astype(int)
    if wrong0.sum() > 20:
        gw = qids[wrong0]
        diag["benefit_auroc_pre_h0"] = round(oof_auroc(H0[wrong0], ben[wrong0], gw), 4)
        diag["benefit_auroc_post_hlast"] = round(oof_auroc(H[:, -1, :][wrong0], ben[wrong0], gw), 4)
        diag["post_minus_pre"] = round(diag["benefit_auroc_post_hlast"] - diag["benefit_auroc_pre_h0"], 4)
        diag["n_wrong0"] = int(wrong0.sum()); diag["benefit_rate_wrong0"] = round(float(ben[wrong0].mean()), 4)
    decod = {}
    for t in range(1, S):
        if len(np.unique(C[:, t])) < 2:
            continue
        a_pre = oof_auroc(H0, C[:, t], qids)
        a_post = oof_auroc(H[:, t, :], C[:, t], qids)
        decod[f"S{t}"] = {"pre_h0": round(a_pre, 4), "post_ht": round(a_post, 4),
                          "post_minus_pre": round(a_post - a_pre, 4)}
    diag["post_retrieval_decodability"] = decod

    # ── Tier-1 router: optimal-stop from h0 (multiclass), argmax stop ──
    P_stop1 = oof_multiclass_proba(H0, s_star, qids, S)
    tier1_stop = P_stop1.argmax(1)
    tier1_conf = P_stop1.max(1)                       # the OLD (overconfident) gate, kept for ref

    # ── NEW gate: P(Tier-1's chosen answer is correct | h0) ── low => escalate to Tier-2
    y_t1correct = C[np.arange(Q), tier1_stop].astype(int)
    gate_score = oof_proba(H0, y_t1correct, qids)
    pcorr0 = oof_proba(H0, C[:, 0], qids)             # correctness@S0 probe (alt gate)

    # ── Tier-2 verifier: per-stage stop prob from POST-retrieval state (+conf) ──
    feat_post = Hflat if not have_conf else np.hstack([Hflat, np.nan_to_num(CONF.reshape(Q * S, 1), nan=0.5)])
    p_post = np.full(Q * S, np.nan)
    for tr, te in GroupKFold(5).split(feat_post, yC, groups=gflat):
        sc = StandardScaler(); Xtr = sc.fit_transform(feat_post[tr]); Xte = sc.transform(feat_post[te])
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=SEED)
        clf.fit(Xtr, yC[tr]); p_post[te] = clf.predict_proba(Xte)[:, 1]
    p_post = p_post.reshape(Q, S)

    # ── Query-only router (Adaptive-RAG-style): TF-IDF char n-grams -> optimal stop ──
    qfeat = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=4000
                            ).fit_transform(QTEXT).toarray().astype(np.float32)
    query_only_stop = oof_multiclass_proba(qfeat, s_star, qids, S).argmax(1)

    # gate separability summary (does the gate's escalation signal differ by regime?)
    gate_sep = {"gate_score_mean": round(float(gate_score.mean()), 4),
                "gate_score_p25_50_75": [round(float(np.percentile(gate_score, q)), 4) for q in (25, 50, 75)],
                "gate_score_std": round(float(gate_score.std()), 4),
                "tier1_conf_mean": round(float(tier1_conf.mean()), 4),
                "note": "gate_score = OOF P(Tier-1 answer correct | h0); low => escalate."}

    # ── dump per-query predictions for local analysis ──
    npz_path = args.out + "_perq.npz"
    np.savez_compressed(
        npz_path, dataset=args.dataset, qids=qids.astype(str),
        C=C.astype(np.int8), s_star=s_star.astype(np.int8),
        tier1_stop=tier1_stop.astype(np.int8), tier1_conf=tier1_conf.astype(np.float32),
        gate_score=gate_score.astype(np.float32), pcorr0=pcorr0.astype(np.float32),
        p_post=p_post.astype(np.float32), conf=CONF.astype(np.float32),
        query_only_stop=query_only_stop.astype(np.int8), have_conf=have_conf,
        stage_acc=C.mean(0).astype(np.float32))

    js = {"_meta": {"dataset": args.dataset, "data": args.data, "n_queries": int(Q),
                    "stages": int(S), "hs_dim": int(D), "have_gen_conf": have_conf,
                    "stage_acc": C.mean(0).round(4).tolist(), "seed": SEED,
                    "npz": os.path.basename(npz_path)},
          "mechanism_pre_vs_post": diag, "gate_separability": gate_sep}
    with open(args.out + ".json", "w") as f:
        json.dump(js, f, indent=2)

    print(f"\n=== {args.dataset}: mechanism (escaping h0 _indep_ R | q) ===")
    if diag.get("benefit_auroc_pre_h0") is not None:
        print(f"  benefit|wrong AUROC  pre-h0={diag['benefit_auroc_pre_h0']}  "
              f"post-h_last={diag['benefit_auroc_post_hlast']}  (post-pre={diag['post_minus_pre']:+})")
    for st, dd in decod.items():
        print(f"  corr@{st} AUROC  pre-h0={dd['pre_h0']}  post-h_t={dd['post_ht']}  (post-pre={dd['post_minus_pre']:+})")
    print(f"=== gate: score mean={gate_sep['gate_score_mean']} p25/50/75={gate_sep['gate_score_p25_50_75']} "
          f"(escalate low) | old tier1_conf mean={gate_sep['tier1_conf_mean']} ===")
    print(f"Saved -> {npz_path}  and  {args.out}.json")


if __name__ == "__main__":
    main()
