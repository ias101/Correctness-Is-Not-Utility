"""
DECISIVE non-oracle CONTENT-embedding control (Loop 33 follow-up).

Loop 33 found: generation-side HS near chance for conditional benefit (0.581),
and a NON-oracle incoming-passage CROSS-ENCODER RELEVANCE descriptor collapses to
CHANCE (0.499) once stage prevalence is controlled. Only ORACLE gold-passage-
presence recovers benefit (0.780). OPEN question (the advisor's thesis in its
strongest form): does the actual incoming-passage CONTENT -- densely embedded,
NON-oracle -- recover the benefit signal once stage is controlled?

This script answers it WITHOUT re-running the 7B LLM. It reuses the existing
BF16 hidden states and reconstructs, per progressive stage transition, the
incoming passages (the 2 newly revealed top-CE passages) from raw HotpotQA using
the SAME cross-encoder ranking as collection (ms-marco-MiniLM-L-6-v2, stages
[2,4,6,8]); alignment is VERIFIED against the stored per-stage ce_scores. The
incoming passages and the query are embedded with a dense retriever (BAAI/
bge-base-en-v1.5). We then test conditional benefit (wrong_t) with HS-only vs
content features vs HS+content, and -- decisively -- WITHIN each stage and vs a
stage-only baseline, to defeat the stage-prevalence confound that killed the
CE-relevance result.

Run on the GPU box (vast.ai):
  python control_content_embedding.py --data bf16_hotpotqa_progressive.jsonl \
      --out control_content_embedding.json
"""
import argparse, json, os, numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

N_BOOT = 2000
SEED = 42
STAGE_SIZES = [2, 4, 6, 8]
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDER = "BAAI/bge-base-en-v1.5"


def load_bf16(path):
    by_q = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            by_q[d["query_id"]].append(d)
    for q in by_q:
        by_q[q].sort(key=lambda x: x["stage_idx"])
    return by_q


def build_paragraphs(sample):
    cd = sample.get("context", {})
    out = []
    if "title" in cd and "sentences" in cd:
        for title, sents in zip(cd["title"], cd["sentences"]):
            text = " ".join(sents) if isinstance(sents, list) else str(sents)
            out.append({"title": title, "text": text})
    return out


def cv_oof(X, y, groups, n_splits=5):
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups=groups):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        lr = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced", random_state=SEED)
        lr.fit(Xtr, y[tr]); oof[te] = lr.predict_proba(Xte)[:, 1]
    return oof


def boot(y, score, groups, rng, n_boot=N_BOOT):
    uq = list(set(groups)); q2i = defaultdict(list)
    for i, q in enumerate(groups):
        q2i[q].append(i)
    aus, aps = [], []
    for _ in range(n_boot):
        idx = []
        for q in rng.choice(uq, len(uq), replace=True):
            idx.extend(q2i[q])
        idx = np.array(idx)
        if len(set(y[idx])) < 2:
            continue
        aus.append(roc_auc_score(y[idx], score[idx])); aps.append(average_precision_score(y[idx], score[idx]))
    a, p = np.array(aus), np.array(aps)
    return {"auroc": float(a.mean()), "auroc_lo": float(np.percentile(a, 2.5)),
            "auroc_hi": float(np.percentile(a, 97.5)), "auprc": float(p.mean()),
            "n": int(len(y)), "n_pos": int(y.sum()), "prev": float(y.mean())}


def evalset(name, X, y, g, rng, out):
    out[name] = boot(y, cv_oof(X, y, g), g, rng)
    m = out[name]
    print(f"  {name:<22} AUROC={m['auroc']:.3f}[{m['auroc_lo']:.3f},{m['auroc_hi']:.3f}] "
          f"AUPRC={m['auprc']:.3f} n={m['n']} pos={m['n_pos']} ({m['prev']*100:.1f}%)")
    return out[name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="bf16_hotpotqa_progressive.jsonl")
    ap.add_argument("--out", default="control_content_embedding.json")
    ap.add_argument("--embedder", default=EMBEDDER)
    args = ap.parse_args()
    rng = np.random.RandomState(SEED)

    import torch
    from datasets import load_dataset
    from sentence_transformers import CrossEncoder, SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}")

    by_q = load_bf16(args.data)
    print(f"bf16 queries: {len(by_q)}")
    print("loading HotpotQA distractor + building id->sample map (validation first, train fallback)...")
    id2s = {}
    for split in ("validation", "train"):
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
        for s in ds:
            if s["id"] in by_q and s["id"] not in id2s:
                id2s[s["id"]] = s
        print(f"  after {split}: matched {len(id2s)}/{len(by_q)}")
        if len(id2s) >= len(by_q):
            break

    ce = CrossEncoder(RERANKER, device=dev)
    emb = SentenceTransformer(args.embedder, device=dev)

    rows, mismatch, missing = [], 0, 0
    for qid, stages in by_q.items():
        if qid not in id2s:
            missing += 1; continue
        s = id2s[qid]
        paras = build_paragraphs(s)
        if len(paras) < STAGE_SIZES[-1]:
            missing += 1; continue
        question = stages[0]["question"]
        ce_all = np.asarray(ce.predict([(question, p["text"]) for p in paras], show_progress_bar=False))
        ranked = np.argsort(ce_all)[::-1]
        # alignment check vs stored stage-0 ce_scores (top-2)
        st0 = stages[0].get("ce_scores", [])
        if st0 is not None and len(st0) >= 2:
            top2 = sorted(ce_all[ranked[:2]], reverse=True)
            if abs(top2[0] - max(st0)) > 0.5:
                mismatch += 1
        q_emb = emb.encode(question, normalize_embeddings=True)
        # cache embeddings of all paragraphs once
        p_embs = emb.encode([p["text"] for p in paras], normalize_embeddings=True, batch_size=32)
        for t in range(len(STAGE_SIZES) - 1):
            kt, kt1 = STAGE_SIZES[t], STAGE_SIZES[t + 1]
            inc_idx = ranked[kt:kt1]                       # incoming passages t -> t+1
            if len(inc_idx) == 0:
                continue
            inc_emb = p_embs[inc_idx].mean(axis=0)         # mean content embedding of incoming
            cosines = p_embs[inc_idx] @ q_emb              # dense query-passage relevance
            rows.append({
                "qid": qid, "t": t,
                "hs": np.asarray(stages[t]["hidden_state"], dtype=np.float32),
                "passage_emb": inc_emb.astype(np.float32),
                "query_emb": q_emb.astype(np.float32),
                "cos_mean": float(cosines.mean()), "cos_max": float(cosines.max()),
                "cur": int(stages[t]["stage_correctness"]), "next": int(stages[t + 1]["stage_correctness"]),
            })
    print(f"transitions={len(rows)} | missing_queries={missing} | ce_alignment_mismatch={mismatch}")

    qid = np.array([r["qid"] for r in rows]); cur = np.array([r["cur"] for r in rows])
    nxt = np.array([r["next"] for r in rows]); tt = np.array([r["t"] for r in rows])
    HS = np.array([r["hs"] for r in rows]); PE = np.array([r["passage_emb"] for r in rows])
    QE = np.array([r["query_emb"] for r in rows])
    COS = np.array([[r["cos_mean"], r["cos_max"]] for r in rows])

    wmask = cur == 0
    yb = nxt[wmask]; gb = qid[wmask]
    print(f"\n[benefit | wrong_t] n={int(wmask.sum())} pos={int(yb.sum())} ({yb.mean()*100:.1f}%)")
    res = {"_meta": {"data": args.data, "embedder": args.embedder, "reranker": RERANKER,
                     "n_transitions": len(rows), "ce_alignment_mismatch": int(mismatch),
                     "hs_dim": int(HS.shape[1]), "emb_dim": int(PE.shape[1]),
                     "note": "BF16 single-layer HS; incoming-passage CONTENT embeddings (non-oracle, dense). "
                             "Decisive stage-controlled test of the information-availability thesis."}}
    out = {}
    feats = {
        "hs_only": HS[wmask],
        "cos_only": COS[wmask],                                   # dense query-passage relevance (2-d, stage-free)
        "passage_emb_only": PE[wmask],                            # incoming-passage CONTENT (768-d)
        "passage+query": np.concatenate([PE, QE], axis=1)[wmask], # content + query (probe can learn matching)
        "hs+passage+query": np.concatenate([HS, PE, QE], axis=1)[wmask],
    }
    for name, X in feats.items():
        evalset(name, X, yb, gb, rng, out)
    # stage baseline + stage-controlled within-stage AUROC for the content features
    out["stage_only"] = boot(yb, cv_oof(tt[wmask].reshape(-1, 1).astype(np.float32), yb, gb), gb, rng)
    print(f"  {'stage_only':<22} AUROC={out['stage_only']['auroc']:.3f}"
          f"[{out['stage_only']['auroc_lo']:.3f},{out['stage_only']['auroc_hi']:.3f}]")
    # within-stage (stage-free) benefit AUROC for passage_emb and cos
    inc_oof = cv_oof(feats["passage_emb_only"], yb, gb)
    cos_oof = cv_oof(feats["cos_only"], yb, gb)
    tw = tt[wmask]; per_stage = {}
    for t in sorted(set(tw)):
        m = tw == t
        if len(set(yb[m])) > 1:
            per_stage[f"S{t}->S{t+1}"] = {
                "n": int(m.sum()), "prev": float(yb[m].mean()),
                "passage_emb_auroc": float(roc_auc_score(yb[m], inc_oof[m])),
                "cos_auroc": float(roc_auc_score(yb[m], cos_oof[m]))}
    out["within_stage"] = per_stage
    print("  within-stage benefit AUROC (passage_emb | cos):")
    for k, v in per_stage.items():
        print(f"    {k}: emb={v['passage_emb_auroc']:.3f} cos={v['cos_auroc']:.3f} (prev {v['prev']*100:.0f}%, n={v['n']})")
    res["benefit_wrong_t"] = out

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
