"""
Regime-control analysis on PopQA data:
1. Within-stage correctness AUROC (control for stage identity leakage)
2. Stage identity / token count / text_length baselines
3. Coverage decomposition (full-set vs covered-subset)
4. Cross-dataset transfer (HotpotQA-trained → PopQA zero-shot, if HotpotQA data available)
"""
import json, sys, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split

POPQA_PATH = "/workspace/popqa_v4_full/popqa_states.jsonl"

def load_popqa(path):
    data = [json.loads(line) for line in open(path)]
    by_qid = {}
    for d in data:
        by_qid.setdefault(d["query_id"], []).append(d)
    for qid in by_qid:
        by_qid[qid].sort(key=lambda x: x["stage"])
    return data, by_qid

def prepare_transitions(by_qid):
    Xl, y_b, y_d, y_c, stages, text_lens = [], [], [], [], [], []
    for qid, tups in by_qid.items():
        for t in range(len(tups) - 1):
            cur = tups[t]
            nxt = tups[t + 1]
            Xl.append(np.array(cur["hs_concat"], dtype=np.float32))
            y_b.append(int(cur["correct"] == 0 and nxt["correct"] == 1))
            y_d.append(int(cur["correct"] == 1 and nxt["correct"] == 0))
            y_c.append(cur["correct"])
            stages.append(cur["stage"])
            text_lens.append(cur.get("text_chars", 0))
    return (np.array(Xl), np.array(y_b), np.array(y_d), np.array(y_c),
            np.array(stages), np.array(text_lens))

def cv_eval(X, y, name, n_folds=5, n_bootstrap=5000):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    probs = np.zeros(len(y)); ys = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        scl = StandardScaler()
        Xtr = scl.fit_transform(X[tr]); Xva = scl.transform(X[va])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(Xtr, y[tr])
        probs[va] = clf.predict_proba(Xva)[:, 1]
        ys[va] = y[va]

    auroc = roc_auc_score(ys, probs)
    auprc = average_precision_score(ys, probs)
    np.random.seed(42)
    n = len(ys)
    aucs = [roc_auc_score(ys[np.random.choice(n,n)], probs[np.random.choice(n,n)]) for _ in range(n_bootstrap)]
    aprs = [average_precision_score(ys[np.random.choice(n,n)], probs[np.random.choice(n,n)]) for _ in range(n_bootstrap)]
    return {"auroc": auroc, "auprc": auprc,
            "auroc_ci": (np.percentile(aucs,2.5), np.percentile(aucs,97.5)),
            "auprc_ci": (np.percentile(aprs,2.5), np.percentile(aprs,97.5)),
            "n": len(y), "pos_rate": y.mean()}

def main():
    print("=" * 60)
    print("REGIME-CONTROL ANALYSIS")
    print("=" * 60)

    data, by_qid = load_popqa(POPQA_PATH)
    X, y_b, y_d, y_c, stages, text_lens = prepare_transitions(by_qid)
    n_q = len(by_qid)
    print(f"\nData: {n_q} queries, {len(X)} transitions")
    print(f"X shape: {X.shape}")

    # === 1. Within-stage correctness AUROC ===
    print("\n--- 1. Within-Stage Correctness AUROC ---")
    for s in range(4):
        mask = stages == s
        n_s = mask.sum()
        if n_s < 20:
            print(f"  S{s}: too few samples ({n_s})")
            continue
        # Train/test split within stage
        X_s, y_s = X[mask], y_c[mask]
        Xtr, Xva, ytr, yva = train_test_split(X_s, y_s, test_size=0.3, random_state=42,
                                                stratify=y_s)
        scl = StandardScaler()
        Xtr_s = scl.fit_transform(Xtr); Xva_s = scl.transform(Xva)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(Xtr_s, ytr)
        probs = clf.predict_proba(Xva_s)[:, 1]
        auroc = roc_auc_score(yva, probs)
        auprc = average_precision_score(yva, probs)
        print(f"  S{s} ({text_lens[mask].mean():.0f} chars avg): AUROC={auroc:.4f}, AUPRC={auprc:.4f}, n={n_s} (train={len(Xtr)}, test={len(Xva)})")

    # Cross-stage (original, for comparison)
    r_across = cv_eval(X, y_c, "cross-stage")
    print(f"  Cross-stage (reference): AUROC={r_across['auroc']:.4f}")

    # === 2. Stage Identity Baseline ===
    print("\n--- 2. Stage Identity + Token Count Baselines ---")
    # Baseline: predict correctness from stage index only
    stage_onehot = np.zeros((len(stages), 4))
    for i, s in enumerate(stages):
        stage_onehot[i, s] = 1
    r_stage = cv_eval(stage_onehot, y_c, "stage-only")
    print(f"  Stage-index only:     AUROC={r_stage['auroc']:.4f}")

    # Baseline: token count (text_chars proxy)
    tok_feat = text_lens.reshape(-1, 1)
    r_tok = cv_eval(tok_feat, y_c, "token-count")
    print(f"  Text-length only:     AUROC={r_tok['auroc']:.4f}")

    # Baseline: stage + text_length
    stage_tok = np.column_stack([stage_onehot, tok_feat])
    r_stok = cv_eval(stage_tok, y_c, "stage+tok")
    print(f"  Stage + text-length:  AUROC={r_stok['auroc']:.4f}")

    # Baseline: HS + stage (check if HS adds over stage identity)
    X_stage = np.column_stack([X, stage_onehot])
    r_hs_stage = cv_eval(X_stage, y_c, "hs+stage")
    print(f"  HS + stage:           AUROC={r_hs_stage['auroc']:.4f}")

    # HS-only vs stage-only delta
    hs_only = cv_eval(X, y_c, "hs-only")
    delta_hs = hs_only["auroc"] - r_stage["auroc"]
    print(f"\n  HS-only - Stage-only delta: {delta_hs:+.4f}")
    print(f"  HS contains {delta_hs/r_stage['auroc']*100:.1f}% additional signal beyond stage identity")

    # === 3. Benefit/Degradation within-stage ===
    print("\n--- 3. Benefit/Degradation Within-Stage ---")
    for s in range(3):  # S0->S1, S1->S2, S2->S3
        mask = (stages == s)
        n_s = mask.sum()
        X_s, yb_s, yd_s = X[mask], y_b[mask], y_d[mask]
        if n_s < 15 or yb_s.sum() < 3:
            print(f"  S{s}->S{s+1}: too few samples ({n_s}, benefit={yb_s.sum()}, degradation={yd_s.sum()})")
            continue
        Xtr, Xva, ytr, yva = train_test_split(X_s, yb_s, test_size=0.3, random_state=42,
                                                stratify=yb_s)
        scl = StandardScaler()
        Xtr_s = scl.fit_transform(Xtr); Xva_s = scl.transform(Xva)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(Xtr_s, ytr)
        auroc_b = roc_auc_score(yva, clf.predict_proba(Xva_s)[:, 1])
        print(f"  S{s}->S{s+1} benefit: AUROC={auroc_b:.4f} (n={n_s}, pos={yb_s.sum()})")

    # === 4. Coverage Decomposition ===
    print("\n--- 4. Coverage Decomposition ---")
    # Covered: has Wikipedia entity page (all 500 queries in v4)
    # Not directly testable with current data (all have entity pages)
    # Instead: analyze accuracy by entity popularity and text length
    pops = [d["popularity"] for d in data if d["stage"] == 3]
    accs = [d["correct"] for d in data if d["stage"] == 3]
    texts = [d["text_chars"] for d in data if d["stage"] == 3]

    # By popularity quartile
    p_bins = [(0, 25), (25, 50), (50, 75), (75, 100)]
    print("  By popularity percentile (S3 accuracy):")
    for lo, hi in p_bins:
        p_lo = np.percentile(pops, lo)
        p_hi = np.percentile(pops, hi)
        mask = (np.array(pops) >= p_lo) & (np.array(pops) <= p_hi)
        if mask.sum() > 0:
            acc = np.mean(np.array(accs)[mask])
            print(f"    {lo}-{hi}%ile: acc={acc:.3f} (n={mask.sum()})")

    # === 5. Summary Table ===
    print("\n" + "=" * 60)
    print("SUMMARY: Confound Controls")
    print("=" * 60)
    print(f"  {'Metric':<30} {'AUROC':>8} {'Interpretation'}")
    print(f"  {'---':<30} {'---':>8} {'---'}")
    print(f"  {'Cross-stage correctness':<30} {hs_only['auroc']:8.4f}  Reference (may include stage leak)")
    print(f"  {'Within-stage S0':<30} {'—':>8}  See above")
    print(f"  {'Stage-index baseline':<30} {r_stage['auroc']:8.4f}  Lower bound")
    print(f"  {'Text-length baseline':<30} {r_tok['auroc']:8.4f}  Lower bound")
    print(f"  {'Stage+text baseline':<30} {r_stok['auroc']:8.4f}  Combined lower bound")
    print(f"  {'HS signal beyond stage':<30} {delta_hs:+.4f}  Unique HS contribution")


if __name__ == "__main__":
    main()
