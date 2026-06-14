import json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

with open("data/collected_richer_signals.jsonl") as f:
    data = [json.loads(l) for l in f if l.strip()]
print("Loaded", len(data), "tuples")

X_hs = np.array([d["hidden_state"] for d in data])
X_ent = np.array([[d["logit_entropy"], d["logit_max_prob"]] for d in data])
X_attn = np.array([[d["attn_max_context"], d["attn_entropy"]] for d in data])
y_correct = np.array([d["stage_correctness"] for d in data])

y_delta = np.zeros(len(data))
queries = {}
for i, d in enumerate(data):
    qid = d["query_id"]
    queries.setdefault(qid, {})[d["stage_idx"]] = d["stage_correctness"]
for i, d in enumerate(data):
    qid, s = d["query_id"], d["stage_idx"]
    if s < 3 and s in queries[qid] and (s+1) in queries[qid]:
        y_delta[i] = int(queries[qid][s]==0 and queries[qid][s+1]==1)

qids = sorted(set(d["query_id"] for d in data))
train_qids, test_qids = train_test_split(qids, test_size=0.3, random_state=42)
train_idx = [i for i,d in enumerate(data) if d["query_id"] in train_qids]
test_idx = [i for i,d in enumerate(data) if d["query_id"] in test_qids]

results = {}
combos = [
    ("hidden_states", X_hs),
    ("logit_entropy", X_ent),
    ("attention", X_attn),
    ("hs+entropy", np.hstack([X_hs, X_ent])),
    ("hs+attention", np.hstack([X_hs, X_attn])),
    ("all_combined", np.hstack([X_hs, X_ent, X_attn])),
]
for name, X in combos:
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X[train_idx], y_correct[train_idx])
    correct_auroc = roc_auc_score(y_correct[test_idx], lr.predict_proba(X[test_idx])[:,1])
    delta_mask = y_delta > 0
    if delta_mask.sum() >= 5:
        lr_d = LogisticRegression(max_iter=1000, class_weight="balanced")
        lr_d.fit(X[train_idx], y_delta[train_idx])
        delta_auroc = roc_auc_score(y_delta[test_idx], lr_d.predict_proba(X[test_idx])[:,1])
        delta_auprc = average_precision_score(y_delta[test_idx], lr_d.predict_proba(X[test_idx])[:,1])
    else:
        delta_auroc = delta_auprc = float("nan")
    results[name] = {"correctness_auroc": round(correct_auroc,4), "delta_auroc": round(delta_auroc,4), "delta_auprc": round(delta_auprc,4)}
    print(name, "correct_auroc=", round(correct_auroc,4), "delta_auroc=", round(delta_auroc,4), "delta_auprc=", round(delta_auprc,4))

with open("results/richer_signals_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results saved")
