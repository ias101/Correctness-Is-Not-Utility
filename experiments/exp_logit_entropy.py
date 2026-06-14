"""Minimal experiment: test whether logit entropy adds routing signal.
Uses existing hidden-state collection + logit extraction in one pass."""
import json, torch, numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_states import (
    set_seed, load_llama_with_hidden_states, build_chat_prompt,
    generate_answer, check_correctness
)
from datasets import load_dataset
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

set_seed(42)
N = 100

# Load dataset
ds = load_dataset("hotpot_qa", "distractor", split=f"train[:{N}]")
print(f"Loaded {len(ds)} queries")

# Load model
model, tokenizer, hook_handles, hs_store = load_llama_with_hidden_states(
    "Qwen/Qwen2.5-7B-Instruct", use_4bit=True
)

results = []
for idx, item in enumerate(tqdm(ds, desc="Collecting")):
    question = item["question"]
    answers = [item["answer"]] if isinstance(item["answer"], str) else item["answer"]
    ctx = item.get("context", {})
    titles = ctx.get("title", [])
    sentences = ctx.get("sentences", [])
    passages = [{"title": t, "text": " ".join(s)} for t, s in zip(titles, sentences)]
    
    for stage_idx, k in enumerate([2, 4, 6, 8]):
        k = min(k, len(passages))
        stage_psgs = passages[:k]
        ctx_text = "\n".join(p["title"] + ": " + p["text"] for p in stage_psgs)
        prompt = f"Context:\n{ctx_text}\n\nQuestion: {question}\nAnswer:"
        chat_prompt = build_chat_prompt(prompt, tokenizer)
        inputs = tokenizer(chat_prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        
        hs_store.clear()
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Hidden state: last layer, last token
        stored_keys = sorted([int(k) for k in hs_store.keys() if not k.endswith("_mean")], reverse=True)
        if stored_keys:
            hs = hs_store[str(stored_keys[0])][0, :].cpu().float().numpy()
        else:
            hs = np.zeros(3584, dtype=np.float32)
        
        # Logit entropy
        logits = outputs.logits[0, -1, :].cpu()
        probs = torch.softmax(logits, dim=-1)
        entropy = float(-(probs * torch.log(probs + 1e-10)).sum())
        max_prob = float(probs.max())
        
        # Generate answer
        ans = generate_answer(model, tokenizer, chat_prompt, max_tokens=48)
        correct = int(check_correctness(ans, answers))
        
        results.append({
            "query_id": idx, "stage_idx": stage_idx,
            "hidden_state": hs.tolist(),
            "logit_entropy": entropy, "logit_max_prob": max_prob,
            "stage_correctness": correct
        })

# Save
with open("data/exp_logit_entropy.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")
print(f"Saved {len(results)} tuples")

# Train probes
data = results
X_hs = np.array([d["hidden_state"] for d in data])
X_ent = np.array([[d["logit_entropy"], d["logit_max_prob"]] for d in data])
y = np.array([d["stage_correctness"] for d in data])

qids = sorted(set(d["query_id"] for d in data))
tq, vq = train_test_split(qids, test_size=0.3, random_state=42)
ti = [i for i,d in enumerate(data) if d["query_id"] in tq]
vi = [i for i,d in enumerate(data) if d["query_id"] in vq]

print("\n=== Correctness Prediction ===")
for name, X in [("hidden_states", X_hs), ("logit_entropy", X_ent), ("hs+entropy", np.hstack([X_hs, X_ent]))]:
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X[ti], y[ti])
    auroc = roc_auc_score(y[vi], lr.predict_proba(X[vi])[:,1])
    print(f"  {name}: AUROC={auroc:.4f}")

print("\n=== Delta (t->t+1 improvement) Prediction ===")
y_delta = np.zeros(len(data))
for i, d in enumerate(data):
    qid, s = d["query_id"], d["stage_idx"]
    if s < 3:
        same_q = [x for x in data if x["query_id"]==qid]
        cur = next((x for x in same_q if x["stage_idx"]==s), None)
        nxt = next((x for x in same_q if x["stage_idx"]==s+1), None)
        if cur and nxt:
            y_delta[i] = int(cur["stage_correctness"]==0 and nxt["stage_correctness"]==1)
pos_rate = y_delta.sum() / len(y_delta)
print(f"  Delta positive rate: {pos_rate:.3f}")
for name, X in [("hidden_states", X_hs), ("logit_entropy", X_ent), ("hs+entropy", np.hstack([X_hs, X_ent]))]:
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X[ti], y_delta[ti])
    auroc = roc_auc_score(y_delta[vi], lr.predict_proba(X[vi])[:,1])
    auprc = average_precision_score(y_delta[vi], lr.predict_proba(X[vi])[:,1])
    print(f"  {name}: AUROC={auroc:.4f}, AUPRC={auprc:.4f}")

for h in hook_handles:
    h.remove()
print("Done")
