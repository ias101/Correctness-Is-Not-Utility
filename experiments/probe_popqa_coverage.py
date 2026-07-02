"""Quick coverage probe: does dense+CE retrieval over wiki_500k supply PopQA gold answers?
Determines whether p=0 is a valid 'reliable retrieval' pole for the causal degradation experiment.
Runs on ~60 queries for speed."""
import json, gzip, re, numpy as np, torch
from collections import OrderedDict

POP = "data/popqa_v4_500q_states.jsonl.gz"
# load unique queries (stage-0 record has all metadata)
seen = OrderedDict()
with gzip.open(POP, "rt", encoding="utf-8") as fh:
    for line in fh:
        d = json.loads(line)
        if d["query_id"] not in seen:
            seen[d["query_id"]] = {"q": d["question"], "gold": d["gold"], "subj": d["subj"]}
queries = list(seen.values())[:60]
print(f"probing {len(queries)} queries")

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder
import faiss

bge_tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
bge = AutoModel.from_pretrained("BAAI/bge-m3", torch_dtype=torch.float32).eval().cuda()
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
index = faiss.read_index("data/wiki_500k/faiss_index.bin")
passages = [json.loads(l) for l in open("data/wiki_500k/passages.jsonl")]
print(f"index ntotal={index.ntotal}, passages={len(passages)}")

def norm(s):
    s = s.lower().strip(); s = re.sub(r"[^\w\s]", " ", s); return re.sub(r"\s+", " ", s).strip()

cov_topk = {1: 0, 4: 0, 10: 0}
for ex in queries:
    qt = f"Represent this sentence for searching relevant passages: {ex['q']}"
    inp = bge_tok(qt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        qe = bge(**inp).last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
    D, I = index.search(qe.reshape(1, -1), 50)
    hits = [passages[idx]["text"] for idx in I[0] if idx < len(passages)]
    pairs = [(ex["q"], t[:500]) for t in hits[:20]]
    sc = ce.predict(pairs, show_progress_bar=False)
    ranked = [t for _, t in sorted(zip(sc, hits[:20]), key=lambda x: x[0], reverse=True)]
    g = norm(ex["gold"])
    for k in cov_topk:
        joined = norm(" ".join(ranked[:k]))
        if g in joined:
            cov_topk[k] += 1

n = len(queries)
print("\n=== GOLD-ANSWER COVERAGE (dense+CE retrieval over wiki_500k) ===")
for k, c in cov_topk.items():
    print(f"  top-{k}: {c}/{n} = {c/n:.1%}")
print("\nIf top-4 coverage > ~50%, p=0 is a valid 'reliable retrieval' pole.")
