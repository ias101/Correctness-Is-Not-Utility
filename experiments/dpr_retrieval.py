#!/usr/bin/env python3
"""DPR retriever: GPU-accelerated chunked search over 21M fp16 passages.

Uses GPU for similarity (RTX 4090 24GB): 2M chunks at ~6GB each.
"""
import json, os, numpy as np, torch
from transformers import AutoTokenizer, AutoModel

CORPUS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "wiki_dpr_corpus")
PASSAGES_FILE = os.path.join(CORPUS_DIR, "passages.jsonl")
EMBEDDINGS_FILE = os.path.join(CORPUS_DIR, "contriever_embeddings.npy")
MODEL = "facebook/contriever"
TOP_K = 20
GPU_CHUNK = 1000000  # 1M per batch (~3GB VRAM fp32)

class DPRRetriever:
    def __init__(self, passages, emb_path, model, tokenizer, device="cuda"):
        self.passages = passages
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self._emb = np.memmap(emb_path, dtype=np.float16, mode="r", shape=(len(passages), 768))
        self.n = len(passages)

    def retrieve(self, query, top_k=TOP_K):
        inp = self.tokenizer(query, return_tensors="pt", truncation=True,
                             max_length=512).to(self.device)
        with torch.no_grad():
            q = self.model(**inp).last_hidden_state.mean(dim=1)
            q = q / q.norm(dim=-1, keepdim=True)  # (1, 768)
            q = q.to(dtype=torch.float16)  # fp16 for efficient dot product

        all_scores, all_idx = [], []
        for start in range(0, self.n, GPU_CHUNK):
            end = min(start + GPU_CHUNK, self.n)
            chunk = torch.from_numpy(
                self._emb[start:end]
            ).to(dtype=torch.float16, device=self.device)  # keep fp16 on GPU
            scores = (q @ chunk.T).squeeze(0)
            k = min(top_k, scores.shape[0])
            vals, idx = torch.topk(scores, k)
            all_scores.append(vals.cpu())
            all_idx.append(idx.cpu() + start)

        all_scores = torch.cat(all_scores)
        all_idx = torch.cat(all_idx)
        final = torch.topk(all_scores, min(top_k, all_scores.shape[0]))
        results = []
        for i in range(final.values.shape[0]):
            idx = int(all_idx[final.indices[i]])
            p = self.passages[idx]
            results.append({"id": p["id"], "title": p["title"], "text": p["text"], "score": float(final.values[i])})
        return results

    def rerank(self, query, passages):
        return passages[:5]

def setup_dpr_retrieval():
    if not os.path.exists(EMBEDDINGS_FILE):
        raise FileNotFoundError(f"DPR index not found at {EMBEDDINGS_FILE}")
    n = sum(1 for _ in open(PASSAGES_FILE))
    print(f"[*] Loading {n} passages")
    passages = [json.loads(l) for l in open(PASSAGES_FILE)]
    print(f"[*] {len(passages)} passages ready, GPU chunk={GPU_CHUNK}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to("cuda")
    model.eval()
    r = DPRRetriever(passages, EMBEDDINGS_FILE, model, tokenizer)
    class P:
        def __init__(self, r):
            self.retrieve = r.retrieve
            self.rerank = r.rerank
            self.passages = passages
        def build_stage_prompt(self, question, retrieved, reranked, stage):
            # Fallback: simple formatting (same as collect_states.py fallback)
            if stage == 0:
                context = "\n\n".join(f"Passage {i+1}: {p['text'][:300]}" for i, p in enumerate(retrieved[:10]) if isinstance(p, dict))
                return f"Question: {question}\n\nRetrieved passages:\n{context}\n\nAnswer the question concisely."
            elif stage in (1, 2):
                ctx = " ".join(p["text"] if isinstance(p, dict) else str(p) for p in reranked[:5])
                return f"Context: {ctx}\n\nQuestion: {question}\n\nAnswer:"
            else:
                ctx = " ".join(p["text"] if isinstance(p, dict) else str(p) for p in reranked[:5])
                return f"Context: {ctx}\n\nQuestion: {question}\n\nProvide a concise answer:"
    return P(r), r

if __name__ == "__main__":
    p, r = setup_dpr_retrieval()
    for q in ["who discovered penicillin", "what is the capital of france"]:
        results = p.retrieve(q)
        print(f"\nQ: {q}")
        for x in results[:3]:
            print(f"  [{x['title']}] {x['text'][:120]}")
