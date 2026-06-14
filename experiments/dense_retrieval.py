"""
Dense retrieval using Contriever (facebook/contriever).
Encodes Wikipedia passages and queries for high-quality retrieval.
Replaces BM25 with semantic dense retrieval.
"""
import json
import os
import sys
import time
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, DPR_TOP_K, DPR_RERANK_K
from retrieval import RAGPipeline

DENSE_EMBEDDINGS_FILE = os.path.join(DATA_DIR, "wiki_corpus", "contriever_embeddings.npy")
DENSE_IDS_FILE = os.path.join(DATA_DIR, "wiki_corpus", "contriever_ids.json")


class DenseRetriever:
    """Dense retrieval using Contriever bi-encoder."""

    def __init__(self, passages, embeddings, model, tokenizer, device="cuda"):
        self.passages = passages
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32, device=device)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def retrieve(self, query: str, top_k: int = DPR_TOP_K) -> list:
        """Retrieve top-k passages using cosine similarity."""
        inputs = self.tokenizer(query, return_tensors="pt",
                                truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            q_emb = self.model(**inputs).last_hidden_state.mean(dim=1)  # mean pooling
            q_emb = q_emb / q_emb.norm(dim=-1, keepdim=True)

        # Cosine similarity
        scores = (q_emb @ self.embeddings.T).squeeze(0)
        top_indices = scores.argsort(descending=True)[:top_k].cpu().numpy()

        results = []
        for idx in top_indices:
            results.append({
                "id": self.passages[idx]["id"],
                "title": self.passages[idx]["title"],
                "text": self.passages[idx]["text"],
                "score": float(scores[idx].item()),
            })
        return results

    def rerank(self, query: str, passages: list) -> list:
        """Simple top-k rerank (no cross-encoder needed for dense retrieval)."""
        return passages[:DPR_RERANK_K]


def build_dense_index(num_passages=200000, model_name="facebook/contriever",
                      force_rebuild=False):
    """Encode all Wikipedia passages with Contriever."""
    if not force_rebuild and os.path.exists(DENSE_EMBEDDINGS_FILE):
        print(f"[*] Loading cached dense embeddings from {DENSE_EMBEDDINGS_FILE}")
        embeddings = np.load(DENSE_EMBEDDINGS_FILE)
        with open(DENSE_IDS_FILE) as f:
            passage_ids = json.load(f)
        return embeddings, passage_ids

    from transformers import AutoTokenizer, AutoModel

    # Load passages from BM25 cache
    passages_file = os.path.join(DATA_DIR, "wiki_corpus", "passages.jsonl")
    if not os.path.exists(passages_file):
        print("[!] No passages found. Run retrieval.py setup first.")
        return None, None

    passages = []
    with open(passages_file, "r", encoding="utf-8") as f:
        for line in f:
            passages.append(json.loads(line))
            if len(passages) >= num_passages:
                    pass  # load ALL passages

    print(f"[*] Encoding {len(passages)} passages with Contriever...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to("cuda")
    model.eval()

    embeddings = []
    batch_size = 256

    for i in tqdm(range(0, len(passages), batch_size), desc="Encoding"):
        batch = passages[i:i+batch_size]
        texts = [p["text"][:512] for p in batch]
        inputs = tokenizer(texts, return_tensors="pt", truncation=True,
                          max_length=512, padding=True).to("cuda")

        with torch.no_grad():
            outputs = model(**inputs)
            cls_emb = outputs.last_hidden_state.mean(dim=1)  # mean pooling (Contriever uses mean, not CLS)
            cls_emb = cls_emb / cls_emb.norm(dim=-1, keepdim=True)
            embeddings.append(cls_emb.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    passage_ids = [p["id"] for p in passages]

    np.save(DENSE_EMBEDDINGS_FILE, embeddings)
    with open(DENSE_IDS_FILE, "w") as f:
        json.dump(passage_ids, f)

    print(f"[*] Saved {len(embeddings)} embeddings to {DENSE_EMBEDDINGS_FILE}")
    return embeddings, passage_ids


def setup_dense_retrieval(num_passages=200000) -> RAGPipeline:
    """Set up dense retrieval pipeline, falling back to BM25 if needed."""
    from transformers import AutoTokenizer, AutoModel

    embeddings, passage_ids = build_dense_index(num_passages)

    if embeddings is None:
        print("[!] Dense retrieval setup failed, falling back to BM25")
        from retrieval import setup_retrieval
        return setup_retrieval(num_passages)

    # Load passages
    passages_file = os.path.join(DATA_DIR, "wiki_corpus", "passages.jsonl")
    passages = []
    with open(passages_file, "r", encoding="utf-8") as f:
        for line in f:
            passages.append(json.loads(line))
            if len(passages) >= num_passages:
                pass  # load ALL passages

    # Load Contriever
    model_name = "facebook/contriever"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to("cuda")
    model.eval()

    retriever = DenseRetriever(passages, embeddings, model, tokenizer)
    pipeline = RAGPipeline()
    pipeline.retrieve = retriever.retrieve
    pipeline.rerank = retriever.rerank
    pipeline.passages = passages

    print("[*] Dense retrieval pipeline ready.")
    return pipeline


if __name__ == "__main__":
    print("Building dense retrieval index...")
    pipeline = setup_dense_retrieval(num_passages=200000)
    results = pipeline.retrieve("What is machine learning?")
    for r in results[:3]:
        print(f"  [{r['title']}] {r['text'][:100]}...")
