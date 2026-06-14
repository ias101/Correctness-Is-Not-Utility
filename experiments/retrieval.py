"""
Real retrieval module for RAG pipeline.

Uses BM25 over Wikipedia passages for efficient, real retrieval.
Replaces placeholder passages from the pilot phase.

Pipeline:
  1. Load Wikipedia passage corpus (subsampled from HF 'wikipedia' dataset)
  2. Build BM25 index
  3. For each query: retrieve top-k passages → optional rerank → return passages

This gives us a real, meaningful retrieval pipeline without the
complexity of DPR + FAISS (which can be upgraded later for final paper).
"""

import json
import os
import pickle
import sys
import time
from typing import List, Dict, Tuple, Optional

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATA_DIR,
    DPR_TOP_K,
    DPR_RERANK_K,
)

# ── Wikipedia Corpus ────────────────────────────────────────────────

WIKI_CACHE_DIR = os.path.join(DATA_DIR, "wiki_corpus")
WIKI_PASSAGES_FILE = os.path.join(WIKI_CACHE_DIR, "passages.jsonl")
BM25_INDEX_FILE = os.path.join(WIKI_CACHE_DIR, "bm25_index.pkl")
DEFAULT_NUM_PASSAGES = 200_000  # 200K passages — good coverage, manageable size


def download_wikipedia_passages(
    num_passages: int = DEFAULT_NUM_PASSAGES,
    cache_dir: str = WIKI_CACHE_DIR,
    min_passage_len: int = 100,
) -> List[Dict]:
    """
    Download Wikipedia passages from HuggingFace 'wikipedia' dataset.

    Each passage is a paragraph from a Wikipedia article.
    Filters out very short passages.

    Returns list of {id, text, title} dicts.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Check cache first
    if os.path.exists(WIKI_PASSAGES_FILE):
        print(f"[*] Loading cached Wikipedia passages from {WIKI_PASSAGES_FILE}")
        passages = []
        with open(WIKI_PASSAGES_FILE, "r", encoding="utf-8") as f:
            for line in f:
                passages.append(json.loads(line))
                if len(passages) >= num_passages:
                    break
        print(f"[*] Loaded {len(passages)} passages from cache")
        return passages[:num_passages]

    print(f"[*] Downloading Wikipedia passages (target: {num_passages})...")
    from datasets import load_dataset

    # Use wikimedia/wikipedia (newer format, no loading script needed)
    dataset = load_dataset(
        "wikimedia/wikipedia", "20231101.en", split="train",
        streaming=True,
    )

    passages = []
    pid = 0

    for article in tqdm(dataset, desc="Processing Wikipedia", unit="articles"):
        title = article.get("title", "")
        text = article.get("text", "")

        if not text:
            continue

        # Split article into paragraphs
        paragraphs = text.split("\n")
        for para in paragraphs:
            para = para.strip()
            if len(para) < min_passage_len:
                continue
            # Truncate very long paragraphs
            if len(para) > 2000:
                para = para[:2000]
            passages.append({
                "id": pid,
                "title": title,
                "text": para,
            })
            pid += 1

            if len(passages) >= num_passages:
                break

        if len(passages) >= num_passages:
            break

    # Save to cache
    with open(WIKI_PASSAGES_FILE, "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"[*] Downloaded and saved {len(passages)} passages to {WIKI_PASSAGES_FILE}")
    return passages


def build_bm25_index(
    passages: List[Dict],
    index_path: str = BM25_INDEX_FILE,
) -> "BM25Okapi":
    """
    Build BM25 index over Wikipedia passages.

    Returns BM25Okapi object ready for retrieval.
    """
    from rank_bm25 import BM25Okapi
    import re

    print(f"[*] Building BM25 index over {len(passages)} passages...")

    # Tokenize passages
    def tokenize(text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return text.split()

    tokenized = []
    for p in tqdm(passages, desc="Tokenizing"):
        tokens = tokenize(p["text"])
        tokenized.append(tokens)

    bm25 = BM25Okapi(tokenized)

    # Save index
    with open(index_path, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "passages": passages,
            "tokenized": tokenized,
        }, f)

    print(f"[*] BM25 index saved to {index_path}")
    return bm25


def load_bm25_index(index_path: str = BM25_INDEX_FILE) -> Tuple:
    """Load cached BM25 index and passages."""
    if not os.path.exists(index_path):
        return None, None, None

    print(f"[*] Loading BM25 index from {index_path}...")
    with open(index_path, "rb") as f:
        data = pickle.load(f)

    print(f"[*] Loaded index with {len(data['passages'])} passages")
    return data["bm25"], data["passages"], data.get("tokenized", [])


# ── Retrieval ───────────────────────────────────────────────────────


def retrieve_bm25(
    query: str,
    bm25: "BM25Okapi",
    passages: List[Dict],
    top_k: int = DPR_TOP_K,
) -> List[Dict]:
    """
    Retrieve top-k passages for a query using BM25.

    Returns list of {id, title, text, score} sorted by relevance (descending).
    """
    import re

    def tokenize(text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return text.split()

    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        p = passages[idx]
        results.append({
            "id": p["id"],
            "title": p["title"],
            "text": p["text"],
            "score": float(scores[idx]),
        })

    return results


def rerank_passages(
    query: str,
    passages: List[Dict],
    top_k: int = DPR_RERANK_K,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> List[Dict]:
    """
    Rerank retrieved passages using a cross-encoder.

    Returns top-k reranked passages.
    """
    from sentence_transformers import CrossEncoder

    print(f"[*] Loading cross-encoder: {model_name}")
    model = CrossEncoder(model_name)

    pairs = [(query, p["text"]) for p in passages]
    scores = model.predict(pairs, show_progress_bar=False)

    # Sort by score
    scored = sorted(
        zip(passages, scores), key=lambda x: x[1], reverse=True
    )
    return [p for p, s in scored[:top_k]]


# ── RAG Pipeline ────────────────────────────────────────────────────


class RAGPipeline:
    """
    Full RAG retrieval pipeline for experiment data collection.

    Stages:
      S1: BM25 retrieval → top-20 passages
      S2: Cross-encoder reranking → top-5 passages
      S3: Context assembly → format passages for LLM prompt
      S4: Generation → LLM generates answer

    At each stage, we capture the LLM's hidden state for failure prediction.
    """

    def __init__(
        self,
        bm25=None,
        passages: List[Dict] = None,
        use_reranker: bool = False,  # Disabled by default: cross-encoder too slow for collection
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k_retrieval: int = DPR_TOP_K,
        top_k_rerank: int = DPR_RERANK_K,
    ):
        self.bm25 = bm25
        self.passages = passages or []
        self.use_reranker = use_reranker
        self.reranker_model = reranker_model
        self.reranker = None
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank

    def retrieve(self, query: str) -> List[Dict]:
        """Stage 1: BM25 retrieval."""
        if self.bm25 is None:
            # Fallback: return empty, use question text as context
            return [{"id": 0, "title": "fallback", "text": query, "score": 0.0}]
        return retrieve_bm25(query, self.bm25, self.passages, self.top_k_retrieval)

    def rerank(self, query: str, passages: List[Dict]) -> List[Dict]:
        """Stage 2: Cross-encoder reranking."""
        if not self.use_reranker or len(passages) <= self.top_k_rerank:
            return passages[: self.top_k_rerank]

        if self.reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder(self.reranker_model)
            except Exception as e:
                print(f"[!] Could not load reranker: {e}")
                return passages[: self.top_k_rerank]

        return rerank_passages(query, passages, self.top_k_rerank)

    def assemble_context(self, passages: List[Dict]) -> str:
        """Stage 3: Assemble passages into a context string."""
        context_parts = []
        for i, p in enumerate(passages[: self.top_k_rerank]):
            context_parts.append(f"[{i+1}] {p['title']}: {p['text']}")
        return "\n\n".join(context_parts)

    def build_stage_prompt(
        self,
        question: str,
        retrieved: List[Dict],
        reranked: List[Dict],
        stage: int,
    ) -> str:
        """
        Build the LLM prompt for a specific RAG stage.

        Stage 0 (retrieval): Show top-20 retrieved passages
        Stage 1 (reranking): Show top-5 reranked passages
        Stage 2 (context_assembly): Formatted context with attribution
        Stage 3 (generation): Final answer prompt
        """
        if stage == 0:
            # Retrieval stage: show raw retrieval results
            context = "\n".join(
                f"[{i+1}] {p['title']}: {p['text'][:150]}..."
                for i, p in enumerate(retrieved[:10])
            )
            return (
                f"Below are retrieved passages for the question:\n\n{context}\n\n"
                f"Question: {question}\n\nBased on the retrieved information, answer the question."
            )

        elif stage == 1:
            # Reranking stage: show top reranked passages
            context = "\n".join(
                f"[{i+1}] {p['title']}: {p['text'][:200]}..."
                for i, p in enumerate(reranked[:5])
            )
            return (
                f"Below are the most relevant passages for the question:\n\n{context}\n\n"
                f"Question: {question}\n\nBased on these passages, answer the question."
            )

        elif stage == 2:
            # Context assembly: full formatted context
            context = self.assemble_context(reranked)
            return (
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\nAnswer:"
            )

        elif stage == 3:
            # Generation: final prompt
            context = self.assemble_context(reranked)
            return (
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\nProvide a concise, accurate answer based on the context.\n"
                f"Answer:"
            )

        else:
            return f"Question: {question}\n\nAnswer:"


# ── Setup ───────────────────────────────────────────────────────────


def setup_retrieval(
    num_passages: int = DEFAULT_NUM_PASSAGES,
    force_rebuild: bool = False,
) -> RAGPipeline:
    """
    Set up the full retrieval pipeline.

    1. Download Wikipedia passages (or load from cache)
    2. Build BM25 index (or load from cache)
    3. Return RAGPipeline ready for use
    """
    print("[*] Setting up retrieval pipeline...")

    # Load or download passages
    if force_rebuild or not os.path.exists(BM25_INDEX_FILE):
        passages = download_wikipedia_passages(num_passages)
        bm25 = build_bm25_index(passages)
    else:
        bm25, passages, _ = load_bm25_index()
        if bm25 is None:
            print("[!] Failed to load BM25 index, rebuilding...")
            passages = download_wikipedia_passages(num_passages)
            bm25 = build_bm25_index(passages)

    pipeline = RAGPipeline(bm25=bm25, passages=passages)
    print("[*] Retrieval pipeline ready.")
    return pipeline


if __name__ == "__main__":
    # Test: build the index
    pipeline = setup_retrieval(num_passages=10000)  # Small test
    results = pipeline.retrieve("What is machine learning?")
    for r in results[:3]:
        print(f"  [{r['title']}] {r['text'][:100]}...")
