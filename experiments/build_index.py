#!/usr/bin/env python3
"""Build BM25 Wikipedia index for RAG retrieval."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from retrieval import setup_retrieval
    pipeline = setup_retrieval(num_passages=200000)
    results = pipeline.retrieve("What is machine learning?")
    for r in results[:3]:
        title = r["title"]
        text = r["text"][:150]
        print(f"[{title}] {text}...")
    print("INDEX_BUILD_COMPLETE")
