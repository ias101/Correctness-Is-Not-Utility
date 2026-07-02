import sys
sys.path.insert(0, ".")
from retrieval import setup_retrieval

print("Rebuilding BM25 index with 2M passages (fresh download)...")
pipeline = setup_retrieval(num_passages=2000000, force_rebuild=True)

print("Testing retrieval...")
results = pipeline.retrieve("when was the last time anyone was on the moon", k=10)
for i, r in enumerate(results[:5]):
    snippet = r["text"][:120]
    print(f"  [{i+1}] [{r[title]}] {snippet}...")
print("Index rebuilt and tested.")
