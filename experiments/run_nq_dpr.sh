#!/bin/bash
# Re-run NQ experiment with DPR Wikipedia corpus + BM25 retrieval
set -e
cd /home/shenjikun/experiments/hazard-early-stopping
eval "$(/home/shenjikun/miniconda3/bin/conda shell.bash hook)" && conda activate research

echo "=== STEP 1: Download DPR Wikipedia corpus ==="
CORPUS_DIR="/home/shenjikun/data/dpr_wiki"
mkdir -p "$CORPUS_DIR"
CORPUS_FILE="$CORPUS_DIR/psgs_w100.tsv"

if [ ! -f "$CORPUS_FILE" ]; then
    echo "Downloading DPR Wikipedia passages (psgs_w100.tsv.gz, ~1.5GB)..."
    wget -q --show-progress -O "${CORPUS_FILE}.gz" \
        "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz"
    echo "Extracting..."
    gunzip -f "${CORPUS_FILE}.gz"
    echo "Corpus ready: $(wc -l < $CORPUS_FILE) passages"
else
    echo "Corpus already exists: $(wc -l < $CORPUS_FILE) passages"
fi

echo "=== STEP 2: Build BM25 index ==="
python3 << 'PY'
import sys
sys.path.insert(0, "/home/shenjikun/experiments/hazard-early-stopping")
import pickle, os
from retrieval import BM25Retriever

corpus_path = "/home/shenjikun/data/dpr_wiki/psgs_w100.tsv"
index_path = "/home/shenjikun/data/dpr_wiki/bm25_index.pkl"
passages_path = "/home/shenjikun/data/dpr_wiki/passages.jsonl"

if os.path.exists(index_path):
    print(f"Index already exists: {index_path}")
else:
    # Load passages (first 2M for speed)
    passages = []
    with open(corpus_path, "r") as f:
        header = f.readline()  # skip header
        for i, line in enumerate(f):
            if i >= 2_000_000:
                break
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                passages.append({"id": parts[0], "text": parts[1], "title": parts[2]})
    
    print(f"Loaded {len(passages)} passages, building BM25 index...")
    retriever = BM25Retriever(passages)
    retriever.build_index()
    with open(index_path, "wb") as f:
        pickle.dump(retriever, f)
    print(f"Index saved: {index_path}")

# Quick test
print("Testing retrieval...")
with open(index_path, "rb") as f:
    retriever = pickle.load(f)
results = retriever.retrieve("when was the last time anyone was on the moon", top_k=5)
for r in results[:3]:
    print(f"  [{r[title]}] {r[text][:100]}...")
print("Index test OK")
PY

echo "=== STEP 3: Collect NQ hidden states (500 queries, 4 stages) ==="
python3 collect_states.py \
    --dataset nq \
    --num_queries 500 \
    --corpus_path /home/shenjikun/data/dpr_wiki/passages.jsonl \
    --index_path /home/shenjikun/data/dpr_wiki/bm25_index.pkl \
    --output_path data/collected_states_nq_dpr.jsonl \
    --real_retrieval \
    --rerank \
    --max_length 2048 \
    2>&1 | tee results/nq_dpr_collection.log

echo "=== STEP 4: Train MLP on NQ data ==="
python3 train_predictor.py \
    --data_path data/collected_states_nq_dpr.jsonl \
    --label_type stage \
    --seed 42 \
    --output_dir results/nq_dpr \
    2>&1 | tee results/nq_dpr_train.log

echo "=== STEP 5: Evaluate ==="
python3 evaluate.py \
    --data_path data/collected_states_nq_dpr.jsonl \
    --model_path checkpoints/predictor_mlp_seed42.pt \
    --output_dir results/nq_dpr_eval \
    2>&1 | tee results/nq_dpr_eval.log

echo "=== DONE ==="
echo "Results in: results/nq_dpr_eval/"
