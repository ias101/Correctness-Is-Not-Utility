#!/bin/bash
# Fix NQ by rebuilding BM25 index with 2M passages (full alphabet coverage)
# Same approach as HotpotQA: BM25 → cross-encoder rerank → progressive context
set -e
cd /home/shenjikun/experiments/hazard-early-stopping
eval "$(/home/shenjikun/miniconda3/bin/conda shell.bash hook)" && conda activate research

echo "=== STEP 1: Remove old broken BM25 index ==="
rm -f ~/wiki_cache/bm25_index.pkl ~/wiki_cache/wiki_passages.jsonl 2>/dev/null || true
echo "Old index removed."

echo "=== STEP 2: Rebuild BM25 index with 2M passages ==="
python3 -c "
import sys; sys.path.insert(0, .)
from retrieval import setup_retrieval
# Force rebuild with 2M passages (full alphabet coverage)
pipeline = setup_retrieval(num_passages=2_000_000, force_rebuild=True)
# Test retrieval quality on a known NQ query
results = pipeline.retrieve(when was the last time anyone was on the moon, top_k=10)
for i, r in enumerate(results[:5]):
    print(f [{i+1}] [{r["title"]}] {r["text"][:120]}...)
print(Index rebuilt and tested.)
" 2>&1 | tee results/nq_rebuild_index.log

echo "=== STEP 3: Collect NQ hidden states (500 queries, cross-encoder rerank, progressive context) ==="
python3 collect_states.py \
    --dataset nq \
    --num_queries 500 \
    --corpus_size 2000000 \
    --real_retrieval \
    --rerank \
    --max_length 2048 \
    --output_path data/collected_states_nq_v2.jsonl \
    2>&1 | tee results/nq_v2_collection.log

echo "=== STEP 4: Train MLP with stage-aligned labels ==="
python3 train_predictor.py \
    --data_path data/collected_states_nq_v2.jsonl \
    --label_type stage \
    --seed 42 \
    --output_dir results/nq_v2 \
    2>&1 | tee results/nq_v2_train.log

echo "=== STEP 5: Evaluate ==="
python3 evaluate.py \
    --data_path data/collected_states_nq_v2.jsonl \
    --model_path checkpoints/predictor_mlp_seed42.pt \
    --output_dir results/nq_v2_eval \
    2>&1 | tee results/nq_v2_eval.log

echo "=== DONE ==="
echo "Per-stage AUROC and accuracy in: results/nq_v2_eval/evaluation_results.json"
