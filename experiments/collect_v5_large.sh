#!/bin/bash
# V5 HotpotQA Collection — Larger Scale + Multi-Layer for Statistical Significance
#
# Runs collect_states.py (which supports multi_layer hooks on last 4 layers)
# with HotpotQA-distractor, progressive context scaling (2→4→6→8).
# Targets 2000+ queries for robust bootstrap + cross-validation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Config ──────────────────────────────────────────────────────────
NUM_QUERIES=${1:-2000}
OUTPUT_FILE="${2:-${SCRIPT_DIR}/data/collected_states_hotpotqa_v5_multi_layer.jsonl}"
REP_TYPE="multi_layer"   # final_token, mean_pool, or multi_layer
MODEL="Qwen/Qwen2.5-7B-Instruct"

echo "============================================"
echo "V5 Collection: ${NUM_QUERIES} queries, ${REP_TYPE} features"
echo "Model: ${MODEL}"
echo "Output: ${OUTPUT_FILE}"
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%S)"
echo "============================================"

# ── Activate conda ──────────────────────────────────────────────────
eval "$(/home/shenjikun/miniconda3/bin/conda shell.bash hook)"
conda activate research

# ── Run collection ──────────────────────────────────────────────────
python experiments/collect_states.py \
    --dataset hotpotqa_distractor \
    --num_queries "${NUM_QUERIES}" \
    --model_name "${MODEL}" \
    --rep_type "${REP_TYPE}" \
    --output "${OUTPUT_FILE}" \
    --use_4bit \
    --progressive_context \
    --context_sizes "2,4,6,8" \
    --reranker cross-encoder/ms-marco-MiniLM-L-6-v2 \
    --max_new_tokens 48 \
    --seed 42

echo ""
echo "============================================"
echo "Collection complete: $(date -u +%Y-%m-%dT%H:%M:%S)"
echo "Output: ${OUTPUT_FILE}"
echo "Lines: $(wc -l < "${OUTPUT_FILE}")"
echo "============================================"
