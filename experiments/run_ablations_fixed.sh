#!/bin/bash
cd /home/shenjikun/experiments/hazard-early-stopping
eval "$(/home/shenjikun/miniconda3/bin/conda shell.bash hook)"
conda activate research
mkdir -p results/ablations
PYTHONPATH=".:experiments" python3 experiments/run_ablation_critical.py \
  --data data/collected_states_hotpotqa_v4.jsonl \
  --ablations all \
  2>&1 | tee results/ablations/ablation_all.log
echo "ALL_DONE"
