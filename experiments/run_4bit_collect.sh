#!/bin/bash
eval "$(/home/shenjikun/miniconda3/bin/conda shell.bash hook)"
conda activate research
cd /home/shenjikun/experiments/hazard-early-stopping
echo "Python: $(which python3)"
echo "Conda env: $CONDA_DEFAULT_ENV"
python3 collect_4bit_compare.py --num_queries 100 --output data/collected_4bit_compare.jsonl 2>&1
echo "EXIT_CODE=$?"
