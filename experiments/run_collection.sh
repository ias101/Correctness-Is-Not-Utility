#!/bin/bash
eval "$(/opt/miniforge3/bin/conda shell.bash hook)"
conda activate main
cd /root/experiments/hazard-early-stopping/experiments

echo "=== Starting NQ data collection with real BM25 retrieval ==="
echo "Date: $(date)"

# Run pilot — real_retrieval is now DEFAULT in collect_states.py (no flag needed)
python run_pilot.py --num_queries 500 2>&1

echo ""
echo "=== Collection complete at $(date) ==="
