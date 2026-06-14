#!/bin/bash
# Deploy experiment code to GPU server and set up environment.
# Usage: bash experiments/deploy.sh [--run-pilot]

set -e

GPU_HOST="shenjikun@192.168.178.108"
GPU_DIR="/home/shenjikun/experiments/hazard-early-stopping"
CONDA_BASE="/home/shenjikun/miniconda3"

echo "=============================================="
echo "  Deploying Hazard Early Stopping Experiments"
echo "=============================================="

# 1. Create remote directory
echo "[*] Creating remote directory..."
ssh "$GPU_HOST" "mkdir -p $GPU_DIR/data $GPU_DIR/results $GPU_DIR/checkpoints"

# 2. Sync experiment code
echo "[*] Syncing experiment code..."
rsync -avz --progress \
    experiments/config.py \
    experiments/models.py \
    experiments/collect_states.py \
    experiments/train_predictor.py \
    experiments/evaluate.py \
    experiments/run_pilot.py \
    experiments/requirements.txt \
    "$GPU_HOST:$GPU_DIR/"

# 3. Activate conda and install dependencies
echo "[*] Installing Python dependencies..."
ssh "$GPU_HOST" "eval \"\$($CONDA_BASE/bin/conda shell.bash hook)\" && \
    conda activate research && \
    cd $GPU_DIR && \
    pip install -r requirements.txt -q"

# 4. Verify GPU availability
echo "[*] Verifying GPU..."
ssh "$GPU_HOST" "nvidia-smi --query-gpu=name,memory.free --format=csv,noheader"

echo ""
echo "✅ Deployment complete. Code at: $GPU_DIR"
echo ""

# Optional: Run pilot immediately
if [[ "$1" == "--run-pilot" ]]; then
    echo "[*] Launching Block 1 Pilot..."
    ssh "$GPU_HOST" "eval \"\$($CONDA_BASE/bin/conda shell.bash hook)\" && \
        conda activate research && \
        cd $GPU_DIR && \
        python run_pilot.py --num_queries 500 2>&1 | tee results/block1_pilot/pilot_run.log"
fi
