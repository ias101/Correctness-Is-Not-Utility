"""
Shared configuration for Hazard-Based Adaptive Early Stopping experiments.
All hyperparameters from EXPERIMENT_PLAN.md are reflected here.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List

# ── Paths ──────────────────────────────────────────────────────────
# Use script's own directory as base (works both locally and on server)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_BASE_DIR) if os.path.basename(_BASE_DIR) == "experiments" else _BASE_DIR
DATA_DIR = os.path.join(_BASE_DIR, "data")
RESULTS_DIR = os.path.join(_BASE_DIR, "results")
MODEL_DIR = os.path.join(_BASE_DIR, "checkpoints")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ── GPU / Hardware ──────────────────────────────────────────────────
DEVICE = "cuda"  # "cuda" or "cpu"
USE_4BIT = True  # 4-bit quantization for LLaMA during data collection
BATCH_SIZE_COLLECT = 1  # batch size during hidden state collection
BATCH_SIZE_TRAIN = 128  # batch size for MLP training

# ── RAG Pipeline Stages ────────────────────────────────────────────
# Ordered list of stage names (0-indexed)
STAGES = ["retrieval", "reranking", "context_assembly", "generation"]
NUM_STAGES = len(STAGES)

# ── Model Paths (configurable) ─────────────────────────────────────
# Primary: Qwen2.5-7B — open-access, no auth required, 7B params, hidden=3584
LLAMA_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
# Alternative for lower VRAM: "Qwen/Qwen2.5-3B-Instruct"
LLAMA_MODEL_NAME_SMALL = "Qwen/Qwen2.5-3B-Instruct"
# If HF token available, use: "meta-llama/Meta-Llama-3-8B"

DPR_QUESTION_ENCODER = "facebook/dpr-question_encoder-single-nq-base"
DPR_CONTEXT_ENCODER = "facebook/dpr-ctx_encoder-single-nq-base"

# ── DPR Retrieval ───────────────────────────────────────────────────
DPR_TOP_K = 20  # number of passages to retrieve
DPR_RERANK_K = 5  # number of passages after reranking
CONTEXT_MAX_LENGTH = 2048  # max tokens for context (was 512, too short for Qwen chat template + passages)

# ── MLP Predictor Architecture (from FINAL_PROPOSAL.md) ────────────
# Frozen backbone hidden_dim depends on the model:
#   Qwen2.5-7B: 3584
#   LLaMA-3-8B: 4096
#   LLaMA-3.2-3B: 3072
#   Mistral-7B: 4096
LLAMA_HIDDEN_DIM = 3584  # Qwen2.5-7B default
STAGE_EMBEDDING_DIM = 16
MLP_HIDDEN_LAYERS = [256, 128]  # 2 hidden layers
MLP_DROPOUT = 0.1
MLP_ACTIVATION = "relu"

# ── Training ────────────────────────────────────────────────────────
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4  # L2 regularization for overfitting mitigation
EARLY_STOPPING_PATIENCE = 10
CLASS_BALANCED_SAMPLING = True  # equal pos/neg per batch

# ── Threshold Search ────────────────────────────────────────────────
TAU_MIN = 0.1
TAU_MAX = 0.9
TAU_STEP = 0.05
LAMBDA_VALUES = [0.2, 0.5, 0.8]  # cost-accuracy trade-off weights

# ── Random Seeds ────────────────────────────────────────────────────
SEEDS = [42, 123, 456]  # for multi-seed MLP training
BASE_SEED = 42  # for data splitting

# ── Datasets ────────────────────────────────────────────────────────
# Block 1 (Pilot): NQ 500 queries
PILOT_NQ_QUERIES = 500

# Block 2 (Main): Full datasets
NQ_FULL_QUERIES = 3610
HOTPOTQA_QUERIES = 7405
TRIVIAQA_QUERIES = 1000  # test set only

DATASET_NAMES = ["nq", "hotpotqa", "triviaqa"]

# ── Block Budgets (GPU-hours) ──────────────────────────────────────
BUDGETS = {
    "block1_pilot": 2,
    "block2_main": 15,
    "block3_ablation": 8,
    "block4_cross_model": 10,
    "block5_latency": 1,
}


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

    # ── Experiment identity ──
    block: str = "block1"
    run_id: str = "1.1"
    description: str = ""

    # ── Data ──
    dataset: str = "nq"
    num_queries: int = 500
    train_split: float = TRAIN_SPLIT
    val_split: float = VAL_SPLIT
    test_split: float = TEST_SPLIT

    # ── Model ──
    llama_model: str = LLAMA_MODEL_NAME
    use_4bit: bool = USE_4BIT
    hidden_dim: int = LLAMA_HIDDEN_DIM

    # ── MLP ──
    stage_emb_dim: int = STAGE_EMBEDDING_DIM
    mlp_hidden: List[int] = field(default_factory=lambda: MLP_HIDDEN_LAYERS)
    mlp_dropout: float = MLP_DROPOUT

    # ── Training ──
    seed: int = BASE_SEED
    batch_size_train: int = BATCH_SIZE_TRAIN
    num_epochs: int = NUM_EPOCHS
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE
    class_balanced: bool = CLASS_BALANCED_SAMPLING

    # ── Evaluation ──
    tau_min: float = TAU_MIN
    tau_max: float = TAU_MAX
    tau_step: float = TAU_STEP
    lambda_values: List[float] = field(default_factory=lambda: LAMBDA_VALUES)

    # ── Output ──
    data_dir: str = DATA_DIR
    results_dir: str = RESULTS_DIR
    model_dir: str = MODEL_DIR

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def save_path(self, suffix: str) -> str:
        """Generate a results file path."""
        return os.path.join(self.results_dir, f"{self.block}_{self.run_id}_{suffix}")
