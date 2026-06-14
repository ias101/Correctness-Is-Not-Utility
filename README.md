# Correctness Is Not Utility: Why Threshold-Based Hidden-State Routing Fails in Adaptive RAG

Official code release for the paper **"Correctness Is Not Utility: Why Threshold-Based Hidden-State Routing Fails in Adaptive RAG"** (AAAI 2027 submission).

## TL;DR

We probe frozen LLM hidden states across 4 RAG stages and find:

1. **Correctness detection is strong** (AUROC 0.78–0.96) — LLMs "know" when they're wrong
2. **Utility prediction is weak** (AUROC ~0.58) — LLMs don't know whether more context will fix a wrong answer
3. **Threshold-based routing fails** because probe scores are bimodal — no threshold separates queries needing more context from those ready to stop
4. The benefit signal hidden states lack is recoverable from **retrieval-side features** (AUROC 0.78 vs 0.52), suggesting hybrid routing as the path forward

**Models**: Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3
**Datasets**: HotpotQA-distractor, PopQA, TriviaQA-open
**GPU**: Single RTX 3080 Laptop (16GB) with 4-bit quantization

## Repository Structure

```
├── experiments/           # All experiment scripts (35 Python files)
│   ├── config.py          # Shared hyperparameters & paths
│   ├── models.py          # MLP probe architectures
│   ├── cost_model.py      # Cost-weighted accuracy (CWA) computation
│   ├── retrieval.py       # BM25 passage retrieval + cross-encoder reranking
│   ├── bootstrap_analysis.py  # Bootstrap CIs, permutation tests
│   │
│   ├── collect_states.py           # Core hidden-state collection (RAG pipeline)
│   ├── collect_v5_multilayer.py    # V5 HotpotQA collection (2000 queries, multi-layer)
│   ├── collect_v3_mistral_matched.py  # Matched Mistral-7B collection
│   ├── collect_triviaqa_open.py    # TriviaQA open-domain collection
│   ├── collect_popqa_v3.py         # PopQA collection
│   │
│   ├── train_predictor.py          # Core MLP probe training + Platt calibration
│   ├── canonical_v5_full.py        # Canonical V5 recompute (all numbers in one run)
│   ├── conditional_delta_v5.py     # Conditional Delta Probe (benefit/degradation)
│   ├── conditional_delta_multi.py  # Multi-regime conditional delta
│   ├── train_popqa_probes.py       # PopQA correctness + delta probes
│   │
│   ├── evaluate.py                 # CWA evaluation, Pareto frontier, ECE calibration
│   ├── routing_baselines.py        # Routing policies (fixed/random/oracle/threshold)
│   ├── routing_hotpotqa_v5.py      # HotpotQA threshold-routing table
│   ├── eval_routing_popqa.py       # PopQA routing evaluation
│   ├── compute_cumulative_cwa.py   # Cumulative CWA for PopQA
│   │
│   ├── oracle_conditional_benefit.py  # Oracle retrieval-side benefit features
│   ├── mlp_v5_delta_probe.py       # MLP confound disentanglement
│   ├── run_ablation_critical.py    # Architecture sweep, LSTM, layer-wise, retrieval
│   │
│   ├── baseline_external_router.py # Query-embedding LR baseline (reviewer request)
│   ├── eval_cross_compare.py       # Cross-dataset transfer evaluation
│   ├── extended_delta_analysis.py  # Multi-step/degradation/net delta label variants
│   ├── collect_multimodel.py       # Multi-model collection utility
│   ├── regime_control_analysis.py  # Within-stage correctness + stage identity baselines
│   │
│   ├── gemini_review.py            # Cross-model adversarial paper review tool
│   ├── assemble_review_context.py  # Paper review context assembler
│   │
│   ├── deploy.sh                   # rsync+ssh deploy to WSL2 GPU server
│   ├── download_mistral.py         # Mistral-7B model downloader
│   ├── download_model_script.py    # Qwen2.5-7B model downloader
│   └── requirements.txt           # Python dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r experiments/requirements.txt
```

### 2. Download Models

```bash
python experiments/download_model_script.py   # Qwen2.5-7B-Instruct
python experiments/download_mistral.py         # Mistral-7B-Instruct-v0.3
```

### 3. Reproduce Main Results

The canonical pipeline has 4 steps:

```bash
# Step 1: Collect hidden states from RAG pipeline
python experiments/collect_v5_multilayer.py \
    --num_queries 2000 --output data/hotpotqa_v5_states.jsonl

# Step 2: Train correctness & delta probes, compute all numbers
python experiments/canonical_v5_full.py \
    --data data/hotpotqa_v5_states.jsonl \
    --output_dir results/v5_experiment/

# Step 3: Run conditional delta probe analysis
python experiments/conditional_delta_v5.py \
    --data data/hotpotqa_v5_states.jsonl \
    --output_dir results/v5_experiment/

# Step 4: Evaluate routing policies
python experiments/routing_hotpotqa_v5.py \
    --data data/hotpotqa_v5_states.jsonl \
    --output_dir results/v5_experiment/
```

## Hardware Requirements

- **GPU**: 16GB VRAM minimum (tested on RTX 3080 Laptop)
- **Quantization**: 4-bit (bitsandbytes) for 7B models
- **Disk**: ~5GB for models, ~2GB for datasets

## Citation

```bibtex
@article{shen2026correctness,
  title={Correctness Is Not Utility: Why Threshold-Based Hidden-State Routing Fails in Adaptive RAG},
  author={Shen, Jikun},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT

## Contact

Jikun Shen — University of Amsterdam — jikun.shen@student.uva.nl
