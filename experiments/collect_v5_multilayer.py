#!/usr/bin/env python3
"""
V5 HotpotQA Collection — Multi-Layer + Large Scale for Statistical Significance.

Key differences from V3/V4:
  - Multi-layer hidden states (last 4 transformer layers concatenated)
  - Larger scale: 2000+ queries for robust bootstrap + CV
  - Query-level data splitting preserved for proper CV

Uses the base collect_states.py infrastructure with register_hooks for
multi-layer extraction on last 4 transformer layers.

Expected runtime: ~6-7 hours on RTX 3080 16GB for 2000 queries.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# ── Self-contained constants (no config.py dependency) ───────────────
STAGES = ["retrieval", "reranking", "context_assembly", "generation"]
NUM_STAGES = len(STAGES)
LLAMA_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
BASE_SEED = 42
CONTEXT_MAX_LENGTH = 2048

os.makedirs(DATA_DIR, exist_ok=True)

from collect_states import (
    set_seed, load_llama_with_hidden_states,
    extract_hidden_state, generate_answer,
    check_correctness, _save_jsonl, load_collected_data,
)

# ── V5-specific constants ─────────────────────────────────────────────
STAGE_CONTEXT_SIZES = [2, 4, 6, 8]  # progressive
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_NEW_TOKENS = 48
BATCH_SIZE = 1  # one query at a time for hidden state extraction


def main():
    parser = argparse.ArgumentParser(description="V5 Multi-Layer HotpotQA Collection")
    parser.add_argument("--num_queries", type=int, default=2000,
                        help="Number of queries to collect (default: 2000)")
    parser.add_argument("--output", type=str,
                        default=os.path.join(DATA_DIR, "collected_states_hotpotqa_v5_multi_layer.jsonl"),
                        help="Output JSONL path")
    parser.add_argument("--model_name", type=str, default=LLAMA_MODEL_NAME,
                        help="HF model name")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                        help="Use 4-bit quantization")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Starting query index (for resuming)")
    args = parser.parse_args()

    set_seed(args.seed)

    # ── Load dataset ──
    print(f"Loading HotpotQA-distractor (train split)...")
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    total_available = len(dataset)
    n_queries = min(args.num_queries, total_available)
    print(f"  Available: {total_available}, Using: {n_queries}")

    # ── Load LLM ──
    print(f"Loading model: {args.model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model, tokenizer, hook_handles, hidden_states_store = load_llama_with_hidden_states(
        args.model_name,
        use_4bit=args.use_4bit,
    )

    # ── Load cross-encoder ──
    print(f"Loading cross-encoder: {RERANKER_MODEL}")
    from sentence_transformers import CrossEncoder
    ce_model = CrossEncoder(RERANKER_MODEL, device=str(device))

    # ── Collect ──
    print(f"\nCollecting {n_queries} queries (indices {args.start_idx}–{args.start_idx + n_queries - 1})...")
    print(f"Output: {args.output}")
    print(f"Stage context sizes: {STAGE_CONTEXT_SIZES}")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    all_records = []
    start_time = time.time()
    query_count = 0

    for idx in tqdm(range(args.start_idx, args.start_idx + n_queries), desc="Queries"):
        sample = dataset[idx]
        question = sample["question"]
        answer = sample["answer"]
        query_id = sample.get("id", str(idx))

        # Get 10 paragraphs (2 gold + 8 distractors)
        context_data = sample.get("context", {})
        paragraphs = []
        if "title" in context_data and "sentences" in context_data:
            titles = context_data["title"]
            sentences_list = context_data["sentences"]
            for i, (title, sents) in enumerate(zip(titles, sentences_list)):
                text = " ".join(sents) if isinstance(sents, list) else str(sents)
                paragraphs.append({"id": str(i), "title": title, "text": text})

        if len(paragraphs) < 2:
            continue

        # ── Cross-encoder ranking ──
        ce_pairs = [(question, p["text"]) for p in paragraphs]
        ce_scores = ce_model.predict(ce_pairs, show_progress_bar=False)
        ranked_indices = np.argsort(ce_scores)[::-1]

        # ── Per-stage collection ──
        query_records = []
        for stage_idx, k in enumerate(STAGE_CONTEXT_SIZES):
            # Get top-k passages
            top_k_indices = ranked_indices[:k]
            context_passages = [paragraphs[i] for i in top_k_indices]

            # Build prompt with context
            context_text = "\n\n".join(
                f"Passage {j+1} (Title: {p['title']}): {p['text']}"
                for j, p in enumerate(context_passages)
            )

            # Apply chat template
            messages = [
                {"role": "user", "content": f"Based on the following passages, answer the question.\n\n{context_text}\n\nQuestion: {question}\nAnswer:"}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # ── Forward pass for hidden states ──
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                              max_length=CONTEXT_MAX_LENGTH).to(device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            # Extract hidden states (multi_layer)
            hidden_state = extract_hidden_state(outputs, tokenizer, model,
                                                 inputs, rep_type="multi_layer")
            # Also get final_token for comparison
            hidden_state_ft = extract_hidden_state(outputs, tokenizer, model,
                                                    inputs, rep_type="final_token")
            # And mean_pool
            hidden_state_mp = extract_hidden_state(outputs, tokenizer, model,
                                                    inputs, rep_type="mean_pool")

            # Generate answer
            raw_answer = generate_answer(model, tokenizer, inputs,
                                         max_new_tokens=MAX_NEW_TOKENS)

            # Check correctness
            stage_correct = check_correctness(raw_answer, answer)
            final_correct = stage_correct  # For stage-aligned labels

            record = {
                "query_id": query_id,
                "question": question,
                "stage_idx": stage_idx,
                "stage_name": STAGES[stage_idx],
                "hidden_state": hidden_state["multi_layer"] if isinstance(hidden_state, dict) else hidden_state,
                "hidden_dim": len(hidden_state["multi_layer"]) if isinstance(hidden_state, dict) and "multi_layer" in hidden_state else len(hidden_state) if isinstance(hidden_state, list) else 14336,
                "final_token_hidden_state": hidden_state_ft["final_token"] if isinstance(hidden_state_ft, dict) else hidden_state_ft,
                "mean_pool_hidden_state": hidden_state_mp["mean_pool"] if isinstance(hidden_state_mp, dict) else hidden_state_mp,
                "multi_layer_hidden_states": hidden_state.get("multi_layer", []) if isinstance(hidden_state, dict) else [],
                "multi_layer_concat": hidden_state.get("multi_layer", []) if isinstance(hidden_state, dict) else hidden_state,
                "stage_answer": raw_answer.strip(),
                "stage_correctness": stage_correct,
                "final_correctness": final_correct,
                "k_passages": k,
            }
            query_records.append(record)

        all_records.extend(query_records)
        query_count += 1

        # Periodic save every 100 queries
        if query_count % 100 == 0:
            elapsed = time.time() - start_time
            rate = query_count / elapsed * 3600
            print(f"\n  [{query_count}/{n_queries}] {elapsed/60:.0f} min, {rate:.0f} q/hr")
            # Save checkpoint
            ckpt_path = args.output.replace(".jsonl", f"_ckpt_{query_count}.jsonl")
            _save_jsonl(all_records, ckpt_path)
            print(f"  Checkpoint: {ckpt_path}")

    # ── Final save ──
    _save_jsonl(all_records, args.output)

    total_time = time.time() - start_time
    print(f"\nCollection complete!")
    print(f"  Queries: {query_count}")
    print(f"  Records: {len(all_records)}")
    print(f"  Time: {total_time/3600:.1f} hours")
    print(f"  Rate: {query_count / total_time * 3600:.0f} q/hr")
    print(f"  Output: {args.output}")
    print(f"  Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
