"""
Hidden state collection from RAG pipeline stages.

ARCHITECTURE CLARIFICATION — How LLM hidden states map to RAG stages:

The RAG pipeline has 4 stages: retrieval → reranking → context_assembly → generation.
However, BM25 and cross-encoder rerankers do NOT produce LLM hidden states.

Our approach: At EACH stage, we build a stage-specific prompt and run it through
the LLM to extract the last-token hidden state. This is NOT the same as the LLM
answering the question at each stage — it's a prefill-based probe that captures
the "state of knowledge" at that stage's prompt.

Specifically:
  Stage 0 (retrieval):  Prompt = raw retrieved passages → get h_0
  Stage 1 (reranking):  Prompt = reranked top passages → get h_1
  Stage 2 (assembly):   Prompt = formatted context → get h_2
  Stage 3 (generation): Prompt = final answer prompt → generate answer + get h_3

Cost model note: Stages 0-2 involve ONLY a single LLM forward pass (prefill),
which is ~1/64 the cost of generation (@64 tokens). Stage 3 includes both
prefill + decode. Therefore:
  - Stage costs [S1, S2, S3] are dominated by prefill (~equal, cheap)
  - S4 (generation) is ~64× more expensive due to autoregressive decode
  - Total overhead of 3 extra prefills ≈ 3/64 ≈ 4.7% — acceptable

Alternative considered: Using retrieval/reranker scores directly as MLP input.
This was rejected because it introduces retrieval-specific features, which
we explicitly avoid for architecture-agnostic generality (Claim 5).

For each query in the dataset, this script:
1. Retrieves passages via BM25 (Stage 1)
2. Reranks passages (Stage 2)
3. Assembles context (Stage 3)
4. Generates answer via LLM, collects hidden states at each stage (Stage 4)

At each stage, we record:
  - h_t: LLM hidden state (last token, final layer) after encoding the prompt at that stage
  - stage: integer stage index [0, 1, 2, 3]
  - final_correctness: whether the generated answer matches ground truth
  - generation_entropy: softmax entropy of first generated token (for confidence baseline)
  - generation_max_prob: max softmax prob of first generated token (for confidence baseline)
  - query_id: for traceability

Output: JSONL file with one line per (query, stage, correctness) tuple.
"""

import argparse
import json
import os
import time
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from tqdm import tqdm

# Add script directory to path (works both locally and on server)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    STAGES,
    NUM_STAGES,
    LLAMA_MODEL_NAME,
    LLAMA_MODEL_NAME_SMALL,
    LLAMA_HIDDEN_DIM,
    DPR_QUESTION_ENCODER,
    DPR_CONTEXT_ENCODER,
    DPR_TOP_K,
    DPR_RERANK_K,
    CONTEXT_MAX_LENGTH,
    DATA_DIR,
    PILOT_NQ_QUERIES,
    USE_4BIT,
    BASE_SEED,
)


def set_seed(seed: int = BASE_SEED):
    """Fix random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_nq_dataset(num_queries: int = PILOT_NQ_QUERIES, split: str = "train"):
    """
    Load Natural Questions dataset.

    Using the 'dev' split from nq_open (open-domain variant) for pilot.
    For full experiments, use 'train' split.

    Returns:
        List of dicts with 'id', 'question', 'answer' keys.
    """
    print(f"[*] Loading NQ dataset ({num_queries} queries, split={split})...")
    try:
        dataset = load_dataset("nq_open", split=split, trust_remote_code=True)
    except Exception:
        # Fallback to a simpler dataset
        print("[!] nq_open not available, trying trivia_qa...")
        dataset = load_dataset(
            "trivia_qa", "rc.nocontext", split="validation", trust_remote_code=True
        )

    # Select subset
    dataset = dataset.select(range(min(num_queries, len(dataset))))

    queries = []
    for i, item in enumerate(dataset):
        # Handle different dataset formats
        if "question" in item:
            question = item["question"]
        elif "query" in item:
            question = item["query"]
        else:
            continue

        # Get ground truth answer(s)
        if "answer" in item:
            answers = item["answer"]
            if isinstance(answers, list):
                answers = answers
            else:
                answers = [answers]
        elif "answers" in item:
            answers = item["answers"]
            if isinstance(answers, list):
                answers = answers
            else:
                answers = [str(answers)]
        else:
            answers = ["<no_answer>"]

        queries.append(
            {
                "id": str(i),
                "question": question,
                "answers": [a for a in answers if a],
                "answer_texts": [],  # filled after generation
            }
        )

    print(f"[*] Loaded {len(queries)} queries.")
    return queries


def load_llama_with_hidden_states(
    model_name: str = LLAMA_MODEL_NAME,
    use_4bit: bool = USE_4BIT,
    device_map: str = "auto",
):
    """
    Load LLaMA model with hooks to capture hidden states.

    Uses 4-bit quantization to fit on RTX 3080 (16GB).

    Returns:
        model, tokenizer, hook_handle (call .remove() to clean up)
    """
    print(f"[*] Loading LLaMA model: {model_name}...")

    # Try smaller model first if full model fails
    quantization_config = None
    if use_4bit:
        try:
            import bitsandbytes  # noqa: F401

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        except ImportError:
            print("[!] bitsandbytes not available, using bfloat16 (may OOM on 16GB)")
            use_4bit = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch.bfloat16 if not use_4bit else None,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"[!] Failed to load {model_name}: {e}")
        print(f"[*] Trying smaller model: {LLAMA_MODEL_NAME_SMALL}")
        model_name = LLAMA_MODEL_NAME_SMALL
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch.bfloat16 if not use_4bit else None,
            trust_remote_code=True,
        )

    model.eval()

    # Storage for captured hidden states
    hidden_states_store = {}

    def get_hidden_state_hook(layer_idx: int):
        """Hook to capture hidden states from a transformer layer.

        Stores:
          - str(layer_idx): last-token hidden state (B, hidden_dim)
          - f"{layer_idx}_mean": mean-pooled hidden state over all tokens (B, hidden_dim)
        """

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hs = output[0]  # (B, seq_len, hidden_dim)
            else:
                hs = output
            # Last token hidden state (for final_token representation)
            hidden_states_store[str(layer_idx)] = hs[:, -1, :].detach().cpu()
            # Mean-pooled over all tokens (for mean_pool representation)
            hidden_states_store[f"{layer_idx}_mean"] = hs.mean(dim=1).detach().cpu()

        return hook_fn

    # Register hooks on the last 4 transformer layers for richer representations
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        all_layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        all_layers = model.transformer.h
    else:
        # Generic: find the last nn.ModuleList
        all_layers = None
        for name, module in model.named_modules():
            if "layers" in name.lower() or "h" in name.lower():
                if hasattr(module, "__len__") and len(module) > 0:
                    all_layers = module
                    break
        if all_layers is None:
            raise ValueError("Cannot find transformer layers in model")

    # Register hooks on last 4 layers (or all if fewer)
    n_layers = len(all_layers)
    hook_layers = min(4, n_layers)
    hook_handles = []
    for offset in range(hook_layers):
        layer_idx = n_layers - 1 - offset  # -1 (last), -2, -3, -4
        hook_handles.append(
            all_layers[layer_idx].register_forward_hook(
                get_hidden_state_hook(layer_idx)
            )
        )
    print(f"[*] Registered hooks on {hook_layers} layers "
          f"(indices: {[n_layers-1-i for i in range(hook_layers)]})")

    return model, tokenizer, hook_handles, hidden_states_store


def build_chat_prompt(user_content: str, tokenizer) -> str:
    """Wrap user content in Qwen2.5-Instruct chat template."""
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Answer questions accurately and concisely based on the provided context."},
        {"role": "user", "content": user_content},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return f"<|im_start|>system\nYou are a helpful AI assistant. Answer questions accurately and concisely based on the provided context.<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"

def extract_hidden_state(
    model,
    tokenizer,
    prompt: str,
    hidden_states_store: dict,
    max_length: int = 2048,
) -> dict:
    """
    Run a prompt through LLaMA and extract hidden state representations.

    Returns dict with keys:
        - "final_token": last-token hidden state from last layer (original)
        - "mean_pool": mean-pooled hidden state over all tokens (last layer)
        - "multi_layer": list of last-token hidden states from last 4 layers
    """
    # Wrap in Qwen2.5 chat template
    chat_prompt = build_chat_prompt(prompt, tokenizer)
    inputs = tokenizer(
        chat_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(model.device)

    # Clear previous stored states
    hidden_states_store.clear()

    with torch.no_grad():
        _ = model(**inputs, output_hidden_states=False)

    # Determine number of layers for multi-layer extraction
    stored_layers = sorted(
        [int(k) for k in hidden_states_store.keys() if not k.endswith("_mean")],
        reverse=True
    )

    # Build result dict
    result = {}

    # 1. Final token from last layer (original behavior)
    last_layer = str(stored_layers[0]) if stored_layers else "-1"
    if last_layer in hidden_states_store:
        result["final_token"] = hidden_states_store[last_layer][0, :]  # (hidden_dim,)
    else:
        # Fallback
        if hasattr(model, "config"):
            hidden_dim = model.config.hidden_size
        else:
            hidden_dim = LLAMA_HIDDEN_DIM
        result["final_token"] = torch.zeros(hidden_dim)

    # 2. Mean-pooled from last layer
    mean_key = f"{last_layer}_mean"
    if mean_key in hidden_states_store:
        result["mean_pool"] = hidden_states_store[mean_key][0, :]  # (hidden_dim,)
    else:
        result["mean_pool"] = result["final_token"].clone()

    # 3. Multi-layer: last-token from last 4 layers
    multi_layer = []
    for layer_idx in stored_layers[:4]:  # up to 4 layers
        if str(layer_idx) in hidden_states_store:
            multi_layer.append(
                hidden_states_store[str(layer_idx)][0, :].tolist()
            )
    result["multi_layer"] = multi_layer if multi_layer else [result["final_token"].tolist()]
    result["multi_layer_indices"] = stored_layers[:4]

    # Backward-compatible return: also set the original "-1" key
    if "-1" not in hidden_states_store and last_layer in hidden_states_store:
        hidden_states_store["-1"] = hidden_states_store[last_layer]

    return result


def generate_answer(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
) -> Tuple[str, Optional[float], Optional[float]]:
    """
    Generate an answer from the LLM.

    Returns:
        (answer_text, entropy, max_prob) — entropy and max_prob are computed
        from the first generated token's logits (for confidence baseline).
    """
    # Wrap in Qwen2.5 chat template
    chat_prompt = build_chat_prompt(prompt, tokenizer)
    inputs = tokenizer(
        chat_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=CONTEXT_MAX_LENGTH,
    ).to(model.device)

    entropy = None
    max_prob = None

    with torch.no_grad():
        # Generate with output_scores to get logits
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for reproducibility
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

    # Extract first-token logits for confidence baseline
    if hasattr(outputs, "scores") and outputs.scores:
        first_token_logits = outputs.scores[0][0]  # (vocab_size,) — logits for first generated token
        first_token_probs = torch.softmax(first_token_logits, dim=-1)

        # Softmax entropy: H = -Σ p_i log(p_i)
        entropy_val = -(first_token_probs * torch.log(first_token_probs + 1e-12)).sum().item()
        # Max probability
        max_prob_val = first_token_probs.max().item()

        entropy = entropy_val
        max_prob = max_prob_val

    # Extract generated tokens
    if hasattr(outputs, "sequences"):
        generated = outputs.sequences[0, inputs.input_ids.shape[1]:]
    else:
        generated = outputs[0, inputs.input_ids.shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True)
    return answer.strip(), entropy, max_prob


def normalize_answer(s: str) -> str:
    """Normalize answer string for comparison (standard QA evaluation)."""
    import re
    import string
    s = s.lower().strip()
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Remove punctuation
    s = s.translate(str.maketrans('', '', string.punctuation))
    # Collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def check_correctness(predicted: str, ground_truths: List[str]) -> bool:
    """
    Check if the generated answer matches any ground truth answer.

    Uses standard QA evaluation: normalized exact match + token F1 ≥ 0.5
    as a fallback for long-form answers. Compares against dataset ground truth
    labels (NOT model-generated pseudo-labels).
    """
    if not ground_truths or all(not gt for gt in ground_truths):
        return False

    pred_norm = normalize_answer(predicted)

    for gt in ground_truths:
        if not gt:
            continue
        gt_norm = normalize_answer(gt)

        # Normalized exact match
        if pred_norm == gt_norm:
            return True

        # Substring containment (for long-form answers)
        if len(gt_norm) > 3 and (gt_norm in pred_norm or pred_norm in gt_norm):
            return True

        # Token F1 ≥ 0.5 fallback for fuzzy matching
        gt_tokens = set(gt_norm.split())
        pred_tokens = set(pred_norm.split())
        if gt_tokens and pred_tokens:
            overlap = gt_tokens & pred_tokens
            precision = len(overlap) / len(pred_tokens) if pred_tokens else 0
            recall = len(overlap) / len(gt_tokens) if gt_tokens else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            if f1 >= 0.5:
                return True

    return False


def build_stage_prompt(
    question: str,
    retrieved: list,
    reranked: list,
    stage: int,
    stage_name: str = "",
    pipeline=None,
) -> str:
    """
    Build a prompt for a specific RAG stage.

    Uses the RAGPipeline's prompt builder if available, otherwise
    uses simple formatting with passage text.
    """
    if pipeline is not None:
        return pipeline.build_stage_prompt(question, retrieved, reranked, stage)

    # Fallback: simple formatting with passage text
    if stage == 0:
        context = "\n\n".join(
            f"Passage {i+1}: {p['text'][:300]}" if isinstance(p, dict) else f"Passage {i+1}: {str(p)[:300]}"
            for i, p in enumerate(retrieved[:10])
        )
        return f"Question: {question}\n\nRetrieved passages:\n{context}\n\nBased on the retrieved information, answer the question."
    elif stage == 1:
        context = "\n\n".join(
            f"Passage {i+1}: {p['text'][:200]}" if isinstance(p, dict) else f"Passage {i+1}: {str(p)[:200]}"
            for i, p in enumerate(reranked[:5])
        )
        return f"Question: {question}\n\nTop relevant passages:\n{context}\n\nBased on these passages, answer the question."
    elif stage == 2:
        context = " ".join(p["text"] if isinstance(p, dict) else str(p) for p in reranked[:5])
        return f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    elif stage == 3:
        context = " ".join(p["text"] if isinstance(p, dict) else str(p) for p in reranked[:5])
        return f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    else:
        return f"Question: {question}\n\nAnswer:"


def collect_hidden_states(
    queries: List[Dict],
    model,
    tokenizer,
    hidden_states_store: dict,
    output_path: str,
    max_queries: Optional[int] = None,
    retrieval_pipeline=None,
) -> str:
    """
    Main collection loop.

    For each query, runs all 4 RAG stages, collecting:
    - Hidden state at each stage
    - Final answer correctness

    If retrieval_pipeline is provided, uses real BM25 retrieval.
    Otherwise falls back to simulated placeholder passages.

    Saves results as JSONL.
    """
    if max_queries:
        queries = queries[:max_queries]

    print(f"\n[*] Collecting hidden states for {len(queries)} queries...")
    print(f"[*] Output: {output_path}")

    collected = []
    start_time = time.time()

    for q_idx, query in enumerate(tqdm(queries, desc="Collecting")):
        question = query["question"]
        answers = query.get("answers", query.get("answer_texts", []))
        query_id = query["id"]

        # ── Real retrieval (or simulated) ──
        retrieved, reranked = _retrieve_passages(question, retrieval_pipeline)

        # ── Process each stage ──
        stage_tuples = []
        final_answer = None

        for stage_idx, stage_name in enumerate(STAGES):
            # Build stage-specific prompt using real or simulated passages
            prompt = build_stage_prompt(
                question, retrieved, reranked, stage_idx, stage_name,
                pipeline=retrieval_pipeline,
            )

            # Extract hidden state
            hs = extract_hidden_state(
                model, tokenizer, prompt, hidden_states_store
            )

            # Generate answer at EVERY stage (short for S0-S2, full for S3)
            is_final = (stage_idx == NUM_STAGES - 1)
            max_tokens = 128 if is_final else 48  # meaningful short answers for early stages
            stage_answer, gen_entropy, gen_max_prob = generate_answer(
                model, tokenizer, prompt, max_new_tokens=max_tokens
            )

            # Check per-stage correctness (against ground truth answers)
            if stage_answer and answers:
                stage_correct = check_correctness(stage_answer, answers)
            else:
                stage_correct = False

            if is_final:
                final_answer = stage_answer
                query["generation_entropy"] = gen_entropy
                query["generation_max_prob"] = gen_max_prob

            # Store tuple with per-stage info
            stage_tuples.append(
                {
                    "query_id": query_id,
                    "question": question,
                    "stage_idx": stage_idx,
                    "stage_name": stage_name,
                    "hidden_state": hs["final_token"].tolist() if isinstance(hs, dict) else hs.tolist(),
                    "hidden_dim": hs["final_token"].shape[0] if isinstance(hs, dict) else hs.shape[0],
                    "stage_answer": stage_answer,
                    "stage_correctness": int(stage_correct),
                }
            )

        # ── Determine final correctness (from stage 3 only) ──
        if final_answer and answers:
            final_correct = check_correctness(final_answer, answers)
        else:
            final_correct = False

        # ── Finalize tuples ──
        for t in stage_tuples:
            t["final_correctness"] = int(final_correct)
            t["generated_answer"] = final_answer or ""
            t["generation_entropy"] = query.get("generation_entropy")
            t["generation_max_prob"] = query.get("generation_max_prob")
            t["generation_max_prob"] = query.get("generation_max_prob")

        collected.extend(stage_tuples)

        # Periodic save every 50 queries
        if (q_idx + 1) % 50 == 0:
            _save_jsonl(collected, output_path)
            elapsed = time.time() - start_time
            rate = (q_idx + 1) / max(elapsed, 1)
            eta = (len(queries) - q_idx - 1) / max(rate, 0.01)
            print(
                f"\n  [{q_idx+1}/{len(queries)}] Saved. "
                f"Rate: {rate:.1f} q/min, ETA: {eta:.0f}s"
            )

    # Final save
    _save_jsonl(collected, output_path)

    elapsed = time.time() - start_time
    print(f"\n[*] Collection complete: {len(collected)} tuples in {elapsed:.0f}s")
    print(f"[*] Saved to: {output_path}")

    return output_path


def _retrieve_passages(question: str, pipeline=None, num_passages: int = 20) -> tuple:
    """
    Retrieve passages for a question.

    REAL RETRIEVAL ONLY — simulated passages have been removed.
    Requires a RAGPipeline with BM25 or DPR index.

    Returns (retrieved_list, reranked_list) where each is a list of dicts
    with keys: id, title, text, score.
    """
    if pipeline is None:
        raise RuntimeError(
            "Retrieval pipeline is required. "
            "Simulated placeholder passages have been removed. "
            "Set up retrieval with: `from retrieval import setup_retrieval; pipeline = setup_retrieval()`"
        )

    retrieved = pipeline.retrieve(question)

    # If BM25 returned nothing useful, at minimum return empty context
    if not retrieved:
        retrieved = [{"id": 0, "title": "empty", "text": "No relevant passages found.", "score": 0.0}]

    reranked = pipeline.rerank(question, retrieved)
    return retrieved, reranked


def _save_jsonl(data: List[Dict], path: str):
    """Save list of dicts as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_collected_data(path: str) -> List[Dict]:
    """Load collected tuples from JSONL."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Collect hidden states from RAG pipeline for failure prediction."
    )
    parser.add_argument(
        "--num_queries", type=int, default=PILOT_NQ_QUERIES,
        help=f"Number of queries to process (default: {PILOT_NQ_QUERIES})"
    )
    parser.add_argument(
        "--model_name", type=str, default=LLAMA_MODEL_NAME,
        help="LLaMA model name or path"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSONL path (default: data/collected_states.jsonl)"
    )
    parser.add_argument(
        "--dataset", type=str, default="nq",
        choices=["nq", "hotpotqa", "triviaqa"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--no_4bit", action="store_true",
        help="Disable 4-bit quantization"
    )
    parser.add_argument(
        "--seed", type=int, default=BASE_SEED,
        help="Random seed"
    )
    parser.add_argument(
        "--real_retrieval", action="store_true", default=True,
        help="Use real BM25 Wikipedia retrieval (now DEFAULT — no simulated fallback)"
    )
    parser.add_argument(
        "--corpus_size", type=int, default=200000,
        help="Number of Wikipedia passages for BM25 index"
    )
    parser.add_argument(
        "--dpr", action="store_true", default=False,
        help="Use DPR-standard Wikipedia corpus (21M passages)",
    )
    parser.add_argument(
        "--dense_retrieval", action="store_true", default=False,
        help="Use Contriever dense retrieval instead of BM25",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    # Output path
    if args.output is None:
        args.output = os.path.join(DATA_DIR, "collected_states.jsonl")

    # Load dataset
    queries = load_nq_dataset(num_queries=args.num_queries)

    # Load retrieval pipeline (if requested)
    retrieval_pipeline = None
    if args.real_retrieval:
        if args.dpr:
            from dpr_retrieval import setup_dpr_retrieval
            retrieval_pipeline, _ = setup_dpr_retrieval()
        elif args.dense_retrieval:
            from dense_retrieval import setup_dense_retrieval
            retrieval_pipeline = setup_dense_retrieval(num_passages=args.corpus_size)
        else:
            from retrieval import setup_retrieval
            retrieval_pipeline = setup_retrieval(num_passages=args.corpus_size)

    # Load model
    model, tokenizer, hook_handle, hidden_states_store = load_llama_with_hidden_states(
        model_name=args.model_name,
        use_4bit=not args.no_4bit,
    )

    try:
        # Collect hidden states
        collect_hidden_states(
            queries=queries,
            model=model,
            tokenizer=tokenizer,
            hidden_states_store=hidden_states_store,
            output_path=args.output,
            max_queries=args.num_queries,
            retrieval_pipeline=retrieval_pipeline,
        )

        # Quick stats
        data = load_collected_data(args.output)
        n_correct = sum(1 for d in data if d["final_correctness"] == 1)
        n_total = len(set(d["query_id"] for d in data))
        print(f"\n[*] Stats:")
        print(f"    Total queries: {n_total}")
        print(f"    Total tuples: {len(data)}")
        print(f"    Correct: {n_correct}/{len(data)} "
              f"({100*n_correct/len(data):.1f}%)")
    finally:
        hook_handle.remove()
        print("[*] Cleaned up model hooks.")


if __name__ == "__main__":
    main()
