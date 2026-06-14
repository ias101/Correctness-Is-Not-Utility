"""
TriviaQA open-domain collection with BM25 Wikipedia retrieval.

Uses the existing BM25 Wikipedia index (2M passages: data/wiki_corpus/passages.jsonl) for retrieval,
then cross-encoder reranking + progressive context scaling (2→4→6→8).
Collects hidden states with 4-layer hooks + mean pooling.

Usage:
  python collect_triviaqa_open.py --num_queries 500 \
      --model Qwen/Qwen2.5-7B-Instruct \
      --output data/collected_triviaqa_open.jsonl
"""

import argparse
import json
import os
import pickle
import sys
import time
from typing import Dict, List

import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from experiments.config import STAGES, NUM_STAGES, BASE_SEED
    from experiments.collect_states import (
        set_seed, load_llama_with_hidden_states,
        generate_answer, check_correctness, _save_jsonl, load_collected_data,
    )
except ImportError:
    from config import STAGES, NUM_STAGES, BASE_SEED
    from collect_states import (
        set_seed, load_llama_with_hidden_states,
        generate_answer, check_correctness, _save_jsonl, load_collected_data,
    )

STAGE_CONTEXT_SIZES = [2, 4, 6, 8]
WIKI_INDEX_PATH = "data/wiki_corpus/bm25_index.pkl"
WIKI_PASSAGES_PATH = "data/wiki_corpus/passages.jsonl"

# ── BM25 Wikipedia retrieval ------------------------------------------------

_wiki_passages = None
_bm25_index = None

def _load_wiki():
    global _wiki_passages, _bm25_index
    if _wiki_passages is not None:
        return _wiki_passages, _bm25_index

    print(f"[*] Loading Wikipedia passages from {WIKI_PASSAGES_PATH}...")
    _wiki_passages = []
    with open(WIKI_PASSAGES_PATH) as f:
        for line in f:
            _wiki_passages.append(json.loads(line))
    print(f"[*] Loaded {len(_wiki_passages)} passages")

    if os.path.exists(WIKI_INDEX_PATH):
        print(f"[*] Loading BM25 index from {WIKI_INDEX_PATH}...")
        with open(WIKI_INDEX_PATH, 'rb') as f:
            loaded = pickle.load(f)
        if isinstance(loaded, dict) and 'bm25' in loaded:
            _bm25_index = loaded['bm25']  # Extract BM25Okapi from dict wrapper
            print(f"[*] BM25 index loaded (from dict wrapper)")
        else:
            _bm25_index = loaded
            print(f"[*] BM25 index loaded")
    else:
        print("[!] No BM25 index found, building from passages...")
        from rank_bm25 import BM25Okapi
        import re
        def _tok(text):
            return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
        docs = [_tok(p["text"]) for p in _wiki_passages]
        _bm25_index = BM25Okapi(docs)

    return _wiki_passages, _bm25_index


def retrieve_wiki(query: str, top_k: int = 10) -> List[Dict]:
    """BM25 retrieve passages from Wikipedia corpus."""
    passages, bm25 = _load_wiki()
    import re
    q_tokens = re.sub(r"[^a-z0-9\s]", " ", query.lower()).split()
    scores = bm25.get_scores(q_tokens)
    order = sorted(range(len(passages)), key=lambda i: scores[i], reverse=True)
    results = []
    for i in order[:top_k]:
        p = passages[i]
        results.append({
            "id": i, "title": p.get("title", ""),
            "text": p["text"], "score": float(scores[i]),
        })
    return results


# ── Cross-encoder ------------------------------------------------------------

_CE_MODEL = None

def _get_cross_encoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    global _CE_MODEL
    if _CE_MODEL is None:
        from sentence_transformers import CrossEncoder
        print(f"[*] Loading cross-encoder: {model_name}...")
        _CE_MODEL = CrossEncoder(model_name)
    return _CE_MODEL


def rank_cross_encoder(query, paragraphs, top_k):
    ce = _get_cross_encoder()
    pairs = [(query, p["text"][:1200]) for p in paragraphs]
    scores = ce.predict(pairs, show_progress_bar=False)
    order = sorted(range(len(paragraphs)), key=lambda i: scores[i], reverse=True)
    return [dict(paragraphs[i], score=float(scores[i])) for i in order[:top_k]]


# ── Prompt building ----------------------------------------------------------

_PARA_CHAR_CAP = 1200

def _ctx(passages, n):
    parts = []
    for i, p in enumerate(passages[:n]):
        parts.append(f"[{i+1}] {p.get('title', '')}: {p['text'][:_PARA_CHAR_CAP]}")
    return "\n\n".join(parts)


def build_stage_prompt(question, ce_ranked, stage):
    k = STAGE_CONTEXT_SIZES[stage]
    ctx = _ctx(ce_ranked, k)
    head = f"Context ({k} retrieved passages):"
    return (
        f"{head}\n{ctx}\n\n"
        f"Question: {question}\n\n"
        f"Answer with only a short phrase or entity, based on the context."
    )


# ── Dataset ------------------------------------------------------------------

def load_triviaqa(num_queries, split="validation"):
    print(f"[*] Loading TriviaQA ({split}[:{num_queries}])...")
    # Try unsplit first, fall back to validation
    try:
        ds = load_dataset("trivia_qa", "rc.nocontext", split=f"{split}[:{num_queries}]")
    except Exception:
        ds = load_dataset("trivia_qa", "rc", split=f"{split}[:{num_queries}]")

    queries = []
    for i, ex in enumerate(ds):
        answers = ex.get("answer", {})
        aliases = []
        if isinstance(answers, dict):
            aliases = answers.get("aliases", []) or answers.get("normalized_aliases", [])
            if not aliases:
                val = answers.get("value", "") or answers.get("normalized_value", "")
                if val:
                    aliases = [val]

        queries.append({
            "id": str(i),
            "question": ex["question"],
            "answers": aliases if aliases else [ex.get("answer", "")],
        })
    print(f"[*] Loaded {len(queries)} TriviaQA queries")
    return queries


# ── Collection ---------------------------------------------------------------

def collect(queries, model, tokenizer, store, output_path, model_name):
    collected = []
    n_final_correct = 0
    n_stage_correct = [0] * NUM_STAGES
    t0 = time.time()
    short_name = model_name.split("/")[-1][:20]

    for q_idx, query in enumerate(tqdm(queries, desc=f"TriviaQA/{short_name}")):
        question = query["question"]
        answers = query["answers"]

        # BM25 retrieval from Wikipedia
        bm25_results = retrieve_wiki(question, top_k=20)

        # Cross-encoder rerank to top-10
        ce_ranked = rank_cross_encoder(question, bm25_results, top_k=10)

        stage_tuples = []
        final_answer = None
        gen_entropy = gen_max_prob = None

        for stage_idx, stage_name in enumerate(STAGES):
            prompt = build_stage_prompt(question, ce_ranked, stage_idx)
            hs_repr = extract_hidden_state_multimodel(model, tokenizer, prompt, store)

            hs = hs_repr["final_token"]
            mean_pool_hs = hs_repr.get("mean_pool", hs)
            multi_layer_hs = hs_repr.get("multi_layer", [hs.tolist()])

            is_final = (stage_idx == NUM_STAGES - 1)
            max_tokens = 64 if is_final else 32
            stage_answer, ent, mp = generate_answer(
                model, tokenizer, prompt, max_new_tokens=max_tokens
            )
            stage_correct = check_correctness(stage_answer, answers) if stage_answer else False
            n_stage_correct[stage_idx] += int(stage_correct)

            if is_final:
                final_answer = stage_answer
                gen_entropy, gen_max_prob = ent, mp

            stage_tuples.append({
                "query_id": query["id"], "question": question,
                "stage_idx": stage_idx, "stage_name": stage_name,
                "hidden_state": hs.tolist() if hasattr(hs, 'tolist') else hs,
                "hidden_dim": hs.shape[0] if hasattr(hs, 'shape') else len(hs),
                "stage_answer": stage_answer,
                "stage_correctness": int(stage_correct),
                "mean_pool_hidden_state": mean_pool_hs.tolist() if hasattr(mean_pool_hs, 'tolist') else mean_pool_hs,
                "multi_layer_hidden_states": multi_layer_hs,
                "model_name": model_name,
                "dataset": "triviaqa",
            })

        final_correct = check_correctness(final_answer, answers) if final_answer else False
        n_final_correct += int(final_correct)
        for t in stage_tuples:
            t["final_correctness"] = int(final_correct)
            t["generated_answer"] = final_answer or ""
            t["generation_entropy"] = gen_entropy
            t["generation_max_prob"] = gen_max_prob
        collected.extend(stage_tuples)

        if (q_idx + 1) % 50 == 0:
            _save_jsonl(collected, output_path)
            elapsed = time.time() - t0
            n = q_idx + 1
            print(f"\n  [{n}/{len(queries)}] saved | acc={100*n_final_correct/n:.1f}% | "
                  f"{n/max(elapsed,1):.2f} q/s")

    _save_jsonl(collected, output_path)
    elapsed = time.time() - t0
    n = len(queries)
    print(f"\n[*] TriviaQA/{short_name}: {n} queries, {len(collected)} tuples in {elapsed:.0f}s")
    print(f"[*] Accuracy: {n_final_correct}/{n} = {100*n_final_correct/n:.1f}%")
    for s in range(NUM_STAGES):
        print(f"  S{s}(k={STAGE_CONTEXT_SIZES[s]}): {100*n_stage_correct[s]/n:.1f}%")
    return output_path


def extract_hidden_state_multimodel(model, tokenizer, prompt, store, max_length=2048):
    """Extract hidden states compatible with any HF causal LM."""
    try:
        from experiments.collect_states import build_chat_prompt
    except ImportError:
        from collect_states import build_chat_prompt
    chat_prompt = build_chat_prompt(prompt, tokenizer)
    inputs = tokenizer(chat_prompt, return_tensors="pt", truncation=True,
                       max_length=max_length).to(model.device)

    store.clear()
    with torch.no_grad():
        _ = model(**inputs, output_hidden_states=False)

    stored_layers = sorted(
        [int(k) for k in store.keys() if not k.endswith("_mean")], reverse=True)

    result = {}
    last_layer = str(stored_layers[0]) if stored_layers else "-1"

    if last_layer in store:
        result["final_token"] = store[last_layer][0, :]
    else:
        hidden_dim = model.config.hidden_size if hasattr(model, "config") else 3584
        result["final_token"] = torch.zeros(hidden_dim)

    mean_key = f"{last_layer}_mean"
    result["mean_pool"] = store[mean_key][0, :] if mean_key in store else result["final_token"].clone()

    multi_layer = []
    for layer_idx in stored_layers[:4]:
        if str(layer_idx) in store:
            multi_layer.append(store[str(layer_idx)][0, :].tolist())
    result["multi_layer"] = multi_layer if multi_layer else [result["final_token"].tolist()]

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--num_queries", type=int, default=500)
    ap.add_argument("--output", type=str, default="data/collected_triviaqa_open.jsonl")
    ap.add_argument("--seed", type=int, default=BASE_SEED)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    _get_cross_encoder()
    print(f"[*] Model: {args.model}")
    print(f"[*] Dataset: TriviaQA (open-domain, BM25 Wikipedia retrieval)")

    queries = load_triviaqa(args.num_queries)
    model, tokenizer, hook_handles, store = load_llama_with_hidden_states(
        model_name=args.model, use_4bit=True,
    )
    try:
        collect(queries, model, tokenizer, store, args.output, args.model)
        data = load_collected_data(args.output)
        n_correct = sum(1 for d in data if d["final_correctness"] == 1)
        print(f"\n[*] Stats: {len(set(d['query_id'] for d in data))} queries, "
              f"{len(data)} tuples, accuracy {100*n_correct/len(data):.1f}%")
    finally:
        for h in hook_handles:
            h.remove()
        print("[*] Cleaned up hooks.")


if __name__ == "__main__":
    main()
