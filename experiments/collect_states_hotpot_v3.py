"""
Hidden-state collection on HotpotQA-distractor — V3 with progressive context narrowing.

WHY V3
------
V1/V2 showed all 4 stages use the same top-8 passages, creating zero stage diversity.
The MLP correctly learned "stop at stage 1, nothing changes later" — degenerating to fixed_S1.

V3 VARIES context size across stages, creating a REAL cost-accuracy tradeoff:
  - Stage 0 (retrieval):   Top-2 passages (fastest, lowest accuracy)
  - Stage 1 (reranking):   Top-4 passages (more context)
  - Stage 2 (assembly):    Top-6 passages (near-full context)
  - Stage 3 (generation):  Top-8 passages (best accuracy, most expensive)

The MLP must now learn: "Can this query be answered with 2 passages, or does it need 8?"
This detaches 'ours' from fixed_S1 and creates a true Pareto curve.

Reviewer fix (Round 3 → 4):
  Fix 1: Progressive context-window scaling ✅ (this file)
  Fix 3: Scale to 2000 queries ✅ (--num_queries 2000)
  Fix 4: Pareto frontier plot (built into evaluate.py)
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import STAGES, NUM_STAGES, LLAMA_MODEL_NAME, DATA_DIR, BASE_SEED
from collect_states import (
    set_seed, load_llama_with_hidden_states,
    extract_hidden_state, generate_answer,
    check_correctness, _save_jsonl, load_collected_data,
)

# -- Progressive context sizes per stage (THE key V3 innovation) -----------
# Stage 0: top-2, Stage 1: top-4, Stage 2: top-6, Stage 3: top-8
STAGE_CONTEXT_SIZES = [2, 4, 6, 8]
assert len(STAGE_CONTEXT_SIZES) == NUM_STAGES

# -- BM25 retriever --------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    import re
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def rank_bm25(query: str, paragraphs: List[Dict], top_k: int) -> List[Dict]:
    q_tokens = _tokenize(query)
    docs = [_tokenize(p["text"]) for p in paragraphs]
    try:
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi(docs)
        scores = bm25.get_scores(q_tokens)
    except Exception:
        qset = set(q_tokens)
        scores = [sum(1 for t in d if t in qset) for d in docs]
    order = sorted(range(len(paragraphs)), key=lambda i: scores[i], reverse=True)
    out = []
    for rank, i in enumerate(order[:top_k]):
        p = paragraphs[i]
        out.append({"id": p["id"], "title": p["title"], "text": p["text"],
                    "score": float(scores[i])})
    return out


# -- Cross-encoder reranker ------------------------------------------------

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
    out = []
    for rank, i in enumerate(order[:top_k]):
        p = paragraphs[i]
        out.append({"id": p["id"], "title": p["title"], "text": p["text"],
                    "score": float(scores[i])})
    return out


# -- Prompt building (V3: per-stage context size) --------------------------

_PARA_CHAR_CAP = 1200


def _ctx(passages, n):
    parts = []
    for i, p in enumerate(passages[:n]):
        parts.append(f"[{i+1}] {p['title']}: {p['text'][:_PARA_CHAR_CAP]}")
    return "\n\n".join(parts)


def build_stage_prompt(question, ce_ranked, stage):
    """Build prompt with progressive context size per stage."""
    k = STAGE_CONTEXT_SIZES[stage]
    ctx = _ctx(ce_ranked, k)

    if stage == 0:
        head = f"Below are the {k} most relevant passages (some may be incomplete):"
    elif stage == 1:
        head = f"Below are the {k} most relevant passages:"
    elif stage == 2:
        head = f"Context ({k} passages):"
    else:
        head = f"Context ({k} passages):"

    return (
        f"{head}\n{ctx}\n\n"
        f"Question: {question}\n\n"
        f"Answer with only a short phrase or entity (or 'yes'/'no'), based on the context."
    )


# -- Dataset --------------------------------------------------------------

def load_hotpotqa(num_queries, split="validation"):
    print(f"[*] Loading HotpotQA distractor ({split}[:{num_queries}])...")
    ds = load_dataset("hotpot_qa", "distractor", split=f"{split}[:{num_queries}]")
    queries = []
    for i, ex in enumerate(ds):
        titles = ex["context"]["title"]
        sents = ex["context"]["sentences"]
        paragraphs = [
            {"id": j, "title": t, "text": "".join(ss)}
            for j, (t, ss) in enumerate(zip(titles, sents))
        ]
        queries.append({
            "id": str(i),
            "question": ex["question"],
            "answers": [ex["answer"]],
            "paragraphs": paragraphs,
            "gold_titles": list(ex["supporting_facts"]["title"]),
        })
    print(f"[*] Loaded {len(queries)} HotpotQA queries (10 paras each).")
    return queries


# -- Collection loop ------------------------------------------------------

def collect(queries, model, tokenizer, store, output_path):
    collected = []
    n_final_correct = 0
    # Track both-gold@k for each stage's context size
    n_both_gold = [0] * NUM_STAGES
    n_stage_correct = [0] * NUM_STAGES  # per-stage correctness
    t0 = time.time()

    gold_sets = [set(q["gold_titles"]) for q in queries]

    for q_idx, query in enumerate(tqdm(queries, desc="Collecting V3")):
        question = query["question"]
        answers = query["answers"]
        paragraphs = query["paragraphs"]
        gold = gold_sets[q_idx]

        # Cross-encoder ranks all 10 passages once (reused for all stages)
        ce_ranked = rank_cross_encoder(question, paragraphs, top_k=10)

        # Track both-gold@k for each stage's context size
        for s, k in enumerate(STAGE_CONTEXT_SIZES):
            topk_titles = {p["title"] for p in ce_ranked[:k]}
            if gold.issubset(topk_titles):
                n_both_gold[s] += 1

        stage_tuples = []
        final_answer = None
        gen_entropy = gen_max_prob = None

        for stage_idx, stage_name in enumerate(STAGES):
            k = STAGE_CONTEXT_SIZES[stage_idx]
            prompt = build_stage_prompt(question, ce_ranked, stage_idx)
            hs_repr = extract_hidden_state(model, tokenizer, prompt, store)

            # Primary hidden state (final token, last layer — original representation)
            hs = hs_repr["final_token"]

            # Richer representations for Delta Probe strengthening (reviewer request)
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
                "query_id": query["id"],
                "question": question,
                "stage_idx": stage_idx,
                "stage_name": stage_name,
                "hidden_state": hs.tolist(),
                "hidden_dim": hs.shape[0],
                "stage_answer": stage_answer,
                "stage_correctness": int(stage_correct),
                # Richer representations for Delta Probe (reviewer request)
                "mean_pool_hidden_state": mean_pool_hs.tolist() if hasattr(mean_pool_hs, 'tolist') else mean_pool_hs,
                "multi_layer_hidden_states": multi_layer_hs,
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
            acc = n_final_correct / n
            bg_parts = ", ".join(
                f"bg@{STAGE_CONTEXT_SIZES[s]}={100*n_both_gold[s]/n:.0f}%"
                for s in range(NUM_STAGES)
            )
            sc_parts = ", ".join(
                f"S{s}={100*n_stage_correct[s]/n:.0f}%"
                for s in range(NUM_STAGES)
            )
            print(f"\n  [{n}/{len(queries)}] saved | final_acc={acc*100:.1f}% | "
                  f"{bg_parts} | per-stage: {sc_parts} | "
                  f"{n/max(elapsed,1):.2f} q/s")

    _save_jsonl(collected, output_path)
    elapsed = time.time() - t0
    n = len(queries)
    print(f"\n[*] Done: {n} queries, {len(collected)} tuples in {elapsed:.0f}s")
    print(f"[*] Final RAG accuracy: {n_final_correct}/{n} = {100*n_final_correct/n:.1f}%")
    print(f"[*] Stage context sizes: {STAGE_CONTEXT_SIZES}")
    print(f"[*] Both-gold@k: " + ", ".join(
        f"@{STAGE_CONTEXT_SIZES[s]}={100*n_both_gold[s]/n:.1f}%"
        for s in range(NUM_STAGES)
    ))
    print(f"[*] Per-stage accuracy: " + ", ".join(
        f"S{s}(k={STAGE_CONTEXT_SIZES[s]})={100*n_stage_correct[s]/n:.1f}%"
        for s in range(NUM_STAGES)
    ))
    return output_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_queries", type=int, default=2000)
    ap.add_argument("--split", type=str, default="validation")
    ap.add_argument("--model_name", type=str, default=LLAMA_MODEL_NAME)
    ap.add_argument("--output", type=str,
                    default=os.path.join(DATA_DIR, "collected_states_hotpotqa_v3.jsonl"))
    ap.add_argument("--seed", type=int, default=BASE_SEED)
    ap.add_argument("--no_4bit", action="store_true", default=True)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    _get_cross_encoder()  # pre-load
    print(f"[*] V3 mode: progressive context sizes {STAGE_CONTEXT_SIZES}")
    print(f"[*] This creates a REAL cost-accuracy tradeoff across stages.")

    queries = load_hotpotqa(args.num_queries, args.split)
    model, tokenizer, hook_handles, store = load_llama_with_hidden_states(
        model_name=args.model_name, use_4bit=not args.no_4bit,
    )
    try:
        collect(queries, model, tokenizer, store, args.output)
        data = load_collected_data(args.output)
        n_correct = sum(1 for d in data if d["final_correctness"] == 1)
        print(f"\n[*] Stats: {len(set(d['query_id'] for d in data))} queries, "
              f"{len(data)} tuples, label-positive rate "
              f"{100*n_correct/len(data):.1f}%")
    finally:
        for h in hook_handles:
            h.remove()
        print("[*] Cleaned up hooks.")


if __name__ == "__main__":
    main()
