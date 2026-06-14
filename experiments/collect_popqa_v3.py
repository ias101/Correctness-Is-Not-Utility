"""
PopQA Wikipedia entity page — V3 progressive text revelation.
Fixes flat S0→S3 curve: gradually increase entity page text length.

DESIGN: Stage 0 gets entity page starting text (150 chars).
       Stage 1 gets more (400 chars).
       Stage 2 gets even more (800 chars).
       Stage 3 gets full context (1500 chars + supplementary passages).
This creates REAL accuracy progression for routing evaluation.
"""
import argparse, json, os, re, sys
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

# Progressive entity page text per stage (chars)
STAGE_TEXT_LENGTHS = [150, 400, 800, 1500]
# Supplementary passages per stage (dense retrieval)
STAGE_PASSAGES = [1, 2, 3, 4]
CONTEXT_MAX = 2048

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_queries", type=int, default=2000)
    parser.add_argument("--output_dir", default="/workspace/popqa_v3")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Step 1: Load entities from CACHE (built by build_popqa_entity_cache.py)
    CACHE_PATH = "/workspace/popqa_entity_cache.json"
    import json as _json
    if not os.path.exists(CACHE_PATH):
        print(f"ERROR: Entity cache not found at {CACHE_PATH}. Run build_popqa_entity_cache.py first.")
        return
    with open(CACHE_PATH) as f:
        found_list = _json.load(f)
    found = {e["subj"].replace("_", " ").lower().strip(): e for e in found_list}
    print(f"Loaded {len(found)} entities from cache ({CACHE_PATH})")

    if len(found) < 50:
        print("ERROR: Too few covered entities")
        return

    # Step 2: Load models
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
    from sentence_transformers import CrossEncoder

    print("Loading Qwen2.5-7B BF16...")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16,
        device_map="cuda:0").eval()

    print("Loading BGE-m3 (CPU, to save GPU memory)...")
    bge_tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    bge_model = AutoModel.from_pretrained(
        "BAAI/bge-m3", torch_dtype=torch.float32).eval()  # CPU

    print("Loading CE + FAISS...")
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    import faiss
    index = faiss.read_index("/workspace/wiki_stream/faiss_index.bin")
    faiss_passages = [json.loads(line) for line in open("/workspace/wiki_stream/passages.jsonl")]

    # Step 3: Collection
    entities_list = list(found.values())
    np.random.seed(42)
    np.random.shuffle(entities_list)
    entities_list = entities_list[:args.n_queries]
    print(f"Collecting {len(entities_list)} queries...")

    results = []
    for e in tqdm(entities_list, desc="PopQA v3"):
        wiki_text = e["wiki_text"]
        wiki_title = e.get("wiki_title", "Entity")
        question = e["question"]
        gold = e["obj"]

        # Dense retrieval for supplementary passages (same for all stages)
        q_text = f"Represent this sentence for searching relevant passages: {question}"
        inputs = bge_tok(q_text, return_tensors="pt", truncation=True,
                        max_length=512)
        with torch.no_grad():
            q_emb = bge_model(**inputs).last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
        D, I = index.search(q_emb.reshape(1, -1), 50)
        dense_hits = [{"text": faiss_passages[idx]["text"], "score": float(-dist)}
                      for dist, idx in zip(D[0], I[0]) if idx < len(faiss_passages)]

        pairs = [(question, p["text"][:500]) for p in dense_hits[:20]]
        ce_scores = ce.predict(pairs, show_progress_bar=False)
        scored = sorted([(float(s), p) for s, p in zip(ce_scores, dense_hits[:20])],
                       key=lambda x: x[0], reverse=True)
        ranked = [p for _, p in scored]

        # Per-stage with PROGRESSIVE text revelation
        for stage in range(4):
            text_len = STAGE_TEXT_LENGTHS[stage]
            n_passages = STAGE_PASSAGES[stage]

            ctx = f"[Wikipedia: {wiki_title}]: {wiki_text[:text_len]}\n\n"
            for p in ranked[:n_passages]:
                ctx += f"[Passage]: {p['text'][:300]}\n\n"

            prompt = (f"<|im_start|>user\nContext:\n{ctx}\n"
                     f"Question: {question}\n\n"
                     f"Answer with just the name or short phrase.<|im_end|>\n"
                     f"<|im_start|>assistant\n")

            inputs = tok(prompt, return_tensors="pt", truncation=True,
                        max_length=CONTEXT_MAX).to("cuda")
            with torch.no_grad():
                mout = model(**inputs, output_hidden_states=True)
                hs = np.concatenate([mout.hidden_states[li][0, -1, :].cpu().float().numpy()
                                    for li in [-4, -3, -2, -1]])
                gen = model.generate(**inputs, max_new_tokens=48, do_sample=False,
                                    pad_token_id=tok.eos_token_id)
                answer = tok.decode(gen[0][inputs.input_ids.shape[1]:],
                                   skip_special_tokens=True).strip()

            def norm(s):
                s = s.lower().strip()
                s = re.sub(r'[^\w\s]', ' ', s)
                s = re.sub(r'\s+', ' ', s).strip()
                return s
            a, g = norm(answer), norm(gold)
            correct = int(a == g or g in a or a in g)

            results.append({
                "query_id": e["qid"], "question": question,
                "gold": gold, "subj": e["subj"],
                "popularity": e["s_pop"], "stage": stage,
                "text_chars": text_len, "n_passages": n_passages,
                "hs_dim": len(hs), "hs_concat": hs.tolist(), "answer": answer, "correct": correct,
            })

        if len(results) % 200 == 0:
            tmp = out / "popqa_states.jsonl"
            with open(tmp, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")

    # Save + stats
    with open(out / "popqa_states.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    n_q = len(set(r["query_id"] for r in results))
    print(f"\nDone: {len(results)} tuples, {n_q} queries")
    for s in range(4):
        rs = [r for r in results if r["stage"] == s]
        acc = np.mean([r["correct"] for r in rs])
        print(f"  S{s} ({STAGE_TEXT_LENGTHS[s]}chars + {STAGE_PASSAGES[s]}p): acc={acc:.3f} n={len(rs)}")
    s0 = np.mean([r["correct"] for r in results if r["stage"] == 0])
    s3 = np.mean([r["correct"] for r in results if r["stage"] == 3])
    gap = (s3 - s0) * 100
    print(f"  S0->S3 gap: {gap:.1f}pp {'✅' if gap > 4 else '❌'}")

    # Popularity stratified
    pops = [r["popularity"] for r in results if r["stage"] == 3]
    if pops:
        med = np.median(pops)
        for label, pset in [("High pop", lambda p: p >= med), ("Low pop", lambda p: p < med)]:
            rs = [r for r in results if r["stage"] == 3 and pset(r["popularity"])]
            if rs:
                print(f"  [{label}] S3 acc={np.mean([r['correct'] for r in rs]):.3f} ({len(rs)}q)")


if __name__ == "__main__":
    main()
