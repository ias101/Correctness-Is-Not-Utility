"""8-bit vs 4-bit Quantization Control for Delta Probe.
Loads Qwen2.5-7B in 8-bit (vs existing 4-bit data) and compares
hidden states + Delta Probe AUROC/AUPRC.

8-bit fits in ~10GB VRAM — feasible on RTX 3080 16GB.
Compares: final-token hidden states between 4-bit and 8-bit.
"""
import torch
import argparse
import json
import os
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_queries", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="results/bit8_control")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda"

    # Load in 8-bit
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    print("[*] Loading Qwen2.5-7B in 8-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    model.eval()

    # Load HotpotQA queries
    from datasets import load_dataset
    hotpot = load_dataset("hotpot_qa", "distractor", split="validation")
    queries = [ex["question"] for ex in hotpot.select(range(args.num_queries))]
    passages = [
        [ex["context"]["sentences"][i] if i < len(ex["context"]["sentences"]) else ""
         for i in range(8)]
        for ex in hotpot.select(range(args.num_queries))
    ]

    results = []
    print(f"[*] Collecting 8-bit hidden states for {args.num_queries} queries...")
    for q_idx, (query, passage_set) in enumerate(tqdm(zip(queries, passages), total=len(queries))):
        for stage_idx, k in enumerate([2, 4, 6, 8]):
            context = passage_set[:k]
            context_text = "\n".join([f"Passage {i+1}: {p}" for i, p in enumerate(context)])
            prompt = f"<|im_start|>user\nQuestion: {query}\n\nContext:\n{context_text}\n\nAnswer the question based on the provided context.<|im_end|>\n<|im_start|>assistant\n"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hs = outputs.hidden_states[-1][0, -1, :].cpu().float().numpy()
                multi = torch.cat([outputs.hidden_states[i][0, -1, :].cpu().float() for i in range(-4, 0)]).numpy()

            results.append({
                "query_idx": q_idx, "stage": stage_idx, "k_passages": k,
                "final_token_hs": hs.tolist(), "multi_layer_hs": multi.tolist(),
                "precision": "8bit",
            })

    out_path = os.path.join(args.output_dir, "8bit_hidden_states.json")
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"[✓] 8-bit states saved ({len(results)} tuples) to {out_path}")

    # Quick comparison: cosine similarity between 8-bit and 4-bit final-token states
    print("\n[*] Loading 4-bit baseline for comparison...")
    import glob as g
    bit4_results = []
    for f in sorted(g.glob("data/hotpotqa_v4/*.jsonl")):
        with open(f) as fh:
            for line in fh:
                bit4_results.append(json.loads(line))
    print(f"[*] 4-bit baseline: {len(bit4_results)} tuples")

    # Match by query_idx and stage
    bit4_by_key = {}
    for r in bit4_results:
        key = (r.get("query_idx", -1), r.get("stage", -1))
        if "final_token_hs" in r or "hidden_state" in r:
            bit4_by_key[key] = r

    cos_sims = []
    for r in results:
        key = (r["query_idx"], r["stage"])
        if key in bit4_by_key:
            r4 = bit4_by_key[key]
            hs4 = np.array(r4.get("final_token_hs", r4.get("hidden_state", [0])))
            hs8 = np.array(r["final_token_hs"])
            if len(hs4) == len(hs8) and len(hs4) > 0:
                cos = np.dot(hs4, hs8) / (np.linalg.norm(hs4) * np.linalg.norm(hs8) + 1e-8)
                cos_sims.append(float(cos))

    if cos_sims:
        mean_cos = np.mean(cos_sims)
        std_cos = np.std(cos_sims)
        print(f"\n[✓] 8-bit vs 4-bit cosine similarity: {mean_cos:.4f} ± {std_cos:.4f}")
        print(f"    Min: {np.min(cos_sims):.4f}, Max: {np.max(cos_sims):.4f}")
        # If cosine > 0.99, hidden states are nearly identical → quantization is NOT the issue
        if mean_cos > 0.99:
            print("[✓] VERDICT: Hidden states nearly identical. Quantization NOT causing the Delta Probe ceiling.")
        elif mean_cos > 0.95:
            print("[!] VERDICT: Minor differences. Quantization unlikely to change Delta Probe conclusion.")
        else:
            print("[!] VERDICT: Substantial differences! Quantization MAY be degrading the signal.")
    else:
        print("[!] No matching tuples found for comparison.")

    mem_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n[*] Peak GPU memory: {mem_gb:.2f} GB")
    print("[✓] DONE")

if __name__ == "__main__":
    main()
