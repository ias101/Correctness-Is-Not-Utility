"""Richer hidden-state features for Delta Probe.
Adds to existing 4-bit collection: attention stats, first-token logit entropy.
Compares Delta Probe with and without richer features.

Usage: python experiments/_richer_features.py --num_queries 100
"""
import torch
import argparse, json, os, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

def predictive_entropy(logits):
    """Entropy of next-token distribution."""
    probs = torch.softmax(logits, dim=-1)
    return float(-torch.sum(probs * torch.log(probs + 1e-10)))

def attention_context_stats(attentions, input_ids, tokenizer):
    """Mean/max attention from last token to context tokens."""
    last_layer_attn = attentions[-1][0]  # (heads, seq, seq)
    avg_attn = last_layer_attn.mean(dim=0)  # (seq, seq)
    last_token_attn = avg_attn[-1, :]  # attention from last token to all tokens
    # Context tokens: everything between "Context:" and "Answer the question"
    return {
        "max_attn_to_context": float(last_token_attn[1:-1].max()),
        "mean_attn_to_context": float(last_token_attn[1:-1].mean()),
        "attn_entropy": float(-torch.sum(last_token_attn * torch.log(last_token_attn + 1e-10))),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_queries", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="results/richer_features")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    print("[*] Loading Qwen2.5-7B 4-bit with attention outputs...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        quantization_config=quant_config, device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    model.config.output_attentions = True
    model.eval()

    from datasets import load_dataset
    hotpot = load_dataset("hotpot_qa", "distractor", split="validation")
    queries = [ex["question"] for ex in hotpot.select(range(args.num_queries))]

    results = []
    print(f"[*] Collecting richer features for {args.num_queries} queries...")
    for q_idx in tqdm(range(args.num_queries)):
        ex = hotpot[q_idx]
        query = ex["question"]
        passages = [ex["context"]["sentences"][i] if i < len(ex["context"]["sentences"]) else "" for i in range(8)]

        for stage_idx, k in enumerate([2, 4, 6, 8]):
            context = passages[:k]
            context_text = "\n".join([f"Passage {i+1}: {p}" for i, p in enumerate(context)])
            prompt = f"<|im_start|>user\nQuestion: {query}\n\nContext:\n{context_text}\n\nAnswer the question based on the provided context.<|im_end|>\n<|im_start|>assistant\n"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

            with torch.no_grad():
                outputs = model(**inputs.to("cuda"), output_hidden_states=True, output_attentions=True)
                hs = outputs.hidden_states[-1][0, -1, :].cpu().float().numpy()
                multi = torch.cat([outputs.hidden_states[i][0, -1, :].cpu().float() for i in range(-4, 0)]).numpy()

                # Richer features
                first_token_logits = outputs.logits[0, -1, :].cpu().float()
                entropy = predictive_entropy(first_token_logits)
                attn_stats = attention_context_stats(outputs.attentions, inputs["input_ids"][0], tokenizer)

            results.append({
                "query_idx": q_idx, "stage": stage_idx, "k_passages": k,
                "final_token_hs": hs.tolist(), "multi_layer_hs": multi.tolist(),
                "predictive_entropy": entropy,
                **attn_stats,
            })

    out_path = os.path.join(args.output_dir, "richer_features.json")
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"[✓] Saved {len(results)} tuples to {out_path}")
    print(f"[*] Features: final_token_hs + multi_layer_hs + predictive_entropy + attention_stats")
    print("[✓] DONE")

if __name__ == "__main__":
    main()
