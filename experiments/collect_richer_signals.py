"""Collect hidden states + logit entropy + attention statistics in one pass.
Combines Experiments #2 and #3 from EXPERIMENT_DESIGN.md.
Target: 200 HotpotQA queries, 4 stages each."""

import argparse, json, os, sys, time, torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import STAGES, NUM_STAGES, LLAMA_MODEL_NAME, DATA_DIR, BASE_SEED
from collect_states import (
    set_seed, load_llama_with_hidden_states, build_chat_prompt,
    generate_answer, check_correctness, _save_jsonl,
)

def collect_with_signals(args):
    set_seed(args.seed)
    
    # Load dataset
    ds = load_dataset("hotpot_qa", "distractor", split=f"train[:{args.num_queries}]")
    print(f"Loaded {len(ds)} HotpotQA queries")
    
    # Load model with hooks for hidden states + attention
    model, tokenizer, hook_handles, hs_store = load_llama_with_hidden_states(args.model_name, args.no_4bit)
    
    # Register attention hooks on last 4 layers
    attention_store = {}
    def make_attn_hook(layer_idx):
        def hook(module, input, output):
            attention_store[layer_idx] = output[0].detach().cpu()
        return hook
    
    attn_handles = []
    for i, layer in enumerate(model.model.layers):
        if i >= len(model.model.layers) - 4:
            handle = layer.self_attn.register_forward_hook(make_attn_hook(i))
            attn_handles.append(handle)
    
    # Also enable output logits
    
    results = []
    for idx, item in enumerate(tqdm(ds, desc="Collecting")):
        question = item["question"]
        answers = [item["answer"]] if isinstance(item["answer"], str) else item["answer"]
        
        # Get passages and rank them (simplified: use context passages directly)
        context = item.get("context", {})
        passages = []
        for title, sentences in zip(context.get("title", []), context.get("sentences", [])):
            passages.append({"title": title, "text": " ".join(sentences)})
        
        # Progressive context
        stage_sizes = [2, 4, 6, 8]
        for stage_idx, k in enumerate(stage_sizes):
            stage_passages = passages[:min(k, len(passages))]
            context_text = "\n".join(p["title"] + ": " + p["text"] for p in stage_passages)
            prompt = f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
            
            chat_prompt = build_chat_prompt(prompt, tokenizer)
            inputs = tokenizer(chat_prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
            
            hs_store.clear()
            attention_store.clear()
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=False)
            
            # Hidden state
            stored_layers = sorted([int(k) for k in hs_store.keys() if not k.endswith("_mean")], reverse=True)
            if stored_layers:
                hs = hs_store[stored_layers[0]][0, -1, :].cpu()
            else:
                hs = torch.zeros(3584)
            
            # Logit entropy
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            entropy = float(-(probs * torch.log(probs + 1e-10)).sum())
            max_prob = float(probs.max())
            
            # Attention statistics
            max_ctx_attn = 0.0
            attn_entropy = 0.0
            if attention_store:
                last_attn = list(attention_store.values())[-1]
                attn_weights = last_attn[0, :, -1, :]
                avg_attn = attn_weights.mean(dim=0)
                # Context tokens: everything after prompt template
                context_start = len(tokenizer.encode(chat_prompt.split("Question:")[0])) if "Question:" in chat_prompt else 50
                if context_start < avg_attn.shape[0]:
                    ctx_attn = avg_attn[context_start:]
                    if ctx_attn.shape[0] > 0:
                        max_ctx_attn = float(ctx_attn.max())
                        attn_entropy = float(-(ctx_attn * torch.log(ctx_attn + 1e-10)).sum())
            
            # Generate answer
            answer_text = generate_answer(model, tokenizer, chat_prompt, max_tokens=48)
            stage_correct = int(check_correctness(answer_text, answers))
            
            results.append({
                "query_id": idx,
                "question": question,
                "stage_idx": stage_idx,
                "hidden_state": hs.tolist(),
                "logit_entropy": entropy,
                "logit_max_prob": max_prob,
                "attn_max_context": max_ctx_attn,
                "attn_entropy": attn_entropy,
                "stage_correctness": stage_correct,
            })
        
        if (idx + 1) % 50 == 0:
            _save_jsonl(results, args.output)
            print(f"  Checkpoint: {len(results)} tuples saved")
    
    _save_jsonl(results, args.output)
    print(f"Done: {len(results)} tuples → {args.output}")
    
    # Cleanup
    for h in attn_handles:
        h.remove()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_queries", type=int, default=200)
    parser.add_argument("--model_name", type=str, default=LLAMA_MODEL_NAME)
    parser.add_argument("--output", type=str, default="data/collected_richer_signals.jsonl")
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    collect_with_signals(args)
