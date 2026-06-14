"""Quick 4-bit hidden state collection for quantization ablation comparison."""
import sys, os, json, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse

from datasets import load_dataset
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from config import LLAMA_MODEL_NAME, NUM_STAGES, STAGES

STAGE_CONTEXT_SIZES = [2, 4, 6, 8]
MAX_NEW_TOKENS = 32

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_queries", type=int, default=100)
    ap.add_argument("--output", type=str, default="data/collected_4bit_compare.jsonl")
    args = ap.parse_args()

    print("[*] Loading HotpotQA...")
    ds = load_dataset("hotpot_qa", "distractor", split=f"validation[:{args.num_queries}]")
    print(f"[*] {len(ds)} queries")

    # Build passages: title + sentences for each of the 10 paragraphs
    print("[*] Building passages...")
    all_queries = []
    for item in ds:
        titles = item["context"]["title"]
        sentences_list = item["context"]["sentences"]
        passages = []
        for title, sents in zip(titles, sentences_list):
            passages.append(f"{title}: " + " ".join(sents))
        all_queries.append({
            "question": item["question"],
            "passages": passages,
        })
    print(f"[*] {len(all_queries)} queries prepared")

    print("[*] Loading Qwen2.5-7B in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"[*] GPU mem: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    hidden_states = {}
    def make_hook(idx):
        def hook(module, input, output):
            hs = output[0] if isinstance(output, tuple) else output
            hidden_states[idx] = hs[:, -1, :].detach().cpu()
        return hook

    hooks = []
    for i, layer_idx in enumerate(range(24, 28)):
        h = model.model.layers[layer_idx].register_forward_hook(make_hook(i))
        hooks.append(h)

    results = []
    t_start = time.time()
    for qi, q in enumerate(all_queries):
        question = q["question"]
        passages = q["passages"]
        eta = (time.time() - t_start) / max(qi, 1) * (len(all_queries) - qi) if qi > 0 else 0

        for si, n_pass in enumerate(STAGE_CONTEXT_SIZES):
            ctx = "\n\n".join(passages[:n_pass])
            prompt = (
                f"<|im_start|>user\n"
                f"Context:\n{ctx}\n\nQuestion: {question}\n"
                f"Answer the question based on the provided context.\n"
                f"<|im_end|>\n<|im_start|>assistant\n"
            )
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=2048).to(model.device)
            hidden_states.clear()
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False, pad_token_id=tokenizer.pad_token_id)
            answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:],
                                     skip_special_tokens=True).strip()
            ml_hs = [hidden_states.get(i, torch.zeros(3584)).tolist() for i in range(4)]
            results.append({
                "query_id": str(qi),
                "question": question,
                "stage_idx": si,
                "stage_name": STAGES[si],
                "hidden_state": ml_hs[-1],
                "hidden_dim": 3584,
                "multi_layer_hidden_states": ml_hs,
                "stage_answer": answer,
                "quantization": "4bit",
            })
        if (qi+1) % 10 == 0:
            print(f"[{qi+1}/{len(all_queries)}] ETA {eta:.0f}s, {len(results)} tuples")

    for h in hooks: h.remove()
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"[*] Done: {len(results)} tuples -> {args.output} in {time.time()-t_start:.0f}s")

if __name__ == "__main__":
    main()
