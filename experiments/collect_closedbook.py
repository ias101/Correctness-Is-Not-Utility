"""
Closed-book (0-passage, no-retrieval) stage collection (Loop 36).

Adds a new CHEAPEST stage to the progressive RAG pipeline: the LLM answers the
question with NO retrieved passages (closed-book / parametric). For easy questions
the model may already know the answer, so routing could stop here and save all
retrieval cost. We collect, per query, the closed-book hidden state (last-4-layer
concat, 14336-d), the generated answer, and its correctness -- joinable by
query_id with the existing 4-stage retrieval data (collected_states_*_v5_2000.jsonl).

Same model/precision/scorer/HS-extraction as collect_canonical_v5.py; the ONLY
change is the prompt has no passages. Iterates HotpotQA-distractor train (first N,
matching the original collection order) so query_ids align.

  python collect_closedbook.py --num_queries 2000 --model Qwen/Qwen2.5-7B-Instruct \
      --output closedbook_hotpotqa_qwen.jsonl
"""
import argparse, json, re, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MAX_LEN = 2048
MAX_NEW = 48


def normalize(s):
    s = s.lower(); s = re.sub(r"\b(a|an|the)\b", " ", s); s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def is_correct(pred, gold):
    p, g = normalize(pred), normalize(gold)
    if not g:
        return False
    if p == g or g in p or p in g:
        return True
    pt, gt = set(p.split()), set(g.split())
    if not pt or not gt:
        return False
    inter = len(pt & gt)
    return inter > 0 and (2 * (inter / len(pt)) * (inter / len(gt)) / (inter / len(pt) + inter / len(gt))) >= 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_queries", type=int, default=2000)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--split", default="train")
    ap.add_argument("--output", default="closedbook_hotpotqa_qwen.jsonl")
    args = ap.parse_args()

    print(f"[*] loading {args.model} (4-bit NF4)...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map="auto", trust_remote_code=True).eval()
    dev = model.device
    print(f"[*] loading HotpotQA distractor [{args.split}]...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=args.split)
    n = min(args.num_queries, len(ds))

    fout = open(args.output, "a", encoding="utf-8")
    t0 = time.time(); done = 0
    for idx in range(n):
        s = ds[idx]
        q, ans, qid = s["question"], s["answer"], s.get("id", str(idx))
        # CLOSED-BOOK prompt: no passages
        msgs = [{"role": "user", "content": f"Answer the question.\n\nQuestion: {q}\nAnswer:"}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(dev)
        with torch.no_grad():
            o = model(**inp, output_hidden_states=True)
        multi = torch.cat([o.hidden_states[-l][0, -1, :].float().cpu() for l in range(1, 5)]).tolist()
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.eos_token_id)
        raw = tok.decode(gen[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        rec = {"query_id": qid, "question": q, "stage_idx": -1, "stage_name": "closed_book",
               "multi_layer_hidden_states": multi, "hidden_dim": len(multi),
               "stage_answer": raw.strip(), "gold_answer": ans,
               "stage_correctness": int(is_correct(raw, ans)), "k_passages": 0,
               "model": args.model, "precision": "4bit-nf4"}
        fout.write(json.dumps(rec) + "\n"); fout.flush()
        done += 1
        if done % 100 == 0:
            el = time.time() - t0
            print(f"  [{done}/{n}] {el/60:.1f} min, ETA {(n-done)/max(done,1)*el/3600:.1f}h", flush=True)
    fout.close()
    cb_acc = None
    print(f"[*] DONE {done} closed-book queries in {(time.time()-t0)/3600:.2f}h -> {args.output}")


if __name__ == "__main__":
    main()
