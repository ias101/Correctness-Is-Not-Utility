"""
Canonical V5 HotpotQA collection (clean, self-contained) — for Loop 33/34 confirmation.

Reproduces the canonical setup: Qwen2.5-7B-Instruct 4-bit (NF4, bf16 compute),
HotpotQA-distractor train split, 10 paragraphs/query ranked by cross-encoder
ms-marco-MiniLM-L-6-v2, progressive stages top-{2,4,6,8}, last-4-layer final-token
hidden states concatenated (14336-d, order [final, L-2, L-3, L-4]), lenient
string-match correctness, greedy 48-token generation. Self-contained (no
collect_states/config import — the repo's collect_v5_multilayer has a stale
extract_hidden_state signature). Saves ce_scores (top-k per stage) so the
deployable-feature control can reconstruct incoming passages.

  python collect_canonical_v5.py --num_queries 2000 \
      --output collected_states_hotpotqa_v5_canon.jsonl
"""
import argparse, json, os, re, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import CrossEncoder

MODEL = "Qwen/Qwen2.5-7B-Instruct"
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
STAGE_SIZES = [2, 4, 6, 8]
MAX_LEN = 2048
MAX_NEW = 48


def normalize(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
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
    if inter == 0:
        return False
    prec, rec = inter / len(pt), inter / len(gt)
    return (2 * prec * rec / (prec + rec)) >= 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_queries", type=int, default=2000)
    ap.add_argument("--output", default="collected_states_hotpotqa_v5_canon.jsonl")
    ap.add_argument("--split", default="train")
    ap.add_argument("--start_idx", type=int, default=0)
    args = ap.parse_args()

    print(f"[*] loading {MODEL} (4-bit NF4)...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=True).eval()
    dev = model.device
    print(f"[*] loading cross-encoder {RERANKER}...")
    ce = CrossEncoder(RERANKER, device="cuda")
    print(f"[*] loading HotpotQA distractor [{args.split}]...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=args.split)
    n = min(args.num_queries, len(ds) - args.start_idx)

    out_path = args.output
    fout = open(out_path, "a", encoding="utf-8")
    t0 = time.time()
    done = 0
    for idx in range(args.start_idx, args.start_idx + n):
        s = ds[idx]
        q, ans = s["question"], s["answer"]
        qid = s.get("id", str(idx))
        cd = s.get("context", {})
        paras = []
        if "title" in cd and "sentences" in cd:
            for title, sents in zip(cd["title"], cd["sentences"]):
                txt = " ".join(sents) if isinstance(sents, list) else str(sents)
                paras.append({"title": title, "text": txt})
        if len(paras) < 2:
            continue
        ce_scores = np.asarray(ce.predict([(q, p["text"]) for p in paras], show_progress_bar=False))
        ranked = np.argsort(ce_scores)[::-1]
        for si, k in enumerate(STAGE_SIZES):
            topk = ranked[:k]
            ctx = "\n\n".join(f"Passage {j+1} (Title: {paras[i]['title']}): {paras[i]['text']}"
                              for j, i in enumerate(topk))
            msgs = [{"role": "user", "content":
                     f"Based on the following passages, answer the question.\n\n{ctx}\n\nQuestion: {q}\nAnswer:"}]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(dev)
            with torch.no_grad():
                o = model(**inp, output_hidden_states=True)
            # last 4 layers, last token, order [final, L-2, L-3, L-4]
            multi = torch.cat([o.hidden_states[-l][0, -1, :].float().cpu() for l in range(1, 5)]).tolist()
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            raw = tok.decode(gen[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
            rec = {"query_id": qid, "question": q, "stage_idx": si,
                   "multi_layer_hidden_states": multi, "hidden_dim": len(multi),
                   "stage_answer": raw.strip(), "gold_answer": ans,
                   "stage_correctness": int(is_correct(raw, ans)),
                   "k_passages": int(k),
                   "ce_scores": [float(ce_scores[i]) for i in topk],
                   "model": MODEL, "precision": "4bit-nf4"}
            fout.write(json.dumps(rec) + "\n")
        fout.flush()
        done += 1
        if done % 50 == 0:
            el = time.time() - t0
            print(f"  [{done}/{n}] {el/60:.1f} min, {done/el*3600:.0f} q/hr, "
                  f"ETA {(n-done)/max(done,1)*el/3600:.1f}h", flush=True)
    fout.close()
    print(f"[*] DONE {done} queries in {(time.time()-t0)/3600:.2f}h -> {out_path}")


if __name__ == "__main__":
    main()
