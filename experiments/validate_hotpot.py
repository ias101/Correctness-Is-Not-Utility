"""
Quick validation: does switching to HotpotQA-distractor (provided context)
fix the catastrophic ~8% RAG accuracy?

Each HotpotQA-distractor question ships with 10 paragraphs (2 gold supporting
paragraphs + 8 distractors), so retrieval recall is 100% by construction.
This isolates the question: "given the right context, can the model+eval
actually score correct answers?"

Run on the remote GPU:
    python validate_hotpot.py [N]      # N = num questions, default 200
"""
import re
import string
import sys
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-7B-Instruct"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200


def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def check_correct(pred: str, gts) -> bool:
    """Same lenient metric as collect_states.check_correctness."""
    p = normalize_answer(pred)
    for gt in gts:
        if not gt:
            continue
        g = normalize_answer(gt)
        if p == g:
            return True
        if len(g) > 3 and (g in p or p in g):
            return True
        gt_t, p_t = set(g.split()), set(p.split())
        if gt_t and p_t:
            ov = gt_t & p_t
            pr = len(ov) / len(p_t)
            rc = len(ov) / len(gt_t)
            f1 = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0
            if f1 >= 0.5:
                return True
    return False


def build_prompt(tok, question: str, context: str) -> str:
    user = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer with only a short phrase or entity (or 'yes'/'no'), based on the context."
    )
    msgs = [
        {"role": "system", "content": "You are a helpful assistant. Answer concisely and accurately."},
        {"role": "user", "content": user},
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    print(f"[*] Loading HotpotQA distractor (validation[:{N}])...")
    ds = load_dataset("hotpot_qa", "distractor", split=f"validation[:{N}]")

    print(f"[*] Loading {MODEL}...")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
    model.eval()

    # ── closed-book and full-context, side by side on the same questions ──
    n_ctx = 0
    n_closed = 0
    samples = []
    t0 = time.time()

    for i, ex in enumerate(ds):
        q = ex["question"]
        ans = ex["answer"]
        titles = ex["context"]["title"]
        sents = ex["context"]["sentences"]
        paras = [f"{t}: {''.join(ss)}" for t, ss in zip(titles, sents)]
        context = "\n\n".join(paras)

        # with-context
        prompt = build_prompt(tok, q, context)
        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=64, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        pred_ctx = tok.decode(out[0, inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
        ok_ctx = check_correct(pred_ctx, [ans])
        n_ctx += ok_ctx

        # closed-book (no context) — only for the first 100 to save time
        ok_closed = None
        if i < 100:
            cb_user = f"Question: {q}\n\nAnswer with only a short phrase or entity (or 'yes'/'no')."
            cb_msgs = [
                {"role": "system", "content": "You are a helpful assistant. Answer concisely and accurately."},
                {"role": "user", "content": cb_user},
            ]
            cb_prompt = tok.apply_chat_template(cb_msgs, tokenize=False, add_generation_prompt=True)
            cb_inp = tok(cb_prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
            with torch.no_grad():
                cb_out = model.generate(**cb_inp, max_new_tokens=64, do_sample=False,
                                        pad_token_id=tok.eos_token_id)
            pred_cb = tok.decode(cb_out[0, cb_inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
            ok_closed = check_correct(pred_cb, [ans])
            n_closed += ok_closed

        if i < 8:
            samples.append((q, ans, pred_ctx, ok_ctx))

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(ds)}] ctx-acc so far: {100*n_ctx/(i+1):.1f}%")

    dt = time.time() - t0
    print("\n" + "=" * 60)
    print(f"  HotpotQA-distractor | Qwen2.5-7B | {len(ds)} questions | {dt:.0f}s")
    print("=" * 60)
    print(f"  WITH provided context : {n_ctx}/{len(ds)} = {100*n_ctx/len(ds):.1f}%")
    print(f"  CLOSED-BOOK (first 100): {n_closed}/100 = {n_closed:.0f}%")
    print("=" * 60)
    print("  (old broken pipeline on NQ was 8.3%)")
    for q, a, p, ok in samples:
        print(f"\n[{'OK' if ok else 'XX'}] Q: {q[:90]}")
        print(f"     gold: {a}")
        print(f"     pred: {p[:140]}")


if __name__ == "__main__":
    main()
