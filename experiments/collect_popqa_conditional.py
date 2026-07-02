"""
PopQA conditional-support collection (entity-page-only, self-contained).

Fixes the PopQA conditional event-sparsity gap: the paper's conditional Delta
probe ran on HotpotQA/TriviaQA (2000q each) but NOT PopQA (500q -> only ~81
benefit events). Here we collect the FULL 2053-entity PopQA set with per-stage
hidden states so the SAME conditional protocol (conditional_delta_multi.py) can
run on PopQA with adequate currently-wrong support (~300 benefit events).

Protocol = collect_popqa_v3.py's progressive ENTITY-PAGE reveal
(150->400->800->1500 chars), BF16 Qwen2.5-7B, last-4-layer final-token concat,
lenient string-match. The supplementary BGE-m3 dense passages are OMITTED: the
wiki_500k corpus had ~8% PopQA gold coverage (mostly noise), and the entity
page IS the retrieval term for entity-grounded PopQA, so entity-page-only is
the cleaner conditional protocol. Self-contained: needs only the entity cache
(qid/question/subj/obj/wiki_title/wiki_text) + Qwen -- NO faiss/wiki corpus.

Outputs the conditional_delta_multi.py schema directly:
  query_id, stage_idx, stage_correctness, multi_layer_hidden_states (14336-d)

  python collect_popqa_conditional.py --cache popqa_entity_cache.json \
      --output popqa_conditional_2k.jsonl
"""
import argparse, json, re, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

STAGE_TEXT_LENGTHS = [150, 400, 800, 1500]
CONTEXT_MAX = 2048
MAX_NEW = 48
MODEL = "Qwen/Qwen2.5-7B-Instruct"


def norm(s):
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_correct(answer, gold):
    a, g = norm(answer), norm(gold)
    if not g or not a:           # empty normalized answer must NOT match ("" in g is always True)
        return 0
    return int(a == g or g in a or a in g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="popqa_entity_cache.json")
    ap.add_argument("--output", default="popqa_conditional_2k.jsonl")
    ap.add_argument("--model", default=MODEL, help="HF name or local dir (e.g. ModelScope download)")
    ap.add_argument("--n_queries", type=int, default=0, help="0 = all cached entities")
    args = ap.parse_args()

    cache = json.load(open(args.cache))
    if args.n_queries > 0:
        cache = cache[: args.n_queries]
    print(f"[*] {len(cache)} entities from {args.cache}", flush=True)

    print(f"[*] loading {args.model} (BF16)...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
    dev = model.device

    fout = open(args.output, "a", encoding="utf-8")
    t0 = time.time()
    done = 0
    stage_correct = [0, 0, 0, 0]
    for e in cache:
        qid = e["qid"]
        question = e["question"]
        gold = e["obj"]
        wiki_text = e.get("wiki_text", "")
        wiki_title = e.get("wiki_title", "Entity")
        if not wiki_text:
            continue
        for stage in range(4):
            text_len = STAGE_TEXT_LENGTHS[stage]
            ctx = f"[Wikipedia: {wiki_title}]: {wiki_text[:text_len]}\n\n"
            msgs = [{"role": "user",
                     "content": f"Context:\n{ctx}\nQuestion: {question}\n\n"
                                f"Answer with just the name or short phrase."}]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tok(prompt, return_tensors="pt", truncation=True,
                         max_length=CONTEXT_MAX).to(dev)
            with torch.no_grad():
                mout = model(**inputs, output_hidden_states=True)
                hs = np.concatenate([mout.hidden_states[li][0, -1, :].cpu().float().numpy()
                                     for li in [-4, -3, -2, -1]])
                gen = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            answer = tok.decode(gen[0][inputs.input_ids.shape[1]:],
                                skip_special_tokens=True).strip()
            correct = is_correct(answer, gold)
            stage_correct[stage] += correct
            rec = {"query_id": qid, "question": question, "subj": e.get("subj"),
                   "popularity": e.get("s_pop"), "stage_idx": stage,
                   "text_chars": text_len,
                   "multi_layer_hidden_states": hs.tolist(), "hidden_dim": int(len(hs)),
                   "stage_answer": answer, "gold_answer": gold,
                   "stage_correctness": int(correct),
                   "model": "Qwen/Qwen2.5-7B-Instruct", "precision": "bf16"}
            fout.write(json.dumps(rec) + "\n")
        fout.flush()
        done += 1
        if done % 50 == 0:
            el = time.time() - t0
            accs = [c / max(done, 1) for c in stage_correct]
            print(f"  [{done}/{len(cache)}] {el/60:.1f}min "
                  f"{done/el*3600:.0f}q/hr ETA {(len(cache)-done)/max(done,1)*el/3600:.1f}h "
                  f"| acc S0-3 {accs[0]:.3f}/{accs[1]:.3f}/{accs[2]:.3f}/{accs[3]:.3f}", flush=True)
    fout.close()
    print(f"[*] DONE {done} queries in {(time.time()-t0)/3600:.2f}h -> {args.output}", flush=True)
    print(f"[*] per-stage acc: " + " ".join(f"S{i}={c/max(done,1):.3f}" for i, c in enumerate(stage_correct)), flush=True)


if __name__ == "__main__":
    main()
