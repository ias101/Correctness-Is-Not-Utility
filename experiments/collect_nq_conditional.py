"""
NQ conditional-support collection (matched 14336-d protocol, good BM25 retrieval).

Adds NQ as the 4th routing dataset. The prior NQ collection used a weak FAISS/wiki_5m
index (model refused: "context does not contain information", flat 0.11 acc, no benefit
events) and single-layer 3584-d states. This re-collects with:
  - BM25 over the PROVEN 2M wiki_corpus (the index TriviaQA used to reach ~0.48 acc),
  - last-4-layer concat (14336-d) matching HotpotQA/PopQA/TriviaQA,
  - per-stage correctness, progressive context (top 2->4->6->8),
so conditional_delta_multi.py can run the identical conditional Delta + routing on NQ.

  python collect_nq_conditional.py --n_queries 2000 \
      --index data/wiki_corpus/bm25_index.pkl --output nq_conditional_2k.jsonl
"""
import argparse, json, re, sys, os, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import CrossEncoder

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)                       # experiments/
sys.path.insert(0, os.path.dirname(_here))      # repo root (retrieval.py / config.py live here)
from retrieval import load_bm25_index, retrieve_bm25

STAGE_SIZES = [2, 4, 6, 8]
MAX_NEW = 16
MAX_LEN = 2400
MODEL = "Qwen/Qwen2.5-7B-Instruct"


def normalize_answer(s):
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def fuzzy_match(pred, golds, threshold=0.3):
    if not golds or not any(golds):
        return 0
    pn = normalize_answer(pred)
    if not pn:
        return 0
    for gold in golds:
        if not gold:
            continue
        gn = normalize_answer(gold)
        if not gn:
            continue
        if gn == pn or gn in pn:
            return 1
        gt = set(gn.split())
        if gt and len(set(pn.split()) & gt) / len(gt) >= threshold:
            return 1
    return 0


def extract_short_answer(text, max_words=10):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return text[:100]
    first = lines[0].split(".")[0].strip()
    return " ".join(first.split()[:max_words])


NQ_FEWSHOT = (
    "Example 1:\nQuestion: who wrote the song wild thing\nAnswer: Chip Taylor\n\n"
    "Example 2:\nQuestion: when did the boston marathon start\nAnswer: 1897\n\n"
    "Example 3:\nQuestion: where is the grand canyon located\nAnswer: Arizona, United States\n\n"
    "Example 4:\nQuestion: how many sides does a pentagon have\nAnswer: five\n\n"
)


def build_prompt(context, question):
    system = ("Answer the question based ONLY on the context. Give a VERY SHORT answer "
              "(a few words or numbers). Never explain.")
    return [{"role": "system", "content": system},
            {"role": "user", "content": NQ_FEWSHOT + f"Context:\n{context}\n\nQuestion: {question}"}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_queries", type=int, default=2000)
    ap.add_argument("--index", default="data/wiki_corpus/bm25_index.pkl")
    ap.add_argument("--nq_qa", default="data/nq_qa.json", help="nq_open q+a json (offline source)")
    ap.add_argument("--output", default="nq_conditional_2k.jsonl")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--start_idx", type=int, default=0)
    args = ap.parse_args()

    print(f"[*] loading BM25 index {args.index} ...", flush=True)
    bm25, passages, _ = load_bm25_index(args.index)
    if bm25 is None:
        print("ERROR: BM25 index not found", flush=True)
        return

    print("[*] loading cross-encoder + Qwen (BF16)...", flush=True)
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
    dev = model.device

    print(f"[*] loading NQ q+a from {args.nq_qa} (offline source)...", flush=True)
    qa = json.load(open(args.nq_qa))
    queries = []
    for i, x in enumerate(qa[args.start_idx:]):
        if len(queries) >= args.n_queries:
            break
        golds = [g.lower().strip() for g in x.get("answers", []) if g]
        if not golds:                      # skip unanswerable
            continue
        queries.append({"query_id": f"nq_{args.start_idx + i}",
                        "question": x["question"], "golds": golds})
    print(f"[*] {len(queries)} answerable NQ queries", flush=True)

    fout = open(args.output, "a", encoding="utf-8")
    t0 = time.time()
    stage_correct = [0, 0, 0, 0]
    done = 0
    for q in queries:
        qid, question, golds = q["query_id"], q["question"], q["golds"]
        hits = retrieve_bm25(question, bm25, passages, top_k=20)
        if not hits:
            continue
        scores = ce.predict([(question, h["text"]) for h in hits], show_progress_bar=False)
        ranked = [h for h, _ in sorted(zip(hits, scores), key=lambda x: -x[1])][:10]
        for stage_idx, k in enumerate(STAGE_SIZES):
            context = "\n\n".join(h["text"] for h in ranked[:k])[:2000]
            prompt = tok.apply_chat_template(build_prompt(context, question),
                                             tokenize=False, add_generation_prompt=True)
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(dev)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
                hs = torch.cat([out.hidden_states[-l][0, -1, :].float().cpu()
                                for l in range(1, 5)]).numpy()
                gen = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            ans = tok.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            short = extract_short_answer(ans)
            correct = fuzzy_match(short, golds)
            stage_correct[stage_idx] += correct
            fout.write(json.dumps({
                "query_id": qid, "question": question, "stage_idx": stage_idx,
                "k_passages": k, "multi_layer_hidden_states": hs.tolist(),
                "hidden_dim": int(len(hs)), "stage_answer": short,
                "gold_answer": " | ".join(golds[:3]), "stage_correctness": int(correct),
                "model": "Qwen/Qwen2.5-7B-Instruct", "precision": "bf16"}) + "\n")
        fout.flush()
        done += 1
        if done % 50 == 0:
            el = time.time() - t0
            accs = [c / max(done, 1) for c in stage_correct]
            print(f"  [{done}/{len(queries)}] {el/60:.1f}min {done/el*3600:.0f}q/hr "
                  f"ETA {(len(queries)-done)/max(done,1)*el/3600:.1f}h | "
                  f"acc S0-3 {accs[0]:.3f}/{accs[1]:.3f}/{accs[2]:.3f}/{accs[3]:.3f}", flush=True)
    fout.close()
    print(f"[*] DONE {done} queries in {(time.time()-t0)/3600:.2f}h -> {args.output}", flush=True)
    print("[*] per-stage acc: " + " ".join(f"S{i}={c/max(done,1):.3f}" for i, c in enumerate(stage_correct)), flush=True)


if __name__ == "__main__":
    main()
