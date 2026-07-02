"""
CAUSAL PopQA retrieval-degradation experiment (Loop 36 centerpiece).

Tests the self-knowledge-share theory CAUSALLY:
  benefit B = [self-knowledge: model lacks fact? -- IN h]  x  [retrieval: will it be supplied? -- NOT in h]
We MANIPULATE the retrieval term (reliability) and predict router gain drops MONOTONICALLY.

Design (5 stages; S0 = closed-book = pure self-knowledge):
  S0 : 0 passages (parametric only)            -> self-knowledge signal s_i, predictable from h_S0
  S1..S4 : progressively MORE passages.
  Retrieval reliability knob p in {0,.25,.5,.75,1}:
     per query, draw r_i ~ Bernoulli(1-p). If reliable, a GOLD-fact passage is among the
     shown passages at S1+ (plus noise fillers); if not, ALL shown passages are random noise.
  => benefit B_i = (1-s_i) * r_i * (model uses passage). At p=0 B governed by self-knowledge
     (routable); as p->1 B governed by the random draw r_i (not in h) -> routability falls.

Reliable passage: prefer a real corpus passage CONTAINING the gold string; else a
naturalistic templated fact sentence. Noise: random corpus passages (seeded).

Collects, per (query, p): h_S0 (closed-book, collected once), and per-stage correctness +
the realized reliability draw. Saves to degrade_popqa_causal.jsonl.

Offline: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1.
"""
import json, gzip, re, os, sys, argparse, random
from collections import OrderedDict
import numpy as np, torch

P_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
STAGE_NPASS = [0, 1, 2, 3, 4]          # S0 closed-book, then progressive
CONTEXT_MAX = 2048
SEED = 42


def norm(s):
    s = s.lower().strip(); s = re.sub(r"[^\w\s]", " ", s); return re.sub(r"\s+", " ", s).strip()


def load_queries(path, limit=None):
    seen = OrderedDict()
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            q = d["query_id"]
            if q not in seen:
                seen[q] = {"qid": q, "q": d["question"], "gold": d["gold"],
                           "subj": d["subj"], "pop": d.get("popularity")}
    qs = list(seen.values())
    if limit:
        qs = qs[:limit]
    return qs


def build_reliable_passages(queries, passages_lc, passages_raw):
    """For each query, find a corpus passage containing the gold string; else templated."""
    rel = {}
    for ex in queries:
        g = norm(ex["gold"]); subj = norm(ex["subj"])
        found = None
        if len(g) >= 3:
            for i, pt in enumerate(passages_lc):
                if g in pt and (subj.split()[0] in pt if subj else True):
                    found = passages_raw[i]["text"][:400]; break
        if found is None:
            # naturalistic templated fact (controlled stimulus)
            found = f"{ex['subj']}: {ex['q'].rstrip('?')} is {ex['gold']}."
        rel[ex["qid"]] = found
    return rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/popqa_v4_500q_states.jsonl.gz")
    ap.add_argument("--corpus", default="data/wiki_500k/passages.jsonl")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--out", default="degrade_popqa_causal.jsonl")
    args = ap.parse_args()

    random.seed(SEED); np.random.seed(SEED)
    queries = load_queries(args.data, args.limit)
    print(f"[*] {len(queries)} queries", flush=True)

    passages_raw = [json.loads(l) for l in open(args.corpus)]
    passages_lc = [norm(p["text"]) for p in passages_raw]
    print(f"[*] corpus {len(passages_raw)} passages", flush=True)

    rel = build_reliable_passages(queries, passages_lc, passages_raw)
    n_templated = sum(1 for ex in queries if rel[ex["qid"]].startswith(ex["subj"] + ":"))
    print(f"[*] reliable passages: {len(queries)-n_templated} natural / {n_templated} templated", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", dtype=torch.bfloat16, device_map="cuda:0").eval()
    print("[*] model loaded", flush=True)

    NOISE_POOL = list(range(len(passages_raw)))

    def gen_and_hs(ctx, question, want_hs):
        if ctx:
            prompt = (f"<|im_start|>user\nContext:\n{ctx}\n"
                      f"Question: {question}\n\nAnswer with just the name or short phrase.<|im_end|>\n"
                      f"<|im_start|>assistant\n")
        else:
            prompt = (f"<|im_start|>user\nQuestion: {question}\n\n"
                      f"Answer with just the name or short phrase.<|im_end|>\n"
                      f"<|im_start|>assistant\n")
        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=CONTEXT_MAX).to("cuda")
        with torch.no_grad():
            hs = None
            if want_hs:
                mo = model(**inp, output_hidden_states=True)
                hs = np.concatenate([mo.hidden_states[li][0, -1, :].cpu().float().numpy()
                                     for li in [-4, -3, -2, -1]])
            gen = model.generate(**inp, max_new_tokens=32, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            ans = tok.decode(gen[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
        return ans, hs

    fout = open(args.out, "w", encoding="utf-8")
    for qi, ex in enumerate(queries):
        g = norm(ex["gold"])
        # S0 closed-book ONCE (independent of p): self-knowledge
        a0, hs0 = gen_and_hs("", ex["q"], want_hs=True)
        s0_correct = int(g in norm(a0) or norm(a0) in g and len(norm(a0)) > 0)
        rec0 = {"qid": ex["qid"], "stage": 0, "p": None, "reliable": None,
                "n_pass": 0, "correct": s0_correct, "answer": a0,
                "hs_concat": hs0.tolist(), "hs_dim": len(hs0),
                "pop": ex["pop"], "closed_book": 1}
        fout.write(json.dumps(rec0) + "\n")

        for p in P_GRID:
            reliable = 1 if random.random() >= p else 0   # Bernoulli(1-p)
            # fixed noise passages for this query/p
            noise_idx = random.sample(NOISE_POOL, 4)
            noise_psg = [passages_raw[i]["text"][:300] for i in noise_idx]
            for st in range(1, 5):
                npass = STAGE_NPASS[st]
                slots = []
                if reliable:
                    slots.append(rel[ex["qid"]])           # gold fact passage first
                    slots += noise_psg[: max(0, npass - 1)]
                else:
                    slots += noise_psg[:npass]
                ctx = "".join(f"[Passage]: {s}\n\n" for s in slots)
                ans, _ = gen_and_hs(ctx, ex["q"], want_hs=False)
                cor = int(g in norm(ans) or (len(norm(ans)) > 0 and norm(ans) in g))
                fout.write(json.dumps({
                    "qid": ex["qid"], "stage": st, "p": p, "reliable": reliable,
                    "n_pass": npass, "correct": cor, "answer": ans,
                    "pop": ex["pop"], "closed_book": 0}) + "\n")
        fout.flush()
        if (qi + 1) % 25 == 0:
            print(f"  [{qi+1}/{len(queries)}] s0_acc-running", flush=True)
    fout.close()
    print("[*] DONE ->", args.out, flush=True)


if __name__ == "__main__":
    main()
