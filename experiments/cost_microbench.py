"""
Cost micro-benchmark (Loop 37, R1 non-negotiable #1 / pitfall #3): defend the
two-tier Pareto cost ratios with measured Qwen2.5-7B wall-clock, not opaque constants.

Measures, on the deployment backbone:
  - prefill latency at the 4 stage context lengths (~ #passages): the READ cost.
  - per-token decode latency: GEN cost = decode * n_answer_tokens,
                              CRIT cost = decode * n_critique_tokens.

The two-tier advantage is that the passive tier decodes ONCE (final answer) while
Self-RAG/FLARE decode at EVERY visited stage; this benchmark quantifies how large
that per-stage decode cost is relative to prefill, so the cost model is auditable.

Outputs review-stage/cost_microbench.json with measured ms and normalized
READ/GEN/CRIT costs (READ rescaled to the paper's [0.25,0.50,0.75,1.08] for
continuity). Plug GEN_COST/CRIT_COST into two_tier_router.py via --gen_cost/--crit_cost.

  python experiments/cost_microbench.py --model Qwen/Qwen2.5-7B-Instruct
"""
import argparse, json, time, os
import numpy as np
import torch

STAGE_PASSAGES = [2, 4, 6, 8]      # S0..S3
TOK_PER_PASSAGE = 110              # ~typical retrieved chunk
TOK_QUESTION = 32
N_ANSWER_TOK = 32                  # a short QA answer
N_CRIT_TOK = 4                     # a Self-RAG-style [ISSUP] yes/no + token
PAPER_READ = np.array([0.25, 0.50, 0.75, 1.08])   # continuity target


def time_prefill(model, tok, n_ctx_tokens, device, reps=5):
    ids = torch.randint(0, tok.vocab_size, (1, n_ctx_tokens), device=device)
    with torch.no_grad():
        for _ in range(2):                      # warmup
            model(ids); torch.cuda.synchronize()
        t = []
        for _ in range(reps):
            torch.cuda.synchronize(); s = time.perf_counter()
            model(ids); torch.cuda.synchronize()
            t.append(time.perf_counter() - s)
    return float(np.median(t))


def time_decode_per_token(model, tok, n_ctx_tokens, n_new, device, reps=3):
    ids = torch.randint(0, tok.vocab_size, (1, n_ctx_tokens), device=device)
    with torch.no_grad():
        for _ in range(1):
            model.generate(ids, max_new_tokens=8, do_sample=False,
                           pad_token_id=tok.eos_token_id); torch.cuda.synchronize()
        t = []
        for _ in range(reps):
            torch.cuda.synchronize(); s = time.perf_counter()
            model.generate(ids, max_new_tokens=n_new, do_sample=False,
                           pad_token_id=tok.eos_token_id)
            torch.cuda.synchronize()
            t.append(time.perf_counter() - s)
    # subtract one prefill to isolate decode; per-token = (gen_total - prefill)/n_new
    pf = time_prefill(model, tok, n_ctx_tokens, device, reps=3)
    per_tok = max(1e-6, (np.median(t) - pf) / n_new)
    return float(per_tok), float(pf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default="review-stage/cost_microbench.json")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda"
    print(f"loading {args.model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map=device)
    model.eval()

    ctx = [TOK_QUESTION + TOK_PER_PASSAGE * k for k in STAGE_PASSAGES]
    prefill_ms = [1000 * time_prefill(model, tok, n, device) for n in ctx]
    # decode/token measured at the mid context length
    per_tok, _ = time_decode_per_token(model, tok, ctx[2], 24, device)
    gen_ms = 1000 * per_tok * N_ANSWER_TOK
    crit_ms = 1000 * per_tok * N_CRIT_TOK

    # normalize READ to the paper scale (preserves continuity); express GEN/CRIT
    # in the SAME normalized units (relative to the stage-3 prefill anchor).
    scale = PAPER_READ[-1] / prefill_ms[-1]
    read_norm = [round(p * scale, 4) for p in prefill_ms]
    gen_norm = round(gen_ms * scale, 4)
    crit_norm = round(crit_ms * scale, 4)

    out = {
        "model": args.model, "device": torch.cuda.get_device_name(0),
        "context_tokens": ctx, "answer_tokens": N_ANSWER_TOK, "critique_tokens": N_CRIT_TOK,
        "measured_ms": {"prefill_per_stage": [round(p, 2) for p in prefill_ms],
                        "decode_per_token": round(1000 * per_tok, 3),
                        "generation": round(gen_ms, 2), "critique": round(crit_ms, 2)},
        "normalized_costs": {"READ_COST": read_norm, "GEN_COST": gen_norm,
                             "CRIT_COST": crit_norm,
                             "note": "READ rescaled to paper [0.25,0.5,0.75,1.08]; "
                                     "GEN/CRIT in same units via the stage-3 prefill anchor."},
        "ratios": {"gen_over_stage3_read": round(gen_ms / prefill_ms[-1], 3),
                   "gen_over_stage0_read": round(gen_ms / prefill_ms[0], 3),
                   "crit_over_gen": round(crit_ms / gen_ms, 3)},
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"\n[*] -> {args.out}")
    print(f"[*] Plug into router:  --gen_cost {gen_norm} --crit_cost {crit_norm}")


if __name__ == "__main__":
    main()
