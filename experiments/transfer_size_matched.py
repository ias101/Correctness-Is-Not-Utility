#!/usr/bin/env python3
"""
Round-2 control: SIZE-MATCHED transfer.

Round 1 found asymmetric transfer (HotpotQA->PopQA MLP 0.775 vs PopQA->HotpotQA
0.632). HotpotQA trains on 2000 queries, PopQA on 500 -> training-set size is a
confound for the asymmetry. Here we subsample HotpotQA queries to 500 (= PopQA)
and re-run both directions. If the asymmetry persists with equal training size,
it is NOT a size artifact.

Reuses the exact probe/protocol from transfer_probe_ablation.py.
"""
import os, sys, json, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transfer_probe_ablation import (load_dataset, transfer, indomain_oof,
                                      SEED, N_STAGES)


def main():
    hotpot = "data/collected_states_hotpotqa_v5_2000.jsonl"
    popqa = "data/popqa_v4_500q_states.jsonl.gz"
    out = "results/transfer_ablation/transfer_size_matched.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    print(f"[*] device={device}")

    Xh, yh, sth, gh = load_dataset(hotpot)
    Xp, yp, stp, gp = load_dataset(popqa, align_popqa=True)

    # subsample HotpotQA to 500 distinct queries (match PopQA), seeded
    rng = np.random.default_rng(SEED)
    uq = np.unique(gh)
    n_target = len(np.unique(gp))                       # = 500
    sub = set(rng.choice(uq, size=n_target, replace=False).tolist())
    mask = np.array([g in sub for g in gh])
    Xh5, yh5, sth5, gh5 = Xh[mask], yh[mask], sth[mask], gh[mask]
    print(f"[*] HotpotQA full {len(np.unique(gh))}q -> subsampled {len(np.unique(gh5))}q "
          f"({len(yh5)} rows, pos {yh5.mean():.3f}); PopQA {len(np.unique(gp))}q ({len(yp)} rows, pos {yp.mean():.3f})")

    res = {"meta": {"hotpot_sub_queries": int(len(np.unique(gh5))),
                    "popqa_queries": int(len(np.unique(gp))),
                    "note": "training-set size matched at 500 queries; eval on full target"}}

    print("[*] in-domain HotpotQA-500 (OOF)..."); res["indomain_hotpot500"] = indomain_oof(Xh5, yh5, sth5, gh5, device)
    print("[*] transfer HotpotQA-500 -> PopQA (full)..."); res["transfer_hotpot500_to_popqa"] = transfer(Xh5, yh5, Xp, yp, stp, gp, device)
    print("[*] transfer PopQA-500 -> HotpotQA (full)..."); res["transfer_popqa_to_hotpot_full"] = transfer(Xp, yp, Xh, yh, sth, gh, device)

    res["meta"]["runtime_sec"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)

    def fmt(d):
        ps = ",".join(f"{d['per_stage'][t]:.3f}" for t in range(N_STAGES))
        return f"{d['pooled_auroc']:.3f} [{d['ci95'][0]:.3f},{d['ci95'][1]:.3f}] per-stage={ps} (n={d['n']})"

    print("\n===== SIZE-MATCHED RESULTS (correctness AUROC) =====")
    for k in ["indomain_hotpot500", "transfer_hotpot500_to_popqa", "transfer_popqa_to_hotpot_full"]:
        print(f"\n[{k}]")
        for mdl in ["mlp", "lr"]:
            print(f"  {mdl.upper()}: {fmt(res[k][mdl])}")
    print(f"\n[*] saved {out} (runtime {res['meta']['runtime_sec']}s)")


if __name__ == "__main__":
    main()
