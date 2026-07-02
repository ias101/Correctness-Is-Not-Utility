"""
Two-tier router analysis (Loop 37, LOCAL): build policies, cost-performance (Pareto)
curves, bootstrap CIs, and the figure from the per-query predictions dumped by
experiments/two_tier_router.py (.npz). No GPU; iterate the gate/cost model freely.

Cost model (decomposed; GEN/CRIT from experiments/cost_microbench.py on Qwen2.5-7B):
  answer-returning @stop s : READ[s] + GEN          (static / query-only / passive Tier-1)
  FLARE @stop s            : READ[s] + (s+1)*GEN     (generate to assess confidence each stage)
  Self-RAG @stop s         : READ[s] + (s+1)*(GEN+CRIT)  (generate + critique each stage)

Two-tier: escalate to Tier-2 the queries the cheap tier is predicted to get wrong
  (lowest gate_score = OOF P(Tier-1 correct | h0)); sweep the escalation fraction to
  trace the frontier between always-passive (f=0) and always-Self-RAG (f=1).

  python experiments/analyze_two_tier.py \
      --popqa review-stage/two_tier_popqa_perq.npz \
      --hotpot review-stage/two_tier_hotpotqa_perq.npz \
      --out review-stage/two_tier_analysis.json
"""
import argparse, json, os
import numpy as np

SEED = 42
N_BOOT = 5000
READ = np.array([0.25, 0.50, 0.75, 1.08])
GEN_DEFAULT = 3.3221
CRIT_DEFAULT = 0.4153


def cost_of(stop, kind, gen, crit):
    rc = READ[stop]
    if kind in ("answer", "passive"):
        return rc + gen
    if kind == "flare":
        return rc + (stop + 1) * gen
    if kind == "selfrag":
        return rc + (stop + 1) * (gen + crit)
    raise ValueError(kind)


def maxcost(S, gen, crit):
    return float(READ[S - 1] + S * (gen + crit))


def stop_first_ge(P, tau, S):
    Q = P.shape[0]
    return np.array([next((s for s in range(S) if P[q, s] >= tau), S - 1) for q in range(Q)])


def boot_diff(a, b, rng, n_boot=N_BOOT):
    """mean(a-b) with query-bootstrap CI (a,b per-query aligned)."""
    d = a - b; n = len(d)
    idx = rng.randint(0, n, (n_boot, n))
    means = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


class Pol:
    """A policy operating point: per-query stop, correctness, cost."""
    def __init__(self, name, stop, C, kind, gen, crit, tag=None, extra=None):
        self.name = name; self.tag = tag; self.kind = kind
        self.stop = stop
        self.corr = C[np.arange(len(stop)), stop].astype(float)
        self.cost = cost_of(stop, kind, gen, crit)
        self.acc = float(self.corr.mean()); self.mcost = float(self.cost.mean())
        self.extra = extra or {}

    def cwa(self, lam, mc):
        return self.corr - lam * self.cost / mc


def build(npz, gen, crit):
    d = np.load(npz, allow_pickle=True)
    C = d["C"].astype(int); Q, S = C.shape
    s_star = d["s_star"]; tier1_stop = d["tier1_stop"].astype(int)
    gate = d["gate_score"]; pcorr0 = d["pcorr0"]; p_post = d["p_post"]
    conf = d["conf"]; qo_stop = d["query_only_stop"].astype(int)
    have_conf = bool(d["have_conf"]); ds = str(d["dataset"])
    mc = maxcost(S, gen, crit)

    pols = {}
    # static (4) + query-only
    pols["static"] = [Pol(f"static_S{s}", np.full(Q, s), C, "answer", gen, crit, tag=f"S{s}")
                      for s in range(S)]
    pols["query_only"] = [Pol("query_only", qo_stop, C, "answer", gen, crit)]

    # passive Tier-1 curve: sweep correctness@S0 threshold (stop at S0 if model "knows",
    #   else run full pipeline) -- the retrieve-or-not passive router; + the argmax point.
    passive = []
    for tau in np.linspace(0, 1, 21):
        stop = np.where(pcorr0 >= tau, 0, S - 1)
        passive.append(Pol("passive_hs", stop, C, "passive", gen, crit, tag=round(float(tau), 3)))
    passive.append(Pol("passive_hs", tier1_stop, C, "passive", gen, crit, tag="argmax"))
    pols["passive_hs"] = passive

    # ── BASELINES that decide by GENERATING (per-stage decode) ──
    # FLARE: generation-confidence threshold sweep (needs gen-confidence)
    if have_conf and np.isfinite(conf).any():
        cf = np.nan_to_num(conf, nan=0.0)
        pols["flare"] = [Pol("flare", stop_first_ge(cf, tau, S), C, "flare", gen, crit,
                             tag=round(float(tau), 3)) for tau in np.linspace(0, 1, 21)]
    # Self-RAG: generates a candidate + reflection tokens at each retrieved stage.
    pols["selfrag"] = [Pol("selfrag", stop_first_ge(p_post, tau, S), C, "selfrag", gen, crit,
                           tag=round(float(tau), 3)) for tau in np.linspace(0, 1, 21)]

    # ── OUR Tier-2: post-retrieval verifier that PROBES the post-retrieval state h_s
    #    (available from the prefill, no per-stage decode) and generates the answer ONCE.
    #    Same decision signal as Self-RAG (post-retrieval), but one generation, not many.
    pols["post_retrieval_verifier"] = [
        Pol("post_retrieval_verifier", stop_first_ge(p_post, tau, S), C, "answer", gen, crit,
            tag=round(float(tau), 3)) for tau in np.linspace(0, 1, 21)]

    # Tier-2 operating point for escalated queries = verifier threshold maximizing accuracy
    sr_acc_best = max(pols["post_retrieval_verifier"], key=lambda p: p.acc)
    tier2_stop = sr_acc_best.stop

    # TWO-TIER: escalate the lowest-gate_score fraction f to Tier-2 (one generation each)
    order = np.argsort(gate)                         # ascending: worst Tier-1 first
    two_tier = []
    for f in np.linspace(0, 1, 21):
        n_esc = int(round(f * Q))
        esc = np.zeros(Q, dtype=bool); esc[order[:n_esc]] = True
        stop = np.where(esc, tier2_stop, tier1_stop)
        corr = C[np.arange(Q), stop].astype(float)
        # both tiers decode ONCE; Tier-2 just reads deeper (post-retrieval) states.
        cost = cost_of(stop, "passive", gen, crit)
        p = Pol.__new__(Pol)
        p.name = "two_tier"; p.tag = round(float(f), 3); p.kind = "mixed"; p.stop = stop
        p.corr = corr; p.cost = cost; p.acc = float(corr.mean()); p.mcost = float(cost.mean())
        p.extra = {"frac_escalated": round(float(esc.mean()), 3)}
        two_tier.append(p)
    pols["two_tier"] = two_tier
    return ds, Q, S, mc, pols, sr_acc_best


def matched(tt_pts, ref, rng):
    """two-tier vs a reference baseline operating point (matched acc / matched cost)."""
    out = {"ref_name": ref.name, "ref_tag": ref.tag, "ref_acc": round(ref.acc, 4),
           "ref_cost": round(ref.mcost, 4)}
    # matched accuracy: cheapest two-tier point with acc >= ref.acc
    feas = [p for p in tt_pts if p.acc >= ref.acc - 1e-9]
    if feas:
        tt = min(feas, key=lambda p: p.mcost)
        g, lo, hi = boot_diff(ref.cost, tt.cost, rng)   # cost saved (ref - tt)
        out["matched_acc"] = {"tt_acc": round(tt.acc, 4), "tt_cost": round(tt.mcost, 4),
                              "cost_saved": round(g, 4), "ci": [round(lo, 4), round(hi, 4)],
                              "cost_saved_pct": round(100 * g / ref.mcost, 1),
                              "frac_escalated": tt.extra.get("frac_escalated")}
    # matched cost: highest-acc two-tier point with cost <= ref.cost
    feas = [p for p in tt_pts if p.mcost <= ref.mcost + 1e-9]
    if feas:
        tt = max(feas, key=lambda p: p.acc)
        g, lo, hi = boot_diff(tt.corr, ref.corr, rng)   # acc gain (tt - ref)
        out["matched_cost"] = {"tt_acc": round(tt.acc, 4), "tt_cost": round(tt.mcost, 4),
                               "acc_gain": round(g, 4), "ci": [round(lo, 4), round(hi, 4)],
                               "frac_escalated": tt.extra.get("frac_escalated")}
    return out


def analyze(npz, gen, crit):
    rng = np.random.RandomState(SEED)
    ds, Q, S, mc, pols, sr_acc_best = build(npz, gen, crit)
    res = {"dataset": ds, "n_queries": Q, "maxcost": round(mc, 4),
           "gen_cost": gen, "crit_cost": crit}

    def best_cwa(pts, lam):
        return max(pts, key=lambda p: p.acc - lam * p.mcost / mc)

    # headline: two-tier vs always-Self-RAG (max-acc) and vs always-FLARE (max-acc)
    tt = pols["two_tier"]
    res["vs_selfrag"] = matched(tt, max(pols["selfrag"], key=lambda p: p.acc), rng)
    if "flare" in pols:
        res["vs_flare"] = matched(tt, max(pols["flare"], key=lambda p: p.acc), rng)
    # also vs Self-RAG at its best CWA(0.5) (a balanced deployment)
    res["vs_selfrag_balanced"] = matched(tt, best_cwa(pols["selfrag"], 0.5), rng)

    # CWA table at lambda grid: best-operating-point per method + CI of two-tier gaps
    cwa_tab = {}
    for lam in (0.25, 0.5, 0.8):
        ttb = best_cwa(tt, lam); srb = best_cwa(pols["selfrag"], lam)
        bsb = best_cwa(pols["static"], lam); qob = best_cwa(pols["query_only"], lam)
        row = {"two_tier": round(ttb.acc - lam * ttb.mcost / mc, 4),
               "selfrag": round(srb.acc - lam * srb.mcost / mc, 4),
               "best_static": round(bsb.acc - lam * bsb.mcost / mc, 4),
               "query_only": round(qob.acc - lam * qob.mcost / mc, 4),
               "frac_escalated_at_tt_opt": ttb.extra.get("frac_escalated")}
        for nm, b in (("vs_selfrag", srb), ("vs_best_static", bsb)):
            g, lo, hi = boot_diff(ttb.cwa(lam, mc), b.cwa(lam, mc), rng)
            row[nm] = {"gap": round(g, 4), "ci": [round(lo, 4), round(hi, 4)]}
        if "flare" in pols:
            fb = best_cwa(pols["flare"], lam)
            row["flare"] = round(fb.acc - lam * fb.mcost / mc, 4)
            g, lo, hi = boot_diff(ttb.cwa(lam, mc), fb.cwa(lam, mc), rng)
            row["vs_flare"] = {"gap": round(g, 4), "ci": [round(lo, 4), round(hi, 4)]}
        cwa_tab[f"lam_{lam}"] = row
    res["cwa_table"] = cwa_tab

    # gate value: does ESCALATION help? compare two-tier(best CWA0.5) vs all-Tier-1 (f=0)
    ttb = best_cwa(tt, 0.5); t1 = tt[0]  # f=0
    g, lo, hi = boot_diff(ttb.cwa(0.5, mc), t1.cwa(0.5, mc), rng)
    res["escalation_value"] = {"frac_escalated": ttb.extra.get("frac_escalated"),
                               "cwa_gain_vs_all_tier1": round(g, 4), "ci": [round(lo, 4), round(hi, 4)],
                               "all_tier1_acc": round(t1.acc, 4), "all_tier1_cost": round(t1.mcost, 4),
                               "tt_acc": round(ttb.acc, 4), "tt_cost": round(ttb.mcost, 4)}

    # gate-threshold SENSITIVITY (R2): is the static-beating result robust to the escalation
    # cutoff, or cherry-picked? Report CWA(0.5) gap vs best-static at EVERY escalation fraction.
    bs05 = best_cwa(pols["static"], 0.5)
    sens = []
    n_pos = 0
    for p in tt:
        g2, lo2, hi2 = boot_diff(p.cwa(0.5, mc), bs05.cwa(0.5, mc), rng)
        sig = lo2 > 0
        n_pos += int(sig)
        sens.append({"frac_esc": p.extra.get("frac_escalated"), "gap_vs_static": round(g2, 4),
                     "ci": [round(lo2, 4), round(hi2, 4)], "ci_excludes_0": bool(sig)})
    gaps = [s["gap_vs_static"] for s in sens]
    res["gate_sensitivity"] = {
        "lambda": 0.5, "n_cutoffs": len(sens),
        "n_cutoffs_beating_static_ci": n_pos,
        "gap_min": round(min(gaps), 4), "gap_max": round(max(gaps), 4),
        "gap_at_f": sens,
        "note": "CWA(0.5)[two-tier@f] - CWA(0.5)[best static]; CI = query-grouped bootstrap."}

    # curves for the figure
    res["curves"] = {n: [{"tag": p.tag, "acc": round(p.acc, 4), "cost": round(p.mcost, 4),
                          **({"frac_escalated": p.extra["frac_escalated"]} if "frac_escalated" in p.extra else {})}
                         for p in pts] for n, pts in pols.items()}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="review-stage/two_tier_analysis.json")
    ap.add_argument("--gen_cost", type=float, default=GEN_DEFAULT)
    ap.add_argument("--crit_cost", type=float, default=CRIT_DEFAULT)
    args = ap.parse_args()
    DATASETS = [("PopQA", "review-stage/two_tier_popqa_perq.npz"),
                ("HotpotQA", "review-stage/two_tier_hotpotqa_perq.npz"),
                ("TriviaQA", "review-stage/two_tier_triviaqa_perq.npz"),
                ("NQ", "review-stage/two_tier_nq_perq.npz"),
                ("MistralHotpot", "review-stage/two_tier_mistral_hotpot_perq.npz")]
    out = {"_cost_model": {"READ": READ.tolist(), "GEN": args.gen_cost, "CRIT": args.crit_cost,
                           "source": "experiments/cost_microbench.py (Qwen2.5-7B, RTX 3080)"}}
    for key, path in DATASETS:
        if os.path.exists(path):
            out[key] = analyze(path, args.gen_cost, args.crit_cost)
            r = out[key]
            print(f"\n##### {key} (n={r['n_queries']}) #####")
            print(f"  vs always-Self-RAG (max-acc {r['vs_selfrag']['ref_acc']}@{r['vs_selfrag']['ref_cost']}):")
            ma = r["vs_selfrag"].get("matched_acc", {})
            print(f"    matched-acc: two-tier {ma.get('tt_acc')}@{ma.get('tt_cost')} "
                  f"=> cost -{ma.get('cost_saved_pct')}% (saved {ma.get('cost_saved')} CI{ma.get('ci')}), "
                  f"escalated={ma.get('frac_escalated')}")
            print(f"  escalation_value: {r['escalation_value']}")
            for lam, row in r["cwa_table"].items():
                print(f"  {lam}: tt={row['two_tier']} sr={row['selfrag']} static={row['best_static']} "
                      f"| tt-sr={row['vs_selfrag']['gap']:+}{row['vs_selfrag']['ci']} "
                      f"tt-static={row['vs_best_static']['gap']:+}{row['vs_best_static']['ci']}")
        else:
            print(f"[skip] {path} missing")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n[*] -> {args.out}")


if __name__ == "__main__":
    main()
