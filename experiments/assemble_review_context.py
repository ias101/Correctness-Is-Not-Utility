#!/usr/bin/env python
"""
Assemble a full review prompt for the Gemini cross-model reviewer.

Bundles, in order:
  1. the per-round instruction file (--instr): the 9/10 bar, what changed this
     round, and the required output format,
  2. the reviewer's persistent memory (REVIEWER_MEMORY.md),
  3. a fixed set of context files (the full paper sections, the canonical result
     JSON artifacts, the scripts that generate them, provenance, and claims),

so the reviewer can verify every headline number against the actual artifacts in
a single call. Re-reads all files fresh each round (picks up paper edits).

Usage:
  python assemble_review_context.py --instr INSTR.txt --out PROMPT.txt
"""
import argparse, os

ap = argparse.ArgumentParser()
ap.add_argument("--instr", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--root", default=".")
args = ap.parse_args()

# Fixed context bundle: paper text + the artifacts/scripts behind the headline numbers.
CONTEXT_FILES = [
    "paper/sections/0_abstract.tex",
    "paper/sections/1_introduction.tex",
    "paper/sections/2_related_work.tex",
    "paper/sections/3_method.tex",
    "paper/sections/4_experiments.tex",
    "paper/sections/5_analysis.tex",
    "paper/sections/6_conclusion.tex",
    "paper/sections/A_appendix.tex",
    "paper/figures/latex_includes.tex",
    "results/v5_experiment/PROVENANCE.md",
    "results/v5_experiment/canonical_v5_full.json",
    "results/v5_experiment/conditional_delta_v5.json",
    "results/v5_experiment/lr_crosseval_addendum.json",
    "results/v5_experiment/conditional_delta_triviaqa.json",   # second regime (if present)
    "experiments/canonical_v5_full.py",
    "experiments/conditional_delta_v5.py",
    "experiments/lr_crosseval_addendum.py",
    "experiments/conditional_delta_multi.py",                   # second-regime script
    "CLAIMS_FROM_RESULTS.md",
]

parts = []
instr = open(args.instr, encoding="utf-8").read()
parts.append(instr)

mem_path = os.path.join(args.root, "REVIEWER_MEMORY.md")
if os.path.exists(mem_path):
    parts.append("\n\n" + "=" * 78 +
                 "\n## YOUR REVIEWER MEMORY (persistent across rounds)\n" +
                 "=" * 78 + "\n" + open(mem_path, encoding="utf-8").read())

parts.append("\n\n" + "=" * 78 +
             "\n## EMBEDDED CONTEXT (verify every number against these)\n" +
             "=" * 78 + "\n")

included, missing = [], []
for rel in CONTEXT_FILES:
    p = os.path.join(args.root, rel)
    if os.path.exists(p):
        body = open(p, encoding="utf-8", errors="replace").read()
        parts.append(f"\n\n----- FILE: {rel} -----\n{body}")
        included.append(rel)
    else:
        missing.append(rel)

if missing:
    parts.append("\n\n----- NOTE: the following referenced files do not yet exist "
                 "(work-in-progress): " + ", ".join(missing) + " -----\n")

out = "".join(parts)
with open(args.out, "w", encoding="utf-8") as f:
    f.write(out)

print(f"Assembled {len(out)} chars (~{len(out)//4} tokens est.)")
print(f"Included {len(included)} files; missing {len(missing)}: {missing}")
