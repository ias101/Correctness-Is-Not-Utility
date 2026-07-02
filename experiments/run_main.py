#!/usr/bin/env python3
"""
Block 2: Main Experiment — Orchestrator.

Runs the full experiment pipeline:
  2.1 Full data collection (NQ 3610 + HotpotQA 7405 + TriviaQA 1000)
  2.2 Train MLP predictor (3 seeds)
  2.3 Platt scaling calibration
  2.4 Grid search tau
  2.5 Test set evaluation with all baselines

Output: results/block2_main/main_results.json
Gate: Cost-weighted accuracy > best baseline + 2 points → PROCEED to Block 3-5
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from config import (
    DATA_DIR,
    RESULTS_DIR,
    MODEL_DIR,
    NQ_FULL_QUERIES,
    HOTPOTQA_QUERIES,
    TRIVIAQA_QUERIES,
    LLAMA_MODEL_NAME,
    SEEDS,
    BASE_SEED,
)

BLOCK2_DIR = os.path.join(RESULTS_DIR, "block2_main")


def run_step(name: str, cmd: list) -> int:
    """Run a subprocess step, print output, return exit code."""
    print(f"\n{'='*60}")
    print(f"  Block 2 Step: {name}")
    print(f"  {' '.join(cmd[:5])}...")
    print(f"{'='*60}")
    start = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start
    if result.returncode == 0:
        print(f"\n✅ {name} completed in {elapsed:.0f}s")
    else:
        print(f"\n❌ {name} FAILED (exit code {result.returncode})")
    return result.returncode


def _script_path(name: str) -> str:
    return os.path.join(_SCRIPT_DIR, name)


def main():
    parser = argparse.ArgumentParser(
        description="Block 2: Main Experiment for Hazard-Based Early Stopping"
    )
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["nq", "hotpotqa", "triviaqa", "all"])
    parser.add_argument("--nq_queries", type=int, default=NQ_FULL_QUERIES)
    parser.add_argument("--hotpotqa_queries", type=int, default=HOTPOTQA_QUERIES)
    parser.add_argument("--triviaqa_queries", type=int, default=TRIVIAQA_QUERIES)
    parser.add_argument("--skip_collection", action="store_true")
    parser.add_argument("--model_name", type=str, default=LLAMA_MODEL_NAME)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--corpus_size", type=int, default=200000)
    args = parser.parse_args()

    os.makedirs(BLOCK2_DIR, exist_ok=True)

    results = {
        "block": "2",
        "title": "Main Experiment",
        "date": datetime.now().isoformat(),
        "datasets": {},
        "gate": {"passed": False, "cwa_gain": 0.0, "threshold": 0.02},
    }

    datasets_to_run = []
    if args.dataset == "all":
        datasets_to_run = [
            ("nq", args.nq_queries),
            ("hotpotqa", args.hotpotqa_queries),
            ("triviaqa", args.triviaqa_queries),
        ]
    else:
        q = getattr(args, f"{args.dataset}_queries", 1000)
        datasets_to_run = [(args.dataset, q)]

    # ═══════════════════════════════════════════════════════════════
    # Step 2.1: Full Data Collection (each dataset)
    # ═══════════════════════════════════════════════════════════════
    all_data_files = []

    for ds_name, num_q in datasets_to_run:
        data_file = os.path.join(DATA_DIR, f"collected_states_{ds_name}.jsonl")

        if not args.skip_collection:
            rc = run_step(f"2.1 Collect {ds_name.upper()} ({num_q} queries)", [
                sys.executable, _script_path("collect_states.py"),
                "--num_queries", str(num_q),
                "--dataset", ds_name,
                "--model_name", args.model_name,
                "--output", data_file,
                "--seed", str(args.seed),
                "--real_retrieval",
                "--corpus_size", str(args.corpus_size),
            ])

            if rc != 0:
                print(f"\n❌ Data collection for {ds_name} failed.")
                results["datasets"][ds_name] = {"status": "FAILED"}
                continue

        all_data_files.append((ds_name, data_file))
        results["datasets"][ds_name] = {"status": "DONE", "file": data_file}

    if not all_data_files:
        print("\n❌ No data collected. Aborting Block 2.")
        _save_results(results)
        sys.exit(1)

    # ═══════════════════════════════════════════════════════════════
    # Step 2.2-2.3: Train MLP (3 seeds) + Platt Calibration
    # ═══════════════════════════════════════════════════════════════
    mlp_results = {}
    for ds_name, data_file in all_data_files:
        ds_results = []
        for seed in SEEDS:
            rc = run_step(f"2.2 MLP {ds_name} seed={seed}", [
                sys.executable, _script_path("train_predictor.py"),
                "--data", data_file,
                "--variant", "mlp",
                "--seed", str(seed),
                "--output_dir", BLOCK2_DIR,
            ])
            if rc == 0:
                ds_results.append({"seed": seed, "status": "DONE"})
            else:
                ds_results.append({"seed": seed, "status": "FAILED"})
        mlp_results[ds_name] = ds_results

    results["training"] = mlp_results

    # ═══════════════════════════════════════════════════════════════
    # Step 2.5: Evaluate (all baselines + tau sweep)
    # ═══════════════════════════════════════════════════════════════
    for ds_name, data_file in all_data_files:
        model_path = os.path.join(MODEL_DIR, f"predictor_mlp_seed{SEEDS[0]}.pt")
        if not os.path.exists(model_path):
            continue

        eval_dir = os.path.join(BLOCK2_DIR, ds_name)
        os.makedirs(eval_dir, exist_ok=True)

        rc = run_step(f"2.5 Evaluate {ds_name}", [
            sys.executable, _script_path("evaluate.py"),
            "--data", data_file,
            "--model", model_path,
            "--variant", "mlp",
            "--hidden_dim", "3584",  # Qwen2.5-7B
            "--output_dir", eval_dir,
            "--seed", str(args.seed),
        ])

        if rc == 0:
            eval_file = os.path.join(eval_dir, "evaluation_results.json")
            if os.path.exists(eval_file):
                with open(eval_file) as f:
                    eval_data = json.load(f)
                results["datasets"][ds_name]["evaluation"] = eval_data

                # Check gate: our CWA > best baseline CWA + 0.02?
                ours_cwa = eval_data.get("ours", {}).get("cost_weighted_accuracy", {})
                best_baseline_cwa = -999
                for name, r in eval_data.items():
                    if name != "ours" and isinstance(r, dict):
                        cwa = r.get("cost_weighted_accuracy", {}).get("lambda_0.5", -999)
                        best_baseline_cwa = max(best_baseline_cwa, cwa)

                our_cwa = ours_cwa.get("lambda_0.5", -999)
                cwa_gain = our_cwa - best_baseline_cwa
                results["datasets"][ds_name]["cwa_gain"] = cwa_gain
                results["datasets"][ds_name]["our_cwa"] = our_cwa
                results["datasets"][ds_name]["best_baseline_cwa"] = best_baseline_cwa

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  BLOCK 2: MAIN EXPERIMENT COMPLETE")
    print("=" * 60)

    for ds_name, ds_info in results["datasets"].items():
        cwa_gain = ds_info.get("cwa_gain", 0)
        gate = cwa_gain > 0.02
        print(f"  {ds_name}: CWA gain={cwa_gain:+.4f} | Gate: {'✅' if gate else '❌'}")

    _save_results(results)
    return 0


def _save_results(results: dict):
    path = os.path.join(BLOCK2_DIR, "main_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n[*] Main results saved to: {path}")


if __name__ == "__main__":
    sys.exit(main())
