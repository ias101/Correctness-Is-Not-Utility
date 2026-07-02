#!/usr/bin/env python3
"""
Block 1: Pilot Validation — Orchestrator.

Runs the full pilot pipeline:
  1.1 Collect hidden states from NQ (500 queries, LLaMA-8B, DPR)
  1.2 Train logistic regression baseline
  1.3 Train MLP predictor
  1.4 Compare AUROC, gate check

Output: results/block1_pilot/pilot_results.json
Gate: AUROC(Stage 1, retrieval) > 0.65 → PROCEED to Block 2
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATA_DIR,
    RESULTS_DIR,
    MODEL_DIR,
    PILOT_NQ_QUERIES,
    LLAMA_MODEL_NAME,
    BASE_SEED,
    SEEDS,
)

BLOCK1_DIR = os.path.join(RESULTS_DIR, "block1_pilot")


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _script_path(name: str) -> str:
    """Get full path to a script in the same directory."""
    return os.path.join(_SCRIPT_DIR, name)


def run_step(name: str, cmd: list) -> int:
    """Run a subprocess step, print output, return exit code."""
    print(f"\n{'='*60}")
    print(f"  Block 1 Step: {name}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}")

    start = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n✅ {name} completed in {elapsed:.0f}s")
    else:
        print(f"\n❌ {name} FAILED (exit code {result.returncode})")

    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Block 1: Pilot Validation for Hazard-Based Early Stopping"
    )
    parser.add_argument(
        "--num_queries", type=int, default=PILOT_NQ_QUERIES,
        help=f"Number of NQ queries for pilot (default: {PILOT_NQ_QUERIES})"
    )
    parser.add_argument(
        "--model_name", type=str, default=LLAMA_MODEL_NAME,
        help="LLaMA model name"
    )
    parser.add_argument(
        "--seed", type=int, default=BASE_SEED,
    )
    parser.add_argument(
        "--skip_collection", action="store_true",
        help="Skip data collection (use existing data)"
    )
    parser.add_argument(
        "--no_4bit", action="store_true",
        help="Disable 4-bit quantization"
    )
    args = parser.parse_args()

    os.makedirs(BLOCK1_DIR, exist_ok=True)

    results = {
        "block": "1",
        "title": "Pilot Validation",
        "date": datetime.now().isoformat(),
        "steps": {},
        "gate": {"passed": False, "auroc": 0.0, "threshold": 0.65},
    }

    data_file = os.path.join(DATA_DIR, "collected_states.jsonl")

    # ═══════════════════════════════════════════════════════════════
    # Step 1.1: Collect hidden states
    # ═══════════════════════════════════════════════════════════════
    if not args.skip_collection:
        rc = run_step("1.1 Collect Hidden States", [
            sys.executable, _script_path("collect_states.py"),
            "--num_queries", str(args.num_queries),
            "--model_name", args.model_name,
            "--output", data_file,
            "--seed", str(args.seed),
        ] + (["--no_4bit"] if args.no_4bit else []))

        if rc != 0:
            print("\n❌ Data collection failed. Aborting Block 1.")
            results["status"] = "FAILED"
            _save_results(results)
            sys.exit(1)

        results["steps"]["1.1_collect"] = {
            "status": "DONE",
            "output": data_file,
            "num_queries": args.num_queries,
        }
    else:
        print(f"[*] Skipping collection, using existing data: {data_file}")
        results["steps"]["1.1_collect"] = {"status": "SKIPPED"}

    # ═══════════════════════════════════════════════════════════════
    # Step 1.2: Logistic Regression baseline
    # ═══════════════════════════════════════════════════════════════
    rc = run_step("1.2 Logistic Regression", [
        sys.executable, _script_path("train_predictor.py"),
        "--data", data_file,
        "--variant", "logistic",
        "--seed", str(args.seed),
        "--output_dir", BLOCK1_DIR,
    ])

    lr_results_path = os.path.join(BLOCK1_DIR, "logistic_regression_results.json")
    lr_auroc = None
    if rc == 0 and os.path.exists(lr_results_path):
        with open(lr_results_path) as f:
            lr_results = json.load(f)
        lr_auroc = lr_results.get("val_auroc", 0)
        results["steps"]["1.2_logistic"] = {
            "status": "DONE",
            "val_auroc": lr_auroc,
        }
        print(f"    Logistic Regression AUROC: {lr_auroc:.4f}")
    else:
        results["steps"]["1.2_logistic"] = {"status": "FAILED"}

    # ═══════════════════════════════════════════════════════════════
    # Step 1.3: MLP Training
    # ═══════════════════════════════════════════════════════════════
    mlp_auroc = None
    for seed in SEEDS[:1]:  # Single seed for pilot
        rc = run_step(f"1.3 MLP (seed={seed})", [
            sys.executable, _script_path("train_predictor.py"),
            "--data", data_file,
            "--variant", "mlp",
            "--seed", str(seed),
            "--output_dir", BLOCK1_DIR,
        ])

        mlp_results_path = os.path.join(
            BLOCK1_DIR, f"predictor_mlp_seed{seed}_results.json"
        )
        if rc == 0 and os.path.exists(mlp_results_path):
            with open(mlp_results_path) as f:
                mlp_results = json.load(f)
            mlp_auroc = mlp_results.get("best_val_auroc", 0)
            results["steps"]["1.3_mlp"] = {
                "status": "DONE",
                "seed": seed,
                "val_auroc": mlp_auroc,
                "best_epoch": mlp_results.get("best_epoch", -1),
                "num_params": mlp_results.get("num_params", 0),
            }
            print(f"    MLP AUROC: {mlp_auroc:.4f}")
        else:
            results["steps"]["1.3_mlp"] = {"status": "FAILED", "seed": seed}

    # ═══════════════════════════════════════════════════════════════
    # Step 1.4: Evaluate + Gate Check
    # ═══════════════════════════════════════════════════════════════
    model_path = os.path.join(MODEL_DIR, f"predictor_mlp_seed{SEEDS[0]}.pt")
    if os.path.exists(model_path):
        rc = run_step("1.4 Evaluate + Gate Check", [
            sys.executable, _script_path("evaluate.py"),
            "--data", data_file,
            "--model", model_path,
            "--variant", "mlp",
            "--output_dir", BLOCK1_DIR,
            "--seed", str(args.seed),
        ])

        eval_results_path = os.path.join(BLOCK1_DIR, "evaluation_results.json")
        if rc == 0 and os.path.exists(eval_results_path):
            with open(eval_results_path) as f:
                eval_results = json.load(f)

            gate = eval_results.get("gate_check", {})
            results["steps"]["1.4_evaluate"] = {
                "status": "DONE",
                "gate_passed": gate.get("passed", False),
            }

            # Update gate check
            retrieval_auroc = gate.get("auroc", 0)
            results["gate"] = {
                "passed": gate.get("passed", False),
                "auroc": retrieval_auroc,
                "threshold": 0.65,
            }

        else:
            results["steps"]["1.4_evaluate"] = {"status": "FAILED"}
    else:
        print(f"\n[!] MLP model not found at {model_path}, computing AUROC manually...")
        # Try to compute AUROC from training results directly
        results["steps"]["1.4_evaluate"] = {
            "status": "SKIPPED",
            "reason": "Model file not found",
        }

        # Use validation AUROC from training as gate proxy
        if mlp_auroc is not None:
            results["gate"]["auroc"] = mlp_auroc
            results["gate"]["passed"] = mlp_auroc > 0.65
        elif lr_auroc is not None:
            results["gate"]["auroc"] = lr_auroc
            results["gate"]["passed"] = lr_auroc > 0.65

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    gate = results["gate"]
    results["status"] = "DONE" if gate["passed"] else "GATE_FAILED"

    print("\n" + "=" * 60)
    print("  BLOCK 1: PILOT VALIDATION COMPLETE")
    print("=" * 60)
    print(f"  Logistic Regression AUROC: {lr_auroc:.4f}" if lr_auroc else "  Logistic Regression: FAILED")
    print(f"  MLP AUROC: {mlp_auroc:.4f}" if mlp_auroc else "  MLP: FAILED")
    print(f"  Gate Check (AUROC > 0.65): {'✅ PASSED' if gate['passed'] else '❌ FAILED'} "
          f"(AUROC = {gate['auroc']:.4f})")

    if gate["passed"]:
        print("\n  ✅ GATE PASSED — Proceed to Block 2 (Main Experiment)")
        print("  → Run: python experiments/run_main.py")
    else:
        print("\n  ❌ GATE FAILED — Hidden states do not contain sufficient pre-failure signal.")
        print("     Per contingency plan: ABANDON project or investigate why signal is weak.")

    results["recommendation"] = (
        "PROCEED_TO_BLOCK2" if gate["passed"] else "ABANDON_OR_INVESTIGATE"
    )

    _save_results(results)
    return 0 if gate["passed"] else 1


def _save_results(results: dict):
    """Save pilot results to JSON."""
    path = os.path.join(BLOCK1_DIR, "pilot_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n[*] Pilot results saved to: {path}")


if __name__ == "__main__":
    sys.exit(main())
