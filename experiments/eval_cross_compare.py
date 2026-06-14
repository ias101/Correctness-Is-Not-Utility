"""
Multi-model + Multi-dataset evaluation pipeline.

After collections complete, run:
  python eval_cross_compare.py --output_dir results/cross_compare

This script loads all available datasets and models, trains predictors,
and produces comparison tables and cross-dataset transfer results.
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
from collections import defaultdict
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_predictor import train_mlp, calibrate_platt, HiddenStateDataset, split_data
from models import build_predictor
from config import LLAMA_HIDDEN_DIM, NUM_STAGES


def load_and_prepare(data_path, label_type="stage"):
    """Load JSONL data and split."""
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))

    train_data, val_data, test_data = split_data(data, seed=42)
    return train_data, val_data, test_data


def train_and_eval(train_data, val_data, test_data, label_type, output_dir, seed=42):
    """Train MLP and return key metrics."""
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, f"predictor_seed{seed}.pt")
    results_path = os.path.join(output_dir, f"train_results_seed{seed}.json")

    results = train_mlp(
        train_data, val_data, results_path, model_path,
        variant="mlp", label_type=label_type, seed=seed, num_epochs=50,
    )

    # Calibrate
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = build_predictor(
        variant="mlp", hidden_dim=LLAMA_HIDDEN_DIM,
        stage_emb_dim=checkpoint.get("stage_emb_dim", 16),
        num_stages=NUM_STAGES,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    calibrator, cal_results = calibrate_platt(
        model, val_data, variant="mlp", label_type=label_type,
    )

    # Per-stage AUROC on test set
    test_dataset = HiddenStateDataset(test_data, label_type=label_type)
    model.eval()
    stage_preds = defaultdict(list)
    stage_labels = defaultdict(list)

    with torch.no_grad():
        for i in range(len(test_dataset)):
            item = test_dataset[i]
            hs = item["hidden_state"].unsqueeze(0).to(model.device)
            si = item["stage_idx"].unsqueeze(0).to(model.device)
            logit = model(hs, si)
            prob = torch.sigmoid(logit).item()
            stage_preds[int(item["stage_idx"])].append(prob)
            stage_labels[int(item["stage_idx"])].append(item["label"].item())

    per_stage_auroc = {}
    for s in range(NUM_STAGES):
        if stage_labels[s]:
            per_stage_auroc[f"stage_{s}"] = float(roc_auc_score(
                stage_labels[s], stage_preds[s]))

    # Routing evaluation (simplified for cross-compare)
    # ... (full routing eval would duplicate evaluate.py logic)

    return {
        "train_auroc": results["best_val_auroc"],
        "per_stage_auroc": per_stage_auroc,
        "calibration": cal_results,
    }


def cross_dataset_transfer(train_data_source, test_data_target, output_dir, label_type="stage"):
    """Train probe on source dataset, evaluate on target (zero-shot)."""
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "transfer_predictor.pt")
    results_path = os.path.join(output_dir, "transfer_results.json")

    # Train on source
    _, val_data, _ = split_data(train_data_source, seed=42)
    results = train_mlp(
        train_data_source, val_data, results_path, model_path,
        variant="mlp", label_type=label_type, seed=42, num_epochs=50,
    )

    # Evaluate zero-shot on target
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = build_predictor(
        variant="mlp", hidden_dim=LLAMA_HIDDEN_DIM,
        stage_emb_dim=checkpoint.get("stage_emb_dim", 16),
        num_stages=NUM_STAGES,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    target_dataset = HiddenStateDataset(test_data_target, label_type=label_type)
    stage_preds = defaultdict(list)
    stage_labels = defaultdict(list)

    with torch.no_grad():
        for i in range(len(target_dataset)):
            item = target_dataset[i]
            hs = item["hidden_state"].unsqueeze(0).to(model.device)
            si = item["stage_idx"].unsqueeze(0).to(model.device)
            prob = torch.sigmoid(model(hs, si)).item()
            stage_preds[int(item["stage_idx"])].append(prob)
            stage_labels[int(item["stage_idx"])].append(item["label"].item())

    per_stage_auroc = {}
    for s in range(NUM_STAGES):
        if stage_labels[s]:
            per_stage_auroc[f"stage_{s}"] = float(roc_auc_score(
                stage_labels[s], stage_preds[s]))

    mean_auroc = float(np.mean(list(per_stage_auroc.values())))

    print(f"\n[*] Cross-dataset transfer AUROC: {mean_auroc:.4f}")
    for s in range(NUM_STAGES):
        if f"stage_{s}" in per_stage_auroc:
            print(f"  Stage {s}: {per_stage_auroc[f'stage_{s}']:.4f}")

    return {
        "train_auroc": results["best_val_auroc"],
        "transfer_per_stage_auroc": per_stage_auroc,
        "transfer_mean_auroc": mean_auroc,
        "threshold_met": mean_auroc > 0.75,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=str, default="results/cross_compare")
    ap.add_argument("--label_type", type=str, default="stage")
    args = ap.parse_args()

    # Available datasets (will be populated after collections)
    configs = {
        "qwen_hotpot": "data/collected_states_hotpotqa_v4_final_token.jsonl",
        "mistral_hotpot": "data/collected_hotpotqa_mistral_v4.jsonl",
        "qwen_triviaqa": "data/collected_triviaqa_open.jsonl",
    }

    results = {}
    available = {}

    for name, path in configs.items():
        if os.path.exists(path):
            print(f"\n{'='*60}")
            print(f"Processing: {name} ({path})")
            print(f"{'='*60}")

            train_data, val_data, test_data = load_and_prepare(path, args.label_type)
            out_dir = os.path.join(args.output_dir, name)
            r = train_and_eval(train_data, val_data, test_data, args.label_type, out_dir)
            results[name] = r
            available[name] = {
                "train_data": train_data, "test_data": test_data,
                "n_train": len(set(d["query_id"] for d in train_data)),
                "n_test": len(set(d["query_id"] for d in test_data)),
            }
            print(f"  Train AUROC: {r['train_auroc']:.4f}")
        else:
            print(f"\n[!] Skipping {name}: data not found at {path}")

    # Cross-dataset transfer (if both datasets available)
    if "qwen_hotpot" in available and "qwen_triviaqa" in available:
        print(f"\n{'='*60}")
        print("CROSS-DATASET TRANSFER: HotpotQA → TriviaQA (Qwen)")
        print(f"{'='*60}")
        transfer_results = cross_dataset_transfer(
            available["qwen_hotpot"]["train_data"],
            available["qwen_triviaqa"]["test_data"],
            os.path.join(args.output_dir, "transfer_hotpot_to_triviaqa"),
            args.label_type,
        )
        results["cross_dataset_transfer"] = transfer_results

    # Save all results
    with open(os.path.join(args.output_dir, "cross_compare_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[*] Cross-compare results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
