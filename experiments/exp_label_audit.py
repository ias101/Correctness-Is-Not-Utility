import json, sys
sys.path.insert(0, ".")
from train_predictor import train_mlp

DATA = "data/collected_states_hotpotqa_v4_final_token.jsonl"
RESULTS = {}
for label_type in ["final", "stage"]:
    print("=== label_type=" + label_type + " ===")
    results = train_mlp(
        data_path=DATA,
        label_type=label_type,
        seed=42,
        output_dir="results/label_audit_" + label_type
    )
    RESULTS[label_type] = results
    val_auroc = results.get("best_val_auroc", "N/A")
    best_epoch = results.get("best_epoch", "N/A")
    print("  Val AUROC: " + str(val_auroc))
    print("  Best epoch: " + str(best_epoch))

print("=== COMPARISON ===")
for k, v in RESULTS.items():
    print("  " + k + ": AUROC=" + str(v.get("best_val_auroc","?")))

def convert(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)

with open("results/label_audit_comparison.json", "w") as f:
    json.dump(RESULTS, f, indent=2, default=convert)
print("Results saved")
