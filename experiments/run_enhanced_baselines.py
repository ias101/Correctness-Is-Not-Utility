"""Runner script for enhanced routing baselines — deployed to GPU server."""
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experiments'))
from enhanced_routing_baselines import *

data = []
with open('data/collected_states_hotpotqa_v4.jsonl') as f:
    for line in f:
        data.append(json.loads(line))
print(f"Loaded {len(data)} tuples")

device = 'cuda'
results = {}

for label_key, name, nc in [
    ('oracle_stop_stage', 'Multi-Class Stage Selector', 4),
    ('cwa_optimal_stop', 'CWA-Optimal Stage Selector', 4),
    ('degradation_avoid', 'Degradation-Avoid Router', 2),
]:
    data = build_routing_labels(data, label_key)
    print(f"\nTraining: {name}")
    r = train_routing_policy(data, label_key, 'results/enhanced_routing', name,
                             num_classes=nc, hidden_dim=3584, num_epochs=100,
                             seed=42, device=device)
    results[label_key] = r

oracle = json.load(open('results/enhanced_routing/oracle_policy_bound.json'))
print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")
print(f"Oracle Policy Upper Bound: CWA(0.5)={oracle['cwa_0.5']:.4f}")
continue_r = json.load(open('results/enhanced_routing/routing_oracle_continue_results.json'))
print(f"Direct Stop/Continue:     CWA(0.5)={continue_r['routing_results']['cwa_0.5']:.4f}")
for lk in ['oracle_stop_stage', 'cwa_optimal_stop', 'degradation_avoid']:
    if lk in results:
        rr = results[lk]['routing_results']
        print(f"{results[lk]['label_name']:<25}: CWA(0.5)={rr['cwa_0.5']:.4f}")

with open('results/enhanced_routing/enhanced_routing_all.json', 'w') as f:
    all_data = {'oracle_policy': oracle, 'oracle_continue': continue_r}
    for k, v in results.items():
        all_data[k] = v
    json.dump(all_data, f, indent=2, default=str)
print("Done!")
