import json, sys
path = sys.argv[1] if len(sys.argv) > 1 else "data/collected_states_nq_v3.jsonl"
with open(path) as f:
    lines = [json.loads(l) for l in f if l.strip()]
queries = {}
for l in lines:
    qid = l["query_id"]
    queries.setdefault(qid, {})[l["stage_idx"]] = l.get("stage_correctness", 0)
print(f"Queries: {len(queries)}")
for s in range(4):
    c = sum(1 for q in queries.values() if s in q and q[s]==1)
    t = sum(1 for q in queries.values() if s in q)
    if t: print(f"  S{s}: {c}/{t} = {c/t:.1%}")
