"""Sample N spam lists per eval prompt (pool >= 300) from one model, for the time-budget simulation.
Usage: .venv/bin/python analysis/sample_pool.py MODEL val|test N   -> runs/eval/pool_<split>_<model>.jsonl (resumable)
"""
import json, sys, time
sys.path[:0] = ["env", "probe", "data", "eval", "analysis"]
import mlx.core as mx
from build import load
from game import split_prompts
from models import MODELS
from play import MLXPolicy, parse_output

name, split, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
mx.set_memory_limit(5 * 2**30); mx.set_cache_limit(512 * 2**20); mx.random.seed(0)
P, A = load(); _, E = split_prompts(P, 0)
V = E[(E["tsplit"] == split) & (E["n_answers"] >= 300)]
m, ad = MODELS[name]
pol = MLXPolicy(m, ad, temp=1.0, top_p=0.95, max_tokens=480, batch=min(n, 64), system="spam")
out = f"runs/eval/pool_{split}_{name}.jsonl"; done = set()
try: done = {json.loads(l)["prompt_id"] for l in open(out)}
except FileNotFoundError: pass
t0 = time.time()
with open(out, "a") as f:
    for i, p in enumerate(V.itertuples()):
        if p.prompt_id in done: continue
        outs = pol.sample(p.text, None, None, n)
        f.write(json.dumps(dict(prompt_id=p.prompt_id, text=p.text,
                                lists=[list(dict.fromkeys(parse_output(o)[1])) for o in outs])) + "\n"); f.flush()
        if i % 10 == 0: print(name, split, i, f"{time.time() - t0:.0f}s", flush=True)
print(name, split, "done", f"{time.time() - t0:.0f}s")
