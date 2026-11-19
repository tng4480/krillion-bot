import json, sys, time
sys.path[:0] = ["env", "probe", "data", "eval"]
from build import load
from game import split_prompts
from play import MLXPolicy, parse_output
import mlx.core as mx
mx.set_memory_limit(5 * 2**30); mx.set_cache_limit(512 * 2**20); mx.random.seed(int(sys.argv[2]))
P, A = load(); _, E = split_prompts(P); V = E[E["tsplit"] == "val"]
pol = MLXPolicy("mlx-community/SmolLM2-360M-Instruct", "checkpoints/spamshape4_smollm2-360m", temp=1.0, top_p=0.95,
                max_tokens=480, batch=8, system="spam")
out = sys.argv[1]; done = set()
try: done = {json.loads(l)["prompt_id"] for l in open(out)}
except FileNotFoundError: pass
t0 = time.time()
with open(out, "a") as f:
    for i, p in enumerate(V.itertuples()):
        if p.prompt_id in done: continue
        outs = pol.sample(p.text, None, None, int(sys.argv[3]))
        f.write(json.dumps(dict(prompt_id=p.prompt_id, text=p.text,
                                lists=[list(dict.fromkeys(parse_output(o)[1])) for o in outs])) + "\n"); f.flush()
        print(i, f"{time.time()-t0:.0f}s", flush=True)
