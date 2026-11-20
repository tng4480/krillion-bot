"""Sampling speed per model: seconds to write n spam lists for one prompt (one batch of n), as in a real-time run.
Usage: .venv/bin/python analysis/speed.py 360m [n ...]   -> appends to runs/speed.jsonl
"""
import json, statistics as st, sys, time
sys.path[:0] = ["env", "probe", "data", "eval", "analysis"]
import mlx.core as mx
from models import MODELS
from play import MLXPolicy

TEXTS = ["Name a fruit that is not red", "Name a famous painter", "Name something you find in a garage",
         "Name a city in South America", "Name a breakfast cereal"]
name = sys.argv[1]; ns = [int(x) for x in sys.argv[2:]] or [4, 8, 16, 32, 64]
mx.set_memory_limit(5 * 2**30); mx.set_cache_limit(512 * 2**20)
m, ad = MODELS[name]
for n in ns:
    pol = MLXPolicy(m, ad, temp=1.0, top_p=0.95, max_tokens=480, batch=n, system="spam")
    pol.sample(TEXTS[0], None, None, n)  # warm-up
    ts = []
    for t in TEXTS[1:]:
        t0 = time.time(); pol.sample(t, None, None, n); ts.append(time.time() - t0)
    r = dict(model=name, n=n, sec=round(st.median(ts), 2), peak_gb=round(mx.get_peak_memory() / 2**30, 2))
    print(r, flush=True)
    with open("runs/speed.jsonl", "a") as f:
        f.write(json.dumps(r) + "\n")
    del pol
