"""Candidate strings for the fixed-list optimiser: what our generators produce, counted.

Samples the spam models on every train prompt (prompt-agnostic models, so this is just a big draw from each) and adds
the char-trigram list (analysis/det_gen.py). Only generated strings: nothing is copied from the answer lists.
Keeps single words (no spaces), normalised. -> runs/fixed/cands.json {string: count}
Usage: .venv/bin/python analysis/cand_pool.py [n_lists_per_prompt=16]
"""
import collections, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / d) for d in ("probe", "data", "env", "eval", "analysis")]
import mlx.core as mx  # noqa: E402
from build import answer_dicts, load  # noqa: E402
from det_gen import Tri, sampled_top  # noqa: E402
from emulator import norm  # noqa: E402
from game import split_prompts  # noqa: E402
from models import MODELS  # noqa: E402
from play import MLXPolicy, parse_output  # noqa: E402

GENS = ["p14_vocab512", "p14_vocab4k", "pythia14m_distill_pipeline", "pythia70m_distill_pipeline"]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    mx.set_memory_limit(5 * 2**30); mx.set_cache_limit(512 * 2**20); mx.random.seed(0)
    P, A = load(); Pall, _ = split_prompts(P, 0)
    tr = Pall[Pall["tsplit"] == "train"]
    cnt = collections.Counter()
    for name in GENS:
        m, ad = MODELS[name]
        pol = MLXPolicy(m, ad, temp=1.0, top_p=0.95, max_tokens=480, batch=n, system="spam")
        t0, c = time.time(), collections.Counter()
        for p in tr.itertuples():
            for o in pol.sample(p.text, None, None, n):
                c.update({norm(x) for x in parse_output(o)[1] if norm(x) and " " not in norm(x)})
        cnt.update(c)
        print(f"{name}: {len(c)} distinct, {time.time() - t0:.0f}s", flush=True)
        del pol
    words = {w for pid in tr["prompt_id"] for a in answer_dicts(A, pid) for w in norm(a["answer"]).split()}
    tri = sampled_top(Tri(words), 20000, m=60000, seed=0)
    cnt.update({norm(x): 1 for x in tri if norm(x) and " " not in norm(x)})
    print(f"trigram: {len(tri)}; total distinct {len(cnt)}")
    out = ROOT / "runs/fixed"; out.mkdir(parents=True, exist_ok=True)
    (out / "cands.json").write_text(json.dumps(dict(cnt.most_common())))


if __name__ == "__main__":
    main()
