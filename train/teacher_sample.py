"""Sample spam lists from a teacher model for distillation: TRAIN-split prompts only (dives 1-100), prompt text only.

Writes one row per prompt: {prompt_id, text, lists: [[guess, ...], ...]} to --out (resumable).
Usage: .venv/bin/python train/teacher_sample.py --model mlx-community/SmolLM2-360M-Instruct \
           --adapter checkpoints/spamshape4_smollm2-360m --min-pool 100 --n 24 --out runs/distill/teacher360.jsonl
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / d) for d in ("env", "probe", "data", "eval")]
from build import load  # noqa: E402
from game import split_prompts  # noqa: E402
from play import MLXPolicy, parse_output  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--min-pool", type=int, default=100)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import mlx.core as mx
    mx.set_memory_limit(4 * 2**30); mx.set_cache_limit(512 * 2**20); mx.random.seed(a.seed)
    P, A = load()
    Pall, _ = split_prompts(P, 0)
    T = Pall[(Pall["tsplit"] == "train") & (Pall["n_answers"] >= a.min_pool)]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["prompt_id"] for l in open(out)} if out.exists() else set()
    pol = MLXPolicy(a.model, a.adapter, temp=1.0, top_p=0.95, max_tokens=480, batch=12, system="spam")
    t0 = time.time()
    with open(out, "a", encoding="utf-8") as f:
        for i, p in enumerate(T.itertuples()):
            if p.prompt_id in done:
                continue
            outs = pol.sample(p.text, None, None, a.n)
            f.write(json.dumps(dict(prompt_id=p.prompt_id, text=p.text,
                                    lists=[list(dict.fromkeys(parse_output(o)[1])) for o in outs]),
                               ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0:
                print(f"{i + 1}/{len(T)} [{time.time() - t0:.0f}s]", flush=True)
    print("done", len(T), "prompts")


if __name__ == "__main__":
    main()
