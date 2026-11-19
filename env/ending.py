"""Fix the list ending: spam from the trained adapter + obvious answers from the SAME base model without the adapter.

The spam adapter's own endings are mostly invalid (10-24%), plausibly because spam training degrades recall.
Here the base model (adapter off) is simply asked for the most obvious answers; its answers become the list's tail.
Reuses the trained model's saved val lists (runs/eval/spam_val_<tag>_sft.jsonl) for the spam part.
Reports: validity of the base model's obvious answers, and the combined score for several tail lengths.
Usage: .venv/bin/python env/ending.py --model mlx-community/SmolLM2-360M-Instruct --tag smollm2-360m
"""
import argparse, json, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "env"), str(ROOT / "probe"), str(ROOT / "data"), str(ROOT / "eval")]
from build import answer_dicts, load  # noqa: E402
from emulator import Dictionary, norm, predict  # noqa: E402
from play import parse_output, play_round  # noqa: E402

ASK = ("List the most common, obvious answers to this prompt - the ones most people would say first. "
       "Reply only as: <guesses> a ; b ; c ; ... (10 answers)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--spam", type=int, default=30)
    a = ap.parse_args()
    import mlx.core as mx
    from mlx_lm import batch_generate, load as mlx_load
    from mlx_lm.sample_utils import make_sampler
    mx.set_memory_limit(5 * 2**30); mx.set_cache_limit(512 * 2**20)
    model, tok = mlx_load(a.model)
    sampler = make_sampler(temp=a.temp, top_p=0.95)

    lists = [json.loads(l) for l in open(ROOT / "runs" / "eval" / f"spam_val_{a.tag}_sft.jsonl")]
    by = {}
    for x in lists:
        by.setdefault(x["prompt_id"], []).append(x)
    P, A = load()
    texts = {x["prompt_id"]: x["text"] for x in lists}
    valid, res = [], {k: [] for k in (0, 3, 5, 8, 10)}
    for pid, xs in by.items():
        D = Dictionary(answer_dicts(A, pid)); exact = set(D.strings); cache = {}
        msgs = [{"role": "user", "content": ASK + "\n\nPrompt: " + texts[pid]}]
        ids = tok.encode(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                                 enable_thinking=False), add_special_tokens=False)
        outs = batch_generate(model, tok, [ids] * len(xs), max_tokens=100, sampler=sampler).texts
        for x, o in zip(xs, outs):
            _, tail = parse_output(o)
            tail = [t for t in dict.fromkeys(tail) if t][:10]
            for t in tail:
                k = norm(t)
                if k not in cache:
                    cache[k] = predict(t, D)
                valid.append(cache[k] is not None)
            spam = x["guesses"][:-3][:a.spam]
            for k in res:
                g = spam[:40 - k] + tail[:k] if k else x["guesses"]
                res[k].append(play_round(g, D, exact, 40, cache=cache)["score"])
    print(f"{a.tag}: base-model obvious answers valid {statistics.mean(valid):.0%}")
    print("  own ending (as trained):", f"{statistics.mean(res[0]):5.1f}")
    for k in (3, 5, 8, 10):
        print(f"  spam + {k:2} base-model obvious answers: {statistics.mean(res[k]):5.1f}")


if __name__ == "__main__":
    main()
