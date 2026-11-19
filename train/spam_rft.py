"""One RFT round for the spam-list students (ReST-EM style): sample, keep the good and diverse lists, rebuild data.

1. The current student (base + adapter) writes --n lists per TRAIN prompt (pool >= 300), from the prompt alone.
2. Each list is scored with the emulator (one-shot, corrections auto-accepted, 40-guess budget).
3. Keep lists that beat that prompt's own mean (advantage > 0, so hard prompts count too) and score >= --min-score,
   dropping near-duplicates (guess-set Jaccard > 0.5 with a kept list): protects diversity.
4. Write runs/sft/<name>/{train,valid}.jsonl = kept lists + the original teacher data (mixed in, guards drift).
Then retrain a fresh LoRA from the base model (train/sft_lora.py) and evaluate on val.
Usage: .venv/bin/python train/spam_rft.py --model M --adapter A --teacher runs/sft/spam_v2 --name spamrft1_x
"""
import argparse, json, random, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "env"), str(ROOT / "probe"), str(ROOT / "data"), str(ROOT / "eval")]
from build import answer_dicts, load  # noqa: E402
from emulator import Dictionary, norm  # noqa: E402
from game import split_prompts  # noqa: E402
from play import MLXPolicy, parse_output, play_round  # noqa: E402


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--teacher", required=True, help="original SFT data dir to mix in")
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--min-score", type=int, default=60)
    ap.add_argument("--keep", type=int, default=4, help="max kept lists per prompt")
    ap.add_argument("--max-tokens", type=int, default=480)
    a = ap.parse_args()
    import mlx.core as mx
    mx.set_memory_limit(5 * 2**30); mx.set_cache_limit(512 * 2**20)
    pol = MLXPolicy(a.model, a.adapter, temp=1.0, top_p=0.95, max_tokens=a.max_tokens, batch=16, system="spam")
    P, A = load()
    _, E = split_prompts(P)
    T = E[E["tsplit"] == "train"]
    kept, means, t0 = [], [], time.time()
    for p in T.itertuples():
        ans = answer_dicts(A, p.prompt_id)
        D = Dictionary(ans); exact = set(D.strings); cache = {}
        outs = pol.sample(p.text, ans, D, a.n)
        scored = []
        for o in outs:
            _, g = parse_output(o)
            g = list(dict.fromkeys(g))[:40]
            scored.append((play_round(g, D, exact, 40, cache=cache)["score"], g))
        m = statistics.mean(s for s, _ in scored)
        means.append(m)
        mine = []
        for s, g in sorted(scored, key=lambda x: -x[0]):
            if s > m and s >= a.min_score and all(jaccard([norm(x) for x in g], [norm(x) for x in h]) <= 0.5
                                                   for h in mine):
                mine.append(g)
            if len(mine) >= a.keep:
                break
        kept += [dict(prompt_id=p.prompt_id, text=p.text, system="spam", completion="<guesses> " + " ; ".join(g),
                      source="rft") for g in mine]
    out = ROOT / "runs" / "sft" / a.name
    out.mkdir(parents=True, exist_ok=True)
    teacher = [json.loads(l) for l in open(Path(a.teacher) / "train.jsonl", encoding="utf-8")]
    rows = kept + teacher
    random.Random(0).shuffle(rows)
    with open(out / "train.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    (out / "valid.jsonl").write_text((Path(a.teacher) / "valid.jsonl").read_text())
    print(f"student mean on train prompts {statistics.mean(means):.1f}; kept {len(kept)} lists "
          f"({len({r['prompt_id'] for r in kept})} prompts) + {len(teacher)} teacher -> {out} "
          f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
