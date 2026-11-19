"""Guess-level RFT for spam students: credit individual guesses, not whole lists.

Why: a spam guess hits a valid rare answer ~2% of the time, so a list that scored 85 is ~1 good guess + 29 random
ones; list-level selection (spam_rft.py) barely moved val. Here every sampled guess is scored on its own:
  hit  = autocorrects (or matches) to an answer worth >= --min-tier points
1. Student (base + adapter) writes --n lists per TRAIN prompt; all samples saved to runs/sft/<name>/samples.jsonl.
2. Per prompt, the hit guesses (deduplicated) are pooled. Each new list = up to --hits of them (shuffled) + the
   student's own other spam to 30, + the 10-answer obvious tail copied from a teacher list of that prompt.
   So lists are enriched in guess shapes that worked, but still in the student's own distribution.
3. Data = --lists enriched lists per prompt (prompts with >= 1 hit) + --teacher-per teacher lists per prompt.
With --shape, the filler follows the prompt's answer FORMAT (not its answers): if the pool has >= 15 single-word
answers worth >= 60, the filler is mostly single words of 5-11 letters (where spam hits ~5-8%, vs <1% for multi-word
and 1.7% for <= 4 letters); otherwise (people, "Los ..." cities: nothing single-word to land on) mostly multi-word; those prompts are rare
(~9% of train), so they get lists even without hits, --multi-boost times as many.
Then retrain a fresh LoRA from base and evaluate on val (train/spam_hits_loop.sh).
"""
import argparse, json, random, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "env"), str(ROOT / "probe"), str(ROOT / "data"), str(ROOT / "eval")]
from build import answer_dicts, load  # noqa: E402
from emulator import Dictionary, norm, predict  # noqa: E402
from game import split_prompts  # noqa: E402
from play import MLXPolicy, parse_output  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--min-tier", type=int, default=60)
    ap.add_argument("--hits", type=int, default=6, help="max hit guesses per enriched list")
    ap.add_argument("--lists", type=int, default=4)
    ap.add_argument("--teacher-per", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=480)
    ap.add_argument("--shape", action="store_true", help="prompt-aware word-count/length mix for the filler")
    ap.add_argument("--shape-frac", type=float, default=0.9, help="share of filler from the preferred shape")
    ap.add_argument("--multi-boost", type=int, default=3,
                    help="with --shape: lists per multi-word-mode prompt = --lists x this, even without hits")
    ap.add_argument("--samples-from", default=None, help="reuse samples.jsonl from this runs/sft dir")
    a = ap.parse_args()
    import mlx.core as mx
    mx.set_memory_limit(5 * 2**30); mx.set_cache_limit(512 * 2**20)
    out = ROOT / "runs" / "sft" / a.name
    out.mkdir(parents=True, exist_ok=True)
    P, A = load()
    _, E = split_prompts(P)
    T = E[E["tsplit"] == "train"]
    teacher = {}
    for l in open(Path(a.teacher) / "train.jsonl", encoding="utf-8"):
        d = json.loads(l)
        teacher.setdefault(d["prompt_id"], []).append(d)
    samples_path = out / "samples.jsonl"
    if a.samples_from and not samples_path.exists():
        samples_path.write_text((Path(a.samples_from) / "samples.jsonl").read_text())
    done = set()
    if samples_path.exists():
        done = {json.loads(l)["prompt_id"] for l in open(samples_path)}
    todo = [p for p in T.itertuples() if p.prompt_id not in done]
    if todo:
        pol = MLXPolicy(a.model, a.adapter, temp=1.0, top_p=0.95, max_tokens=a.max_tokens, batch=16, system="spam")
        t0 = time.time()
        with open(samples_path, "a", encoding="utf-8") as f:
            for i, p in enumerate(todo):
                ans = answer_dicts(A, p.prompt_id)
                outs = pol.sample(p.text, ans, Dictionary(ans), a.n)
                gs = [list(dict.fromkeys(parse_output(o)[1])) for o in outs]
                f.write(json.dumps(dict(prompt_id=p.prompt_id, text=p.text, lists=gs), ensure_ascii=False) + "\n")
                f.flush()
                if i % 20 == 0:
                    print(f"  sampled {i + 1}/{len(todo)} [{time.time() - t0:.0f}s]", flush=True)
    rng = random.Random(0)
    rows, n_hit, n_guess, prompts_hit, modes = [], 0, 0, 0, {"single": 0, "multi": 0}
    for l in open(samples_path, encoding="utf-8"):
        s = json.loads(l)
        ans = answer_dicts(A, s["prompt_id"])
        D = Dictionary(ans)
        n_single = sum(1 for x in ans if 60 <= x["score"] < 100 and len(norm(x["answer"]).split()) == 1)
        mode = "single" if n_single >= 15 else "multi"
        cache, hits, other = {}, {}, {}
        for g in s["lists"]:
            spam = g[:-10] if len(g) > 10 else g
            for x in spam:
                k = norm(x)
                if not k:
                    continue
                if k not in cache:
                    cache[k] = predict(x, D)
                n_guess += 1
                ans = cache[k]
                if ans is not None and D.rank[ans][0] >= a.min_tier:
                    n_hit += 1
                    hits.setdefault(k, x)
                elif ans is None:
                    other.setdefault(k, x)
        tl = teacher.get(s["prompt_id"], [])
        if not tl:
            continue
        if hits or (a.shape and mode == "multi"):
            prompts_hit += bool(hits)
            modes[mode] += 1
            H, O = list(hits.values()), list(other.values())
            def good(x):
                t = norm(x).split()
                return (len(t) == 1 and 5 <= len(t[0]) <= 11) if mode == "single" else len(t) >= 2
            Og, Ob = [x for x in O if good(x)], [x for x in O if not good(x)]
            for _ in range(a.lists * (a.multi_boost if a.shape and mode == "multi" else 1)):
                h = rng.sample(H, min(a.hits, len(H)))
                k = 30 - len(h)
                if a.shape:
                    fg = rng.sample(Og, min(round(k * a.shape_frac), len(Og)))
                    fill = fg + rng.sample(Ob, min(k - len(fg), len(Ob)))
                else:
                    fill = rng.sample(O, min(k, len(O)))
                g = h + fill
                rng.shuffle(g)
                tail = tl[rng.randrange(len(tl))]["completion"].removeprefix("<guesses> ").split(" ; ")[-10:]
                rows.append(dict(prompt_id=s["prompt_id"], text=s["text"], system="spam", source="hits",
                                 completion="<guesses> " + " ; ".join(g + tail)))
        rows += rng.sample(tl, min(a.teacher_per, len(tl)))
    rng.shuffle(rows)
    with open(out / "train.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    (out / "valid.jsonl").write_text((Path(a.teacher) / "valid.jsonl").read_text())
    print(f"spam hit rate {n_hit / max(1, n_guess):.2%} ({n_hit}/{n_guess}); prompts with hits {prompts_hit}; "
          f"{len(rows)} rows -> {out}; modes {modes}; shape={a.shape}")


if __name__ == "__main__":
    main()
