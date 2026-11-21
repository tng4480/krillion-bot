"""Replay whole dives in the emulator, many random seeds per model: how much of a live score is luck?

Per seed, per prompt of each day: mx.random.seed(seed), sample n lists, build the live list (spam + 10-answer tail,
G guesses from the budget model in budget_sim.py and runs/speed.jsonl), play it in the emulator. Game score = sum
over the day's prompts. Days are read straight from data/reveal/<date>.json (answers the game revealed), so they
can be days that were never in the training data. Writes runs/seed_runs.json; prints mean, sd, min, max per model/day.
Usage: .venv/bin/python analysis/seed_runs.py --dates 2026-11-18 2026-11-19 2026-11-20 --seeds 20 \
           --models 360m:8 135m:16 p14_vocab4k:32 p14_vocab512:32
"""
import argparse, json, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / d) for d in ("probe", "data", "env", "eval", "analysis")]
import mlx.core as mx  # noqa: E402
from budget_sim import BUILD, CLOCK, FREE, PER_GUESS, build_lists, play  # noqa: E402
from emulator import Dictionary  # noqa: E402
from live import train_lm  # noqa: E402
from models import MODELS  # noqa: E402
from play import MLXPolicy, parse_output  # noqa: E402


def day(date):
    d = json.load(open(ROOT / f"data/reveal/{date}.json"))
    out = []
    for p in d["prompts"]:
        ans = [dict(answer=a["answer"], aliases=a.get("aliases", []), tier=a["tier"], score=a["score"])
               for a in p["answers"]]
        out.append((p["text"], ans, Dictionary(ans)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="name:n_lists")
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args()
    mx.set_memory_limit(5 * 2**30); mx.set_cache_limit(512 * 2**20)
    speed = {}
    for l in open(ROOT / "runs/speed.jsonl"):
        r = json.loads(l); speed[(r["model"], r["n"])] = r["sec"]
    lm, days = train_lm(), {d: day(d) for d in a.dates}
    out_f = ROOT / "runs/seed_runs.json"
    res = json.loads(out_f.read_text()) if out_f.exists() else {}
    for spec in a.models:
        name, n = spec.split(":"); n = int(n)
        G = int((CLOCK - max(0.0, speed[(name, n)] + BUILD - FREE)) / PER_GUESS)
        base, ad = MODELS[name]
        pol = MLXPolicy(base, ad, temp=1.0, top_p=0.95, max_tokens=480, batch=min(n, 64), system="spam")
        t0 = time.time()
        for date, prompts in days.items():
            totals, per = [], []
            for s in range(a.seeds):
                mx.random.seed(s)
                sc = []
                for text, ans, D in prompts:
                    lists = [list(dict.fromkeys(parse_output(o)[1])) for o in pol.sample(text, None, None, n)]
                    sc.append(play(ans, D, {}, build_lists(lists, G - 10, lm), G))
                totals.append(sum(sc)); per.append(sc)
            res[f"{name}|{date}"] = dict(model=name, n=n, guesses=G, date=date, totals=totals, per_prompt=per)
            out_f.write_text(json.dumps(res, indent=1))
            print(f"{name:14} n={n:3} G={G} {date}  mean {st.mean(totals):6.1f}  sd {st.stdev(totals):5.1f}  "
                  f"min {min(totals):4}  max {max(totals):4}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
