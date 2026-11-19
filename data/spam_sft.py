"""Build SFT data for the knowledge-free ladder: '<guesses> spam x30 ; obvious x3' lists for train prompts.

Spam = rung 2 of env/game.py: a char model over the answers of the 10 most similar OTHER train prompts, bent to
the prompt's literal rules. The list ends with 3 obvious answers (the prompt's basic answers, score <= 30): the
only knowledge the recipe needs. Lists are NOT filtered by score, so the student learns the generator (shapes for
a prompt), not the lucky hits. Writes runs/sft/<name>/{train,valid}.jsonl.
Usage: .venv/bin/python data/spam_sft.py --name spam_v1 --lists 8
"""
import argparse, json, random, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "env"), str(ROOT / "probe"), str(ROOT / "data"), str(ROOT / "eval")]
from build import answer_dicts, load  # noqa: E402
from emulator import Dictionary, norm  # noqa: E402
from game import char_model, known_common, prompt_rules, ruled, shape_source, split_prompts  # noqa: E402
from play import play_round  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="spam_v1")
    ap.add_argument("--lists", type=int, default=8)
    ap.add_argument("--spam", type=int, default=30)
    ap.add_argument("--obvious", type=int, default=3)
    a = ap.parse_args()
    P, A = load()
    Pall, E = split_prompts(P)
    T = E[E["tsplit"] == "train"]
    rows, scores = [], []
    for p in T.itertuples():
        ans = answer_dicts(A, p.prompt_id)
        D = Dictionary(ans); exact = set(D.strings); cache = {}
        rng = random.Random(hash(p.prompt_id) % 2**32)
        gen = ruled(char_model(shape_source("near", p.prompt_id, p.text, ans, Pall, A), seed=rng.randrange(10**6)),
                    prompt_rules(p.text), rng)
        basics = known_common(ans)
        for _ in range(a.lists):
            spam, seen = [], set()
            for _ in range(a.spam * 10):
                s = gen()
                if s and s not in seen and len(s) <= 40:
                    seen.add(s); spam.append(s)
                if len(spam) >= a.spam:
                    break
            tail = rng.sample(basics, min(a.obvious, len(basics)))
            g = spam + tail
            scores.append(play_round(g, D, exact, 40, cache=cache)["score"])
            rows.append(dict(prompt_id=p.prompt_id, text=p.text, system="spam",
                             completion="<guesses> " + " ; ".join(g), score=scores[-1]))
        print(f"{p.prompt_id} {statistics.mean(scores[-a.lists:]):5.1f}  {p.text[:50]}", flush=True)
    out = ROOT / "runs" / "sft" / a.name
    out.mkdir(parents=True, exist_ok=True)
    random.Random(0).shuffle(rows)
    with open(out / "train.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[16:])
    with open(out / "valid.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[:16])
    print(f"\n{len(rows)} lists over {len(T)} train prompts, teacher list mean score {statistics.mean(scores):.1f} -> {out}")


if __name__ == "__main__":
    main()
