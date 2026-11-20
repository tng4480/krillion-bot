"""Build sequence-level distillation sets from teacher samples (train/teacher_sample.py).

raw       every teacher list as one SFT row (the teacher's own distribution)
pipeline  per prompt, --k random groups of 8 teacher lists, each turned into what env/live.py would play: pooled spam,
          sorted by shape then char-trigram word-likeness (LM over TRAIN answer words), first 40 + the group's first
          10-answer tail. The student learns to write the filtered list directly, with no sampling or reordering.
Writes runs/sft/distill_<name>_{raw,pipeline}/{train,valid}.jsonl (valid = spam_v2's, for loss monitoring).
Usage: .venv/bin/python train/distill_data.py --samples runs/distill/teacher360.jsonl --name t360
"""
import argparse, json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / d) for d in ("env", "probe", "data", "eval", "analysis")]
from emulator import norm  # noqa: E402
from live import shape, train_lm  # noqa: E402


def row(pid, text, g):
    return dict(prompt_id=pid, text=text, system="spam", source="distill", completion="<guesses> " + " ; ".join(g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--spam", type=int, default=40)
    a = ap.parse_args()
    rng = random.Random(0)
    lm = train_lm()
    raw, pipe = [], []
    for l in open(a.samples, encoding="utf-8"):
        s = json.loads(l)
        lists = [g for g in s["lists"] if len(g) >= 20]
        raw += [row(s["prompt_id"], s["text"], g) for g in lists]
        for _ in range(a.k):
            grp = rng.sample(lists, min(8, len(lists)))
            pool, seen = [], set()
            for g in grp:
                for x in g[:-10]:
                    k = norm(x)
                    if k and k not in seen:
                        seen.add(k); pool.append(x)
            pool.sort(key=lambda x: (shape(x), -lm.score(norm(x))))
            spam = pool[:a.spam]; ss = {norm(x) for x in spam}
            pipe.append(row(s["prompt_id"], s["text"], spam + [x for x in grp[0][-10:] if norm(x) not in ss]))
    for kind, rows in (("raw", raw), ("pipeline", pipe)):
        rng.shuffle(rows)
        out = ROOT / "runs" / "sft" / f"distill_{a.name}_{kind}"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "train.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        (out / "valid.jsonl").write_text((ROOT / "runs/sft/spam_v2/valid.jsonl").read_text())
        print(f"{kind}: {len(rows)} rows from {len({r['prompt_id'] for r in rows})} prompts -> {out}")


if __name__ == "__main__":
    main()
