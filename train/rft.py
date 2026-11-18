"""One RFT round: sample from the current policy, edit and filter with the emulator, write an SFT dataset.

For each train/val prompt (test prompts are never touched):
  1. sample --n guess lists from the model (+ optional adapter) with the eval prompt,
  2. edit each list: drop duplicates and every guess the emulator maps to a basic answer (score <= 30),
     cap at --k guesses; the model's own rejected guesses stay in (a rejection is free),
  3. score the edited list under the retry rule and keep lists scoring >= --min-score, best first, at most
     --keep per prompt.
Targets therefore stay in the model's own distribution, conditioned on the prompt, which is what should
transfer to unseen prompts; the emulator only teaches it what to leave out and which lists were good.

Writes runs/sft/<name>/{train,valid}.jsonl (same format as data/sft.py) and samples.jsonl (raw, for reuse).
Then train with train/sft_lora.py (--resume to continue the previous adapter).
Usage:
  .venv/bin/python train/rft.py --model mlx-community/Qwen3-0.6B-4bit --name rft1_qwen3-0.6b --n 64
"""
import argparse, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "probe"), str(ROOT / "data"), str(ROOT / "eval")]
from emulator import Dictionary, norm, predict  # noqa: E402
from build import answer_dicts, load  # noqa: E402
from play import MLXPolicy, parse_output, play_round  # noqa: E402

BASIC = 30


def edit(guesses, D, cache, k):
    out, seen = [], set()
    for g in guesses:
        n = norm(g)
        if not n or n in seen:
            continue
        seen.add(n)
        if n not in cache:
            cache[n] = predict(g, D)
        a = cache[n]
        if a is not None and D.rank[a][0] <= BASIC:
            continue
        out.append(g)
        if len(out) >= k:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--keep", type=int, default=48)
    ap.add_argument("--min-score", type=int, default=60)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()

    P, A = load()
    pol = MLXPolicy(a.model, a.adapter, temp=a.temp, max_tokens=a.max_tokens, batch=a.batch)
    out_dir = ROOT / "runs" / "sft" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {s: open(out_dir / f"{s}.jsonl", "w", encoding="utf-8") for s in ("train", "valid")}
    raw = open(out_dir / "samples.jsonl", "w", encoding="utf-8")
    for p in P.itertuples():
        split = {"train": "train", "val": "valid"}.get(p.split)
        if split is None:
            continue
        answers = answer_dicts(A, p.prompt_id)
        D = Dictionary(answers); exact = set(D.strings); cache = {}
        outs = pol.sample(p.text, answers, D, a.n)
        scored = []
        for o in outs:
            plan, guesses = parse_output(o)
            before = play_round(guesses, D, exact, a.k, cache=cache)["score"]
            ed = edit(guesses, D, cache, a.k)
            after = play_round(ed, D, exact, a.k, cache=cache)["score"]
            raw.write(json.dumps(dict(prompt_id=p.prompt_id, output=o, before=before, after=after),
                                 ensure_ascii=False) + "\n")
            if ed:
                plan = re.sub(r"\s+", " ", plan or "recall")[:80]
                scored.append((after, plan, ed))
        scored.sort(key=lambda x: -x[0])
        kept = [x for x in scored if x[0] >= a.min_score][:a.keep]
        for s, plan, ed in kept:
            files[split].write(json.dumps(dict(prompt_id=p.prompt_id, text=p.text, plan=plan, score=s,
                                               completion=f"<plan> {plan} <guesses> " + " ; ".join(ed)),
                                          ensure_ascii=False) + "\n")
        mb = sum(x[0] for x in scored) / max(1, len(scored))
        print(f"{p.prompt_id} {split:5} sampled {len(outs)}, usable {len(scored)}, mean edited score {mb:.1f}, "
              f"kept {len(kept)}  {p.text[:45]}", flush=True)
    for f in [*files.values(), raw]:
        f.close()
    print(f"-> {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
