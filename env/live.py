"""Build live-play guess lists for a dive: sample several lists per prompt, pool the spam, order it by shape.

Per prompt: --n lists from the model (system "spam"); spam = every list minus its last 10 (the obvious ending),
pooled and de-duplicated in sample order, then stably sorted by shape (single words of 5-7 letters first, then
8-11, then other single words, then multi-word; hit rates on val 8.3% / 4.4% / <2% / <1%). The list = first --spam
of those + the first sample's own 10-answer ending. On val (360M spamshape4) this scores 54.3 at 40 guesses vs 50.3
for one raw list. Default 40 spam + 10 = 50 guesses: on val (32 samples/prompt, emulator) 40 -> 50 guesses is
+2.3-3.4, 60-70 adds ~+1 per 10 more; 16 or 32 samples instead of 8 changes little (only 40 of a ~240-guess pool are
used). Preview/decline and modifier stacking gained nothing (HANDOFF, "Preview, stacking, budget"), so play with
answer corrections OFF. Prompts are texts only (no answers). Writes runs/live/<date>_lists.json.
Within each shape, spam is ordered by a char-trigram LM over train-split answer words (most word-like first):
test 56.2 -> 61.4 (360M), 54.5 -> 57.8 (135M) at 8 samples, 50 guesses (analysis/select_sim.py).
Usage: .venv/bin/python env/live.py --date 2026-11-19 --prompts prompts.json
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "probe"), str(ROOT / "data"), str(ROOT / "env"), str(ROOT / "analysis")]
from emulator import norm  # noqa: E402
from play import MLXPolicy, parse_output  # noqa: E402


def shape(g):
    t = norm(g).split()
    if not t:
        return 9
    if len(t) > 1:
        return 3
    L = len(t[0])
    return 0 if 5 <= L <= 7 else 1 if 8 <= L <= 11 else 2


def train_lm():
    """Char-trigram LM over the words of TRAIN-split answers (dives 1-100): a word-likeness prior, no eval answers."""
    from build import answer_dicts, load
    from game import split_prompts
    from select_sim import CharLM
    P, A = load()
    Pall, _ = split_prompts(P, 0)
    words = set()
    for pid in Pall[Pall["tsplit"] == "train"]["prompt_id"]:
        for a in answer_dicts(A, pid):
            words.update(norm(a["answer"]).split())
    return CharLM(words)


def build(outs, n_spam=40, n_tail=10, lm=None):
    lists = [list(dict.fromkeys(x.strip() for x in parse_output(o)[1] if x.strip())) for o in outs]
    pool, seen = [], set()
    for g in lists:
        for x in g[:-10]:
            k = norm(x)
            if k and k not in seen:
                seen.add(k); pool.append(x)
    pool.sort(key=(lambda x: (shape(x), -lm.score(norm(x)))) if lm else shape)
    spam = pool[:n_spam]
    tail = [x for x in lists[0][-n_tail:] if norm(x) not in {norm(s) for s in spam}]
    return spam + tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--prompts", required=True, help="json list of {pid, text}")
    ap.add_argument("--model", default="mlx-community/SmolLM2-360M-Instruct")
    ap.add_argument("--adapter", default="checkpoints/spamshape4_smollm2-360m")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--spam", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default runs/live/<date>_lists.json")
    a = ap.parse_args()
    import mlx.core as mx
    mx.set_memory_limit(5 * 2**30); mx.random.seed(a.seed)
    lm = train_lm()
    pol = MLXPolicy(a.model, a.adapter, temp=1.0, top_p=0.95, max_tokens=480, batch=8, system="spam")
    out = []
    for p in json.load(open(a.prompts)):
        g = build(pol.sample(p["text"], None, None, a.n), a.spam, lm=lm)
        out.append(dict(pid=p["pid"], text=p["text"], guesses=g))
        print(f'{p["pid"]} ({len(g)}): ' + " ; ".join(g), flush=True)
    path = Path(a.out) if a.out else ROOT / "runs" / "live" / f"{a.date}_lists.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("->", path)


if __name__ == "__main__":
    main()
