"""Which 40 of the pooled spam to play? Emulated live play (preview off, 50 guesses = 40 spam + 10 tail), val prompts.
Selection uses only the model's own samples and train-split text, never this prompt's answers:
  base    pooled spam, stable sort by shape (env/live.py)
  spread  base order, skip a guess whose typo window overlaps an already chosen one (d <= allowed(x) + allowed(y))
  dens    within shape, most "neighbours" in the pool first (guesses the samples agree on)
  lm      within shape, highest char-trigram log-prob per char first (LM over words of TRAIN-split answers)
  combos  dens/lm order + spread
Usage: .venv/bin/python analysis/select_sim.py runs/eval/spamshape4_val_smollm2-360m.jsonl [runs/eval/val_pool24_360m.jsonl]
"""
import collections, json, math, statistics as st, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / d) for d in ("probe", "data", "env", "eval")]
from build import answer_dicts, load
from emulator import Dictionary, allowed, norm
from game import Round, split_prompts
from rapidfuzz.distance import Levenshtein


def shape(k):
    t = k.split()
    if not t: return 9
    if len(t) > 1: return 3
    L = len(t[0]); return 0 if 5 <= L <= 7 else 1 if 8 <= L <= 11 else 2


class CharLM:
    def __init__(self, words, n=3):
        self.n, self.c, self.h = n, collections.Counter(), collections.Counter()
        for w in words:
            w = "^" * (n - 1) + w + "$"
            for i in range(len(w) - n + 1):
                self.c[w[i:i + n]] += 1; self.h[w[i:i + n - 1]] += 1

    def score(self, s):
        tot = 0.0; parts = s.split()
        L = 0
        for w in parts:
            w = "^" * (self.n - 1) + w + "$"
            for i in range(len(w) - self.n + 1):
                tot += math.log((self.c[w[i:i + self.n]] + 0.1) / (self.h[w[i:i + self.n - 1]] + 3.0)); L += 1
        return tot / max(1, L)


def near(x, y):
    lim = allowed(len(x)) + allowed(len(y))
    return Levenshtein.distance(x, y, score_cutoff=lim) <= lim


def spread(order, n):
    out = []
    for x in order:
        if len(out) >= n: break
        if " " in x or not any(near(x, y) for y in out if " " not in y):
            out.append(x)
    return out


def main():
    P, A = load()
    Pall, E = split_prompts(P, 0)
    words = set()
    for pid in Pall[Pall["tsplit"] == "train"]["prompt_id"]:
        for a in answer_dicts(A, pid):
            words.update(norm(a["answer"]).split())
    lm = CharLM(words); lm4 = CharLM(words, 4); lm2 = CharLM(words, 2)
    print(f"char LM over {len(words)} train words", flush=True)
    lists = collections.defaultdict(list)
    for l in open(ROOT / sys.argv[1]):
        r = json.loads(l); lists[r["prompt_id"]].append(list(dict.fromkeys(r["guesses"])))
    for l in (open(sys.argv[2]) if len(sys.argv) > 2 else []):
        r = json.loads(l); lists[r["prompt_id"]] += r["lists"]
    res = collections.defaultdict(lambda: collections.defaultdict(list))
    for pid, ls in lists.items():
        ans = answer_dicts(A, pid); D = Dictionary(ans); cache = {}
        for NS in ((8, 32) if len(sys.argv) > 2 else (8,)):
            for gi in range(len(ls) // NS):
                grp = ls[gi * NS:(gi + 1) * NS]
                pool = list(dict.fromkeys(k for g in grp for k in (norm(x) for x in g[:-10]) if k))
                tail = [x for x in grp[0][-10:]]
                base = sorted(pool, key=shape)
                nb = {x: sum(near(x, y) for y in pool if y != x and " " not in y) for x in pool if " " not in x}
                dens = sorted(pool, key=lambda x: (shape(x), -nb.get(x, 0)))
                lmo = sorted(pool, key=lambda x: (shape(x), -lm.score(x)))
                V = dict(base=base[:40], dens=dens[:40], lm=lmo[:40], lm_spread=spread(lmo, 40),
                         lm4=sorted(pool, key=lambda x: (shape(x), -lm4.score(x)))[:40],
                         lm2=sorted(pool, key=lambda x: (shape(x), -lm2.score(x)))[:40],
                         lm_noshape=sorted(pool, key=lambda x: (shape(x) == 3, -lm.score(x)))[:40],
                         lm_short=sorted(pool, key=lambda x: (min(shape(x), 1) if shape(x) < 3 else 3, -lm.score(x)))[:40])
                for name, spam in V.items():
                    ss = set(spam)
                    R = Round(ans, tries=50, D=D, cache=cache)
                    for g in spam + [x for x in tail if norm(x) not in ss]:
                        if R.done: break
                        kind, a = R.submit(g)
                        if kind == "corrected": R.submit(a)
                    res[(NS, name)][pid].append(R.score)
    for k, byp in sorted(res.items()):
        print(k, f"{st.mean(st.mean(v) for v in byp.values()):.1f}")


if __name__ == "__main__":
    main()
