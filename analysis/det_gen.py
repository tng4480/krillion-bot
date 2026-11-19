"""Deterministic spam, no language model: the most likely letter strings under a char-trigram LM.

The LM is built from the words of TRAIN-split answers (dives 1-100), from one of these sources:
  all       every train prompt (one generic list, the same for every prompt)
  cluster   train prompts in the same wording cluster as this prompt (k-means on tf-idf of prompt texts; the
            prompt is assigned to the nearest centroid from its text alone)
  near      the k train prompts whose wording is most similar (tf-idf), a soft per-prompt category
Spam = the top strings of 5-7 letters by per-char log-prob (beam search), so the same prompt always gets the
same list. `novel` variants drop strings that are themselves words of the source answers (pure pseudo-words), to
separate "letter statistics" from "reusing other prompts' answer words".
Played with preview off: 50 spam guesses, or 40 spam + the 360M model's 10-answer tail (to compare with the
model pipeline at 50 guesses). Never uses the answers of the prompt being played, except to score.
Usage: .venv/bin/python analysis/det_gen.py val|test
"""
import collections, json, math, re, statistics as st, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / d) for d in ("probe", "data", "env", "eval")]
from build import answer_dicts, load  # noqa: E402
from emulator import Dictionary, norm  # noqa: E402
from game import STOP, Round, split_prompts  # noqa: E402

ALPHA = "abcdefghijklmnopqrstuvwxyz"


class Tri:
    """Char trigram with add-k smoothing; contexts are 2 chars, '^' pads the start, '$' ends a word."""
    def __init__(self, words, k=0.1):
        self.c = collections.Counter()
        self.h = collections.Counter()
        for w in words:
            w = "^^" + w + "$"
            for i in range(len(w) - 2):
                self.c[w[i:i + 3]] += 1; self.h[w[i:i + 2]] += 1
        self.k = k

    def lp(self, ctx, ch):
        return math.log((self.c[ctx + ch] + self.k) / (self.h[ctx] + self.k * 27))

    def top(self, n, lens=(5, 6, 7), beam=400, banned=()):
        """Most likely strings of the given lengths (beam search), ranked by log-prob per char incl. the end."""
        out = []
        beams = [("", 0.0)]
        for L in range(1, max(lens) + 1):
            cand = []
            for s, p in beams:
                ctx = ("^^" + s)[-2:]
                for ch in ALPHA:
                    cand.append((s + ch, p + self.lp(ctx, ch)))
            cand.sort(key=lambda x: -x[1])
            beams = cand[:beam]
            if L in lens:
                for s, p in beams:
                    out.append((s, (p + self.lp(("^^" + s)[-2:], "$")) / (L + 1)))
        out.sort(key=lambda x: -x[1])
        res, seen = [], set(banned)
        for s, _ in out:
            if s not in seen:
                seen.add(s); res.append(s)
            if len(res) >= n:
                break
        return res


def spread_top(g, n, lim=2, banned=()):
    """Top strings, skipping any within edit distance lim of one already chosen (diversity)."""
    from rapidfuzz.distance import Levenshtein
    out = []
    for x in g.top(n * 30, banned=banned):
        if all(Levenshtein.distance(x, y, score_cutoff=lim) > lim for y in out):
            out.append(x)
        if len(out) >= n:
            break
    return out


def sampled_top(g, n, m=3000, seed=0):
    """Seeded samples of 5-7 letters (deterministic), deduped, ranked by log-prob per char."""
    import random
    r = random.Random(seed); pool = set()
    while len(pool) < m:
        s, ctx = "", "^^"
        while len(s) < 8:
            ws = [math.exp(g.lp(ctx, ch)) for ch in ALPHA + "$"]
            ch = r.choices(ALPHA + "$", ws)[0]
            if ch == "$": break
            s += ch; ctx = ctx[1:] + ch
        if 5 <= len(s) <= 7: pool.add(s)
    def sc(x):
        c = "^^"; t = 0
        for ch in x + "$": t += g.lp(c, ch); c = (c + ch)[-2:]
        return t / (len(x) + 1)
    return sorted(pool, key=lambda x: (-sc(x), x))[:n]


def toks(t):
    return [w for w in re.findall(r"[a-z]+", t.lower()) if w not in STOP]


class TfIdf:
    def __init__(self, texts):
        df = collections.Counter(w for t in texts for w in set(toks(t)))
        self.vocab = {w: i for i, w in enumerate(sorted(df))}
        self.idf = np.array([math.log(len(texts) / df[w]) for w in sorted(df)])

    def vec(self, t):
        v = np.zeros(len(self.vocab))
        for w in toks(t):
            if w in self.vocab:
                v[self.vocab[w]] += 1
        v *= self.idf
        n = np.linalg.norm(v)
        return v / n if n else v


def kmeans(X, k, iters=30, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        lab = np.argmax(X @ C.T, axis=1)
        for j in range(k):
            m = X[lab == j]
            if len(m):
                c = m.mean(0); C[j] = c / (np.linalg.norm(c) or 1)
    return C, np.argmax(X @ C.T, axis=1)


def play(ans, D, cache, guesses, tries=50):
    R = Round(ans, tries=tries, D=D, cache=cache)
    for g in guesses:
        if R.done:
            break
        kind, a = R.submit(g)
        if kind == "corrected":
            R.submit(a)
    return R.score


def main():
    split = sys.argv[1]
    P, A = load()
    Pall, E = split_prompts(P, 0)
    T = Pall[Pall["tsplit"] == "train"]
    ev = E[(E["tsplit"] == split) & (E["n_answers"] >= 300)]
    words_of = {pid: {w for a in answer_dicts(A, pid) for w in norm(a["answer"]).split()} for pid in T["prompt_id"]}
    tf = TfIdf(list(T["text"]))
    X = np.stack([tf.vec(t) for t in T["text"]])
    tids = list(T["prompt_id"])
    clusters = {}
    for k in (10, 30):
        C, lab = kmeans(X, k)
        clusters[k] = (C, lab)
    allw = set().union(*words_of.values())
    gen_all = Tri(allw)
    top_all = gen_all.top(50)
    top_all_novel = gen_all.top(50, banned=allw)
    var_all = {"all_spread1": spread_top(gen_all, 50, 1), "all_spread2": spread_top(gen_all, 50, 2),
               "all_sampled": sampled_top(gen_all, 50)}
    for kk, v in var_all.items(): print(kk, " ".join(v[:15]), flush=True)
    print("generic top:", " ".join(top_all[:20]), "| novel:", " ".join(top_all_novel[:20]), flush=True)
    tails = collections.defaultdict(list)
    tail_file = ROOT / ("runs/eval/spamshape4_val_smollm2-360m.jsonl" if split == "val"
                        else "runs/eval/test_spamshape4_smollm2-360m.jsonl")
    for l in open(tail_file):
        r = json.loads(l); tails[r["prompt_id"]].append(r["guesses"][-10:])
    res = collections.defaultdict(list)
    cl_cache = {}
    for p in ev.itertuples():
        ans = answer_dicts(A, p.prompt_id); D = Dictionary(ans); cache = {}
        q = tf.vec(p.text)
        srcs = {"all": (top_all, top_all_novel)}
        for kk, v in var_all.items(): srcs[kk] = (v, v)
        for k, (C, lab) in ([] if "--fast" in sys.argv else clusters.items()):
            j = int(np.argmax(C @ q))
            if (k, j) not in cl_cache:
                w = set().union(*(words_of[tids[i]] for i in np.where(lab == j)[0]))
                g = Tri(w)
                cl_cache[(k, j)] = (g.top(50), g.top(50, banned=w))
            srcs[f"cluster{k}"] = cl_cache[(k, j)]
        sims = X @ q
        for kn in (10,):
            ids = np.argsort(-sims)[:kn]
            w = set().union(*(words_of[tids[i]] for i in ids))
            g = Tri(w)
            srcs[f"near{kn}"] = (g.top(50), g.top(50, banned=w))
            srcs[f"near{kn}_spread2"] = (spread_top(g, 50, 2),) * 2
            srcs[f"near{kn}_sampled"] = (sampled_top(g, 50),) * 2
            mix = list(dict.fromkeys(x for pair in zip(var_all["all_spread2"], srcs[f"near{kn}_spread2"][0]) for x in pair))
            srcs[f"mix_all+near{kn}_spread2"] = (mix, mix)
        for name, (top, novel) in srcs.items():
            for vn, sp in ((("", top), ("_novel", novel)) if novel is not top else (("", top),)):
                res[name + vn + " 50 spam"].append(play(ans, D, cache, sp[:50]))
                tl = [st_ for st_ in tails[p.prompt_id]]
                if tl:
                    s = st.mean(play(ans, D, cache, sp[:40] + [x for x in t if norm(x) not in set(sp[:40])])
                                for t in tl)
                    res[name + vn + " 40 + model tail"].append(s)
    print(f"{split}: {len(ev)} prompts")
    for k in sorted(res):
        print(f"  {k:34} {st.mean(res[k]):.1f}")


if __name__ == "__main__":
    main()
