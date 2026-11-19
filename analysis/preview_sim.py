"""Emulate live play with the preview ON: pooled spam (shape-sorted), then the model's own 10-answer tail.
On a correction to X: auto = accept; decline = decline if X is one of the model's own obvious answers (any tail
of the samples used), else accept; stack = decline as above, then try K modifiers on X ("m X"/"X m")."""
import json, sys, collections, statistics as st
sys.path[:0] = ["probe", "data", "env", "eval"]
from build import answer_dicts, load
from emulator import Dictionary, norm
from game import Round, train_modifiers
from rapidfuzz.distance import Levenshtein

def shape(g):
    t = norm(g).split()
    if not t: return 9
    if len(t) > 1: return 3
    L = len(t[0]); return 0 if 5 <= L <= 7 else 1 if 8 <= L <= 11 else 2

P, A = load()
lists = collections.defaultdict(list)
for l in open("runs/eval/spamshape4_val_smollm2-360m.jsonl"):
    r = json.loads(l); lists[r["prompt_id"]].append(list(dict.fromkeys(r["guesses"])))
for l in open(sys.argv[1]):
    r = json.loads(l); lists[r["prompt_id"]] += r["lists"]
MODS = train_modifiers(int(sys.argv[2]) if len(sys.argv) > 2 else 40)

def build(ls, n_spam):
    pool, seen = [], set()
    for g in ls:
        for x in g[:-10]:
            k = norm(x)
            if k and k not in seen: seen.add(k); pool.append(x)
    pool.sort(key=shape)
    spam = pool[:n_spam]; ss = {norm(s) for s in spam}
    tail = [x for x in ls[0][-10:] if norm(x) not in ss]
    obvious = {norm(x) for g in ls for x in g[-10:]}
    return spam, tail, obvious

def is_obv(a, O):
    k = norm(a)
    return k in O or any(Levenshtein.distance(k, o, score_cutoff=1) <= 1 for o in O)

def play(ans, D, cache, spam, tail, O, B, mode, K, keep=True):
    R = Round(ans, tries=B, D=D, cache=cache)
    budget_spam = B - len(tail)
    declined = []
    for g in spam:
        if R.done or R.n >= budget_spam: break
        kind, a = R.submit(g)
        if kind != "corrected": continue
        if mode == "auto" or not is_obv(a, O):
            R.submit(a); break
        declined.append(a)
        if mode == "stack":
            b = norm(a); n = 0
            for m, pos in MODS:
                if R.done or R.n >= budget_spam or n >= K: break
                if m in b.split(): continue
                k2, a2 = R.submit(f"{m} {b}" if pos == "pre" else f"{b} {m}"); n += 1
                if k2 == "corrected" and not is_obv(a2, O) and norm(a2) != b:
                    R.submit(a2); break
            if R.done: break
    fb = declined + tail if keep else tail
    for g in fb:
        if R.done: break
        kind, a = R.submit(g)
        if kind == "corrected": R.submit(a)
    return R.score

res = collections.defaultdict(list)
for pid, ls in lists.items():
    ans = answer_dicts(A, pid); D = Dictionary(ans); cache = {}
    for NS in (8, 16, 32):
        for gi in range(len(ls) // NS):
            grp = ls[gi * NS:(gi + 1) * NS]
            for B in (40, 50):
                spam, tail, O = build(grp, B - 10)
                for mode, K, keep in (("auto", 0, 1), ("decline", 0, 0), ("decline", 0, 1), ("stack", 4, 1), ("stack", 8, 1), ("stack", 16, 1)):
                    res[(NS, B, mode, K, keep)].append((pid, play(ans, D, cache, spam, tail, O, B, mode, K, keep)))
for k, v in sorted(res.items()):
    # average per prompt first, then over prompts
    byp = collections.defaultdict(list)
    for pid, s in v: byp[pid].append(s)
    print(k, f"{st.mean(st.mean(x) for x in byp.values()):.1f}  (n={len(v)})")
