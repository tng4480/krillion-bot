"""Turn-based Krillion round, driven by the autocorrect emulator: the policy reacts to what the game shows.

What the game shows, and so what the policy sees after each submission:
  rejected               nothing matched: free, keep going
  corrected -> Y         a near-miss matched answer Y; the preview pauses on it (no score shown).
                         Resubmitting Y exactly accepts it; anything else declines it.
  scored <tier> <pts>    an exact answer (or an accepted correction): the round is over
The budget is --tries submissions (stand-in for the 25 s clock; declines and rejections both cost one).

Policies act in turns: each turn is a list of guesses, processed in order until something other than a rejection
happens, so a turn ends at the first correction (the policy gets to decide on it) or at a score.
The observation after a turn is text, e.g.
  rejected: vexillo, morbanta, zinfeli
  corrected: "tricerops" -> Triceratops (resubmit to accept, or keep guessing to decline)
Usage (scripted policies, all eval prompts):
  .venv/bin/python env/game.py --policy stack --split val
"""
import argparse, json, random, re, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "probe"), str(ROOT / "data"), str(ROOT / "analysis")]
from emulator import Dictionary, norm, predict  # noqa: E402
from build import answer_dicts, load  # noqa: E402

TIER_NAME = {10: "plankton", 15: "too clever", 30: "schooler", 60: "rare", 85: "deep cut", 100: "krillion"}


class Round:
    def __init__(self, answers, tries=40, D=None, cache=None):
        self.D = D or Dictionary(answers)
        self.exact = set(self.D.strings)
        self.cache = {} if cache is None else cache
        self.tries, self.n, self.seen = tries, 0, set()
        self.pending = None        # canonical answer shown in the preview, if any
        self.score, self.answer, self.done = 0, None, False
        self.log = []              # (guess, outcome, answer)

    def _match(self, k, g):
        if k not in self.cache:
            self.cache[k] = predict(g, self.D)
        return self.cache[k]

    def submit(self, g):
        """One submission -> ('rejected'|'corrected'|'scored'|'skip'|'over', answer)."""
        if self.done or self.n >= self.tries:
            self.done = True
            return "over", None
        k = norm(g)
        if not k:
            return "skip", None
        if self.pending is not None and k == norm(self.pending):   # accept the previewed correction
            return self._score(g, self.pending)
        self.pending = None                                        # anything else declines it
        if k in self.seen:
            return "skip", None
        self.seen.add(k); self.n += 1
        a = self._match(k, g)
        if a is None:
            self.log.append((g, "rejected", None))
            return "rejected", None
        if k in self.exact:
            return self._score(g, a)
        self.pending = a
        self.log.append((g, "corrected", a))
        return "corrected", a

    def _score(self, g, a):
        self.score, self.answer, self.done = self.D.rank[a][0], a, True
        self.log.append((g, "scored", a))
        return "scored", a

    def turn(self, guesses):
        """Process a list in order until something other than a rejection. -> observation text."""
        rej = []
        for g in guesses:
            kind, a = self.submit(g)
            if kind == "rejected":
                rej.append(g)
            elif kind == "corrected":
                head = f"rejected: {', '.join(rej)}\n" if rej else ""
                return head + f'corrected: "{g}" -> {a} (resubmit to accept, or keep guessing to decline)'
            elif kind == "scored":
                return f"scored: {a} - {TIER_NAME.get(self.score, self.score)}, {self.score} pts. Round over."
            elif kind == "over":
                break
        if self.n >= self.tries:
            self.done = True
            return (f"rejected: {', '.join(rej)}\n" if rej else "") + "time's up. Round over, 0 pts."
        return f"rejected: {', '.join(rej)}" if rej else "nothing new submitted"


# ---------- scripted policies (references; they get the observations, not the answer list) ----------

def known_common(answers):
    """Stand-in for 'any LM knows the obvious answers': the prompt's basic answers (score <= 30)."""
    return sorted(a["answer"] for a in answers if a["score"] <= 30)


def policy_common(answers, R, rng):
    pool = known_common(answers)
    R.turn(rng.sample(pool, len(pool)))


def train_modifiers(top=120):
    """Modifiers that stack on another answer of the same prompt, mined from TRAIN-split prompts only
    (cached in analysis/modifiers_train.json), ranked by how many prompts they recur in."""
    path = ROOT / "analysis" / "modifiers_train.json"
    if path.exists():
        return [tuple(m) for m in json.loads(path.read_text())][:top]
    import collections
    P, A = load()
    P, _ = split_prompts(P, 0)
    n_prompts, hits = collections.Counter(), collections.Counter()
    for pid in P[P["tsplit"] == "train"]["prompt_id"]:
        ans = answer_dicts(A, pid)
        single = {norm(a["answer"]) for a in ans if " " not in norm(a["answer"])}
        seen = set()
        for a in ans:
            t = norm(a["answer"]).split()
            if len(t) != 2:
                continue
            for mod, base, pos in ((t[0], t[1], "pre"), (t[1], t[0], "post")):
                if base in single and len(mod) >= 3 and mod.isalpha():
                    hits[(mod, pos)] += 1; seen.add((mod, pos)); break
        n_prompts.update(seen)
    ranked = sorted(n_prompts, key=lambda k: (-n_prompts[k], -hits[k]))[:500]
    path.write_text(json.dumps(ranked))
    return ranked[:top]


def policy_stack(answers, R, rng, mods=None, fallback=2):
    """Your strategy, played from observations only: misspell an obvious answer so it previews instead of
    scoring, decline, stack learned modifiers on whatever was revealed, repeat on new corrections; finish
    with exact obvious answers if nothing landed."""
    mods = mods or train_modifiers()
    known = set(known_common(answers))
    bases = []
    probes = rng.sample(sorted(known), len(known))
    budget = R.tries - fallback
    while not R.done and R.n < budget:
        if not bases:
            if not probes:
                break
            w = norm(probes.pop())
            if len(w) < 5:
                continue
            i = rng.randrange(1, len(w) - 1)
            R.turn([w[:i] + w[i + 1:]])
            if R.pending is not None:
                bases.append(R.pending)
            continue
        b = norm(bases[0])
        tried = 0
        for w, pos in rng.sample(mods, len(mods)):
            if R.done or R.n >= budget:
                break
            if w in b.split():
                continue
            kind, a = R.submit(f"{w} {b}" if pos == "pre" else f"{b} {w}")
            tried += 1
            if kind == "corrected":
                if a in known or norm(a) == b:
                    if a not in bases:
                        bases.append(a)          # declined; still a base to stack on later
                else:
                    R.submit(a)                  # an unfamiliar multi-word answer: accept it
            if tried >= 12:
                break
        bases.append(bases.pop(0)) if len(bases) > 1 else bases.pop(0)
    if not R.done:
        R.turn(rng.sample(sorted(known), len(known)))


def char_model(words, order=3, seed=0):
    """Character n-gram generator over a word list (the 'plausible-shaped nonsense' source)."""
    import collections
    chains = collections.defaultdict(list)
    for w in words:
        w = "^" * order + w + "$"
        for i in range(len(w) - order):
            chains[w[i:i + order]].append(w[i + order])
    r = random.Random(seed)

    def gen():
        s, out = "^" * order, ""
        while len(out) < 40:
            c = r.choice(chains[s])
            if c == "$":
                break
            out += c; s = s[1:] + c
        return out
    return gen


def policy_shape(answers, R, rng, gen, fallback=2):
    """Spam generated strings; accept any correction that isn't an obvious answer, decline those; fall back to
    exact obvious answers at the end."""
    known = set(known_common(answers))
    for _ in range(R.tries * 30):
        if R.done or R.n >= R.tries - fallback:
            break
        kind, a = R.submit(gen())
        if kind == "corrected" and a not in known:
            R.submit(a)
    if not R.done:
        R.turn(rng.sample(sorted(known), len(known)))


_SHAPE = {}
STOP = set("name a an the of or and in on to for with that is are from by any its it as at be".split())


def shape_source(kind, pid, text, answers, P, A):
    """Word lists for the shape generator. own = this prompt's answers (LEAKY reference); near = answers of the
    10 train prompts with the most similar wording (what training on the archive could teach); all = all
    train answers (generic shape)."""
    if kind == "own":
        return [norm(a["answer"]) for a in answers]
    import math, re, collections
    toks = lambda t: [w for w in re.findall(r"[a-z]+", t.lower()) if w not in STOP]
    if "docs" not in _SHAPE:  # tf-idf over train prompt texts (no extra dependency)
        T = P[P["tsplit"] == "train"]
        docs = {pid: collections.Counter(toks(t)) for pid, t in zip(T["prompt_id"], T["text"])}
        df = collections.Counter(w for c in docs.values() for w in c)
        _SHAPE.update(docs=docs, idf={w: math.log(len(docs) / df[w]) for w in df})
    docs, idf = _SHAPE["docs"], _SHAPE["idf"]
    ids = [i for i in docs if i != pid]   # never the prompt itself
    if kind == "near":
        q = collections.Counter(toks(text))
        def sim(c):
            num = sum(q[w] * c[w] * idf.get(w, 0) ** 2 for w in q)
            den = math.sqrt(sum((v * idf.get(w, 0)) ** 2 for w, v in c.items()) or 1)
            return num / den
        ids = sorted(ids, key=lambda i: -sim(docs[i]))[:10]
    else:
        ids = random.Random(0).sample(ids, 60)
    sub = A[A["prompt_id"].isin(ids)]
    return [norm(x) for x in sub["answer"]]


POLICIES = {"common": policy_common, "stack": policy_stack, "shape-own": None, "shape-near": None, "shape-all": None}


def split_prompts(P, min_pool=300):
    """Time split by dive date: train dives 1-100, val 101-113, test 114+. Eval uses big pools only."""
    dates = sorted(P["date"].unique())
    val0, test0 = dates[100], dates[113]
    P = P.assign(tsplit=["train" if d < val0 else "val" if d < test0 else "test" for d in P["date"]])
    return P, P[P["n_answers"] >= min_pool]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="stack", choices=sorted(POLICIES))
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--tries", type=int, default=40)
    ap.add_argument("--min-pool", type=int, default=300)
    a = ap.parse_args()
    P, A = load()
    P, E = split_prompts(P, a.min_pool)
    E = E[E["tsplit"] == a.split]
    scores = []
    for p in E.itertuples():
        ans = answer_dicts(A, p.prompt_id)
        D, cache, rng = Dictionary(ans), {}, random.Random(0)
        if a.policy.startswith("shape-"):
            gen = char_model(shape_source(a.policy[6:], p.prompt_id, p.text, ans, P, A))
            play = lambda ans, R, rng: policy_shape(ans, R, rng, gen)
        else:
            play = POLICIES[a.policy]
        s = []
        for _ in range(a.n):
            R = Round(ans, a.tries, D, cache)
            play(ans, R, rng)
            s.append(R.score)
        scores.append(statistics.mean(s))
        print(f"{p.prompt_id} pool {p.n_answers:6} E={scores[-1]:5.1f}  {p.text[:60]}", flush=True)
    print(f"\n{a.policy} on {a.split} (pool >= {a.min_pool}): {len(scores)} prompts, mean {statistics.mean(scores):.1f}")




# ---------- knowledge-free ladder (rungs 0-2): literal prompt rules + shape, no idea what is famous ----------

LETTERS = "eeeeeeeeeeeeaaaaaaaaarrrrrrrriiiiiiiioooooooootttttttnnnnnnnsssssslllllcccccuuuudddpppmmmhhhggbbffyywkvxzjq"


def prompt_rules(text):
    """Literal constraints a rule follower can read off the prompt, no knowledge involved."""
    t = text.replace("“", '"').replace("”", '"').replace("’", "'")
    r = dict(prefixes=[], suffixes=[], contains=[], length=None, min_len=None, one_word=False)
    m = re.search(r"starts? with (?:the word |the letters? |a )?(.+?)(?:$|\(| and | that | or more)", t, re.I)
    if m:
        r["prefixes"] = [x.strip(' "\'.-') for x in re.split(r",| or ", m.group(1)) if x.strip(' "\'.-')]
    m = re.search(r"end(?:s|ing)? (?:in|with) (.+?)(?:$|\(| and | that )", t, re.I)
    if m:
        r["suffixes"] = [x.strip(' "\'.-') for x in re.split(r",| or ", m.group(1)) if x.strip(' "\'.-')]
    r["contains"] = re.findall(r'containing "([^"]+)"', t)
    m = re.search(r"\((\d+) letters\)|(\d+) letters\b", t)
    if m and "or more" not in t[m.start():m.end() + 10]:
        r["length"] = int(m.group(1) or m.group(2))
    m = re.search(r"(\d+) or more letters", t)
    if m:
        r["min_len"] = int(m.group(1))
    r["one_word"] = bool(re.search(r"one-word|single word|\(\d+ letters\)", t, re.I)) or r["length"] is not None
    return r


def obeys(s, r):
    k = norm(s)
    w = k.replace(" ", "")
    if r["one_word"] and " " in k:
        return False
    if r["prefixes"] and not any(k.startswith(norm(p)) for p in r["prefixes"]):
        return False
    if r["suffixes"] and not any(k.endswith(norm(x)) for x in r["suffixes"]):
        return False
    if r["contains"] and not all(norm(c) in k for c in r["contains"]):
        return False
    if r["length"] and len(w) != r["length"]:
        return False
    if r["min_len"] and len(w) < r["min_len"]:
        return False
    return True


def ruled(gen, r, rng):
    """Bend generated strings to the literal rules (prefix/suffix spliced on), then filter."""
    def g():
        for _ in range(200):
            s = gen()
            if r["prefixes"]:
                p = norm(rng.choice(r["prefixes"]))
                s = p + ("" if " " in p and False else "") + s[rng.randrange(0, 3):] if " " not in p else p + " " + s
            if r["suffixes"]:
                x = norm(rng.choice(r["suffixes"]))
                s = s[:max(2, len(s) - rng.randrange(0, 3))] + x
            if r["one_word"]:
                s = s.replace(" ", "")
            if obeys(s, r):
                return s
        return s
    return g


def random_letters(rng):
    def g():
        words = [ "".join(rng.choice(LETTERS) for _ in range(rng.randint(4, 10))) ]
        if rng.random() < 0.3:
            words.append("".join(rng.choice(LETTERS) for _ in range(rng.randint(4, 8))))
        return " ".join(words)
    return g


def policy_blind(answers, R, rng, gen):
    """No knowledge: spam, accept every correction (can't tell famous from rare), no obvious-answer fallback."""
    for _ in range(R.tries * 30):
        if R.done or R.n >= R.tries:
            break
        kind, a = R.submit(gen())
        if kind == "corrected":
            R.submit(a)


def ladder_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args(sys.argv[2:])
    P, A = load()
    P, E = split_prompts(P)
    E = E[E["tsplit"] == a.split]
    rungs = ["0 random letters + rules", "1 generic shape + rules", "2 similar-prompt shape + rules",
             "2 without rules", "2 + knowledge (decline famous, fallback)"]
    res = {k: [] for k in rungs}
    for p in E.itertuples():
        ans = answer_dicts(A, p.prompt_id)
        D, cache = Dictionary(ans), {}
        rules = prompt_rules(p.text)
        srcs = {"all": char_model(shape_source("all", p.prompt_id, p.text, ans, P, A), seed=1),
                "near": char_model(shape_source("near", p.prompt_id, p.text, ans, P, A), seed=2)}
        for k in rungs:
            rng = random.Random(0)
            s = []
            for _ in range(a.n):
                R = Round(ans, 40, D, cache)
                if k.startswith("0"):
                    policy_blind(ans, R, rng, ruled(random_letters(rng), rules, rng))
                elif k.startswith("1"):
                    policy_blind(ans, R, rng, ruled(srcs["all"], rules, rng))
                elif k == "2 without rules":
                    policy_blind(ans, R, rng, srcs["near"])
                elif k.startswith("2 +"):
                    policy_shape(ans, R, rng, ruled(srcs["near"], rules, rng))
                else:
                    policy_blind(ans, R, rng, ruled(srcs["near"], rules, rng))
                s.append(R.score)
            res[k].append(statistics.mean(s))
        print(f"{p.prompt_id} " + " ".join(f"{res[k][-1]:5.1f}" for k in rungs) + f"  {p.text[:45]}", flush=True)
    ruled_ids = [i for i, p in enumerate(E.itertuples()) if any(prompt_rules(p.text)[x] for x in
                                                               ("prefixes", "suffixes", "contains", "length", "min_len"))]
    print(f"\n{'rung':44} all({len(E)})  with-rules({len(ruled_ids)})")
    for k in rungs:
        v = res[k]
        print(f"{k:44} {statistics.mean(v):6.1f}  {statistics.mean([v[i] for i in ruled_ids]) if ruled_ids else float('nan'):6.1f}")


if __name__ == "__main__":
    ladder_main() if sys.argv[1:2] == ["ladder"] else main()
