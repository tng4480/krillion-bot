"""Interactive search: spam until something autocorrects, then stack modifiers on what it corrected to.

The strategy:
  1. Spam garbage (or deliberately misspelled common answers) until a guess is autocorrected. With preview
     on, the correction shows the canonical answer X before it is scored, so X can be declined. That
     reveals a real answer without spending the round.
  2. Spam modifiers on X: "M X" / "X M". Multi-word guesses only match answers with the same word count,
     each word within its own typo allowance, so each try asks "is there an answer like 'M X'?".
  3. A modifier guess that is corrected to Y (and declined) becomes a new base: "then again".
An exact answer skips the preview and scores at once; a corrected one can be declined. The first accepted
guess ends the round. Budget = --tries submissions (rejections and declines both count, as both cost time).

Decline rules (what the player sees in the preview is only the canonical answer, not its score):
  oracle  decline if the answer scores <= 30 (upper bound: needs knowing which answers are common)
  known   decline if the answer is one of the obvious answers we already know (the proxy for "a model
          knows this one is common"; here the prompt's answers scoring <= 30)
  multi   accept only multi-word corrections that differ from the base (never accept a bare base word)

Knowledge sources (leaky ones are marked):
  bases:     known  = obvious answers (score <= 30), the stand-in for a model's own recall; no probe cost
             probe  = misspell an obvious answer so it corrects instead of scoring (costs one try each)
             markov = char-model garbage over the prompt's own answers (LEAKY; reference only)
  modifiers: loo    = first/last words of 2-word answers in the OTHER prompts (non-leaky)
             own    = first/last words of 2-word answers in THIS prompt (LEAKY upper bound)
Usage:
  .venv/bin/python analysis/search.py [--n 100] [--tries 40]
"""
import argparse, collections, json, random, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "probe"), str(ROOT / "data"), str(ROOT / "analysis")]
from emulator import Dictionary, norm, predict  # noqa: E402
from build import answer_dicts, load  # noqa: E402
import strategies  # noqa: E402

BASIC = 30
# what any model knows: generic descriptors (non-leaky, hand-written, not taken from any answer list)
GENERIC = [(w, "pre") for w in (
    "east west north south new old upper lower port fort mount little big great grand saint "
    "red green white black yellow purple golden blue pink brown wild giant dwarf baby sweet sour bitter "
    "japanese chinese african indian mexican american french italian spanish korean thai greek russian "
    "british english irish scottish german brazilian australian canadian "
    "royal common true false mini super ancient modern classic magic dark sea mountain desert water "
    "river forest snow ice fire star moon sun").split()] + [(w, "post") for w in (
    "city town ville burg island bay springs falls beach heights park hill "
    "juice pie cake soup tea wine beer salad bread sauce jam fruit berry tree root "
    "ball racing riding hockey jumping skiing boxing "
    "academy school lab online plus pro kids learn prep tutor").split()]


def two_word_mods(answer_lists, top=80):
    """-> [(word, 'pre'|'post')] most common first/last words of 2-word answers, counted once per prompt
    and then by frequency, so generic modifiers (shared across prompts) come first."""
    n_prompts, freq = collections.Counter(), collections.Counter()
    for answers in answer_lists:
        seen = set()
        for a in answers:
            t = norm(a["answer"]).split()
            if len(t) == 2:
                for key in ((t[0], "pre"), (t[1], "post")):
                    if len(key[0]) < 3 or not key[0].isalpha():  # initials ('j smith') aren't modifiers
                        continue
                    freq[key] += 1
                    seen.add(key)
        n_prompts.update(seen)
    keys = sorted(freq, key=lambda k: (-n_prompts[k], -freq[k]))
    return keys[:top]


def misspell(w, rng):
    """One edit that keeps the guess inside the typo allowance for words of 5+ letters."""
    if len(w) < 5:
        return None
    i = rng.randrange(1, len(w) - 1)
    return w[:i] + w[i + 1] + w[i] + w[i + 2:] if rng.random() < 0.5 else w[:i] + w[i + 1:]


class Round:
    def __init__(self, D, exact, cache, decline, known, tries):
        self.D, self.exact, self.cache, self.decline, self.known, self.tries = D, exact, cache, decline, known, tries
        self.n, self.seen = 0, set()

    def submit(self, g, base=None):
        """-> ('done', score) | ('corrected', answer) | ('rejected', None) | ('skip', None)"""
        k = norm(g)
        if not k or k in self.seen:
            return "skip", None
        self.seen.add(k); self.n += 1
        if k not in self.cache:
            self.cache[k] = predict(g, self.D)
        a = self.cache[k]
        if a is None:
            return "rejected", None
        s = self.D.rank[a][0]
        if k in self.exact:
            return "done", s
        if self.decline == "oracle":
            dec = s <= BASIC
        elif self.decline == "known":
            dec = a in self.known
        else:  # multi
            dec = " " not in norm(a) or (base is not None and norm(a) == norm(base))
        return ("corrected", a) if dec else ("done", s)


def play(D, exact, cache, rng, *, bases_from, mods, decline, known, garbage, tries=40, max_bases=12,
         fallback=0):
    """fallback = submissions kept back at the end for obvious answers (exact, so they score 10-30)."""
    s = _play(D, exact, cache, rng, bases_from=bases_from, mods=mods, decline=decline, known=known,
              garbage=garbage, tries=tries - fallback, max_bases=max_bases)
    if s == 0 and fallback and known:
        return D.rank[rng.choice(sorted(known))][0]
    return s


def _play(D, exact, cache, rng, *, bases_from, mods, decline, known, garbage, tries=40, max_bases=12):
    R = Round(D, exact, cache, decline, known, tries)
    queue, used = [], set()  # bases waiting for modifiers
    if bases_from == "known":
        queue = rng.sample(sorted(known), min(len(known), max_bases))
    pending_probe = rng.sample(sorted(known), len(known)) if bases_from == "probe" else []
    mod_iter = {}
    while R.n < R.tries:
        if queue:  # stack a modifier on a base
            b = queue[0]
            it = mod_iter.setdefault(b, iter(rng.sample(mods, len(mods))))
            m = next(it, None)
            if m is None:
                queue.pop(0); continue
            w, pos = m
            nb = norm(b)
            if w in nb.split():
                continue
            g = f"{w} {nb}" if pos == "pre" else f"{nb} {w}"
            kind, v = R.submit(g, base=b)
            if kind == "done":
                return v
            if kind == "corrected" and v not in used and len(queue) < max_bases:
                used.add(v); queue.append(v)  # then again, on top of the new answer
            # move to the next base after a few misses so one dead base doesn't eat the budget
            if kind in ("rejected", "corrected") and rng.random() < 0.25:
                queue.append(queue.pop(0))
            continue
        # no base yet: find one
        if pending_probe:
            w = pending_probe.pop()
            g = misspell(norm(w), rng)
            if g is None:
                continue
        elif garbage is not None:
            g = garbage()
        else:
            return 0
        kind, v = R.submit(g)
        if kind == "done":
            return v
        if kind == "corrected" and v not in used:
            used.add(v); queue.append(v)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--tries", type=int, default=40)
    ap.add_argument("--only", default=None, help="comma-separated prompt_ids")
    a = ap.parse_args()
    P, A = load()
    allans = {pid: answer_dicts(A, pid) for pid in P["prompt_id"]}
    configs = [  # name, bases_from, modifier source, decline rule, garbage source[, fallback tries]
        ("known bases + generic mods + fallback", "known", "generic", "known", None, 2),
        ("misspelled-known + generic mods + fallback", "probe", "generic", "known", None, 2),
        ("markov + own mods + fallback  [LEAKY]", "markov", "own", "oracle", "markov", 2),
        ("misspelled-known + loo mods, decline known", "probe", "loo", "known", None),
        ("known bases + loo mods, decline known", "known", "loo", "known", None),
        ("misspelled-known + generic mods, decline known", "probe", "generic", "known", None),
        ("known bases + generic mods, decline known", "known", "generic", "known", None),
        ("known bases + generic mods, oracle", "known", "generic", "oracle", None),
        ("known bases + own mods, decline known  [LEAKY]", "known", "own", "known", None),
        ("markov garbage + own mods, oracle  [LEAKY]", "markov", "own", "oracle", "markov"),
        ("markov alone, oracle  [LEAKY reference]", "markov", None, "oracle", "markov"),
    ]
    res = collections.defaultdict(dict)
    rows = P if not a.only else P[P["prompt_id"].isin(a.only.split(","))]
    for p in rows.itertuples():
        ans = allans[p.prompt_id]
        D = Dictionary(ans); exact = set(D.strings); cache = {}
        known = {x["answer"] for x in ans if x["score"] <= BASIC}
        loo = two_word_mods([v for k, v in allans.items() if k != p.prompt_id])
        mods = {"loo": loo, "generic": GENERIC + [m for m in loo if m not in GENERIC][:40],
                "own": two_word_mods([ans]), None: []}
        common = strategies.common_answers(ans, D)
        base = statistics.mean(_common_round(common, D, exact, cache, a.tries) for _ in range(a.n))
        res[p.prompt_id]["common answers (floor)"] = base
        t0 = time.time()
        for name, bf, ms, dec, gsrc, *fb in configs:
            rng = random.Random(0)
            garbage = strategies.markov(ans, D) if gsrc == "markov" else None
            sc = [play(D, exact, cache, rng, bases_from=bf, mods=mods[ms], decline=dec, known=known,
                       garbage=garbage, tries=a.tries, fallback=fb[0] if fb else 0) for _ in range(a.n)]
            res[p.prompt_id][name] = statistics.mean(sc)
        print(f"{p.prompt_id} {p.category:12} {time.time() - t0:5.0f}s  {p.text[:60]}", flush=True)
    names = list(next(iter(res.values())))
    print(f"\n{'strategy':50} " + " ".join(f"{pid[-5:]:>6}" for pid in res) + "   mean")
    for nm in names:
        v = [res[pid][nm] for pid in res]
        print(f"{nm:50} " + " ".join(f"{x:6.1f}" for x in v) + f"  {statistics.mean(v):5.1f}")
    (ROOT / "analysis" / "search-results.json").write_text(json.dumps(res, indent=1))


def _common_round(gen, D, exact, cache, tries):
    seen = set()
    for _ in range(tries * 20):
        if len(seen) >= tries:
            break
        g = gen(); k = norm(g)
        if k in seen:
            continue
        seen.add(k)
        if k not in cache:
            cache[k] = predict(g, D)
        if cache[k] is not None:
            return D.rank[cache[k]][0]
    return 0


if __name__ == "__main__":
    main()
