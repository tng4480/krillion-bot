"""Build SFT data: teacher guess lists made from each prompt's own answer list, filtered by the emulator.

For every prompt, several generic strategies are fitted to its answer list (nothing prompt-specific is hand
written, so the same code works on any new reveal):
  rare      real answers scoring >= 60, sampled
  shape     character 3-gram nonsense shaped like the prompt's answers
  splice    a real word's opening + a frequent ending (fake genera, diminutives, -ville towns)
  names     first-name x surname spam (only when two-word names dominate the pool)
  modifier  a learned modifier word + a common answer ("East Nashville", "Chinese yam"); modifiers are the
            first words of multi-word answers whose remainder is itself a common answer
Each strategy is scored with the emulator (E[round] under the retry rule). Examples are drawn from the
strategies within --keep-within points of the best one, weighted by score. In a teacher list:
  - guesses the emulator maps to a basic answer (score <= 30) are dropped (they would end the round low),
  - the first --best-of candidate lists are played and the best kept,
  - --fallback basic answers go last, so a list that misses everything still scores ~10-30.
Target text: "<plan> {strategy} <guesses> g1 ; g2 ; ...".

Writes runs/sft/<name>/{train,valid}.jsonl rows {prompt_id, text, completion, plan, score} and strategy
scores to strategies.json. Usage:
  .venv/bin/python data/sft.py --name v1 --per-prompt 300
"""
import argparse, collections, json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "probe"), str(ROOT / "data"), str(ROOT / "eval")]
from emulator import Dictionary, norm, predict  # noqa: E402
from build import answer_dicts, load  # noqa: E402
from play import play_round  # noqa: E402

BASIC = 30


class Teacher:
    def __init__(self, answers, rng):
        self.rng = rng
        self.answers = answers
        self.D = Dictionary(answers)
        self.exact = set(self.D.strings)
        self.cache = {}
        self.basic = [a["answer"] for a in answers if a["score"] <= BASIC]
        self.rare = [a["answer"] for a in answers if a["score"] >= 60]
        self.words = [norm(a["answer"]) for a in answers]
        self.gens = {k: g for k, g in [("rare", self._rare()), ("shape", self._shape()), ("splice", self._splice()),
                                       ("names", self._names()), ("modifier", self._modifier())] if g}

    def mapped_score(self, g):
        k = norm(g)
        if k not in self.cache:
            self.cache[k] = predict(g, self.D)
        a = self.cache[k]
        return None if a is None else self.D.rank[a][0]

    # ----- strategies; each returns a () -> str or None if it doesn't apply -----
    def _rare(self):
        return (lambda: self.rng.choice(self.rare)) if self.rare else None

    def _shape(self, order=3):
        chains = collections.defaultdict(list)
        for w in self.words:
            w = "^" * order + w + "$"
            for i in range(len(w) - order):
                chains[w[i:i + order]].append(w[i + order])

        def gen():
            s, out = "^" * order, ""
            while len(out) < 40:
                c = self.rng.choice(chains[s])
                if c == "$":
                    break
                out += c; s = s[1:] + c
            return out.title() if self.rng.random() < 0.5 else out
        return gen

    def _splice(self, stem=4, n_suf=40, suf=4):
        w = [x for x in self.words if " " not in x and len(x) > 6]
        if len(w) < 50:
            return None
        sufs = [s for s, _ in collections.Counter(x[-suf:] for x in w).most_common(n_suf)]
        return lambda: (self.rng.choice(w)[:stem] + self.rng.choice(["", "a", "o", "i", "e"])
                        + self.rng.choice(sufs)).capitalize()

    def _names(self):
        two = [a["answer"].split() for a in self.answers if len(a["answer"].split()) == 2]
        if len(two) < 0.3 * len(self.answers) or len(two) < 200:
            return None
        F = collections.Counter(t[0] for t in two).most_common(300)
        L = collections.Counter(t[1] for t in two).most_common(1000)
        (f, fw), (l, lw) = zip(*F), zip(*L)
        return lambda: f"{self.rng.choices(f, fw)[0]} {self.rng.choices(l, lw)[0]}"

    def _modifier(self):
        basics = {norm(b) for b in self.basic}
        mods = collections.Counter()
        for w in self.words:
            t = w.split()
            if len(t) >= 2 and " ".join(t[1:]) in basics:
                mods[t[0]] += 1
        if sum(mods.values()) < 10:
            return None
        M, Mw = zip(*mods.most_common(40))
        pool = self.basic + [a["answer"] for a in self.answers if a["score"] == 60][:200]
        return lambda: f"{self.rng.choices(M, Mw)[0].title()} {self.rng.choice(pool)}"

    # ----- lists -----
    def raw_list(self, plan, k):
        parts = plan.split("+")
        out, seen = [], set()
        for _ in range(k * 6):
            g = self.gens[self.rng.choice(parts)]()
            n = norm(g)
            if not n or n in seen:
                continue
            seen.add(n)
            out.append(g)
            if len(out) >= k:
                break
        return out

    def teacher_list(self, plan, k, fallback):
        """k guesses with nothing that lands on a basic answer, then `fallback` basic answers."""
        out = []
        for g in self.raw_list(plan, k * 2):
            s = self.mapped_score(g)
            if s is not None and s <= BASIC:
                continue
            out.append(g)
            if len(out) >= k:
                break
        return out + self.rng.sample(self.basic, min(fallback, len(self.basic)))

    def score_list(self, guesses, tries):
        return play_round(guesses, self.D, self.exact, tries, cache=self.cache)["score"]

    def eval_strategy(self, plan, rounds, k, tries):
        return sum(self.score_list(self.raw_list(plan, k), tries) for _ in range(rounds)) / rounds


def fmt(plan, guesses):
    return f"<plan> {plan} <guesses> " + " ; ".join(guesses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v1")
    ap.add_argument("--per-prompt", type=int, default=300)
    ap.add_argument("--k", type=int, default=30, help="guesses per list before the fallback")
    ap.add_argument("--fallback", type=int, default=2)
    ap.add_argument("--best-of", type=int, default=2)
    ap.add_argument("--keep-within", type=float, default=15)
    ap.add_argument("--tries", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    P, A = load()
    out_dir = ROOT / "runs" / "sft" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {s: open(out_dir / f"{s}.jsonl", "w", encoding="utf-8") for s in ("train", "valid")}
    report = {}
    for p in P.itertuples():
        split = {"train": "train", "val": "valid"}.get(p.split)
        if split is None:
            continue  # test prompts never enter training data, not even for validation loss
        T = Teacher(answer_dicts(A, p.prompt_id), rng)
        names = list(T.gens)
        plans = names + [f"{x}+{y}" for i, x in enumerate(names) for y in names[i + 1:]]
        scores = {pl: T.eval_strategy(pl, 30, a.k, a.tries) for pl in plans}
        best = max(scores.values())
        keep = {pl: s for pl, s in scores.items() if s >= best - a.keep_within}
        report[p.prompt_id] = dict(text=p.text, scores=scores, keep=list(keep))
        print(f"{p.prompt_id} {p.split:5} {p.text[:55]}")
        print("   " + "  ".join(f"{pl}={s:.0f}{'*' if pl in keep else ''}" for pl, s in
                                sorted(scores.items(), key=lambda x: -x[1])))
        n = a.per_prompt if split == "train" else max(50, a.per_prompt // 4)
        pls, w = zip(*keep.items())
        got = []
        for _ in range(n):
            pl = rng.choices(pls, w)[0]
            cands = [T.teacher_list(pl, a.k, a.fallback) for _ in range(a.best_of)]
            sc = [T.score_list(c, a.tries) for c in cands]
            i = max(range(len(cands)), key=sc.__getitem__)
            got.append(sc[i])
            files[split].write(json.dumps(dict(prompt_id=p.prompt_id, text=p.text, plan=pl,
                                               completion=fmt(pl, cands[i]), score=sc[i]),
                                          ensure_ascii=False) + "\n")
        print(f"   -> {n} {split} examples, mean teacher round score {sum(got) / len(got):.1f}")
    for f in files.values():
        f.close()
    (out_dir / "strategies.json").write_text(json.dumps(report, indent=1))
    print(f"-> {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
