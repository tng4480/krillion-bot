"""Measure guessing strategies against the autocorrect emulator.

Each strategy is a generator of guesses for a prompt. We score it two ways:
  per-guess:  acceptance rate, mean score when accepted, expected score per guess
  game:       the real retry rule - submit guesses in order, a rejection is free, the first
              accepted guess ends the round. E[score] over N tries (default 40).

Usage: .venv/Scripts/python analysis/strategies.py [--n 400] [--tries 40]
"""
import argparse, collections, json, random, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "probe"))
from emulator import Dictionary, norm, predict  # noqa: E402

REVEAL = "data/reveal/2026-11-17.json"
rng = random.Random(0)


# ---------- generators: each takes (prompt answers, Dictionary) and returns a () -> str ----------

def markov(answers, D, order=3):
    """Character n-gram model over the prompt's own answers: 'plausible-shaped bullshit'."""
    words = [norm(a["answer"]) for a in answers]
    chains = collections.defaultdict(list)
    for w in words:
        w = "^" * order + w + "$"
        for i in range(len(w) - order):
            chains[w[i:i + order]].append(w[i + order])
    exact = set(D.strings)

    def gen():
        for _ in range(50):
            s, out = "^" * order, ""
            while len(out) < 40:
                c = rng.choice(chains[s])
                if c == "$": break
                out += c; s = s[1:] + c
            if out not in exact:  # novel strings only: that's the point of bullshit
                return out
        return out
    return gen


def common_pool(answers, max_score=30):
    return [a["answer"] for a in answers if a["score"] <= max_score] or [a["answer"] for a in answers]


def prefix_modifier(mods, max_score=30):
    """'East Nashville' / 'Chinese yam': put a modifier word before a common answer."""
    def make(answers, D):
        pool = common_pool(answers, max_score)
        return lambda: f"{rng.choice(mods)} {rng.choice(pool)}"
    return make


def suffix_modifier(sufs, max_score=30):
    """'Scooby-Doo 2' / 'Bambi the movie': append to a common answer."""
    def make(answers, D):
        pool = common_pool(answers, max_score)
        return lambda: f"{rng.choice(pool)}{rng.choice(sufs)}"
    return make


def name_spam(answers, D):
    """Random first name x random surname, both drawn from the prompt's own two-word answers."""
    two = [norm(a["answer"]).split() for a in answers if len(norm(a["answer"]).split()) == 2]
    firsts = collections.Counter(t[0] for t in two)
    lasts = collections.Counter(t[1] for t in two)
    F, Fw = zip(*firsts.most_common(300)); L, Lw = zip(*lasts.most_common(1000))
    return lambda: f"{rng.choices(F, Fw)[0]} {rng.choices(L, Lw)[0]}"


def splice(stem_len=4, top_suffixes=40, suf_len=4):
    """Glue a real answer's opening to a frequent ending: fake Latin genus / fake diminutive."""
    def make(answers, D):
        w = [norm(a["answer"]) for a in answers if " " not in norm(a["answer"]) and len(norm(a["answer"])) > 6]
        sufs = [s for s, _ in collections.Counter(x[-suf_len:] for x in w).most_common(top_suffixes)]
        return lambda: rng.choice(w)[:stem_len] + rng.choice(["", "a", "o", "i", "e"]) + rng.choice(sufs)
    return make


def common_answers(answers, D):
    """Baseline: just say the obvious answers (what a naive model does)."""
    pool = common_pool(answers, 30)
    return lambda: rng.choice(pool)


DIRECTIONS = ["East", "West", "North", "South", "New", "Old", "Upper", "Lower", "Port", "Fort", "Mount", "Little"]
DESCRIPTORS = ["Chinese", "Japanese", "African", "Wild", "Giant", "Bush", "Prairie", "Mountain", "Green", "White",
               "Black", "Red", "Sea", "Desert", "Indian", "Mexican", "Dwarf", "Sweet", "Bitter", "Water"]
SEQUELS = [" 2", " 3", " II", ": The Movie", " Christmas", " Returns", " (1998)", " Forever", " Again"]

STRATEGIES = {
    "p1": {"common": common_answers, "markov": markov, "sequel/extension": suffix_modifier(SEQUELS),
           "descriptor prefix": prefix_modifier(["The", "Little", "Baby", "Super", "Return of"])},
    "p2": {"common": common_answers, "markov": markov, "diminutive splice": splice(4, 40, 4)},
    "p3": {"common": common_answers, "markov": markov, "latin genus splice": splice(4, 60, 4)},
    "p4": {"common": common_answers, "markov": markov, "name spam": name_spam},
    "p5": {"common": common_answers, "markov": markov, "direction prefix": prefix_modifier(DIRECTIONS, 60),
           "suffix splice": splice(5, 4, 4)},
    "p6": {"common": common_answers, "markov": markov, "descriptor prefix": prefix_modifier(DESCRIPTORS, 60)},
    "p7": {"common": common_answers, "markov": markov},
}


def evaluate(n=400, tries=40):
    R = json.load(open(REVEAL, encoding="utf-8"))
    out = []
    for p in R["prompts"]:
        D = Dictionary(p["answers"])
        print(f'\n{p["id"]}  {p["text"][:70]}')
        print(f'  {"strategy":20} {"accept":>7} {"score|acc":>9} {"E/guess":>8} {"P(85+|acc)":>10} {"E[game]":>8} {"+preview":>8}  examples')
        for name, make in STRATEGIES[p["id"]].items():
            gen = make(p["answers"], D)
            res = []
            for _ in range(n):
                g = gen(); a = predict(g, D)
                res.append((g, a, D.rank[a][0] if a else 0))
            acc = [r for r in res if r[1]]
            # retry game: chunks of `tries` guesses, score = first accepted guess
            games, games_pv = [], []
            exact = set(D.strings)
            for i in range(0, len(res) - tries + 1, tries):
                chunk = res[i:i + tries]
                games.append(next((s for _, a, s in chunk if a), 0))
                # preview rule: a corrected (non-exact) guess is shown before scoring, so a bot can
                # decline it when the canonical answer is basic (<=30); exact answers score at once
                games_pv.append(next((s for g, a, s in chunk if a and (s > 30 or norm(g) in exact)), 0))
            row = dict(prompt=p["id"], strategy=name, accept=len(acc) / n,
                       score_acc=sum(s for *_, s in acc) / max(1, len(acc)),
                       e_guess=sum(s for *_, s in res) / n,
                       p85=sum(s >= 85 for *_, s in acc) / max(1, len(acc)),
                       e_game=sum(games) / max(1, len(games)),
                       e_game_preview=sum(games_pv) / max(1, len(games_pv)))
            out.append(row)
            ex = ", ".join(f"{g}->{a}" for g, a, s in acc[:2])
            print(f'  {name:20} {row["accept"]:7.1%} {row["score_acc"]:9.0f} {row["e_guess"]:8.1f} '
                  f'{row["p85"]:10.0%} {row["e_game"]:8.1f} {row["e_game_preview"]:8.1f}  {ex[:70]}')
    json.dump(out, open("analysis/strategies-results.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=400); ap.add_argument("--tries", type=int, default=40)
    a = ap.parse_args(); evaluate(a.n, a.tries)
