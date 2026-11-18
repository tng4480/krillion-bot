"""Turn reveal JSONs into the training/eval tables.

Reads every data/reveal/<date>.json and writes:
  data/prompts.parquet   prompt_id, date, slot, text, category, split, n_answers, n_<tier>
  data/answers.parquet   prompt_id, pos, answer, aliases, tier, score, n_words, quip

`pos` is the answer's position in the reveal; the emulator uses it as the last tie-break, so keep
answers ordered by (prompt_id, pos) when rebuilding a Dictionary.

Categories come from `data/categories.json` ({prompt text: category}) when listed there, otherwise from
the keyword rules below; anything unmatched is "other" and printed so it can be labelled by hand.

Split is by CATEGORY (never by answer, and never splitting one category across train/test), so held-out
numbers measure transfer to a new kind of prompt. A category goes to test/val by a stable hash, unless
`data/categories.json` pins it under "_split": {category: split}.

Usage: .venv/bin/python data/build.py [--test-frac 0.15] [--val-frac 0.15]
"""
import argparse, hashlib, json, re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
TIERS = ["plankton", "tooclever", "schooler", "rare", "deepcut", "krillion"]

# First matching rule wins; order from specific to general. Keys are regexes on the lowercased prompt.
RULES = [
    ("games",   r"\b(video game|valorant|overwatch|league of legends|apex|pokemon|pokémon|minecraft|fortnite|zelda|mario|nintendo|playable character)\b"),
    ("film_tv", r"\b(movie|film|tv show|series|sitcom|cartoon|anime|actor|actress|character from|disney|pixar|marvel|netflix)\b"),
    ("music",   r"\b(song|band|singer|album|rapper|musician|music)\b"),
    ("books",   r"\b(book|novel|author|poem|poet|comic)\b"),
    ("sport_people", r"\b(player|athlete|footballer|nba|nfl|premier league|olympian|cricketer|tennis|golfer|boxer|driver)\b"),
    ("people",  r"\b(person|people|celebrity|president|politician|scientist|artist|king|queen|famous)\b"),
    ("places",  r"\b(city|cities|town|country|countries|capital|state|island|river|mountain|lake|place|-ville|-burg)\b"),
    ("animals", r"\b(animal|bird|fish|insect|mammal|reptile|dog breed|cat breed|dinosaur|species)\b"),
    ("food",    r"\b(food|drink|fruit|vegetable|dish|snack|dessert|cheese|beverage|cocktail|spice|herb|candy|cereal)\b"),
    ("brands",  r"\b(brand|company|logo|car make|car model)\b"),
    ("words",   r"\b(word|letter|rhymes|anagram|starts with|ends in|ends with)\b"),
]


def categorize(text, overrides):
    if text in overrides:
        return overrides[text]
    t = text.lower()
    for cat, pat in RULES:
        if re.search(pat, t):
            return cat
    return "other"


def assign_split(cat, pinned, test_frac, val_frac):
    if cat in pinned:
        return pinned[cat]
    h = int(hashlib.sha1(cat.encode()).hexdigest(), 16) % 10_000 / 10_000
    return "test" if h < test_frac else "val" if h < test_frac + val_frac else "train"


def build(test_frac, val_frac):
    ov_path = ROOT / "categories.json"
    ov = json.loads(ov_path.read_text(encoding="utf-8")) if ov_path.exists() else {}
    pinned = ov.pop("_split", {})

    prompts, answers = [], []
    for f in sorted((ROOT / "reveal").glob("*.json")):
        R = json.loads(f.read_text(encoding="utf-8"))
        for p in R["prompts"]:
            pid = f'{R["date"]}/{p["id"]}'
            tiers = pd.Series([a["tier"] for a in p["answers"]]).value_counts()
            prompts.append(dict(prompt_id=pid, date=R["date"], slot=p["id"], text=p["text"],
                                category=categorize(p["text"], ov), n_answers=len(p["answers"]),
                                **{f"n_{t}": int(tiers.get(t, 0)) for t in TIERS}))
            for pos, a in enumerate(p["answers"]):
                answers.append(dict(prompt_id=pid, pos=pos, answer=a["answer"],
                                    aliases=list(a.get("aliases") or []), tier=a["tier"], score=int(a["score"]),
                                    n_words=len(a["answer"].split()), quip=a.get("quip")))

    P = pd.DataFrame(prompts)
    P["split"] = [assign_split(c, pinned, test_frac, val_frac) for c in P["category"]]
    A = pd.DataFrame(answers)
    unknown = sorted(set(A["tier"]) - set(TIERS))
    assert not unknown, f"unknown tiers {unknown}: add them to TIERS"
    P.to_parquet(ROOT / "prompts.parquet", index=False)
    A.to_parquet(ROOT / "answers.parquet", index=False)

    print(f"{len(P)} prompts, {len(A)} answers from {P['date'].nunique()} dives")
    print(P.groupby(["split", "category"]).size().rename("prompts").to_string())
    other = P[P["category"] == "other"]["text"].tolist()
    if other:
        print(f"\nunlabelled ({len(other)}), add to data/categories.json:")
        for t in other[:30]:
            print("  ", t)
    return P, A


def load():
    """(prompts, answers) DataFrames; answers sorted by (prompt_id, pos)."""
    P = pd.read_parquet(ROOT / "prompts.parquet")
    A = pd.read_parquet(ROOT / "answers.parquet").sort_values(["prompt_id", "pos"])
    return P, A


def answer_dicts(A, prompt_id):
    """The prompt's answers in the reveal's own shape, for emulator.Dictionary."""
    sub = A[A["prompt_id"] == prompt_id]
    return [dict(answer=r.answer, aliases=list(r.aliases), tier=r.tier, score=r.score)
            for r in sub.itertuples()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--val-frac", type=float, default=0.15)
    a = ap.parse_args()
    build(a.test_frac, a.val_frac)
