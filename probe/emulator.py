"""Offline emulator of Krillion's autocorrect, scored against real probe outcomes.

Confirmed rule (single-word queries, 2026-11-18 boundary run, ~250 probes):
    accepted iff min Levenshtein(query, any answer-or-alias) <= allowed(len(query))
    allowed: len<=4 -> 0, 5-7 -> 1, 8-11 -> 2, 12+ -> 3
Multi-word queries: same word count, and every word within allowed(len(word)) of its counterpart
(136/149; the 13 misses are unexplained acceptances, possibly a semantic fallback).
Ties: lowest total distance, then lowest score (most common answer), then reveal position.

Usage: .venv/Scripts/python probe/emulator.py
"""
import json, re
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

REVEAL = "data/reveal/2026-11-17.json"
norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def allowed(L):
    return 0 if L <= 4 else 1 if L <= 7 else 2 if L <= 11 else 3


class Dictionary:
    def __init__(self, answers):
        self.strings, self.owner = [], []
        # tie-break key: lower score = more commonly submitted; reveal position approximates
        # popularity within a tier (true submission counts aren't in the reveal)
        self.rank = {a["answer"]: (a["score"], pos) for pos, a in enumerate(answers)}
        for a in answers:
            for s in [a["answer"]] + (a.get("aliases") or []):
                self.strings.append(norm(s)); self.owner.append(a["answer"])
        # token-level index, handy for multi-word hypotheses
        self.tokens = sorted({t for s in self.strings for t in s.split()})

    def nearest(self, q, limit=10):
        """[(string, distance, answer)] sorted by distance."""
        hits = process.extract(q, self.strings, scorer=Levenshtein.distance, limit=limit)
        return [(m, d, self.owner[i]) for m, d, i in hits]


def match_single(q, D):
    """Nearest answer within the allowance; ties go to the most common answer.

    Observed: server always picks min distance (85/85); on ties with differing scores it picks
    the LOWEST score, i.e. most common (37/37). Same-score ties (46) follow hidden popularity.
    """
    hits = process.extract(q, D.strings, scorer=Levenshtein.distance,
                           score_cutoff=allowed(len(q)), limit=None)
    if not hits:
        return None
    best = {}
    for m, d, i in hits:
        a = D.owner[i]
        best[a] = min(best.get(a, 99), d)
    return min(best, key=lambda a: (best[a], *D.rank[a]))


def match_multi(q, D):
    """Return the canonical answer a multi-word query maps to, or None if rejected.

    Evidence so far (query -> outcome):
      accepted: 'whioe lquat'->White loquat, 'lcevi greetwowd'->Levi Greenwood,
                'sield bldedwit'->Field blewit, 'eddzge shaw'->Eddie Shaw,
                'jfitte melon'->Bitter melon, 'manuqel fmrande'->Manuel Fernandes
      rejected: 'jloe retlly' (Joe Reilly?), 'baiana stm' (Banana stem), 'liply bulh' (Lily bulb),
                'ircv bolete', 'gea losvt', 'ccmea bug'
    """
    # Rule that fit best (136/149, and never accepts something the server rejected): the candidate must
    # have the same number of words, and every query word must be within allowed(len(word)) of its
    # counterpart. Whole-string Levenshtein (81/149), anchored variants (~105) and fuzz ratios with a
    # tuned threshold (<=132) all do worse. The 13 remaining misses are acceptances no string metric
    # explains ('littlev rd en' -> Raven the Little Rascal, 'canin the aat' -> The Cat in the Hat),
    # which suggests a semantic fallback on the server.
    qt = q.split()
    best = {}
    for s, owner in zip(D.strings, D.owner):
        ct = s.split()
        if len(ct) != len(qt):
            continue
        total = 0
        for a, b in zip(qt, ct):
            d = Levenshtein.distance(a, b, score_cutoff=allowed(len(a)))
            if d > allowed(len(a)):
                break
            total += d
        else:
            best[owner] = min(best.get(owner, 99), total)
    return min(best, key=lambda a: (best[a], *D.rank[a])) if best else None


def predict(q, D):
    q = norm(q)
    return match_multi(q, D) if " " in q else match_single(q, D)


def load_probes():
    rows = []
    for l in open("data/probes/2026-11-18_dive125.jsonl", encoding="utf-8"):
        p = json.loads(l)
        if p["res"] in ("corr", "rej"):
            rows.append({"prompt": p["prompt"], "raw": p["raw"], "acc": p["res"] == "corr", "to": p.get("to")})
    texts = list(json.load(open("data/probes/plan-2026-11-17.json", encoding="utf-8")))
    for x in json.load(open("data/probes/run-2026-11-18-boundary.joined.json", encoding="utf-8")):
        rows.append({"prompt": texts[x["r"]], "raw": x["raw"], "acc": x["acc"], "to": x["to"]})
    return rows


def evaluate():
    R = json.load(open(REVEAL, encoding="utf-8"))
    dicts = {norm(p["text"]): Dictionary(p["answers"]) for p in R["prompts"]}
    stats = {"single": [0, 0, 0], "multi": [0, 0, 0]}  # [n, accept/reject correct, target correct]
    misses = []
    for p in load_probes():
        D = dicts[norm(p["prompt"])]
        pred = predict(p["raw"], D)
        k = "multi" if " " in norm(p["raw"]) else "single"
        s = stats[k]; s[0] += 1
        ok = (pred is not None) == p["acc"]
        s[1] += ok
        s[2] += ok and (not p["acc"] or norm(pred) == norm(p["to"]))
        if k == "multi" and not (ok and (not p["acc"] or norm(pred) == norm(p["to"]))):
            misses.append(f'{"ACC" if p["acc"] else "rej"} {p["raw"]!r:24} want {p["to"]}  got {pred}')
    for k, (n, ok, tgt) in stats.items():
        print(f"{k:6}: accept/reject {ok}/{n} ({ok / n:.0%})   exact outcome {tgt}/{n} ({tgt / n:.0%})")
    print(f"\nmulti-word misses ({len(misses)}):")
    for m in misses[:25]: print("  ", m)


if __name__ == "__main__":
    evaluate()
