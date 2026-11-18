"""Precompute boundary/tie probes for a dive from its reveal dictionary.

Hypothesis under test: accepted iff min Levenshtein distance to any answer/alias <= floor(len(q)/4).
For each prompt we emit probes whose true nearest distance is known:
  kind=edge   d == floor(L/4)      (should be accepted, unique nearest answer)
  kind=over   d == floor(L/4) + 1  (should be rejected)
  kind=tie    d <= floor(L/4), nearest strings belong to 2+ different answers (learn tie-break)
Writes data/probes/plan-<date>.json  {prompt_text: [{raw, kind, d, L, nearest: [answers]}...]}
"""
import json, random, re, sys
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

random.seed(7)
ABC = "abcdefghijklmnopqrstuvwxyz"
norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

def garble(s, k):
    s = list(s)
    for _ in range(k):
        op = random.choice("sdi")
        i = random.randrange(len(s))
        if op == "s": s[i] = random.choice(ABC)
        elif op == "d" and len(s) > 3: del s[i]
        else: s.insert(i, random.choice(ABC))
    return "".join(s)

def nearest(q, strings, owner):
    hits = process.extract(q, strings, scorer=Levenshtein.distance, limit=25, score_cutoff=6)
    if not hits: return 99, []
    d = min(h[1] for h in hits)
    return d, sorted({owner[i] for _, dd, i in hits if dd == d})

def plan_prompt(p, per_kind=45):
    strings, owner = [], []
    for a in p["answers"]:
        for s in [a["answer"]] + (a.get("aliases") or []):
            strings.append(norm(s)); owner.append(a["answer"])
    exact = set(strings)
    pool = [norm(a["answer"]) for a in p["answers"] if 4 <= len(norm(a["answer"])) <= 16]
    want = {"edge": per_kind, "over": per_kind, "tie": per_kind // 2}
    got = {k: [] for k in want}
    lens_seen = {k: {} for k in want}
    for _ in range(20000):
        if all(len(got[k]) >= want[k] for k in want): break
        src = random.choice(pool)
        L0 = len(src)
        k = random.choice([max(1, L0 // 4), L0 // 4 + 1, L0 // 4 + 1])
        q = garble(src, k)
        if q in exact or len(q) < 4 or " " in q.strip() and len(q) < 6: continue
        L = len(q); allow = L // 4
        d, near = nearest(q, strings, owner)
        kind = ("tie" if len(near) > 1 and d <= allow else
                "edge" if d == allow and len(near) == 1 and d > 0 else
                "over" if d == allow + 1 else None)
        if not kind or len(got[kind]) >= want[kind]: continue
        if lens_seen[kind].get(L, 0) >= 6: continue  # spread over lengths
        lens_seen[kind][L] = lens_seen[kind].get(L, 0) + 1
        got[kind].append({"raw": q, "kind": kind, "d": d, "L": L, "nearest": near[:5]})
    # interleave kinds so an early time-out still samples all three
    out, its = [], [iter(got["edge"]), iter(got["over"]), iter(got["tie"])]
    while its:
        for it in list(its):
            x = next(it, None)
            if x is None: its.remove(it)
            else: out.append(x)
    return out

if __name__ == "__main__":
    reveal = sys.argv[1]
    R = json.load(open(reveal, encoding="utf-8"))
    plan = {p["text"]: plan_prompt(p) for p in R["prompts"]}
    path = f"data/probes/plan-{R['date']}.json"
    json.dump(plan, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    for t, v in plan.items():
        c = {k: sum(x["kind"] == k for x in v) for k in ("edge", "over", "tie")}
        print(f"{len(v):3} probes {c} | {t[:55]}")
    print("wrote", path)
