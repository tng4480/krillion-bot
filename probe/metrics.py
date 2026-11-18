"""Which string-similarity metric reproduces Krillion's autocorrect?

For every probe we find the best-matching dictionary string (canonical answers + aliases)
under each candidate metric, then check:
  - top1 agreement: for corrected probes, does the metric's best match = the observed target?
  - separability: is there a threshold that accepts every corrected probe and rejects every
    rejected one? (gap = min accepted score - max rejected score; > 0 means separable)

Usage: .venv/Scripts/python probe/metrics.py data/reveal/2026-11-17.json data/probes/*.jsonl
"""
import json, re, sys
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein, OSA, JaroWinkler, Indel

def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

def trigrams(s):
    s = f"  {s} "
    return {s[i:i + 3] for i in range(len(s) - 2)}

def trigram_sim(a, b, **_):  # Postgres pg_trgm-style similarity, 0..100
    A, B = trigrams(a), trigrams(b)
    return 100 * len(A & B) / len(A | B) if A | B else 0

METRICS = {
    "levenshtein": lambda a, b, **_: 100 * Levenshtein.normalized_similarity(a, b),
    "osa(damerau)": lambda a, b, **_: 100 * OSA.normalized_similarity(a, b),
    "indel(ratio)": lambda a, b, **_: 100 * Indel.normalized_similarity(a, b),
    "jaro_winkler": lambda a, b, **_: 100 * JaroWinkler.similarity(a, b),
    "trigram": trigram_sim,
    "partial_ratio": fuzz.partial_ratio,
    "token_set": fuzz.token_set_ratio,
    "WRatio": fuzz.WRatio,
}

def load_dict(path):
    R = json.load(open(path, encoding="utf-8"))
    out = {}
    for p in R["prompts"]:
        strings, owner = [], []
        for a in p["answers"]:
            for s in [a["answer"]] + (a.get("aliases") or []):
                strings.append(norm(s)); owner.append(a["answer"])
        out[norm(p["text"])] = (strings, owner)
    return out

def main(reveal, *probe_files):
    D = load_dict(reveal)
    probes = [json.loads(l) for f in probe_files for l in open(f, encoding="utf-8")]
    probes = [p for p in probes if p["res"] in ("corr", "rej")]
    for name, scorer in METRICS.items():
        agree = total_corr = 0
        acc, rej, misses = [], [], []
        for p in probes:
            strings, owner = D[norm(p["prompt"])]
            q = norm(p["raw"])
            _, score, i = process.extractOne(q, strings, scorer=scorer)
            if p["res"] == "corr":
                total_corr += 1
                ok = norm(owner[i]) == norm(p["to"])
                agree += ok
                acc.append(score)
                if not ok: misses.append(f'{p["raw"]}->{owner[i]} (want {p["to"]})')
            else:
                rej.append(score)
        gap = min(acc) - max(rej)
        print(f"{name:14} top1 {agree}/{total_corr}  accepted>= {min(acc):5.1f}  rejected<= {max(rej):5.1f}  gap {gap:+6.1f}")
        for m in misses[:4]: print("      miss:", m)

if __name__ == "__main__":
    main(*sys.argv[1:])
