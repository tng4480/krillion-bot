"""Mine the answer archive for patterns a bot could exploit. Prints a report; no emulator calls.

Sections:
  1. tiers vs pool size      how many answers are basic (<=30) and how the rest splits, by pool size
  2. stacking                P(a 2-word answer = modifier + another answer of the same prompt), its tier,
                             and the modifiers that recur across many prompts (a data-driven generic list)
  3. cross-prompt reuse      how much of a prompt's rare/deep answers already appear in OTHER prompts'
                             lists: the ceiling of "retrieve from the archive, filter by the prompt"
  4. answer shape            tier by word count, length, non-ASCII / non-English
  5. Krillion (100) answers  what the single top answer looks like
  6. repeats                 near-duplicate prompts across dives
Usage: .venv/bin/python analysis/patterns.py
"""
import collections, json, re, statistics, sys, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "probe"))
from emulator import norm  # noqa: E402


def load_all():
    prompts = []
    for f in sorted((ROOT / "data" / "reveal").glob("*.json")):
        d = json.loads(f.read_text())
        for p in d["prompts"]:
            prompts.append(dict(pid=f"{d['date']}/{p['id']}", text=p["text"], answers=p["answers"]))
    return prompts


def pct(x):
    return f"{100 * x:4.0f}%"


def main():
    P = load_all()
    print(f"{len(P)} prompts, {sum(len(p['answers']) for p in P):,} answers\n")

    # 1. tiers vs pool size
    print("## 1. tiers vs pool size")
    bins = [(0, 100), (100, 300), (300, 1000), (1000, 5000), (5000, 10**9)]
    for lo, hi in bins:
        ps = [p for p in P if lo <= len(p["answers"]) < hi]
        if not ps:
            continue
        basic = [sum(a["score"] <= 30 for a in p["answers"]) for p in ps]
        rare = [sum(a["score"] == 60 for a in p["answers"]) / len(p["answers"]) for p in ps]
        deep = [sum(a["score"] >= 85 for a in p["answers"]) / len(p["answers"]) for p in ps]
        print(f"  pool {lo:>5}-{hi if hi < 10**9 else 'inf':<5} n={len(ps):3}  basic answers/prompt median "
              f"{statistics.median(basic):4.0f}  rare {pct(statistics.mean(rare))}  deep+ {pct(statistics.mean(deep))}")

    # 2. stacking: 2-word answer whose other word is an answer of the same prompt
    print("\n## 2. modifier stacking (2-word answers built on another answer of the same prompt)")
    mod_prompts, mod_hits, stack_tiers, base_tiers = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    frac = []
    for p in P:
        single = {norm(a["answer"]): a["score"] for a in p["answers"] if " " not in norm(a["answer"])}
        two = [(norm(a["answer"]).split(), a["score"]) for a in p["answers"] if len(norm(a["answer"]).split()) == 2]
        n_st, seen = 0, set()
        for (w1, w2), s in two:
            for mod, base in ((w1, w2), (w2, w1)):
                if base in single and len(mod) >= 3 and mod.isalpha():
                    n_st += 1; stack_tiers[s] += 1; base_tiers[single[base]] += 1
                    key = (mod, "pre" if mod == w1 else "post")
                    mod_hits[key] += 1
                    if key not in seen:
                        seen.add(key); mod_prompts[key] += 1
                    break
        if p["answers"]:
            frac.append(n_st / len(p["answers"]))
    tot = sum(stack_tiers.values())
    print(f"  share of answers that are modifier+answer: mean {pct(statistics.mean(frac))}, "
          f"prompts with >=5%: {sum(f >= .05 for f in frac)}/{len(frac)}")
    print("  tier of the stacked answer: " + ", ".join(f"{s}:{pct(c / tot)}" for s, c in sorted(stack_tiers.items())))
    print("  tier of the base it stacks on: " + ", ".join(f"{s}:{pct(c / tot)}" for s, c in sorted(base_tiers.items())))
    top = sorted(mod_prompts, key=lambda k: (-mod_prompts[k], -mod_hits[k]))[:40]
    print("  modifiers recurring across prompts (prompts, hits): " +
          ", ".join(f"{w}{'_' if pos == 'pre' else ''}{'' if pos == 'pre' else '_'}".replace("__", "_") +
                    f"({mod_prompts[(w, pos)]},{mod_hits[(w, pos)]})" for w, pos in top))
    (ROOT / "analysis" / "modifiers.json").write_text(json.dumps(
        [dict(word=w, pos=pos, prompts=mod_prompts[(w, pos)], hits=mod_hits[(w, pos)]) for w, pos in
         sorted(mod_prompts, key=lambda k: (-mod_prompts[k], -mod_hits[k]))[:500]], indent=0))

    # 3. cross-prompt reuse
    print("\n## 3. cross-prompt reuse (archive retrieval ceiling)")
    where = collections.defaultdict(set)
    for i, p in enumerate(P):
        for a in p["answers"]:
            where[norm(a["answer"])].add(i)
    cov_hi, cov_basic = [], []
    for i, p in enumerate(P):
        hi = [norm(a["answer"]) for a in p["answers"] if a["score"] >= 60]
        if hi:
            cov_hi.append(sum(len(where[x]) > 1 for x in hi) / len(hi))
    print(f"  share of a prompt's rare/deep answers that appear in some other prompt: mean {pct(statistics.mean(cov_hi))}, "
          f"median {pct(statistics.median(cov_hi))}")
    # tier stability: answers in several prompts - how often rare in one, basic in another
    multi = [x for x, s in where.items() if len(s) >= 3]
    flips = 0
    score_of = collections.defaultdict(list)
    for i, p in enumerate(P):
        for a in p["answers"]:
            score_of[norm(a["answer"])].append(a["score"])
    for x in multi:
        v = score_of[x]
        flips += min(v) <= 30 < max(v)
    print(f"  answers seen in >=3 prompts: {len(multi):,}; basic in one but rare+ in another: {pct(flips / max(1, len(multi)))}")

    # 4. answer shape
    print("\n## 4. answer shape vs tier (share scoring >= 60)")
    shape = collections.defaultdict(lambda: [0, 0])
    for p in P:
        for a in p["answers"]:
            s = a["answer"]
            nw = min(len(s.split()), 4)
            na = any(ord(c) > 127 for c in s)
            for key in (f"words={nw}{'+' if nw == 4 else ''}", f"len<=6" if len(s) <= 6 else "len7-12" if len(s) <= 12 else "len>12",
                        "non-ascii" if na else "ascii", "has digit" if re.search(r"\d", s) else "no digit",
                        "has ()" if "(" in s else "no ()"):
                shape[key][0] += a["score"] >= 60; shape[key][1] += 1
    for k in sorted(shape):
        h, n = shape[k]
        print(f"  {k:10} {pct(h / n)} of {n:,}")

    # 5. Krillion answers
    print("\n## 5. Krillion (100) answers")
    kr = [(p["text"], a) for p in P for a in p["answers"] if a["score"] == 100]
    print(f"  {len(kr)} over {len(P)} prompts; mean words {statistics.mean(len(a['answer'].split()) for _, a in kr):.1f}, "
          f"mean length {statistics.mean(len(a['answer']) for _, a in kr):.1f}")
    for t, a in kr[:: max(1, len(kr) // 12)][:12]:
        print(f"    {t[:50]:50} -> {a['answer']}")

    # 6. repeated prompts
    print("\n## 6. near-duplicate prompts")
    key = lambda t: re.sub(r"[^a-z ]", "", t.lower())
    groups = collections.defaultdict(list)
    for p in P:
        groups[key(p["text"])].append(p["pid"])
    rep = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"  exact repeats: {len(rep)}")
    for k, v in list(rep.items())[:8]:
        print(f"    {k[:60]}  {v}")


if __name__ == "__main__":
    main()
