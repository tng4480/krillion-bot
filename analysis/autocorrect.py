"""Fit the server's autocorrect: test candidate matching rules against every logged probe.

Probes: dive 125 hand probes + the 2026-11-18 boundary run (both against data/reveal/2026-11-17.json) and the live
Dive #127 guesses (data/probes/2026-11-19_dive127.jsonl, against data/reveal/2026-11-19.json).
A rule maps (query, Dictionary) -> canonical answer or None. Scored on accept/reject and on the exact target.
Usage: .venv/bin/python analysis/autocorrect.py
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "probe")]
from rapidfuzz.distance import Levenshtein  # noqa: E402
import emulator as em  # noqa: E402
from emulator import Dictionary, allowed, norm  # noqa: E402


def load_all():
    rows = [dict(p, reveal="data/reveal/2026-11-17.json") for p in em.load_probes()]
    live = ROOT / "data/probes/2026-11-19_dive127.jsonl"
    for l in open(live, encoding="utf-8"):
        p = json.loads(l)
        rows.append(dict(prompt=p["prompt"], raw=p["raw"], acc=p["res"] == "corr", to=p.get("to"),
                         reveal="data/reveal/2026-11-19.json"))
    return rows


def strip_the(s):
    return s[4:] if s.startswith("the ") else s


class Index:
    """Extra views of a Dictionary for the candidate rules."""
    def __init__(self, D):
        self.D = D
        self.squash = [(strip_the(s).replace(" ", ""), o) for s, o in zip(D.strings, D.owner)]
        self.tokens = [(t, o) for s, o in zip(D.strings, D.owner) for t in s.split() if len(s.split()) > 1]


def pick(best, D):
    return min(best, key=lambda a: (best[a], *D.rank[a])) if best else None


def whole(q, I, by="query"):
    """Whole string, spaces removed, leading 'the' stripped; allowance from the squashed length."""
    qs = strip_the(q).replace(" ", "")
    best = {}
    for c, o in I.squash:
        lim = allowed(len(qs) if by == "query" else len(c))
        d = Levenshtein.distance(qs, c, score_cutoff=lim)
        if d <= lim:
            best[o] = min(best.get(o, 99), d)
    return pick(best, I.D)


def token(q, I):
    """Single-word query against single words of multi-word answers."""
    best = {}
    for t, o in I.tokens:
        lim = allowed(len(q))
        d = Levenshtein.distance(q, t, score_cutoff=lim)
        if d <= lim:
            best[o] = min(best.get(o, 99), d + 0.5)  # rank below a full-string match at the same distance
    return pick(best, I.D)


def rule_current(q, I):
    return em.predict(q, I.D)


def rule_whole_q(q, I):
    return em.predict(q, I.D) or whole(q, I, "query")


def rule_whole_c(q, I):
    return em.predict(q, I.D) or whole(q, I, "cand")


def rule_whole_q_multi(q, I):
    return em.predict(q, I.D) or (whole(q, I, "query") if " " in q else None)


def rule_tok(q, I):
    return rule_whole_q_multi(q, I) or (token(q, I) if " " not in q else None)


RULES = dict(current=rule_current, whole_query=rule_whole_q, whole_cand=rule_whole_c,
             whole_multi_only=rule_whole_q_multi, whole_multi_plus_token=rule_tok)


def main():
    rows = load_all()
    cache = {}
    for name, f in RULES.items():
        st = {"single": [0, 0, 0], "multi": [0, 0, 0]}
        bad = []
        for p in rows:
            key = (p["reveal"], norm(p["prompt"]))
            if key not in cache:
                R = json.load(open(ROOT / p["reveal"], encoding="utf-8"))
                for x in R["prompts"]:
                    cache[(p["reveal"], norm(x["text"]))] = Index(Dictionary(x["answers"]))
            I = cache.get(key)
            if I is None:
                continue
            q = norm(p["raw"])
            pred = f(q, I)
            k = "multi" if " " in q else "single"
            s = st[k]; s[0] += 1
            ok = (pred is not None) == p["acc"]
            tgt = ok and (not p["acc"] or norm(pred) == norm(p["to"]))
            s[1] += ok; s[2] += tgt
            if not tgt:
                bad.append(f'{"ACC" if p["acc"] else "rej"} {p["raw"]!r:22} want {p["to"]!s:28} got {pred}')
        print(f"{name:24}" + "  ".join(f"{k}: acc/rej {o}/{n} target {t}/{n}" for k, (n, o, t) in st.items()))
        if "-v" in sys.argv:
            for b in bad:
                print("     ", b)


# ---------- pg_trgm-style trigram similarity as a fallback ----------
import re as _re


def trigrams(s):
    out = set()
    for w in _re.findall(r"[a-z0-9]+", s):
        w = "  " + w + " "
        out |= {w[i:i + 3] for i in range(len(w) - 2)}
    return out


def trgm_sim(a, b):
    A, B = trigrams(a), trigrams(b)
    return len(A & B) / max(1, len(A | B))


def word_sim(q, c):
    """pg_trgm word_similarity: best match of q's trigrams against a contiguous word-extent of c (approximated by
    trying every contiguous run of c's words)."""
    Q = trigrams(q)
    ws = c.split()
    best = 0.0
    for i in range(len(ws)):
        for j in range(i + 1, len(ws) + 1):
            E = trigrams(" ".join(ws[i:j]))
            best = max(best, len(Q & E) / max(1, len(Q | E)) if E else 0)
    return best


def make_trgm(th, kind="sim", multi_only=False):
    def rule(q, I):
        a = em.predict(q, I.D)
        if a is not None or (multi_only and " " not in q):
            return a
        if not hasattr(I, "tri"):
            I.tri = [(s, o, trigrams(s)) for s, o in zip(I.D.strings, I.D.owner)]
        Q = trigrams(q)
        best, bo = th, None
        for s, o, T in I.tri:
            if not (Q & T):
                continue
            v = len(Q & T) / len(Q | T) if kind == "sim" else word_sim(q, s)
            if v >= best and (bo is None or v > best or I.D.rank[o] < I.D.rank[bo]):
                best, bo = v, o
        return bo
    return rule


for th in (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6):
    RULES[f"trgm_sim>={th}"] = make_trgm(th)
    RULES[f"trgm_sim>={th} multi"] = make_trgm(th, multi_only=True)

if __name__ == "__main__" and "--trgm" in sys.argv:
    for k in list(RULES):
        if not k.startswith("trgm") and k != "current":
            del RULES[k]
    main()


if __name__ == "__main__" and "--trgm" not in sys.argv:
    main()


def make_token_rule(which="last", extra=0):
    """Single-word query also matches one word of a multi-word answer (first/last/any), within allowed(len)+extra."""
    def rule(q, I):
        a = em.predict(q, I.D)
        if a is not None or " " in q:
            return a
        if not hasattr(I, "tok3"):
            I.tok3 = []
            for s, o in zip(I.D.strings, I.D.owner):
                t = s.split()
                if len(t) > 1:
                    I.tok3.append((t[0], t[-1], t, o))
        best = {}
        lim = allowed(len(q)) + extra
        for first, last, t, o in I.tok3:
            cands = {"last": [last], "first": [first], "any": t}[which]
            for c in cands:
                if len(c) < 4:
                    continue
                d = Levenshtein.distance(q, c, score_cutoff=lim)
                if d <= lim:
                    best[o] = min(best.get(o, 99), d)
        return pick(best, I.D)
    return rule


if __name__ == "__main__" and "--token" in sys.argv:
    RULES.clear()
    RULES["current"] = rule_current
    for w in ("last", "first", "any"):
        for e in (0, 1):
            RULES[f"token_{w}+{e}"] = make_token_rule(w, e)
    main()
