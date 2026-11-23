"""The statistical fixed list: one prompt-agnostic guess list from per-guess hit statistics on train prompts.

For each candidate (runs/fixed/hits.npz, analysis/hit_table.py): p = hit rate over train prompts, e = mean points
when it hits. Rank by p * e^alpha, skip any candidate within `dedup` edits of one already taken (near-duplicates hit
the same answers), take G, then order by e descending: with independent hits that order is optimal for a fixed set
(an earlier guess only blocks a later one when it hits, so put the most valuable hits first).
Unlike analysis/opt_list.py (exact per-prompt coverage: train 80, held-out ~61-68) this barely overfits.
alpha and dedup were picked on val (alpha 0.5, dedup 2). -> runs/fixed/stat_list.json (list + scores per set)
Usage: .venv/bin/python analysis/stat_list.py [--alpha 0.5] [--dedup 2] [--G 111]
"""
import argparse, json
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parent.parent


def score(S, L):
    H = S[L] > 0
    fp = np.where(H.any(0), H.argmax(0), len(L))
    return np.where(fp < len(L), S[np.array(L)[np.minimum(fp, len(L) - 1)], np.arange(S.shape[1])], 0).mean()


def build(Z, alpha, dedup, G):
    St = Z["S_train"].astype(float); cands = Z["cands"]
    hits = (St > 0).sum(1); p = hits / (St.shape[1] + 2)
    e = np.where(hits > 0, St.sum(1) / np.maximum(hits, 1), 0)
    L = []
    for c in np.argsort(-(p * np.power(np.maximum(e, 1e-9), alpha))):
        if dedup and any(Levenshtein.distance(cands[c], cands[d], score_cutoff=dedup) <= dedup for d in L):
            continue
        L.append(int(c))
        if len(L) == G:
            break
    return sorted(L, key=lambda c: -e[c]), p, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--dedup", type=int, default=2)
    ap.add_argument("--G", type=int, default=111)
    a = ap.parse_args()
    Z = np.load(ROOT / "runs/fixed/hits.npz"); cands = Z["cands"]
    L, p, e = build(Z, a.alpha, a.dedup, a.G)
    sets = [k[2:] for k in Z.files if k.startswith("S_")]
    res = {s: float(score(Z[f"S_{s}"].astype(float), L)) for s in sets}
    out = dict(alpha=a.alpha, dedup=a.dedup, G=a.G, mean=res,
               game={s: round(res[s] * Z[f"S_{s}"].shape[1]) for s in sets if s[0].isdigit()},
               list=[str(cands[c]) for c in L],
               stats=[dict(word=str(cands[c]), hit_rate=round(float(p[c]), 4), points=round(float(e[c]), 1)) for c in L])
    (ROOT / "runs/fixed/stat_list.json").write_text(json.dumps(out, indent=1))
    print(" ".join(f"{s} {v:5.1f}" for s, v in res.items()))
    print("games:", out["game"])
    print("first 30:", " ".join(out["list"][:30]))


if __name__ == "__main__":
    main()
