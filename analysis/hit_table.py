"""Score table for the fixed-list optimiser: what every candidate guess scores on every prompt, alone.

Rejections don't change a round's state (env/game.py Round), so a guess's outcome on a prompt is fixed: the points of
the answer the emulator matches it to (probe/emulator.py match_single: nearest answer/alias within allowed(len),
ties to the lowest score then reveal position), or 0. Bulk version of that rule with rapidfuzz cdist, checked against
emulator.predict on random pairs.
Sets: train / val / test prompts with >= 300 answers (data split), and whole days from data/reveal/<date>.json.
-> runs/fixed/hits.npz: cands, and per set S_<set> (uint8, cands x prompts) + ids_<set>
Usage: .venv/bin/python analysis/hit_table.py [K=30000] [days ...]
"""
import json, random, sys, time
from pathlib import Path

import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / d) for d in ("probe", "data", "env", "eval", "analysis")]
from build import answer_dicts, load  # noqa: E402
from emulator import Dictionary, allowed, norm, predict  # noqa: E402
from game import split_prompts  # noqa: E402

BIG = np.int64(1) << 62


def scores_for(cands, lens, D):
    """uint8 points per candidate (single words) on one prompt's Dictionary."""
    strings = D.strings
    sc = np.array([D.rank[o][0] for o in D.owner], np.int64)
    pos = np.array([D.rank[o][1] for o in D.owner], np.int64)
    key = sc * (1 << 24) + pos
    out = np.zeros(len(cands), np.uint8)
    exact = {}
    for j, s in enumerate(strings):  # allowed 0: exact match; lowest (score, pos) owner wins
        if s not in exact or key[j] < key[exact[s]]:
            exact[s] = j
    for a in (0, 1, 2, 3):
        rows = np.flatnonzero(np.array([allowed(L) for L in lens]) == a)
        if not len(rows):
            continue
        if a == 0:
            for r in rows:
                j = exact.get(cands[r])
                if j is not None:
                    out[r] = sc[j]
            continue
        step = max(1, int(2.5e8 // (8 * len(strings))))
        for s in range(0, len(rows), step):
            rr = rows[s:s + step]
            M = process.cdist([cands[r] for r in rr], strings, scorer=Levenshtein.distance,
                              score_cutoff=a, dtype=np.uint8, workers=-1)
            K = np.where(M <= a, M.astype(np.int64) * (1 << 40) + key[None, :], BIG)
            j = K.argmin(1); ok = K[np.arange(len(rr)), j] < BIG
            out[rr[ok]] = sc[j[ok]]
    return out


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    days = sys.argv[2:] or ["2026-11-18", "2026-11-19", "2026-11-20"]
    cnt = json.loads((ROOT / "runs/fixed/cands.json").read_text())
    cands = [c for c in cnt if c.isascii()][:K]
    lens = [len(c) for c in cands]
    P, A = load(); Pall, _ = split_prompts(P, 0)
    sets = {}
    for sp in ("train", "val", "test"):
        sub = Pall[(Pall["tsplit"] == sp) & (Pall["n_answers"] >= 300)]
        sets[sp] = [(pid, answer_dicts(A, pid)) for pid in sub["prompt_id"]]
    for d in days:
        r = json.load(open(ROOT / f"data/reveal/{d}.json"))
        sets[d] = [(f"{d}/{p['id']}", [dict(answer=a["answer"], aliases=a.get("aliases", []), tier=a["tier"],
                                             score=a["score"]) for a in p["answers"]]) for p in r["prompts"]]
    res = dict(cands=np.array(cands))
    rng = random.Random(0)
    for name, prompts in sets.items():
        t0 = time.time()
        S = np.zeros((len(cands), len(prompts)), np.uint8)
        for k, (pid, ans) in enumerate(prompts):
            D = Dictionary(ans)
            S[:, k] = scores_for(cands, lens, D)
            if k < 3:  # check against the emulator itself
                bad = 0
                for i in rng.sample(range(len(cands)), 300) + list(np.flatnonzero(S[:, k])[:100]):
                    a = predict(cands[i], D)
                    bad += (D.rank[a][0] if a else 0) != S[i, k]
                assert bad == 0, f"{pid}: {bad} mismatches vs emulator.predict"
        res[f"S_{name}"] = S; res[f"ids_{name}"] = np.array([p for p, _ in prompts])
        print(f"{name}: {len(prompts)} prompts, hit rate {(S > 0).mean():.4f}, {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(ROOT / "runs/fixed/hits.npz", **res)


if __name__ == "__main__":
    main()
