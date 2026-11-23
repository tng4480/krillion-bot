"""Optimise one fixed, prompt-agnostic guess list on train prompts; score it on val / test / held-out days.

Objective: mean over prompts of the points of the first guess in the list that the emulator accepts (later guesses
never get typed on that prompt). Table from analysis/hit_table.py. Coordinate ascent over slots: for slot i, the
gain of putting candidate c there is sum over prompts still open at i that c hits of (points(c) - what that prompt
would otherwise score from slot i on). Take the best, next slot, repeat passes until no slot changes. Restarts from
shuffled greedy starts. --min-support m: only candidates that hit >= m train prompts (guards against one-offs).
Writes runs/fixed/list_m<m>.json; checks the table score against a real emulator replay (budget_sim.play).
Usage: .venv/bin/python analysis/opt_list.py [--min-support 1 2 3 5 8] [--G 111] [--restarts 4]
"""
import argparse, json, statistics as st, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / d) for d in ("probe", "data", "env", "eval", "analysis")]


def first_hits(S, L):
    """per prompt: (position of first hit or len(L), its points)"""
    H = S[L] > 0
    fp = np.where(H.any(0), H.argmax(0), len(L))
    v = np.where(fp < len(L), S[np.array(L)[np.minimum(fp, len(L) - 1)], np.arange(S.shape[1])], 0)
    return fp, v.astype(np.float64)


def score(S, L):
    return first_hits(S, L)[1].mean()


def optimise(S, G, rng, cand_ok):
    C, P = S.shape
    Sf = S.astype(np.float32); Hit = S > 0
    # greedy start: best immediate gain over open prompts, a little shuffled
    L, open_ = [], np.ones(P, bool)
    for _ in range(G):
        g = (Sf[:, open_]).sum(1) + rng.random(C) * 1e-3
        g[~cand_ok] = -1; g[L] = -1
        c = int(g.argmax()); L.append(c); open_ &= ~Hit[c]
    best = score(S, L)
    for _ in range(50):  # passes
        changed = False
        for i in range(G):
            fp, v = first_hits(S, L)
            rest = L[i + 1:]
            if rest:
                fp2, v2 = first_hits(S, rest)
            else:
                v2 = np.zeros(P)
            alt = np.where(fp == i, v2, v)          # what each prompt scores from slot i on without L[i]
            op = fp >= i
            gain = (np.where(Hit[:, op], Sf[:, op] - alt[op][None, :], 0)).sum(1)
            gain[~cand_ok] = -1e9
            taken = np.zeros(C, bool); taken[L] = True; taken[L[i]] = False
            gain[taken] = -1e9
            c = int(gain.argmax())
            if c != L[i] and gain[c] > gain[L[i]] + 1e-9:
                L[i] = c; changed = True
        s = score(S, L)
        if not changed or s <= best + 1e-12:
            best = max(best, s)
            if not changed:
                break
        best = max(best, s)
    return L, score(S, L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-support", type=int, nargs="+", default=[1, 2, 3, 5, 8])
    ap.add_argument("--G", type=int, default=111)
    ap.add_argument("--restarts", type=int, default=4)
    a = ap.parse_args()
    Z = np.load(ROOT / "runs/fixed/hits.npz")
    cands = list(Z["cands"]); sets = [k[2:] for k in Z.files if k.startswith("S_")]
    St = Z["S_train"]; support = (St > 0).sum(1)
    days = [s for s in sets if s[0].isdigit()]
    print(f"{len(cands)} candidates, train prompts {St.shape[1]}, sets {sets}")
    rows = {}
    for m in a.min_support:
        ok = support >= m
        best = None
        for r in range(a.restarts):
            L, s = optimise(St, a.G, np.random.default_rng(r), ok)
            if best is None or s > best[1]:
                best = (L, s)
        L = best[0]
        res = {s: score(Z[f"S_{s}"], L) for s in sets}
        day_tot = {d: res[d] * Z[f"S_{d}"].shape[1] for d in days}
        rows[m] = dict(list=[cands[i] for i in L], **{f"mean_{k}": v for k, v in res.items()},
                       **{f"game_{k}": v for k, v in day_tot.items()}, eligible=int(ok.sum()))
        print(f"min-support {m:2} (eligible {ok.sum():6}): train {res['train']:5.1f}  val {res['val']:5.1f}  "
              f"test {res['test']:5.1f}  days " + " ".join(f"{d[5:]} {t:4.0f}" for d, t in day_tot.items()), flush=True)
        (ROOT / f"runs/fixed/list_m{m}.json").write_text(json.dumps(rows[m], indent=1))
    # replay check with the real emulator round for one list
    from budget_sim import play
    from build import answer_dicts, load
    from emulator import Dictionary
    _, A = load(); m = a.min_support[0]; L = rows[m]["list"]
    ids = list(Z["ids_val"])
    rep = st.mean(play(answer_dicts(A, pid), Dictionary(answer_dicts(A, pid)), {}, L, a.G) for pid in ids)
    print(f"check: val replay through emulator Round {rep:.1f} vs table {rows[m]['mean_val']:.1f}")


if __name__ == "__main__":
    main()
