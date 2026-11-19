import sys, collections
sys.argv = [sys.argv[0], sys.argv[1]]
exec(open("analysis/preview_sim.py").read().split("res = collections")[0])
ev = collections.Counter(); stk = collections.Counter(); gain = []
for pid, ls in lists.items():
    ans = answer_dicts(A, pid); D = Dictionary(ans); cache = {}
    for gi in range(4):
        grp = ls[gi*8:(gi+1)*8]; spam, tail, O = build(grp, 40)
        R = Round(ans, tries=50, D=D, cache=cache)
        for g in spam:
            kind, a = R.submit(g)
            if kind == "corrected":
                o = is_obv(a, O); ev["corr_obvious" if o else "corr_other"] += 1
                ev[f"score_{'obv' if o else 'oth'}_{D.rank[a][0]}"] += 1
                if o:
                    b = norm(a)
                    for m, pos in MODS[:16]:
                        k2, a2 = R.submit(f"{m} {b}" if pos == "pre" else f"{b} {m}")
                        stk[k2] += 1
                        if k2 in ("corrected", "scored"): stk[f"{k2}_{D.rank[a2][0]}"] += 1
                    R = Round(ans, tries=50, D=D, cache=cache)
                break
        else:
            ev["no_corr_in_spam"] += 1
for k, v in sorted(ev.items()): print(k, v)
print(dict(stk))
