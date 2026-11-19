"""Seed/size sweep for the generic deterministic list (analysis/det_gen.py sampled_top): one fixed list for every
prompt, from a char trigram over TRAIN answer words. Usage: .venv/bin/python analysis/det_seeds.py val|test [seeds]"""
import collections, json, statistics as st, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "analysis")]
from det_gen import Tri, sampled_top, play, answer_dicts, load, split_prompts, Dictionary, norm  # noqa: E402

split = sys.argv[1]; seeds = range(int(sys.argv[2]) if len(sys.argv) > 2 else 6)
P, A = load(); Pall, E = split_prompts(P, 0)
T = Pall[Pall["tsplit"] == "train"]
ev = E[(E["tsplit"] == split) & (E["n_answers"] >= 300)]
words = {w for pid in T["prompt_id"] for a in answer_dicts(A, pid) for w in norm(a["answer"]).split()}
g = Tri(words)
tails = collections.defaultdict(list)
tf = {"val": "runs/eval/spamshape4_val_smollm2-360m.jsonl", "test": "runs/eval/test_spamshape4_smollm2-360m.jsonl"}
for l in (open(ROOT / tf[split]) if split in tf else []):
    r = json.loads(l); tails[r["prompt_id"]].append(r["guesses"][-10:])
dicts = {p.prompt_id: (answer_dicts(A, p.prompt_id),) for p in ev.itertuples()}
dicts = {k: (v[0], Dictionary(v[0]), {}) for k, v in dicts.items()}
for m in [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1000,3000,10000").split(",")]:
    pure, withtail = [], []
    for sd in seeds:
        L = sampled_top(g, 50, m=m, seed=sd)
        a = [play(*dicts[pid], L) for pid in dicts]
        b = [st.mean(play(*dicts[pid], L[:40] + [x for x in t if norm(x) not in set(L[:40])]) for t in tails[pid])
             for pid in dicts if tails[pid]] or [0]
        pure.append(st.mean(a)); withtail.append(st.mean(b))
    print("  per seed (50 spam):", " ".join(f"{x:.1f}" for x in pure))
    print(f"{split} m={m}: 50 spam {st.mean(pure):.1f} (sd {st.pstdev(pure):.1f}, {min(pure):.1f}-{max(pure):.1f})   "
          f"40 + model tail {st.mean(withtail):.1f} (sd {st.pstdev(withtail):.1f})", flush=True)
