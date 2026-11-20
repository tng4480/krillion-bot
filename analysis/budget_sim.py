"""Real-time budget: more lists cost sampling time, which costs guesses. Which model and list count score best?

Measured live (Unlimited, 2026-11-20): the answer box opens ~3.2 s after the prompt text appears, then the 25 s clock
runs; typing takes ~0.22 s per guess at 100+ guesses. Sampling n lists takes t(n) (runs/speed.jsonl, one batch of
n) + 0.3 s to build the list; only the part past 3.2 s costs clock. So
    guesses G(n) = floor((24.5 - max(0, t(n) + 0.3 - 3.2)) / 0.22)
The list: spam pooled from the n lists, sorted by shape then char-trigram word-likeness, first G-10 + the first
list's 10-answer tail (env/live.py). Emulator, val and test prompts (pool >= 300), averaged over up to 4 disjoint
draws of n lists from the sampled pool (runs/eval/pool_<split>_<model>.jsonl, analysis/sample_pool.py).
Also the fixed no-model list (char trigram, analysis/det_gen.py) with the whole clock: G = 111.
Usage: .venv/bin/python analysis/budget_sim.py val|test
"""
import collections, json, statistics as st, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / d) for d in ("probe", "data", "env", "eval", "analysis")]
from build import answer_dicts, load  # noqa: E402
from emulator import Dictionary, norm  # noqa: E402
from game import Round  # noqa: E402
from live import shape, train_lm  # noqa: E402

CLOCK, FREE, BUILD, PER_GUESS = 24.5, 3.2, 0.3, 0.22
SPEED_ALIAS = {"pythia14m_distill_pipeline": "pythia14m_distill_raw", "135m_distill_pipeline": "135m_distill_raw"}


def build_lists(lists, n_spam, lm):
    pool, seen = [], set()
    for g in lists:
        for x in g[:-10]:
            k = norm(x)
            if k and k not in seen:
                seen.add(k); pool.append(x)
    pool.sort(key=lambda x: (shape(x), -lm.score(norm(x))))
    spam = pool[:n_spam]; ss = {norm(s) for s in spam}
    return spam + [x for x in lists[0][-10:] if norm(x) not in ss]


def play(ans, D, cache, guesses, tries):
    R = Round(ans, tries=tries, D=D, cache=cache)
    for g in guesses:
        if R.done:
            break
        kind, a = R.submit(g)
        if kind == "corrected":
            R.submit(a)
    return R.score


def main():
    split = sys.argv[1]
    speed = collections.defaultdict(dict)
    for l in open(ROOT / "runs/speed.jsonl"):
        r = json.loads(l); speed[r["model"]][r["n"]] = r["sec"]
    lm = train_lm()
    P, A = load()
    rows = []
    for f in sorted((ROOT / "runs/eval").glob(f"pool_{split}_*.jsonl")):
        name = f.stem[len(f"pool_{split}_"):]
        sp = speed[SPEED_ALIAS.get(name, name)]
        pools = [json.loads(l) for l in open(f)]
        N = min(len(p["lists"]) for p in pools)
        for n in sorted(sp):
            if n > N:
                continue
            G = int((CLOCK - max(0.0, sp[n] + BUILD - FREE)) / PER_GUESS)
            scores = []
            for p in pools:
                ans = answer_dicts(A, p["prompt_id"]); D = Dictionary(ans); cache = {}
                draws = [p["lists"][i * n:(i + 1) * n] for i in range(min(4, N // n))]
                scores.append(st.mean(play(ans, D, cache, build_lists(d, G - 10, lm), G) for d in draws))
            rows.append((name, n, sp[n], G, st.mean(scores), len(pools)))
            print(f"{name:28} n={n:4} sample {sp[n]:5.2f}s  guesses {G:4}  E[round] {st.mean(scores):5.1f}  "
                  f"({len(pools)} prompts)", flush=True)
    # fixed list, whole clock
    from det_gen import Tri, sampled_top
    from game import split_prompts
    Pall, E = split_prompts(P, 0)
    words = {w for pid in Pall[Pall["tsplit"] == "train"]["prompt_id"] for a in answer_dicts(A, pid)
             for w in norm(a["answer"]).split()}
    G = int(CLOCK / PER_GUESS)
    fixed = sampled_top(Tri(words), G, m=max(1000, 8 * G), seed=0)
    ids = [json.loads(l)["prompt_id"] for l in open(next((ROOT / "runs/eval").glob(f"pool_{split}_*.jsonl")))]
    s = st.mean(play(answer_dicts(A, pid), Dictionary(answer_dicts(A, pid)), {}, fixed, G) for pid in ids)
    print(f"{'fixed_list_no_model':28} n=   0 sample  0.00s  guesses {G:4}  E[round] {s:5.1f}  ({len(ids)} prompts)")
    (ROOT / f"runs/budget_{split}.json").write_text(json.dumps(rows + [("fixed_list_no_model", 0, 0, G, s, len(ids))]))


if __name__ == "__main__":
    main()
