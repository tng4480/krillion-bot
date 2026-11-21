"""Does the spam model read the prompt, or is it a fixed list in disguise?

For each model, the live list (32 lists -> 101 spam + 10-answer tail, 111 guesses, as in budget_sim) is scored:
  matched   : list built for prompt i, played against prompt i's answers
  swapped   : list built for prompt i+k, played against prompt i's answers (k = 1, 7, 23; mean)
  spam only : matched, without the 10-answer tail (111 spam guesses)
  spam swap : spam only, swapped
and the overlap of spam sets: mean Jaccard of the 101 spam words between two different prompts, vs between two
disjoint draws for the same prompt (needs >= 64 lists in the pool; else n/a).
Usage: .venv/bin/python analysis/prompt_ablation.py val|test [model ...]
"""
import json, statistics as st, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / d) for d in ("probe", "data", "env", "eval", "analysis")]
from budget_sim import build_lists, play  # noqa: E402
from build import answer_dicts, load  # noqa: E402
from emulator import Dictionary, norm  # noqa: E402
from live import train_lm  # noqa: E402

G, N, SHIFTS = 111, 32, (1, 7, 23)
DEFAULT = ["360m", "135m", "pythia14m_distill_pipeline", "p14_vocab4k", "p14_vocab512"]


def jac(a, b):
    a, b = {norm(x) for x in a}, {norm(x) for x in b}
    return len(a & b) / len(a | b)


def main():
    split, models = sys.argv[1], sys.argv[2:] or DEFAULT
    lm = train_lm(); _, A = load()
    out = {}
    for m in models:
        pools = [json.loads(l) for l in open(ROOT / f"runs/eval/pool_{split}_{m}.jsonl")]
        full = [build_lists(p["lists"][:N], G - 10, lm) for p in pools]
        spam = [build_lists(p["lists"][:N], G + 50, lm)[:G] for p in pools]  # spam only (tail pushed past G)
        spam101 = [f[:G - 10] for f in full]
        ctx = []
        for p in pools:
            ans = answer_dicts(A, p["prompt_id"]); ctx.append((ans, Dictionary(ans), {}))
        L = len(pools)
        sc = lambda lists, k: st.mean(play(*ctx[i][:2], ctx[i][2], lists[(i + k) % L], G) for i in range(L))  # noqa
        r = dict(matched=sc(full, 0), swapped=st.mean(sc(full, k) for k in SHIFTS),
                 spam_only=sc(spam, 0), spam_swapped=st.mean(sc(spam, k) for k in SHIFTS),
                 overlap_other_prompt=st.mean(jac(spam101[i], spam101[(i + k) % L]) for i in range(L) for k in SHIFTS))
        if min(len(p["lists"]) for p in pools) >= 2 * N:
            r["overlap_same_prompt"] = st.mean(jac(spam101[i], build_lists(p["lists"][N:2 * N], G - 10, lm)[:G - 10])
                                               for i, p in enumerate(pools))
        out[m] = r
        print(f"{m:28} " + "  ".join(f"{k} {v:5.2f}" if "overlap" in k else f"{k} {v:5.1f}" for k, v in r.items()),
              flush=True)
    (ROOT / f"runs/prompt_ablation_{split}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
