"""Score every model's live lists for one dive in the emulator, on that day's answers (data/reveal/<date>.json).

Each list is played in order with corrections accepted, 50 guesses, as in live play. Also scores the fixed no-model
list (char trigram over train answer words, analysis/det_gen.py). The site archive has no aliases, so the emulator
undercounts real hits a little (see HANDOFF, "Autocorrect: what's left to explain").
Usage: .venv/bin/python analysis/score_day.py 2026-11-20
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / d) for d in ("probe", "data", "env", "eval", "analysis")]
from emulator import Dictionary  # noqa: E402
from game import Round  # noqa: E402


def play(ans, guesses, tries=50):
    R = Round(ans, tries=tries, D=Dictionary(ans))
    for i, g in enumerate(guesses):
        if R.done:
            break
        kind, a = R.submit(g)
        if kind == "corrected":
            R.submit(a)
        if R.done:
            return R.score, f"{g} -> {R.answer} (#{i + 1})"
    return R.score, ""


def main():
    date = sys.argv[1]
    rev = json.load(open(ROOT / "data" / "reveal" / f"{date}.json"))
    ans = {p["id"]: [dict(a, aliases=a.get("aliases", [])) for a in p["answers"]] for p in rev["prompts"]}
    runs = {}
    for f in sorted((ROOT / "runs" / "live" / f"{date}_all").glob("*.json")):
        runs[f.stem] = {p["pid"]: p["guesses"] for p in json.load(open(f))}
    from det_gen import Tri, sampled_top
    from live import train_lm  # noqa: F401  (same train-split words as the model pipeline)
    from build import answer_dicts, load
    from emulator import norm
    from game import split_prompts
    P, A = load()
    Pall, _ = split_prompts(P, 0)
    words = {w for pid in Pall[Pall["tsplit"] == "train"]["prompt_id"] for a in answer_dicts(A, pid)
             for w in norm(a["answer"]).split()}
    fixed = sampled_top(Tri(words), 50, m=1000, seed=0)
    runs["fixed_list_no_model"] = {pid: fixed for pid in ans}
    out = {}
    print(f"{'model':30} " + " ".join(f"{p:>4}" for p in ans) + "  total")
    for name, L in runs.items():
        res = {pid: play(ans[pid], L.get(pid, [])) for pid in ans}
        out[name] = {pid: dict(score=s, hit=h) for pid, (s, h) in res.items()}
        print(f"{name:30} " + " ".join(f"{res[p][0]:>4}" for p in ans) + f"  {sum(s for s, _ in res.values()):>5}")
    print()
    for name, r in out.items():
        print(name + ": " + "; ".join(f"{p} {v['hit']}" for p, v in r.items() if v["hit"]))
    (ROOT / "runs" / "live" / f"{date}_all_scores.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
