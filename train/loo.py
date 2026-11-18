"""Leave-one-prompt-out check of one RFT round: does training on the other prompts help an unseen one?

1. Sample --n lists per prompt ONCE from the policy (model + optional adapter); this is shared by all folds.
2. For each prompt h: edit/filter the other prompts' samples (train/rft.py rules), train a fresh LoRA from
   the base model, evaluate on h only. Compare with the policy's own score on h (no training on anything).
Prints a per-prompt table and writes runs/loo/<name>/results.json.
Usage:
  .venv/bin/python train/loo.py --model mlx-community/Qwen3-0.6B-4bit --name qwen3-0.6b_r1
"""
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "probe"), str(ROOT / "data"), str(ROOT / "eval"), str(ROOT / "train")]
from emulator import Dictionary  # noqa: E402
from build import answer_dicts, load  # noqa: E402
from play import MLXPolicy, parse_output, play_round  # noqa: E402
from rft import edit  # noqa: E402

PY = str(ROOT / ".venv" / "bin" / "python")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--keep", type=int, default=48)
    ap.add_argument("--min-score", type=int, default=60)
    ap.add_argument("--iters", type=int, default=0, help="0 = 2 per kept list, clamped to [60, 400]")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--eval-n", type=int, default=16)
    ap.add_argument("--method", default="sft", choices=["sft", "dpo"],
                    help="sft = RFT round (SFT on kept lists); dpo = DPO, kept lists vs the policy's lists "
                         "scoring <= --max-rejected")
    ap.add_argument("--max-rejected", type=int, default=30)
    ap.add_argument("--pairs", default="self", choices=["self", "other"],
                    help="self = rejected is the same sample before editing (differs only by its basic answers); "
                         "other = a different failing sample from the prompt (v1; learned length/garbage, degenerated)")
    ap.add_argument("--dpo-args", default="", help="extra args for train/dpo.py, e.g. '--alpha 1.0'")
    ap.add_argument("--samples-from", default=None, help="reuse runs/loo/<name>/samples.json (same policy)")
    a = ap.parse_args()

    P, A = load()
    out = ROOT / "runs" / "loo" / a.name
    out.mkdir(parents=True, exist_ok=True)
    samples_path = out / "samples.json"
    if a.samples_from and not samples_path.exists():
        samples_path.write_text((ROOT / "runs" / "loo" / a.samples_from / "samples.json").read_text())
    samples = json.loads(samples_path.read_text()) if samples_path.exists() else {}
    missing = [p for p in P.itertuples() if p.prompt_id not in samples]
    if missing:  # sample only prompts not seen yet (new captured days), so earlier samples stay comparable
        pol = MLXPolicy(a.model, a.adapter, max_tokens=512, batch=16)
        for p in missing:
            samples[p.prompt_id] = pol.sample(p.text, answer_dicts(A, p.prompt_id), None, a.n)
            samples_path.write_text(json.dumps(samples, ensure_ascii=False))
            print(f"sampled {p.prompt_id}", flush=True)
        del pol

    # per prompt: the policy's own score (round under the retry rule) and the filtered training lists
    fmt = lambda pl, g: f"<plan> {' '.join(pl.split())[:80]} <guesses> " + " ; ".join(g)
    base, kept, bad = {}, {}, {}
    for p in P.itertuples():
        D = Dictionary(answer_dicts(A, p.prompt_id)); exact = set(D.strings); cache = {}
        sc, lists, bad[p.prompt_id] = [], [], []
        for o in samples[p.prompt_id]:
            plan, g = parse_output(o)
            sc.append(play_round(g, D, exact, a.k, cache=cache)["score"])
            if sc[-1] <= a.max_rejected and g:
                bad[p.prompt_id].append(fmt(plan or "recall", g[:a.k]))
            ed = edit(g, D, cache, a.k)
            if ed:
                lists.append((play_round(ed, D, exact, a.k, cache=cache)["score"], plan or "recall", ed,
                              sc[-1], g[:a.k]))
        lists.sort(key=lambda x: -x[0])
        base[p.prompt_id] = sum(sc) / len(sc)
        kept[p.prompt_id] = [dict(prompt_id=p.prompt_id, text=p.text, plan=" ".join(pl.split())[:80], score=s,
                                  completion=fmt(pl, ed), raw_score=rs, raw=fmt(pl, raw))
                             for s, pl, ed, rs, raw in lists if s >= a.min_score][:a.keep]

    results = json.loads((out / "results.json").read_text()) if (out / "results.json").exists() else {}
    for p in P.itertuples():
        h = p.prompt_id
        if h in results:
            continue
        rows = [r for pid, rs in kept.items() if pid != h for r in rs]
        if a.method == "dpo":  # pair each kept list with a failing list from the same prompt
            rng = __import__("random").Random(0)
            if a.pairs == "self":
                rows = [dict(text=r["text"], chosen=r["completion"], rejected=r["raw"])
                        for r in rows if r["raw_score"] <= a.max_rejected]
            else:
                rows = [dict(text=r["text"], chosen=r["completion"], rejected=rng.choice(bad[r["prompt_id"]]))
                        for r in rows if bad[r["prompt_id"]]]
        d = out / f"fold_{h.replace('/', '_')}"
        d.mkdir(exist_ok=True)
        with open(d / "train.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        with open(d / "valid.jsonl", "w", encoding="utf-8") as f:  # loss-only sanity set, never the held-out prompt
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[:: max(1, len(rows) // 8)][:8])
        ck = ROOT / "checkpoints" / f"loo_{a.name}_{h.replace('/', '_')}"
        if a.method == "dpo":  # 2 pairs per step, ~2 epochs
            iters = a.iters or max(40, min(300, len(rows) // 2))  # ~1 epoch at 2 pairs/step
            cmd = [PY, str(ROOT / "train" / "dpo.py"), "--model", a.model, "--data", str(d), "--out", str(ck),
                   "--iters", str(iters)] + a.dpo_args.split()
        else:
            iters = a.iters or max(60, min(400, 2 * len(rows)))
            cmd = [PY, str(ROOT / "train" / "sft_lora.py"), "--model", a.model, "--data", str(d),
                   "--out", str(ck), "--iters", str(iters), "--lr", str(a.lr), "--eval-every", "100000"]
        with open(d / "train.log", "w") as lf:
            subprocess.run(cmd, check=True, stdout=lf, stderr=subprocess.STDOUT)
        # eval on the held-out prompt only, via a one-prompt view of the tables
        name = f"loo_{a.name}_{h.replace('/', '_')}"
        subprocess.run([PY, str(ROOT / "eval" / "play.py"), "--policy", f"mlx:{a.model}", "--adapter", str(ck),
                        "--n", str(a.eval_n), "--max-tokens", "512", "--only", h, "--name", name],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        S = json.loads((ROOT / "runs" / "eval" / f"{name}.summary.json").read_text())["prompts"][0]
        results[h] = dict(text=p.text, category=p.category, base=base[h], trained=S["e_round"], se=S["se"],
                          n_train_lists=len(rows), iters=iters)
        (out / "results.json").write_text(json.dumps(results, indent=1))
        print(f"{h} base {base[h]:5.1f} -> held-out after RFT {S['e_round']:5.1f} ±{S['se']:4.1f}  "
              f"({len(rows)} lists, {iters} it)  {p.text[:45]}", flush=True)

    b = sum(r["base"] for r in results.values()) / len(results)
    t = sum(r["trained"] for r in results.values()) / len(results)
    print(f"\nmean over {len(results)} held-out prompts: base {b:.1f} -> RFT {t:.1f}")


if __name__ == "__main__":
    main()
