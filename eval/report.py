"""Table of E[round] per run from runs/eval/*.summary.json (one column per prompt, plus split means).

Usage: .venv/bin/python eval/report.py [glob, default 'zs_*']
"""
import glob, json, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
pat = sys.argv[1] if len(sys.argv) > 1 else "zs_*"
rows = []
for f in sorted(glob.glob(str(ROOT / "runs" / "eval" / f"{pat}.summary.json"))):
    S = json.loads(Path(f).read_text())
    ps = S["prompts"]
    by = lambda s: statistics.mean([p["e_round"] for p in ps if p["split"] == s] or [float("nan")])
    rows.append((Path(f).name.removesuffix(".summary.json"), S["args"]["n"], ps,
                 statistics.mean(p["e_round"] for p in ps), by("train"), by("val"), by("test")))
rows.sort(key=lambda r: -r[3])
if not rows:
    sys.exit("no runs")
cats = [f'{p["prompt_id"].split("/")[1]}:{p["category"][:6]}' for p in rows[0][2]]
print(f'| run | n | all | train | val | test | ' + " | ".join(cats) + " |")
print("|---" * (6 + len(cats)) + "|")
for name, n, ps, a, tr, va, te in rows:
    print(f"| {name} | {n} | **{a:.1f}** | {tr:.1f} | {va:.1f} | {te:.1f} | "
          + " | ".join(f'{p["e_round"]:.0f}' for p in ps) + " |")
