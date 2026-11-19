"""Eval harness: play Daily Dive rounds against the emulator and report E[round score].

For each prompt, a policy produces N guess lists. Each list is played as one round under the real rule:
guesses are submitted in order (duplicates skipped, at most --tries of them), a rejection is free, and the
first accepted guess ends the round with that answer's score. With --preview, a corrected (non-exact) guess
that maps to an answer scoring <= --decline-at is declined and play continues; exact answers score at once.

Policies:
  mlx:<model>   an mlx-lm model (HF id or local path), optional --adapter for LoRA weights
  common        say the obvious answers (sanity floor, ~28)
  markov        char-model nonsense trained on the prompt's own answers (leaky upper reference)

Every sample is written to runs/eval/<name>.jsonl (prompt, raw output, guesses, outcome) so RFT can reuse
them. Usage:
  .venv/bin/python eval/play.py --policy common
  .venv/bin/python eval/play.py --policy mlx:mlx-community/SmolLM2-135M-Instruct --n 8 --split all
"""
import argparse, json, re, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "probe"), str(ROOT / "data"), str(ROOT / "analysis")]
from emulator import Dictionary, norm, predict  # noqa: E402
from build import answer_dicts, load  # noqa: E402

# ---------- the model's input/output format (shared with SFT data prep) ----------

SYSTEM = (
    "You play Krillion. For a prompt like 'Name a ...', list guesses. Rare valid answers score high, "
    "common ones score low, and wrong guesses cost nothing. The first accepted guess ends the round, so put "
    "rare, specific answers first and obvious answers last. Reply as: <plan> strategy <guesses> g1 ; g2 ; ..."
)


SYSTEM_SPAM = (
    "You play Krillion. For a prompt like 'Name a ...', list about 30 guesses. Wrong guesses are free and near-misses "
    "get autocorrected to real answers, so write many plausible-looking entries of the right kind that obey the "
    "prompt's rules, then end with 3 obvious answers. Reply as: <guesses> g1 ; g2 ; ..."
)
SYSTEMS = {"plan": SYSTEM, "spam": SYSTEM_SPAM}


def render_prompt(tokenizer, text, system="plan"):
    msgs = [{"role": "system", "content": SYSTEMS[system]}, {"role": "user", "content": text}]
    if getattr(tokenizer, "chat_template", None):
        # enable_thinking=False: Gemma 4 / Qwen 3 templates otherwise spend the token budget reasoning
        kw = dict(add_generation_prompt=True, tokenize=False, enable_thinking=False)
        try:
            return tokenizer.apply_chat_template(msgs, **kw)
        except Exception:  # some templates reject a system role
            msgs = [{"role": "user", "content": SYSTEMS[system] + "\n\n" + text}]
            return tokenizer.apply_chat_template(msgs, **kw)
    return f"{SYSTEMS[system]}\n\nPrompt: {text}\n"


def parse_output(out):
    """-> (plan or None, [guesses]). Tolerates missing tags, numbering and bullets."""
    plan = None
    # drop reasoning blocks (Gemma 4 '<|channel>thought ... <channel|>', Qwen '<think> ... </think>');
    # an unterminated one means the model never got to answering
    out = re.sub(r"<\|channel>.*?(<channel\|>|$)|<think>.*?(</think>|$)", "", out, flags=re.S)
    G = r"<?\s*guesses\s*>"  # small models often drop the '<'
    m = re.search(r"<plan>(.*?)(?=" + G + r"|$)", out, re.S)
    if m:
        plan = m.group(1).strip() or None
    m = re.search(G + r"(.*)", out, re.S)
    body = m.group(1) if m else out
    parts = re.split(r"[;\n]|,(?=\s)", body)
    guesses = []
    for g in parts:
        g = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", g)
        g = re.sub(r"\s*\([^)]*\)\s*$", "", g).strip().strip("\"'`.*")  # 'Elephant (8)' -> 'Elephant'
        if 0 < len(g) <= 60 and not g.startswith("<"):
            guesses.append(g)
    return plan, guesses


# ---------- round simulation ----------

def play_round(guesses, D, exact, tries=40, preview=False, decline_at=30, cache=None):
    """-> dict(score, answer, guess, n_submitted, n_accepted_declined)."""
    seen, declined = set(), 0
    for g in guesses:
        k = norm(g)
        if not k or k in seen:
            continue
        seen.add(k)
        if len(seen) > tries:
            break
        if cache is not None and k in cache:
            a = cache[k]
        else:
            a = predict(g, D)
            if cache is not None:
                cache[k] = a
        if a is None:
            continue
        s = D.rank[a][0]
        if preview and k not in exact and s <= decline_at:
            declined += 1
            continue
        return dict(score=s, answer=a, guess=g, n_submitted=len(seen), declined=declined)
    return dict(score=0, answer=None, guess=None, n_submitted=min(len(seen), tries), declined=declined)


# ---------- policies ----------

class MLXPolicy:
    def __init__(self, model, adapter=None, temp=1.0, top_p=0.95, max_tokens=256, batch=8, system="plan"):
        self.system = system
        from mlx_lm import load as mlx_load
        from mlx_lm.sample_utils import make_sampler
        self.model, self.tok = mlx_load(model, adapter_path=adapter)
        self.sampler = make_sampler(temp=temp, top_p=top_p)
        self.max_tokens, self.batch = max_tokens, batch

    def sample(self, text, answers, D, n):
        from mlx_lm import batch_generate
        ids = self.tok.encode(render_prompt(self.tok, text, self.system), add_special_tokens=False)
        outs = []
        while len(outs) < n:
            k = min(self.batch, n - len(outs))
            r = batch_generate(self.model, self.tok, [ids] * k, max_tokens=self.max_tokens, sampler=self.sampler)
            outs += r.texts
        return outs


class GeneratorPolicy:
    """Wraps an analysis/strategies.py generator: each sample is a list of `tries` draws."""
    def __init__(self, make, tries):
        self.make, self.tries = make, tries

    def sample(self, text, answers, D, n):
        gen = self.make(answers, D)
        return ["<guesses> " + " ; ".join(gen() for _ in range(self.tries)) for _ in range(n)]


def make_policy(a):
    if a.policy.startswith("mlx:"):
        return MLXPolicy(a.policy[4:], a.adapter, a.temp, a.top_p, a.max_tokens, a.batch, a.system)
    import strategies
    gens = {"common": strategies.common_answers, "markov": strategies.markov}
    return GeneratorPolicy(gens[a.policy], a.tries)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="common")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", default="all", help="train|val|test|all")
    ap.add_argument("--only", default=None, help="comma-separated prompt_ids to evaluate")
    ap.add_argument("--n", type=int, default=16, help="samples (rounds) per prompt")
    ap.add_argument("--tries", type=int, default=40)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--decline-at", type=int, default=30)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", default=None, help="output name under runs/eval/")
    ap.add_argument("--system", default="plan", choices=["plan", "spam"])
    ap.add_argument("--tsplit", default=None, help="date split (env/game.py): train|val|test, pools >= --min-pool")
    ap.add_argument("--min-pool", type=int, default=300)
    a = ap.parse_args()

    P, A = load()
    if a.tsplit:
        sys.path.insert(0, str(ROOT / "env"))
        from game import split_prompts
        _, P = split_prompts(P, a.min_pool)
        P = P[P["tsplit"] == a.tsplit].assign(split=a.tsplit)
    elif a.split != "all":
        P = P[P["split"] == a.split]
    if a.only:
        P = P[P["prompt_id"].isin(a.only.split(","))]
    policy = make_policy(a)
    name = a.name or re.sub(r"[^A-Za-z0-9_.-]+", "_", a.policy) + ("_pv" if a.preview else "")
    out_path = ROOT / "runs" / "eval" / f"{name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(out_path, "w", encoding="utf-8") as f:
        for p in P.itertuples():
            answers = answer_dicts(A, p.prompt_id)
            D = Dictionary(answers)
            exact, cache = set(D.strings), {}
            t0 = time.time()
            outs = policy.sample(p.text, answers, D, a.n)
            t_gen = time.time() - t0
            scores, n_guesses, n_plan = [], [], 0
            for o in outs:
                plan, guesses = parse_output(o)
                r = play_round(guesses, D, exact, a.tries, a.preview, a.decline_at, cache)
                scores.append(r["score"]); n_guesses.append(len(guesses)); n_plan += plan is not None
                f.write(json.dumps(dict(prompt_id=p.prompt_id, text=p.text, output=o, plan=plan,
                                        guesses=guesses, **r), ensure_ascii=False) + "\n")
            mean = statistics.mean(scores)
            se = statistics.stdev(scores) / len(scores) ** 0.5 if len(scores) > 1 else 0.0
            rows.append(dict(prompt_id=p.prompt_id, category=p.category, split=p.split, e_round=mean, se=se,
                             p_zero=sum(s == 0 for s in scores) / len(scores),
                             guesses=statistics.mean(n_guesses), plan_rate=n_plan / len(outs),
                             gen_s=t_gen, play_s=time.time() - t0 - t_gen))
            r = rows[-1]
            print(f'{p.prompt_id:14} {p.split:5} {p.category:13} E={mean:5.1f}±{se:4.1f}  zero={r["p_zero"]:4.0%} '
                  f'guesses={r["guesses"]:5.1f}  plan={r["plan_rate"]:4.0%}  gen {t_gen:5.1f}s  {p.text[:50]}',
                  flush=True)

    print("\nby split:")
    for s in sorted({r["split"] for r in rows}):
        v = [r["e_round"] for r in rows if r["split"] == s]
        print(f"  {s:5}  E[round]={statistics.mean(v):5.1f}  over {len(v)} prompts")
    allv = [r["e_round"] for r in rows]
    print(f"  all    E[round]={statistics.mean(allv):5.1f}")
    summary = dict(policy=a.policy, adapter=a.adapter, args=vars(a), prompts=rows)
    (out_path.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=1))
    print(f"\nsamples -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
