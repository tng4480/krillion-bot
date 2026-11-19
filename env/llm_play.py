"""Let an MLX language model play turn-based Krillion rounds (env/game.py) and report how well it plays.

Each game is a short chat: the model gets the prompt (+ the game state) and writes a batch of guesses; the game
processes them until the first correction or score, and the next turn shows what happened. Many games run in one
batched generation (several prompts x several games), so a turn costs one batch_generate call.

Prompt styles:
  rules  the strategy in the instructions (invent plausible entries, use typo windows, stack modifiers, decline
         famous corrections, obvious answers only at the end), plus a per-game tactic for diversity
  plain  just the game rules (control: what the model does untold)
Reports per prompt and overall: mean round score, zero rate, basic rate (accepted <= 30), share of points won
through autocorrect vs exact hits (spam skill vs knowledge), distinct guesses across games, time.
Games are saved to runs/games/<name>.jsonl (full transcripts) for distillation.
Usage:
  .venv/bin/python env/llm_play.py --model <path-or-hf-id> --style rules --split val --prompts 20 --games 4
"""
import argparse, json, random, re, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "env"), str(ROOT / "probe"), str(ROOT / "data"), str(ROOT / "eval")]
from game import Round, split_prompts  # noqa: E402
from emulator import Dictionary, norm  # noqa: E402
from build import answer_dicts, load  # noqa: E402
from play import parse_output  # noqa: E402

RULES = (
    "You are playing Krillion. Each round has a prompt like 'Name a type of cheese'. Answers score by rarity: "
    "obvious answers score 10-30, rare ones 85. Wrong guesses are free and you have 40 guesses. The game "
    "autocorrects near-misses: a guess within one to three typos of a real answer counts as that answer, and "
    "every word of a multi-word guess gets its own typo allowance.\n"
    "So you don't need to know rare answers. Each guess must look exactly like a real entry for the prompt (a "
    "real-looking name, title or word of that kind - never a description or a phrase about the prompt), obey every "
    "rule in the prompt, and be close in spelling to things that might exist: half-remembered lesser-known items "
    "with a letter changed, foreign-looking forms, long specific names, or a modifier plus an item. Never write a "
    "famous, obvious answer exactly. Never repeat a guess.\n"
    "Example for 'Name a type of cheese': Gorgonzella ; Caciotto ; Brie de Mauxe ; Smoked Havarty ; Tallegio ; "
    "Vacherine ; Aged Mimolett ; Kasseri Blanco ; Ossau-Irraty ; Pecorina Sardo ; Bleu de Gexx ; Limburgo\n"
    "Example for 'Name a city ending in -ville': Marbleville ; Crandonville ; Oakhurstville ; Lyndeville ; "
    "Port Ashville ; Bexville ; Thornville ; Harroldsville ; Granvillette ; East Pikeville ; Selbyville ; Ruthville\n"
    "When the game says a guess was corrected to Y: if Y is famous and obvious, don't accept it; keep guessing and "
    "try modifiers on top of Y (e.g. Y -> 'Smoked Y', 'Y Blanco', 'Wild Y'). If Y is unfamiliar or specific, write "
    "Y exactly as your first guess to accept it.\n"
    "If the message says it is your last turn and nothing was accepted, give the most obvious answers instead.\n"
    "Reply only with: <guesses> g1 ; g2 ; ... (12 guesses, no explanations)."
)
PLAIN = (
    "You are playing Krillion. Each round has a prompt like 'Name a type of dress'. Rare valid answers score "
    "high, common ones low, wrong guesses cost nothing, and the first accepted guess ends the round. You have 40 "
    "guesses. Reply only with: <guesses> g1 ; g2 ; ... (12 guesses, no explanations)."
)
TACTICS = [
    "This game, favour slight misspellings of lesser-known real items.",
    "This game, favour invented foreign-looking forms (Latin, Italian, Japanese, Slavic, ...).",
    "This game, favour two-word combinations: a modifier plus an item.",
    "This game, favour long, specific-sounding names.",
    "This game, start many guesses with these letters: {L}.",
    "This game, mix every kind of invention.",
]


class Game:
    def __init__(self, pid, text, answers, D, cache, tries, style, rng):
        self.pid, self.text, self.style = pid, text, style
        self.R = Round(answers, tries, D, cache)
        self.revealed, self.tried, self.turns, self.last_obs, self.outputs = [], [], 0, None, []
        t = rng.choice(TACTICS)
        self.tactic = t.replace("{L}", ", ".join(rng.sample(["ka", "mo", "tr", "gr", "be", "sa", "lu", "ve", "po",
                                                             "ch", "da", "ri", "no", "fe", "zu"], 3)))

    def messages(self):
        sys_ = RULES + "\n" + self.tactic if self.style == "rules" else PLAIN
        left = self.R.tries - self.R.n
        u = f"Prompt: {self.text}\nGuesses left: {left}."
        if self.tried:
            u += "\nAlready tried (don't repeat): " + ", ".join(self.tried[-40:])
        if self.revealed:
            u += "\nRevealed by corrections (declined unless you accept): " + ", ".join(self.revealed)
        if self.last_obs:
            u += "\nLast turn: " + self.last_obs
        if left <= 12:
            u += "\nThis is your last turn."
        return [{"role": "system", "content": sys_}, {"role": "user", "content": u}]

    def step(self, out):
        self.outputs.append(out)
        _, guesses = parse_output(out)
        self.turns += 1
        before = self.R.n
        self.last_obs = self.R.turn(guesses[:16])
        self.tried = [g for g, o, _ in self.R.log]
        if self.R.pending is not None and self.R.pending not in self.revealed:
            self.revealed.append(self.R.pending)
        if self.R.n == before and not self.R.done:   # nothing new submitted: count it as a wasted turn
            self.R.n = min(self.R.tries, self.R.n + 4)
        return self.R.done or self.R.n >= self.R.tries


def render(tok, msgs):
    kw = dict(add_generation_prompt=True, tokenize=False, enable_thinking=False)
    return tok.apply_chat_template(msgs, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--style", default="rules", choices=["rules", "plain"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--prompts", type=int, default=20, help="first N eval prompts of the split (0 = all)")
    ap.add_argument("--games", type=int, default=4, help="games per prompt")
    ap.add_argument("--group", type=int, default=4, help="prompts played concurrently (batch = group x games)")
    ap.add_argument("--tries", type=int, default=40)
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--mem-gb", type=float, default=6.0)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import batch_generate, load as mlx_load
    from mlx_lm.sample_utils import make_sampler
    mx.set_memory_limit(int(a.mem_gb * 2**30)); mx.set_cache_limit(512 * 2**20)
    model, tok = mlx_load(a.model, adapter_path=a.adapter)
    sampler = make_sampler(temp=a.temp, top_p=a.top_p)

    P, A = load()
    P, E = split_prompts(P)
    E = E[E["tsplit"] == a.split]
    if a.prompts:
        E = E.iloc[:a.prompts]
    out_path = ROOT / "runs" / "games" / f"{a.name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    rows, t_all = [], time.time()
    rows_e = list(E.itertuples())
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(0, len(rows_e), a.group):
            chunk, games = rows_e[i:i + a.group], []
            for p in chunk:
                ans = answer_dicts(A, p.prompt_id)
                D, cache = Dictionary(ans), {}
                games += [Game(p.prompt_id, p.text, ans, D, cache, a.tries, a.style, rng) for _ in range(a.games)]
            t0 = time.time()
            for _ in range(a.max_turns):
                live = [g for g in games if not (g.R.done or g.R.n >= g.R.tries)]
                if not live:
                    break
                ids = [tok.encode(render(tok, g.messages()), add_special_tokens=False) for g in live]
                r = batch_generate(model, tok, ids, max_tokens=a.max_tokens, sampler=sampler)
                for g, o in zip(live, r.texts):
                    g.step(o)
            dt = time.time() - t0
            for p in chunk:
                gs = [g for g in games if g.pid == p.prompt_id]
                sc = [g.R.score for g in gs]
                won = [g.R.log[-1] for g in gs if g.R.score]
                via_corr = sum(1 for g in gs if g.R.score and len(g.R.log) >= 2 and g.R.log[-2][1] == "corrected"
                               and norm(g.R.log[-2][2] or "") == norm(g.R.log[-1][2]))
                guesses = [norm(x) for g in gs for x, _, _ in g.R.log]
                row = dict(prompt_id=p.prompt_id, text=p.text, n_answers=int(p.n_answers), e=statistics.mean(sc),
                           zero=sum(s == 0 for s in sc) / len(sc), basic=sum(0 < s <= 30 for s in sc) / len(sc),
                           via_correction=via_corr / max(1, len(won)),
                           distinct=len(set(guesses)) / max(1, len(guesses)),
                           submitted=statistics.mean(g.R.n for g in gs), turns=statistics.mean(g.turns for g in gs))
                rows.append(row)
                print(f"{p.prompt_id} E={row['e']:5.1f} zero={row['zero']:4.0%} basic={row['basic']:4.0%} "
                      f"viaCorr={row['via_correction']:4.0%} distinct={row['distinct']:4.0%} "
                      f"sub={row['submitted']:4.1f} turns={row['turns']:3.1f}  {p.text[:45]}", flush=True)
                for g in gs:
                    f.write(json.dumps(dict(prompt_id=g.pid, text=g.text, tactic=g.tactic, style=g.style,
                                            score=g.R.score, answer=g.R.answer, log=g.R.log, outputs=g.outputs),
                                       ensure_ascii=False) + "\n")
            print(f"  [{len(rows)}/{len(rows_e)} prompts, {dt:.0f}s for {len(games)} games, "
                  f"peak {mx.get_peak_memory() / 2**30:.2f} GB]", flush=True)
    m = lambda k: statistics.mean(r[k] for r in rows)
    summ = dict(name=a.name, model=a.model, adapter=a.adapter, style=a.style, split=a.split, prompts=len(rows),
                games=a.games, e=m("e"), zero=m("zero"), basic=m("basic"), via_correction=m("via_correction"),
                distinct=m("distinct"), submitted=m("submitted"), seconds=time.time() - t_all, rows=rows)
    out_path.with_suffix(".summary.json").write_text(json.dumps(summ, indent=1))
    print(f"\n{a.name}: E={summ['e']:.1f} zero={summ['zero']:.0%} basic={summ['basic']:.0%} "
          f"viaCorr={summ['via_correction']:.0%} distinct={summ['distinct']:.0%} "
          f"over {len(rows)} prompts x {a.games} games in {summ['seconds'] / 60:.1f} min")


if __name__ == "__main__":
    main()
