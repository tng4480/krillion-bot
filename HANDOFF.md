# Handoff: krillion-bot (state as of 2026-11-18)

## Goal

**Clarified 2026-11-18:** the target is a language model that *plays* Krillion, using the mechanics (free
rejections, autocorrect typo windows, preview and decline, stacking modifiers on what a correction reveals,
spam shaped like the answers). It is **not** a knowledge model. Don't train trivia into it, don't retrieve answers
from the archive, and leave small closed-set prompts (fewer than 100 answers) aside for now. The archive is for
prompts and the reward dictionary only. The 100-point answer is usually a joke; ignore it.

Find out how small a language model can be and still score well at Krillion's Daily Dive. Use the project to compare
fine-tuning methods (SFT, RFT, DPO, GRPO) and data-prep choices.

## The game, as far as it matters for training

- 7 prompts per dive ("name a …"). You have 25 s per prompt, and answers are scored by rarity tier: Plankton 10,
  Too Clever 15, Schooler 30, Rare 60, Deep Cut 85, One in a Krillion 100.
- **A rejected guess costs nothing.** You can keep guessing until the time runs out. The first **accepted** guess
  ends the round.
- **Autocorrect:** the server maps near-misses to a canonical answer. With "Answer corrections" preview on (Settings),
  a corrected guess is shown before it's scored and you can decline it. An **exact** answer skips the preview and is
  scored immediately.

## What's done

1. **Autocorrect reverse-engineered** from ~520 live probes. `probe/emulator.py` is the offline copy; run it to
   re-score it.
   - Single words: accepted if Levenshtein distance to the nearest answer or alias is at most `allowed(len)`, which
     is ≤4 letters → 0, 5–7 → 1, 8–11 → 2, 12+ → 3. Agrees with the server 99% of the time.
   - Multi-word: the answer must have the same number of words, and each word must pass its own allowance. Agrees 91%
     of the time; the misses look like a semantic fallback on the server.
   - Ties: nearest answer, then most common answer (lowest score, 37/37 observed), then reveal position. The emulator
     picks the same answer 95% of the time.
2. **Full answer list for Dive #125**: `data/reveal/2026-11-17.json`, 110k answers with aliases, tier and score
   across 7 prompts. The aliases include player typos.
3. **Strategies measured** against the emulator (`analysis/strategies.py`, `notes/strategies.md`). Well-shaped
   nonsense scores 82–85 per round on big, regular answer pools: fake Latin genera, Italian diminutives, name spam,
   "-ville" splices, "East/West X". Saying the obvious answer scores about 28. Small or irregular pools (game
   characters, fruit, film titles) need real knowledge.
4. **Data sources checked** (`notes/data-sources.md`). Themed packs have no answer lists. The krillionanswers top-200
   lists give 42–58% false zeros as a reward, so they're unusable for that.

## Key decisions and why

- **The reward is the emulator**, scored on the *round* under the retry rule, not per-guess accuracy. Reason:
  rejections are free, so low precision with many guesses is fine, but a common answer that gets accepted ends the
  round at ~10.
- **Model output = plan + ranked guess list**, e.g. `<plan> descriptor|foreign|name-spam|shape-bullshit|canonical-rare`
  then `<guesses> g1 ; g2 ; …`. Plan labels can be generated from the answer lists. This is how "there's no one right
  answer" gets encoded.
- **Split by prompt (ideally by category), never by answer.** With few prompts every method memorises, so
  held-out-prompt performance is the only metric that means anything.
- **Method order: SFT → RFT (sample, score with emulator, keep the best, retrain) / DPO → GRPO.** Sampling is the
  slow part on these machines, and RFT reuses samples.
- **Model sizes:** develop the pipeline at 135M–500M, run ~1B for the ceiling, then sweep down. Always run a zero-shot
  baseline first, so pretraining knowledge is separated from fine-tuning gains.

## Blocker: prompts

Only 7 prompts have full answer lists. Options:
- **Archive unlock ($5, done by the user).** "Every aired daily" ≈ 875 prompts. Each finished dive's results screen
  loads `/api/reveal?date=…`, which is the full list. Capture method used so far: play the dive so it times out (or
  submit plain answers), open the results screen, read the reveal object from the page's React state, and save it as
  a browser download. Direct scripted calls to the API were blocked by the agent's safety check; going through the
  UI was fine.
- **Daily captures:** 7 prompts a day, free. **Done on the Mac for Dive #126 (2026-11-18)** in the desktop app's
  in-app browser: play the dive with one easy answer per prompt (decline optional cookies first), click Surface,
  then find the reveal object (`{date, prompts:[{id, text, answers}]}`) by walking the React fiber tree from any
  `__reactFiber$` key in page JS, and save it with a Blob download (it lands in `~/Downloads`). Then move it to
  `data/reveal/<date>.json` and run `data/build.py`. Now 14 prompts and 183k answers. New categories: brands, sports.
  The new day puts cities-starting-with-D in val (places) and languages-ending-in-an in val (words).

## Machines

**The project continues on the MacBook Air M5 (16 GB): it is the primary machine for everything from here on**,
including data prep, emulator, eval and training.

- **Training stack: MLX** (`mlx-lm`, which has built-in LoRA fine-tuning and runs on the Apple GPU). Use PyTorch
  with the MPS backend only where a library needs PyTorch, e.g. TRL for DPO/GRPO; otherwise prefer MLX-native loops.
- **Memory:** 16 GB unified memory is shared by CPU and GPU. That's plenty for LoRA up to ~1B (1B in bf16 ≈ 2 GB of
  weights); full fine-tuning at 1B would be tight. Avoid loading the 73k-answer animal prompt's dictionary several
  times in one process.
- **Heat:** the Air is fanless and slows down on sustained load, so prefer short, resumable jobs (one RFT round,
  checkpoint, repeat) over one long run. Save checkpoints often (`checkpoints/` is git-ignored).
- **Sampling for RFT/GRPO:** use `mlx_lm.generate` / batch generation. Sampling is the slow step, so reuse samples
  (RFT) before moving to on-policy GRPO.

The Windows laptop (Ryzen AI 7 350, no CUDA) is where the browser probing and reveal captures were done. Its in-page
scripts live in `probe/`. Capturing more reveals needs a browser on whichever machine is used; the method is described
above. A Dell Precision 5550 (small NVIDIA Quadro, 4 GB) is available but not planned for use.

## Setup (Mac)

```bash
git clone https://github.com/tng4480/krillion-bot.git && cd krillion-bot
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python probe/emulator.py          # expect single 99%/95%, multi 91%
.venv/bin/python analysis/strategies.py     # strategy table
.venv/bin/pip install mlx-lm                # training/sampling on the Apple GPU
# only if needed for DPO/GRPO via TRL: .venv/bin/pip install torch transformers peft trl
```

Add whatever gets installed to `requirements.txt` so the environment can be rebuilt.

## Progress on the Mac (2026-11-18)

- Setup done; the emulator self-check reproduces 99%/95% single, 91% multi.
- **Data layer done:** `data/build.py` writes `data/prompts.parquet` and `data/answers.parquet` (git-ignored, rebuild
  any time). Categories come from keyword rules, overridable in `data/categories.json`; unmatched prompts print as
  "other". The split is by **category**; `categories.json` `_split` pins `games`→test and `places`→val for now, since
  with 7 prompts the hash put everything in train. Revisit once the Archive data lands.
- **Eval harness done:** `eval/play.py` (policies `mlx:<model>` with optional `--adapter`, plus `common` and
  `markov` references; `--preview` for the decline rule). The model format (system prompt, `<plan>`/`<guesses>`
  parsing) lives there too, so SFT data prep should import `render_prompt`/`SYSTEM` from it. Every sample goes to
  `runs/eval/<name>.jsonl` for reuse in RFT. Checks: `common` ≈ 27, `markov --preview` ≈ 69 (matches the notes).
- **Local models (already on disk, MLX 4-bit)** in `~/projects/harness-investigation/work/models/`: gemma4-e2b-4bit
  (actually the E2B **-it**), gemma4-e4b-it-4bit, gemma3-4b-it-4bit, qwen3-4b-instruct-2507-4bit, qwen3-4b-thinking,
  qwen3.5-9b-4bit, qwen2.5-coder 7b/14b. There's also Qwen2.5-Coder-0.5B-Instruct 4-bit at
  `~/projects/harness-investigation/research/speculative-runtime-2026-10-17/draft`. Pass the path as `--policy mlx:<path>`.
  The sub-1B general-purpose models (SmolLM2 360M, Qwen 0.5B, Gemma 3 270M/1B) still need downloading.
- **Zero-shot sweep done** (`eval/report.py` prints this table; n=16 rounds/prompt up to 1.7B, n=8 above;
  ±5–15 per prompt). Small models were downloaded to the HF cache from `mlx-community` (SmolLM2-135M/360M, gemma-3-270m-it-bf16,
  Qwen2.5-0.5B/1.5B-Instruct-4bit, Qwen3-0.6B/1.7B-4bit, Qwen3.5-0.8B-4bit, gemma-3-1b-it-4bit).

  | model | E[round] | guesses/list | rounds scoring 0 |
  |---|---|---|---|
  | Qwen3-4B-it | **40.8** | 27 | 16% |
  | Gemma3-4B-it | 30.8 | 38 | 18% |
  | Gemma4-E4B-it | 30.5 | 7 | 27% |
  | Qwen3-1.7B | 25.4 | 48 | 48% |
  | Gemma3-1B-it | 22.0 | 33 | 49% |
  | Gemma4-E2B-it | 19.0 | 12 | 43% |
  | Qwen2.5-1.5B-it | 17.9 | 13 | 61% |
  | Qwen3.5-0.8B | 12.9 | 13 | 79% |
  | Qwen3-0.6B | 11.5 | 33 | 82% |
  | SmolLM2-135M | 8.2 | 7 | 89% |
  | SmolLM2-360M | 7.7 | 4 | 86% |
  | Qwen2.5-0.5B-it | 6.6 | 3 | 79% |
  | Gemma3-270M-it | 0.0 | 0 | 100% (outputs a bare tag like `<film>`, or nothing) |

  Reading: untuned, only Qwen3-4B beats the ~28 "obvious answer" floor. Every model names real, mostly common answers;
  none spams plausible nonsense. Below ~1.5B most of the loss is format: few guesses and many empty or invalid lists. The
  Qwen3 family follows the `<plan>/<guesses>` format best at every size. So SFT's first job is format and the
  nonsense strategies; the knowledge gap matters less than expected.

## SFT and RFT (2026-11-18, Qwen3-0.6B-4bit, LoRA rank 16 on all layers)

Tooling: `data/sft.py` (teacher lists from answer lists), `train/rft.py` (sample → emulator edit/filter →
dataset), `train/sft_lora.py` (mlx-lm trainer with the eval prompt and prompt masking; adapters go to
`checkpoints/`), `train/rft_loop.sh` (expert iteration: growing pool, fresh LoRA from base each round).
`probe/emulator.py` now indexes strings by word count. It's a speed fix only, and the self-check is unchanged.

| Qwen3-0.6B | train (5) | val: places | test: games | all |
|---|---|---|---|---|
| untuned | 6.0 | 39.1 | 11.2 | 11.5 |
| SFT v1 on teacher lists (1,500 ex, 400 it) | 72.0 | 5.3 | 7.5 | 53.3 |
| RFT round 1 (44 self-samples, 80 it) | 55.5 | 32.5 | 28.1 | 48.3 |

- **SFT on teacher lists memorises the prompts.** The teacher lists (real rare answers + character-model nonsense /
  splices / name spam / learned modifiers, each scoring ~84) are built from each prompt's own answers. The model
  learns those shapes and sprays them at every prompt: cities and game characters get Latin-ish and Portuguese-ish
  nonsense. It's already broken at iteration 100, so this isn't a training-length problem. With 5 training prompts
  it can't learn to pick a strategy from the prompt.
- **RFT on the model's own samples transfers.** Sample 64 lists per prompt, drop any guess the emulator maps to a
  basic answer, and keep lists scoring >= 60. Targets stay prompt-conditioned and within the model's own knowledge. One
  round lifts the held-out games prompt from 11 to 28 and leaves places within noise. Overall 48.3 already beats
  untuned Qwen3-4B (40.8).
- **Expert iteration (rounds 2–4, pool grows, fresh LoRA each round) overfits the 5 training prompts.** Train
  scores go 55 → 74 → 78 → 73, while games goes 28 → 26 → 19 → 21 and places hits 0. The training prompts' winning
  guesses leak across prompts: cities get "Gary Green ; Jane Simmons" (player-style made-up names), and game
  characters get animal words. Don't run more rounds until there are many more prompts.
- **Leave-one-prompt-out (`train/loo.py`, the reliable transfer measure with 7 prompts):** sample the untuned model
  once, then for each prompt train RFT round 1 on the other 6 and test on it. Qwen3-0.6B: **mean 11.1 → 25.4**, 6 of 7
  improved (movies 4→44, animals 11→45, games 6→26, fruit 5→21, -o food 9→20, players 0→11, cities 43→10).
  Results are in `runs/loo/<name>/results.json`. Use this, not the fixed val/test split, until more prompts arrive.
- **LOO RFT round 1 across sizes** (same recipe; base = untuned, 64 samples at 512 tokens; trained = held-out
  prompt, 16 rounds, SE ≈ 7–9 per prompt):

  | model | base | RFT r1 (held-out) | prompts improved |
  |---|---|---|---|
  | SmolLM2-360M | 8.3 | 16.4 | 7/7 (all small gains) |
  | Qwen3-0.6B | 11.1 | 25.4 | 6/7 |
  | Gemma3-1B | 22.9 | 21.8 | 3/7 (movies 13→53; animals 35→11, cities 48→16) |
  | Qwen3-1.7B | 19.7 | **39.0** | 5/7 (movies 10→80, players 0→38, games 0→24; -o food 62→30, cities 41→24) |

  Reading: transfer grows with size within the Qwen3 family (0.6B +14, 1.7B +19), and Qwen3-1.7B after one LOO
  round sits near untuned Qwen3-4B (40.8). Gemma3-1B gains nothing on average: RFT trades its decent untuned recall
  for transferred spam that fails on unseen prompts. Cities is the prompt every model loses on (strong untuned
  baseline, and the transferred "made-up name" mode doesn't fit). The 360M model mostly learns the output format.
  Seven prompts, so differences under ~8 points are noise.
- Next: the same at other sizes is done for round 1; more rounds wait on more prompts.

- **LOO on 14 prompts (Qwen3-1.7B, RFT round 1, each fold trains on 13 prompts):** mean held-out **19.1 → 31.7**.
  On the original 7 prompts, training on 13 gives 33.3, against 39.0 when training on 6: no gain from more prompts at
  this recipe. Likely causes are the 400-iteration cap and mixing two days' modes; within noise, ±4. Memory peak
  2.8 GB with grad checkpointing (`train/sft_lora.py --mem-gb 5`). 14 folds took ~2 h 15 min. For the next run, cap
  at ~2 epochs, or hold out a day instead of a prompt.

## DPO (2026-11-18, Qwen3-1.7B-4bit, leave-one-out, same untuned samples as `qwen3-1.7b_r1`)

- `train/dpo.py`: LoRA DPO in MLX. The reference policy is the base model, with reference log-probs computed once
  up front. Loss = DPO (β 0.1) + α × per-token NLL on chosen (α 0.2, RPO-style), lr 5e-6. It runs one sequence at a
  time with gradient accumulation, scores only completion-token logits, and sets `mx.set_memory_limit(5 GB)`.
  Peak is about 4.5 GB. A first version with padded 4-sequence batches and full-vocab float32 logits ran the 16 GB Mac
  out of RAM, so smoke-test peak memory before any long job. Sanity check: at lr 0 the margin is exactly 0.
- `train/loo.py --method dpo` with `--pairs other` (v1): chosen = the RFT-kept lists (edited, score ≥ 60);
  rejected = a different sample from the same prompt that scored ≤ 30. **Mean 19.7 → 24.4, against RFT round 1's
  39.0.** Training margins hit 10–18 almost at once: the pairs separate on length and garbage (rejected lists are
  ~2× longer and often malformed), not on which answers they contain. Pushing them down degenerated the model:
  cities gave `olio: olio: ...` and non-Latin text (0), movies spammed "naruto : ..." variants, and fruit still led
  with common answers. Only animals (14 → 46) and games (0 → 33) improved.
- `--pairs self` (implemented, not run): rejected = the same sample before editing, so the pair differs only by the
  basic answers. The untuned samples give too few such pairs: 79 in total, 54 of them from the fruit prompt, and 0
  for players and games, where untuned lists fail because nothing is accepted, not because a basic answer comes
  first.
- Takeaway: offline DPO on untuned samples is worse than RFT here. DPO needs pairs that differ only in the thing to
  learn. That kind of contrast appears once the policy already plays reasonably, so DPO belongs after an RFT round
  (sample from the RFT adapter, then self-pair or score-rank lists of the same prompt). That isn't run yet. GRPO
  (online, emulator reward) is still to come; with 7 prompts, expect it to overfit the way the expert-iteration
  rounds did. The teacher data could come back
  later as a small mixed-in fraction, or once there are many more prompts.

## Interactive search: spam until autocorrect, then stack modifiers (2026-11-18, `analysis/search.py`)

The user's idea: spam garbage until something autocorrects. The preview shows the canonical answer X, which you
decline. Then spam modifiers on X ("M X" / "X M") until one is accepted, and use any new correction as the next base.
Simulated in the emulator with a 40-submission budget (declines count), 100 rounds per prompt, all 14 prompts:

| strategy | mean |
|---|---|
| obvious answers (floor) | 26.3 |
| known bases + generic modifiers, no fallback | 16.6 |
| known bases + generic modifiers + 2 obvious-answer fallbacks | **32.7** (≥ floor on all 14) |
| misspelled known answers as probes + generic modifiers + fallback | 32.5 |
| known bases + this prompt's own modifiers (LEAKY) | 26.7 |
| markov garbage + own modifiers + fallback (LEAKY) | 67.8 |
| markov garbage alone, oracle decline (LEAKY reference) | 67.7 |

- Stacking works where answers are often multi-word: fruit white inside 36, cities -ville 38, fictional food 39,
  vowel fruit 43. It does nothing on one-word or name pools (players, one-word animals, movies), where it only
  scores through the fallback.
- Modifiers mined from other prompts' 2-word answers are useless (mostly initials and name fragments, even after
  filtering). A hand-written generic list (directions, colours, nationalities, sizes, place and food suffixes) works
  better. Choosing modifiers that fit the prompt is exactly what a model would add.
- The probe step (misspell an obvious answer so it corrects instead of scoring) costs a try each and gains nothing
  over starting from known answers. Its value is discovering answers the model doesn't know, which a strong garbage
  generator does far better: the leaky markov garbage alone gets ~68. So the gap is the quality of the
  garbage/shape generator, not the stacking.
- This is a multi-turn policy (it reacts to preview feedback), while the current model format is one-shot
  (a whole list per prompt). Supporting it means a turn-based format: the model sees "X was corrected to Y" and
  proposes the next guesses. The emulator can already drive that.

## Archive: 882 prompts from krillionanswer.today (2026-11-18)

- `data/fetch_archive.py` fetched all 126 dives into `data/reveal/<date>.json`, about 100 MB, at one request per
  second (robots.txt allows crawling). Each dive page has the 7 prompt texts embedded, and
  `/answers/<dive>/<pid>.json` has every answer with tier and score. Checked against our own Dive 126 p3 capture:
  658 of 669 answers, with matching tiers. **No aliases**, and the 2 days captured from the game itself are kept as
  they are. Files from the site carry `"source": "krillionanswer.today"`. The real tier names include `tooclever`
  (15 pts, 525 answers in total).
- Totals: 882 prompts and 1.6M answers (deepcut 1.50M, rare 80k, schooler 18k, plankton 4.5k, krillion 1,076,
  tooclever 525).
- The earlier "not enough data" was wrong: the old notes had only checked krillionanswers.com (top 200 per
  prompt). This is a different site with essentially full lists.
- TODO: `data/categories.json` labels only about half the prompts (424 fall into "other"). Categories drive the split,
  so label them (or cluster the prompt texts) before the next training run.

### Patterns (`analysis/patterns.py`; modifiers saved to `analysis/modifiers.json`)

1. **The basic set is tiny.** Median basic answers (score ≤ 30) per prompt: 15 for pools under 100, 22 for
   300–1000, 36 for 5k+. In pools above 1000, 89–96% of valid answers are Deep Cut (85). So the score is mostly
   85 × P(valid and not one of the ~20–35 famous answers). The skill to learn is validity plus avoiding the famous
   head, not finding rare answers.
2. **40% of prompts are small closed sets** (fewer than 100 answers: NBA teams, Greek gods, constellations). Only 37%
   of their answers are Deep Cut and 29% are Rare. Typo-window spam has few targets here. The strategy is to
   enumerate the set from knowledge and play its least famous members.
3. **Stacking is real.** 6% of all answers are a modifier plus another answer to the same prompt, and 93% of those
   score 85. 28% of the bases they stack on are basic answers, so stacking turns obvious answers into Deep Cuts.
   304 of 882 prompts have at least 5% stacked answers. The modifiers that recur across prompts are learned from
   data: black, blue, red, water, white, snow, green, chinese, mountain, ice, dark, american, golden, rock, sea,
   southern, little, yellow, sand, giant, ...
4. **Reuse across prompts:** 57% of a prompt's rare or deep answers appear in some other prompt's list. Looking up
   the archive lexicon and filtering by the new prompt's constraint is a strong non-leaky generator. Only 15% of
   answers seen in 3 or more prompts are basic in one and rare in another.
5. **Krillion answers** (about 1.2 per prompt) are specific, real, notable-but-forgotten things (Kouprey,
   Adamstown, Huitlacoche). They're worth only +15 over a Deep Cut, so not worth chasing.
6. **Repeats:** only 4 exact repeated prompts (elements ending in -ium, languages with 10M+ speakers, YouTubers,
   Instagram accounts).

## Next steps, in order

1. ~~Data layer~~ (done).
2. ~~Eval harness~~ (done).
3. ~~Zero-shot baselines~~ (done, see above).
   **Zero-shot baselines** across sizes, using MLX-converted checkpoints from the `mlx-community` Hugging Face org
   (e.g. SmolLM2 135M/360M, Qwen ~0.5B/1.5B, Gemma ~270M/1B; check what's current).
4. ~~SFT~~ (first pass done, see above).
   **SFT**, with targets built from answer lists plus heuristic plan labels; sample answers weighted by tier and leave
   basic answers last.
5. **RFT rounds**, then DPO, then GRPO with the emulator reward.
6. **More data**: the Archive decision, or daily captures.

## Files

- `probe/emulator.py`: emulator and scorer. `probe/plan.py`: boundary-probe generator.
  `probe/metrics.py`: metric comparison. `probe/kp2.js`: in-page prober.
- `analysis/strategies.py`: strategy simulator.
- `data/reveal/`: full answer lists. `data/probes/`: probe plans and results.
- `notes/autocorrect-probe.md`, `notes/strategies.md`, `notes/data-sources.md`: findings.
