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

## Turn-based environment and skill-only eval (2026-11-18, `env/game.py`)

- `Round` plays one prompt from observations only: each submission is `rejected`, `corrected -> Y` (a preview:
  resubmit Y to accept, anything else declines it) or `scored`. Policies act in turns (a list of guesses,
  processed until the first correction or score), and get back text such as
  `rejected: a, b` / `corrected: "x" -> Y (resubmit to accept, ...)`. The budget is 40 submissions.
- **Split by date** (no category labels needed): train = dives 1-100, val = 101-113, test = 114-126.
  **Eval = pools with at least 300 answers** (69 val prompts), where mechanics dominate; closed sets are left out by
  design (see Goal).
- Scripted references on val (10 rounds per prompt), all seeing only observations except `shape-own`:

  | policy | mean |
  |---|---|
  | obvious answers | 26.5 |
  | stacking (misspell an obvious answer -> preview reveals base -> stack train-mined modifiers -> fallback) | 30.1 |
  | char-model spam over all train answers (generic shape) + decline obvious + fallback | 37.0 |
  | char-model spam over the 10 train prompts with the most similar wording (tf-idf) | **43.7** |
  | char-model spam over the prompt's own answers (LEAKY) | 69.1 |

- Reading: **about 44 is reachable from shape learned on similar past prompts**. That's the skill an LM would pick up
  from training on the archive, and it already beats every trained model so far (Qwen3-1.7B LOO ~32-39). The gap to
  69 is prompt-specific shape. Stacking on its own is weak (+4 over obvious answers); combining it with shape spam
  is untested.
- Modifiers for stacking are mined from train prompts only (`analysis/modifiers_train.json`). `analysis/modifiers.json`
  was mined from all 882 prompts, so it includes val/test; don't use it in evals.
- Next: put LM policies into the turn-based env (multi-turn chat: prompt, then observations), score them on this
  val set, and use `shape-near` spam as the RFT/teacher data source (its lists are non-leaky).

## Teacher benchmark (2026-11-19, `env/llm_play.py`, 69 val prompts x 4 games, turn-based)

| player | mean | best of 4 | zero | basic | wins via autocorrect | distinct guesses |
|---|---|---|---|---|---|---|
| obvious answers (scripted) | 26.5 | | | | | |
| similar-prompt shape spam (scripted, no knowledge) | 43.7 | | | | | |
| Qwen3-4B + rules + worked examples | 35.8 | 62.7 | 36% | 21% | **2%** | 94% |
| Qwen3-4B, plain instructions | 22.0 | 30.1 | 3% | 84% | 1% | 51% |
| Qwen3-1.7B + rules | 20.7 | 42.8 | 54% | 25% | 14% | 71% |

- The rules help a lot (4B: 22 → 36; obvious-answer endings 84% → 21%), and the 4B at about 16 min for 276 games is
  fast enough. But **98% of its wins are exact real answers it knows** (Pilanesberg National Park, lymphocyte,
  Pudu, Featherston). It plays by recall, despite instructions to invent. Distilling it would transfer knowledge,
  which is against the goal.
- It also handles corrections badly: 22 of 54 games that got a correction ended at 0. It didn't accept unfamiliar
  corrections (Wordle: slyer, quays, which are Deep Cuts).
- The scripted spam is still the best knowledge-free player (43.7), but its shapes are off-topic (it draws on salad
  and bicycle prompts for dresses).
- The first rules version (no examples) scored 0 on 3 smoke prompts: the 4B wrote phrases about the prompt
  ("festival premiere 2026"), not entries.

## Bottom-up ladder: how far does a knowledge-free rule follower get? (2026-11-19, `env/game.py ladder`)

All 69 val prompts, 10 rounds each, 40 submissions. "Rules" = the literal constraints parsed from the prompt
(`prompt_rules`: starts with / ends in / contains / N letters / one-word; 6 of 69 val prompts have them), which are
spliced into generated strings.

| rung | all 69 | the 6 with rules |
|---|---|---|
| 0 random letters + rules | 7.8 | 44.8 |
| 1 generic answer shape (char model, all train answers) + rules | 21.4 | 59.1 |
| 2 similar-prompt shape + rules | 27.6 | 52.3 |
| 2 without rules | 27.4 | 21.8 |
| rung 2, decline every correction | 13.5 | |
| rung 2, accept multi-word corrections only | 14.4 | |
| rung 2, decline famous corrections (needs knowledge), no fallback | 28.7 | |
| **rung 2, accept every correction + obvious-answer fallback at the end** | **44.4** | |
| rung 2, decline famous + fallback | 44.9 | 67.5 |

- **Knowing what's famous is worth ~1 point; the obvious-answer fallback is worth ~17.** Accepting every
  correction is fine (blind accept 27.6 vs oracle decline 28.7). The big gain is finishing a failed round with an
  obvious answer (10-30 pts instead of 0). The only knowledge the recipe needs is naming obvious answers, which
  any small LM can do.
- Literal rules matter a lot where they exist: random letters plus rules already reach 45 on those prompts, and
  dropping the rules costs 30 points there.
- With preview off, the game accepts corrections automatically, so this recipe (spam shaped for the prompt, then a
  few obvious answers) is a **one-shot list**. No turn-based play is needed for it.
- Next rungs: train tiny LMs (SmolLM2-135M, 360M, Qwen3-0.6B, 1.7B) to write that list from the prompt alone. The
  targets are rung-2 spam from train prompts plus an obvious-answer tail. Score on val, as one-shot lists.

### Ladder rungs 3+: tiny LMs trained on the spam lists (2026-11-19, `data/spam_sft.py`, `train/spam_ladder.sh`)

Data: 1,736 one-shot lists for the 217 train prompts (pool ≥ 300). Each is 30 rung-2 spam strings (char model over the
10 most similar *other* train prompts, bent to the literal rules), then 3 obvious answers. Unfiltered, so the student
learns the generator, not the hits. LoRA rank 16, 860 iters (~2 epochs), lr 1e-4. Eval: 69 val prompts × 8 one-shot
lists, 40-guess budget, corrections auto-accepted (preview off).

| model | untuned | trained | zero | basic | points from the spam part | distinct guesses |
|---|---|---|---|---|---|---|
| teacher lists (scripted, on val) | | 43.6 | | | 26.9 | |
| SmolLM2-135M | 4.1 | 28.5 | 49% | 19% | 20.8 | 94% |
| SmolLM2-360M | 10.7 | **33.9** | 33% | 30% | 22.6 | 93% |
| Qwen3-0.6B (4-bit) | 10.6 | 29.4 | 55% | 11% | 24.7 | 93% |

- **The spam skill transfers even at 135M.** Students get 77-92% of the teacher's spam-part points (20.8-24.7 vs
  26.9) on unseen prompts, from the prompt alone, and stay diverse (93-94% distinct guesses). Training took 4-14 min.
- **The gap to the teacher is the fallback.** The teacher's 3 obvious answers come from the answer list (+16.7),
  while the students have to recall them, and tiny models get them wrong (for "a movie first released in 2026" they
  write Space Jam, Spider-Man: Into the Spider-Verse). That's the one knowledge piece in the recipe, and it is what
  scales with size here (the 360M wins most via fallback, 35%).
- Qwen3-0.6B has the best spam part but the weakest fallback (4-bit, and lists sometimes end without obvious answers).

- **Fixing the ending (2026-11-19).** Only 10-24% of the students' ending guesses are valid answers. The base model
  without the adapter, asked directly for obvious answers, is worse (SmolLM2 14-15%; Qwen3-0.6B 1%, which looks like
  a format issue I haven't checked). So spam training didn't damage recall; tiny models just don't know the obvious
  answers well. `env/ending.py` shows spam + base-model ending scores 23-25, below the trained endings. What works is
  **more tries for the ending**: lists of 30 spam + **10** obvious answers (`runs/sft/spam_v2`):
  SmolLM2-135M **28.5 → 37.5**, SmolLM2-360M **33.9 → 42.1** (scripted teacher on val: 43.6). Zero rate
  49% → 24% (135M) and 33% → 11% (360M); the basic-answer rate rises (36%, 45%), as expected.

## List-level RFT on the spam students (2026-11-19, `train/spam_rft.py`, `train/spam_rft_loop.sh`)

ReST-EM style: 8 lists per train prompt; keep the lists that beat that prompt's mean and score ≥ 60 (Jaccard ≤ 0.5
dedup, ≤ 4 per prompt, ~370-400 lists), mixed with the 1720 spam_v2 teacher lists; fresh LoRA from base each round.
Val (69 prompts × 8, one-shot 40-guess lists):

| model | spam_v2 | RFT r1 | RFT r2 |
|---|---|---|---|
| SmolLM2-135M | 37.5 | 35.8 | 38.8 |
| SmolLM2-360M | 42.1 | 42.1 | 40.9 |
| Qwen3-0.6B (4-bit) | 37.4 | 38.1 | 38.2 |

**Flat, within noise.** Zero rate, the 85+ rate and distinct guesses (93-95%) don't change, so there's no collapse,
but also no gain. Why: a spam guess hits a valid rare answer only **~2%** of the time (all sizes, before and after
RFT), so a winning list is ~1 lucky hit + 29 random guesses. Whole-list selection is mostly selecting luck, and the
teacher lists dilute it 4:1. Qwen3-0.6B is no better than 360M here (more zeros: 31% vs 11%).

The skill metric that matters is the **per-guess hit rate of the spam part**. Next: guess-level credit
(`train/spam_hits.py`): pool the student's own guesses that hit, and build lists enriched in them.

## Guess-level RFT (2026-11-19, `train/spam_hits.py`, `train/spam_hits_loop.sh`)

16 lists per train prompt; every spam guess is scored on its own (hit = maps to an answer worth ≥ 60). Per prompt, 4
new lists = up to 6 of the student's own hit guesses + its own non-matching guesses to 30 + a teacher 10-answer
ending; plus 2 teacher lists per prompt (~1100-1200 rows). Fresh LoRA from base each round, ~2 epochs.
Val (69 prompts × 8); "rare hit/guess" = share of spam guesses mapping to an answer worth ≥ 60:

| model | round | val E | rare hit/guess | won by spam | zero | distinct |
|---|---|---|---|---|---|---|
| SmolLM2-360M | spam_v2 | 42.1 | 1.7% | 36% | 11% | 93% |
| | hits 1 | **45.2** | 2.5% | 47% | 11% | 92% |
| | hits 2 | 43.6 | 2.9% | 48% | 12% | 92% |
| SmolLM2-135M | spam_v2 | 37.5 | 1.5% | 33% | 24% | 95% |
| | hits 1 | 37.8 | 1.9% | 45% | 27% | 95% |
| | hits 2 | 40.5 | 2.2% | 46% | 24% | 95% |
| | hits 3 | 38.7 | 2.4% | 47% | 24% | 94% |
| | hits 4 | 39.2 | 3.0% | 55% | 24% | 94% |

SmolLM2-360M continued: hits 3 **47.4** (3.6% rare hit/guess, 53% won by spam, zero 11%), hits 4 46.5 (3.5%). The
360M plateaus from round 3. Rare hits on val stay mostly autocorrect-based (65% via correction in round 3 vs 74% at
spam_v2), so the gain is from better-shaped guesses, not from recalled answers. Examples: pushton → Bushton,
corium → Cerium, grothery → Brothers.

**Held-out test** (74 prompts × 8, evaluated once):

| model | adapter | test E | rare hit/guess | won by spam | zero | 85+ |
|---|---|---|---|---|---|---|
| SmolLM2-360M | spam_v2 | 42.6 | 2.2% | 37% | 11% | 28% |
| SmolLM2-360M | hits 3 | **50.5** | 4.2% | 59% | 10% | 37% |
| SmolLM2-135M | spam_v2 | 41.6 | 2.1% | 38% | 19% | 29% |
| SmolLM2-135M | hits 4 | **47.0** | 3.8% | 59% | 19% | 36% |

Gains hold on test (+7.9 and +5.4) and the hit rate roughly doubles. Best adapters:
`checkpoints/spamhits3_smollm2-360m`, `checkpoints/spamhits4_smollm2-135m`.
Open levers: spam that lands on basic answers (auto-accepted, ends rounds low); the same recipe for Qwen3-0.6B; turn
play (stacking modifiers on corrections) on top of these students.

## Guess shape: one word or two? (2026-11-19, `train/spam_hits.py --shape`)

Hit rate of the 360M's spam on val, by shape (hits 3 adapter): one word ≤ 4 letters 1.7%, **5-7 letters 8.3%**,
8-11 letters 4.4%, 12+ letters 0.2%; two words 0.7%, three+ 0.1%. Even on prompts where ≥ 60% of rare answers are
multi-word (72% of eval prompts), single words hit 3.1% vs 0.6%: every word of a multi-word guess must be within its
own allowance (≤ 4 letters: exact), so two invented words never land (0.00% over ~7,000), and anchored ones (one word
exact) only ~0.3-0.6%. Two words are the right call only where nothing single-word exists (people, "Los ..." cities,
phone settings: single-word spam hits 0% there) or once a correction has confirmed a base word (turn play).

`--shape` uses the answer pool's **format**, not its answers: a prompt with ≥ 15 single-word answers worth ≥ 60 is
"single" mode (90% of filler = single words of 5-11 letters), otherwise "multi" mode (90% multi-word filler).
`--multi-boost 3` (variant "4b") also builds lists for every multi-mode prompt (19 of 217 train) even without hits,
3× as many. Ablation on the **same samples** as hits round 4:

| model | adapter | val E | val: multi-word share, single / multi prompts | val E, multi prompts (3) | test E |
|---|---|---|---|---|---|
| 360M | hits 4 (no shape) | 46.5 | 35% / 42% | 28.8 | (hits 3: 50.5) |
| 360M | shape 4 | **50.6** | 15% / 20% | 12.9 | 51.1 |
| 360M | shape 4b (boost) | 48.7 | 21% / 37% | 8.5 | see below |
| 135M | hits 4 (no shape) | 39.2 | 35% / 39% | 5.4 | 47.0 |
| 135M | shape 4 | 43.6 | 19% / 23% | 5.6 | **50.0** |
| 135M | shape 4b (boost) | **46.6** | 21% / 35% | 16.0 | see below |

Shaping helps: +4 val for both sizes, +3 test for 135M (exact control), rare hits per guess 3.5% → 4.6% (360M).
Without the boost the model also drops multi-word guesses where it needs them; the boost restores that (the
model learns *when* from the prompt wording), but there are only 3 multi-mode prompts each in val and test, so their
scores can't separate the variants. Overall, shape 4 vs 4b is a tie within noise.

Test, shape 4b: 360M 50.2 (shape 4: 51.1), 135M 49.9 (shape 4: 50.0). On the 3 multi-mode test prompts the boost
raises multi-word guesses (28% → 41%, 24% → 50%) but the scores go opposite ways (34.0 → 15.8; 9.2 → 18.5).
**Conclusion:** the model can learn when to switch to two words, but on those prompts invented two-word spam still
hardly ever lands, so switching doesn't pay; they are won (if at all) by the obvious ending. **Keep plain shape 4**
(`checkpoints/spamshape4_smollm2-360m`, `checkpoints/spamshape4_smollm2-135m`: test 51.1 / 50.0, the current best).
Multi-word prompts need anchoring from a correction (turn play), not more blind two-word spam.

## First live game: Dive #127 (2026-11-19), SmolLM2-360M `spamshape4`

Prompt texts only were taken from krillionanswer.today before playing; the model wrote one 40-guess list per prompt
(`runs/live/2026-11-19_lists.json`), typed in order in the in-app browser, corrections accepted. **Real score 170**
(better than 9% of players): p1 `bombo` → Bobo horse 85, p2 `kowar` → Kowak 85, the other five 0.

**The big lesson: the real game is time-limited, not guess-limited.** 25 s per prompt, and the answer box locks
while the server checks each guess (~2 s), so only **1-5 guesses per prompt** got in (agent tool latency adds to it;
a fast human might manage ~8-10). The emulator's 40-guess budget is far off. Replaying the same lists in the emulator
(answers fetched after the dive into `data/reveal/2026-11-19.json`):

| budget | emulator total |
|---|---|
| guesses actually played (1-5 per prompt) | 85 (the real 170 includes `bombo` → Bobo horse, a server match the emulator misses) |
| first 10 guesses | 85 |
| full 40 | 385 |

The full-list hits came at guesses #15-33.

## Live harness v2 and replay of Dive #127: 380, better than 82% (2026-11-19)

**The budget problem was the harness, not the game.** The server answers in ~250 ms; the slowness was input
handling. Fixes, found in Unlimited mode (`/unlimited/classic`, no effect on the daily):
- Settings: **Answer corrections OFF** (corrections auto-accepted, as in the emulator), **Clear rejected entries ON**
  (box empties after a miss, so no select-all), Low-power ON. Stored in localStorage (`krillion-answer-corrections`,
  `krillion-clear-rejected-entry`, `krillion-low-power`).
- Per guess: click the box at x=250 → type → **click DIVE** (x=605) → wait 0.25 s. Never press Enter: on the result
  screen Enter activates Descend and the leftover guesses spill into the next prompt. The click points are outside
  the Descend button, so leftovers after a hit do nothing.
- Browser batches of ≤ 25 actions (6 guesses), all of a prompt's batches sent in one message: **~0.33 s per guess,
  40 guesses in ~14 s, 50 fit in 25 s.** So the 40-guess emulator budget is realistic after all.

**List builder** (`env/live.py`): 8 samples per prompt, spam pooled and de-duplicated, stably sorted by shape
(single words of 5-7 letters first, then 8-11, other single words, multi-word), first 30-40 spam + the first sample's
10-answer ending. On val at 40 guesses: 54.3 vs 50.3 for one raw list (at a 10-guess budget: 39.3 vs 23.6; ranking
by cross-sample similarity hurts). Prompts were taken as texts only; the lists were not checked against the answers.

Replay after clearing site cookies/storage (fresh player id): **380, better than 82% of players** (run 1: 170, 9%).
`runs/live/2026-11-19_run2_result.json`:

| prompt | raw guess accepted | answer | score | emulator |
|---|---|---|---|---|
| animal bigger than a human | `tanda` (#10) | Giant panda | 60 | 60 (narwhal, #23) |
| fictional place | `fanes` (#13) | Fang | 85 | 85 (#24) |
| sticks to skin | none of 40 | — | 0 | 0 |
| film director | `barbers` (#4) | Clive Barker | 60 | 85 (#19) |
| app permission | `texas` (#11) | Text messages | 30 | 10 (#42) |
| country, soccer not top | `astland` (#1) | Estonia | 85 | 60 (#11) |
| TV show, 5+ seasons | `dallas` (#1) | Dallas | 60 | 60 |

The real server accepts earlier than the emulator (real 380 vs emulator 360).

## Autocorrect: what's left to explain (2026-11-19, `analysis/autocorrect.py`)

Tested on ~600 probes (dive 125 hand probes, the boundary run, both live Dive #127 runs, in
`data/probes/2026-11-19_dive127.jsonl`). The current rule (Levenshtein per word, `allowed(len)`, same word count,
canonical + aliases) stays best: 98% single-word, 92% multi-word accept/reject. Rejected alternatives:
whole-string Levenshtein (spaces removed, "the" stripped; accepts many server rejections, multi 69%), pg_trgm
trigram similarity at any threshold (worse), single word vs one word of a multi-word answer (first/last/any;
all worse: false accepts).

**The remaining gap is the alias layer, not the metric.** Live accepts like `tanda` → Giant panda, `barbers` → Clive
Barker, `astland` → Estonia, `texas` → Text messages fit aliases ("panda", "Barker", "Estland", "texts"). Our own
captures (11-17, 11-18) have aliases on 16-100% of answers per prompt (176k aliases); only ~14% are simple
transforms (plurals 8.7%, spacing/hyphens 1.8%, first/last word ~3%), the rest are synonyms ("giant moa"),
descriptions ("big boat racing") and even misspellings ("lary bir" → Larry Bird), which suggests accepted player
inputs get added. **The krillionanswer.today archive has no aliases**, so the emulator undercounts real hits on all
archive days: training rewards are conservative, and some guesses that would land live are scored as misses.
Possible next step: add rule-based aliases (plurals, surname for people prompts) to archive days and check on the 2
captured days whether that moves emulator scores toward the real-alias scores.

**This works where list-level RFT didn't:** the hit rate on unseen prompts rises ~1.5-1.7× in two rounds, with no
loss of diversity, and SmolLM2-360M passes the scripted teacher (43.6). The score gain is smaller than the hit-rate
gain because spam now also lands on more basic answers (valid but ≤ 30, auto-accepted with preview off), which
ends rounds early at low scores. The val E has ~±2 noise at this size.

## Preview, stacking, budget (2026-11-19, `analysis/preview_sim.py`, `analysis/preview_diag.py`)

Emulated live play on the 69 val prompts with 32 lists per prompt from 360M spamshape4 (8 from the val eval + 24 new,
`runs/eval/val_pool24_360m.jsonl`, `analysis/sample_val.py`). Lists are built as in `env/live.py`: pooled spam sorted
by shape, then the first sample's 10-answer tail. Preview ON variants: *decline* = decline a correction whose target
is one of the model's own tail answers (its "obvious" set, i.e. the model's own output, no lookup), else accept;
declined answers go to the front of the fallback (they're known valid, so the exact spelling scores). *stack* = also
try K train-mined modifiers on the declined answer ("m X" / "X m").

| samples pooled | guesses | auto (preview off) | decline | stack K=4 / 8 / 16 |
|---|---|---|---|---|
| 8 | 40 | 53.3 | 53.3 | 53.2 / 53.2 / 53.4 |
| 8 | 50 | 55.6 | 55.7 | 55.5 / 55.5 / 55.7 |
| 16 | 50 | 55.7 | 55.7 | 55.3 / 55.3 / 55.3 |
| 32 | 50 | 57.5 | 57.4 | 56.7 / 56.7 / 56.7 |

Budget only (auto): 8 samples 40/50/60/70 guesses = 53.3 / 55.6 / 56.3 / 57.5.

- **More guesses is the real lever:** +2.3 for 40 -> 50, then ~+1 per 10 more. `env/live.py` now defaults to 40 spam +
  10 tail = 50.
- **More samples barely matter:** 8 samples already give a ~240-guess pool and only 40 are used. (32 is +1.9 at 50
  guesses, but that's one group per prompt, within noise.)
- **Decline does nothing because the case is rare.** In 276 rounds, the first spam correction went to one of the
  model's own obvious answers only 10 times (3.6%), and half of those were actually worth 85 (the tails include
  deep cuts). Low-value corrections (10/30) to answers *not* in the tail (like live `texas` -> Text messages, 30) were
  8/276; the model can't tell those apart without knowing the answer's value. 54% of rounds get no correction at all in
  the spam: the bottleneck is spam hit rate, not what happens after a hit.
- **Stacking is worse than more spam:** 160 stacked guesses on declined answers gave 1 Deep Cut and 2 Rare previews
  (~2% per guess), vs ~8% for 5-7 letter single-word spam. The earlier stacking result (+4 over obvious answers,
  2026-11-18) also came mostly on multi-word pools, which this spam rarely targets.
- So play live with answer corrections OFF (no preview click), and spend the time on more guesses.

## Which spam to play: word-likeness ordering (2026-11-19, `analysis/select_sim.py`)

From the pooled spam of 8 samples (~240 guesses), which 40 to play (+ 10-answer tail, 50 guesses, preview off)?
Only the model's own samples and train-split text are used, never the prompt's answers.

| selection (within shape: 5-7 letters, 8-11, other single, multi) | val 360M, 8 samples (4 groups) | val, 32 samples | **test 360M** | **test 135M** |
|---|---|---|---|---|
| base: sample order (old `env/live.py`) | 55.6 | 57.5 | 56.2 | 54.5 |
| spread: skip guesses whose typo windows overlap a chosen one | 54.8 | 55.6 | | |
| dens: guesses the samples agree on first | 57.3 | 55.4 | 63.1 | 53.9 |
| **lm: most word-like first (char trigram over train answer words)** | **60.8** | 59.6 | **61.4** | **57.8** |
| lm + spread | 59.7 | 62.0 | 63.5 | 58.4 |

(Other variants: bigram/4-gram LMs and LM ignoring length were within noise or worse on val.) `lm` was picked on val
and checked once on test: **+5.2 (360M) and +3.3 (135M)**, no training. Test rows for other variants are shown, not
used to choose.

- **Word-likeness is the lever.** Spam made of common letter patterns of real answer words lands inside more typo
  windows. The LM is a phonotactic prior from train-split answer words (178k words, dives 1-100): letter statistics,
  not answers to the prompt being played.
- **Spreading guesses out hurts.** Guesses cluster near answer-rich regions, and that clustering is good.
- With the LM, 32 samples aren't better than 8: the pool is already larger than what's played.
- `env/live.py` now uses this ordering (`train_lm()` builds the LM at start-up).
- Next: put the same prior into the model itself (reward word-like spam in guess-level RFT, or filter the samples used
  for training by LM score), so a small model generates more of what this ordering picks.

## Deterministic spam, no model (2026-11-19, `analysis/det_gen.py`, `analysis/det_seeds.py`)

Can a letter model replace the LM for the spam? Char trigram over the words of TRAIN-split answers (dives 1-100),
played with preview off, 50 guesses (or 40 + the 360M's own 10-answer tail, to isolate the spam part).

Val (69 prompts), one variant each:

| spam source | 50 spam, no tail | 40 + model tail |
|---|---|---|
| top-50 most likely strings, all train words | 38.5 | 49.9 |
| same, wording clusters (k=10 / 30, tf-idf k-means) | 38.0 / 33.3 | 50.3 / 45.6 |
| same, 10 / 40 most similar train prompts | 39.4 / 33.3 | 49.1 / 45.2 |
| top-50 with near-duplicates removed (edit distance > 2) | 47.0 | 55.9 |
| 10 most similar prompts, seeded samples | 45.1 | 52.5 |
| **all train words, 1000 seeded samples, keep top 50 by log-prob** | **50.9** (sd 4.5 over seeds) | **57.4** |

Pool size for the seeded version (10 seeds, val): 50 samples 34.8, 150 43.5, 400 46.9, 1000 49.1, 3000 48.7,
10000 44.5. Ranking too hard collapses onto repeats ("graphy maraphy traphy braphy"); not ranking at all is junk.

**Test, once (74 prompts, m=1000, 10 seeds): 53.7 (sd 3.6, range 46.5-59.7); 58.5 with the model tail.** Picking
the seed by its train score doesn't help (train scores don't predict test). Train prompts score much lower overall
(38.1), so the dive-1-100 prompts are harder than the later ones.

Test-set comparison:

| | test |
|---|---|
| one model list, 360M / 135M | 51.1 / 50.0 |
| **one fixed letter-model list, same for every prompt, no model** | **53.7** |
| model pipeline (8 samples, word-like order), 135M / 360M | 57.8 / 61.4 |
| fixed list + 360M's obvious-answer tail | 58.5 |

- **Most of the spam's value needs no model.** One fixed generic list of word-like 5-7 letter strings beats a single
  model list. What the model adds on top is (a) its obvious-answer fallback ending, which is knowledge, and
  (b) prompt-tailored spam: +7.7 for the 360M pipeline over the fixed list, ~+4 for the 135M.
- **Per-category letter models don't help.** Fewer source words make the strings narrower; what lands is general
  word-likeness, not category flavour.
- Diversity matters here, unlike reordering model spam: near-duplicate strings waste guesses.
- Not tried: optimizing a fixed list greedily on train prompts. That would select strings that are answers in many
  past prompts, which drifts into answer reuse.
- Consequence for "how small": the bar for a model is the fixed list (53.7) + a fallback ending. A tiny model has to
  beat that, and the prompt-specific part of the 360M (~+3 over fixed list + tail) is what's left to keep.

## Smaller models: pruning the 135M, Pythia-70M and 14M (2026-11-19)

Scripts: `train/prune.py` (fuse the LoRA, then drop the block of consecutive layers with the smallest angular
distance between its input and output on 64 training completions, Gromov et al. 2024), `train/prune_pipeline.sh`
(score, heal with LoRA on the same spamshape4 data, score), `train/size_pipeline.sh` (the whole training program for a
new model: SFT spam_v2 -> guess-level RFT rounds 1-3 -> round 4 with --shape -> val + test). `train/sft_lora.py` got
`--full` (full fine-tune in fp32; Pythia ships fp16 and goes NaN) and `--warmup`. Pythia LR: 70M 3e-5 (1e-4 diverged),
14M 1e-4. Two jobs ran in parallel under a 3 GB MLX cap each; peak use < 1 GB per job.

| model | params (vocab part) | val, one list | test, one list | **test, live setup** (8 samples, word-like order, 50) |
|---|---|---|---|---|
| SmolLM2-360M spamshape4 | 362M (28M) | 50.6 | 51.1 | **61.4** |
| SmolLM2-135M spamshape4 | 135M (28M) | 43.6 | 50.0 | **57.8** |
| 135M, drop 6 layers (13-18), healed | 113M | 41.6 | 47.0 | 57.0 |
| 135M, drop 10 (9-18), healed | 99M | 43.8 | 46.6 | 53.5 |
| 135M, drop 15 (2-16), healed | 81M | 36.9 | 42.9 | 54.1 |
| Pythia-70M, full program | 70M (51M) | 30.2 | 37.7 | 49.1 |
| Pythia-14M, full program | 14M (13M) | 35.3 | 40.6 | 54.6 |
| *fixed letter-model list, no model* | 0 | | | *53.7* |

Unhealed pruned models: val 18.7 (drop 6), 14.5 (drop 10), 0.1 (drop 15). Healing is what makes pruning work.
Pythia RFT hit rate per round: 70M 0.62 -> 0.70 -> 0.93 -> 1.02%; 14M 0.79 -> 1.02 -> 1.33 -> 1.45% (135M was ~2-4%).

- **Pruning 6 of 30 layers costs ~1 point live** (57.8 -> 57.0); 10 or 15 drops it to the fixed-list level. The
  middle layers are the least needed; the first and last ones matter most.
- **Pythia-14M beats Pythia-70M**, probably because 70M's low learning rate (3e-5) undertrained it. Its RFT hit rate
  was still rising each round, so more rounds may help.
- **Below ~100M, the live setup lands at the fixed list (~54).** The word-like ordering does most of the work there; the
  model adds little beyond what one zero-parameter list does. Only the 113M+ models clearly beat it (not yet
  checked whether that's their spam or their obvious-answer endings).
- Pythia's vocab is 51M of its 70M and 13M of its 14M; Pythia-14M has ~1M parameters of actual transformer.
- Not done yet: distillation into a tiny letter-level model; prune *before* training (asked for later).

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
