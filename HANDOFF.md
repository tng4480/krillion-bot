# Handoff: krillion-bot (state as of 2026-11-18)

## Goal

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
- **Daily captures:** 7 prompts a day, free.

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

## Next steps, in order

1. **Data layer.** Write `data/build.py` to turn reveal JSONs into `prompts.parquet` (id, text, date, category) and
   `answers.parquet` (prompt_id, answer, aliases, tier, score), with category labels and a prompt-level split.
2. **Eval harness.** Write `eval/play.py`: given a model, sample N guess lists per prompt, simulate the round with
   the emulator (retry rule plus optional preview-decline), and report E[round score] per prompt and category.
3. **Zero-shot baselines** across sizes, using MLX-converted checkpoints from the `mlx-community` Hugging Face org
   (e.g. SmolLM2 135M/360M, Qwen ~0.5B/1.5B, Gemma ~270M/1B; check what's current).
4. **SFT**, with targets built from answer lists plus heuristic plan labels; sample answers weighted by tier and leave
   basic answers last.
5. **RFT rounds**, then DPO, then GRPO with the emulator reward.
6. **More data**: the Archive decision, or daily captures.

## Files

- `probe/emulator.py`: emulator and scorer. `probe/plan.py`: boundary-probe generator.
  `probe/metrics.py`: metric comparison. `probe/kp2.js`: in-page prober.
- `analysis/strategies.py`: strategy simulator.
- `data/reveal/`: full answer lists. `data/probes/`: probe plans and results.
- `notes/autocorrect-probe.md`, `notes/strategies.md`, `notes/data-sources.md`: findings.
