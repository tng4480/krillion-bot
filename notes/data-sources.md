# Data sources for more prompts (checked 2026-11-18)

More distinct prompts matter more than more answers per prompt: they're what lets the model learn which strategy
fits which kind of prompt. A source is only usable for the reward if it gives the **complete** answer list, with tiers
and aliases.

| Source | Prompts | Complete answer list? | Status |
|---|---|---|---|
| Daily Dive reveal (results screen) | 7 per day | Yes, with aliases, tier and score | Working: `data/reveal/2026-11-17.json` |
| Archive unlock ($5, "every aired daily") | ~125 dives × 7 ≈ 875 | Very likely; same results screen as the daily | Not bought |
| Themed packs (free beta) | 4 × 30 × 7 ≈ 840 | **No** | Dead end, see below |
| krillionanswers.com catalogue | 322 | **No**, top 200 per prompt and no aliases | Not usable as a reward |

## Themed packs

I played Words & Language chapter 1 and let every round time out, so nothing was scored. The pack made three API
calls: `/api/packs/lexicon/1` (1 KB, prompts only), `/api/packs/lexicon/1/submit` (an empty submission on timeout)
and `/api/announcements/current`. The results screen never requested a reveal, so packs give prompt text but no
answers. They could still be used as unlabelled prompts for zero-shot testing.

Timing out is a clean way to page through games: blank submissions score nothing and add nothing to anyone's stats.

## Top-200 truncation test

I simulated the catalogue's lists by keeping the top 200 answers per prompt (score descending, then alphabetical), and
compared emulator rewards with the full list. Guesses came from the strategy generators in `analysis/strategies.py`.

| Prompt | Full list | Reward unchanged | False 0 | Mean reward, full → top 200 |
|---|---|---|---|---|
| p1 movies | 4501 | 71% | 29% | 9.0 → 0.0 |
| p2 food ending -o | 3133 | 44% | 55% | 28.0 → 2.3 |
| p3 animals | 73295 | 42% | 58% | 29.9 → 0.0 |
| p4 players | 25158 | 57% | 43% | 16.6 → 0.0 |
| p5 -ville cities | 3325 | 52% | 41% | 28.5 → 10.6 |
| p6 white-inside fruit | 453 | 56% | 42% | 14.6 → 1.5 |
| p7 game characters | 288 | 43% | 56% | 27.1 → 10.6 |

Truncation drops every common answer, so the reward can no longer teach "avoid the obvious", and it drops the long
tail too. Within a tier the cut is alphabetical, so what's left is mostly Deep Cuts starting with "A". Use the catalogue
only for prompt text, and treat its answers as a few rare positive examples with a strong A-bias, never as a reward
dictionary.

## Hardware (training runs locally)

- Windows laptop: Ryzen AI 7 350, Radeon 860M integrated GPU (shares system RAM), 30 GB RAM, no NVIDIA, no WSL.
- MacBook Air M5: 16 GB unified memory, Apple GPU usable from MLX / PyTorch MPS.
