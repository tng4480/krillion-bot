# krillion-bot

How small a language model can be trained to score well at [Krillion](https://krillion.io)'s Daily Dive? The game
autocorrects guesses, so close-but-wrong strings ("garbles") still count, and a rejected guess costs nothing.

## Layout

- `probe/emulator.py`: offline copy of the autocorrect, which also serves as the training reward. Run it to score it
  against real probe outcomes.
- `probe/plan.py`: makes boundary and tie probes from an answer list.
- `probe/metrics.py`: compares similarity metrics against probe outcomes.
- `probe/kp2.js`: in-page prober for the game (paste into the console during a dive).
- `probe/receive.py`: localhost receiver for saving page data. The browser pane blocked it, so downloads were used instead.
- `data/reveal/<date>.json`: full official answer lists (answers, aliases, tier, score) for a Daily Dive.
- `data/probes/`: probe plans and logged results.
- `notes/autocorrect-probe.md`: findings so far.

## Autocorrect, as reverse-engineered (2026-11-18)

- **Single words:** accepted if the nearest answer or alias is within `allowed(len)` edits. That's 0 edits for 4
  letters or fewer, 1 for 5–7, 2 for 8–11 and 3 for 12+. Agrees with the server on 99% of accept/reject outcomes.
- **Multi-word:** the answer must have the same number of words, and every word must pass its own allowance (91%).
- **Ties:** the nearest answer wins, then the most common answer (lowest score), then probably a hidden popularity
  count. The emulator picks the same answer 95% of the time.

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt    # macOS: .venv/bin/pip
.venv/Scripts/python probe/emulator.py
```
