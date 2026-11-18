# krillion-bot

Read `HANDOFF.md` first: it has the project state, decisions and next steps.

- Primary machine: MacBook Air M5, 16 GB. Train and sample with MLX (`mlx-lm`); keep jobs short and checkpointed
  (the machine is fanless).
- Python venv in `.venv` (`.venv/bin/python`); dependencies in `requirements.txt`.
- `probe/emulator.py` is the reward function. Re-run it after any change to matching logic (it scores itself
  against the logged probes).
- Split data by prompt, never by answer. Report round score under the retry rule, not per-guess accuracy.
- `data/reveal/*.json` is the game's answer data. Keep the repo private.
