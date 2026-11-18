# Krillion autocorrect: what we know and how to probe it

## From reading the client code (krillion.io, 2026-11-18)

- Answers are judged on the server. The client sends `POST /api/submit` with
  `{date, promptId, answer, previewCorrection, playerId, runId}`. No word list or fuzzy matcher ships to the browser.
- The server replies in one of three ways:
  1. `{correction: "<canonical>"}`: only when `previewCorrection` is true. The client puts the corrected text in the
     input box and shows "from -> to"; nothing is scored yet, and you press enter again to submit it.
  2. `{accepted: true, answer, tier, score, quip, receipt}`: scored. `answer` is the canonical form, and the client
     stores your raw text separately as `rawAnswer`.
  3. Rejected, with a reason string shown to the player. You can keep typing until the 25 s run out.
- Settings has an "Answer corrections" toggle: "Preview corrected answers before scoring. Off accepts them
  immediately." With it off, a garbled input that the server can map is scored straight away as the canonical answer.
- The post-game reveal screen filters the answer list on `answer` and on `aliases[]`. So each prompt's answer key holds
  a canonical name plus a list of aliases. The fuzzy layer, whatever it is, probably runs on top of those.
- A rejection doesn't end the round, so within 25 s you can retry: try a cheap garble first, then a clean answer if
  it's rejected.

## Results: Daily Dive #125, 2026-11-18 (37 probes, final score 440/700)

Raw log: `data/probes/2026-11-18_dive125.jsonl`.

- **Typos and dropped letters are corrected generously.** Examples: `espreso`, `rizzoto`, `axolotel`, `blobfsh`,
  `kohlrbi`, `manggostin`, `tendulkr`, `nashvil`, `ratatui` (4 letters short of Ratatouille).
- **Prefixes don't work on their own.** `ratat` was rejected and `ratatui` was accepted, so it's a similarity score
  with a threshold, not prefix matching.
- **Vowel-stripped strings fail.** `gzpcho` was rejected while `gazpach` was accepted, so the vowels carry weight.
- **The matcher is loose enough that invented words can land.** `kalamato` became Affogato and `blorpville` became
  Bloomville (an obscure town), and `velkor` became Vel'Koz. Shared suffix plus similar length seems to be enough.
  Fully random strings (`zxqvbnmk`, `zorak`, `blorp`, `blorpnip`, `flurbington`) were rejected.
- **A surname alone resolves to the most famous player.** `jordan` → Michael Jordan, `henry` → Thierry Henry,
  `beckam` → David Beckham. Aliases also work: `spud` → Potato. So the answer key has an alias layer.
- **It picks the nearest string, not the most famous match.** `strasburgh` → Strasburg, not Strasbourg.
- **Constraints are enforced.** `quokka` (6 letters) was rejected on the 7+ letter prompt, with the same "no echo"
  message that unknown strings get.
- **Exact canonical answers skip the preview and score immediately**, so a probe list must not contain real answers
  unless you mean to bank them.

Scored: Fantastic Mr. Fox 30 · Pisco 60 · Pademelon 85 · Matthew Pavlich 60 · Orangeburg 60 · Salsify 60 · Kalista 85.

## Boundary run, 2026-11-18 (482 probes, nothing banked)

The dictionary is the game's own reveal, `data/reveal/2026-11-17.json` (110k answers, 93k aliases). It came from the
results screen after a throwaway dive. The aliases contain player typos, so part of the "autocorrect" is just alias
lookup.

Probes were generated offline with a known nearest distance (`probe/plan.py`), run in the page, then joined in
`data/probes/run-2026-11-18-boundary.joined.json`.

**Single-word rule (99% accept/reject agreement over 356 probes):** accept if min Levenshtein to any answer or alias is
at most `allowed(len(query))`. Allowed edits are: length ≤4 → 0, 5–7 → 1, 8–11 → 2, 12+ → 3. **Tie-break: the most common answer wins.** The server always picks the nearest answer (85/85, never a farther
one). When nearest answers tie and have different scores, it picked the lowest score, the most common answer, every
time (37/37). Ties within one tier (46) aren't explained by alphabetical order, reveal order, alias count or any string
similarity tested (best: Jaro-Winkler 19/46), so they most likely follow a hidden submission count. The emulator ranks
by (distance, score, reveal position) and now gets the exact answer 95% of the time (337/356), up from 85%.

**Multi-word rule: every word must pass on its own (136/149).** The candidate must have the same number of words,
and each query word must be within `allowed(len(word))` of the matching word. That's why `baiana stm` is rejected:
"stm" is 3 letters, so it needs an exact match. The rule never accepts something the server rejected.

Rules that did worse:
- whole-string Levenshtein: 81
- fuzz ratios with a tuned threshold: ≤132
- "per-word, or whole-string with one word anchored": ~105, because it accepts many near-misses the server rejects

The 13 misses are all acceptances that no string metric explains, e.g. `littlev rd en` → Raven the Little Rascal and
`canin the aat` → The Cat in the Hat. This suggests a semantic fallback (embeddings or an LLM) when the string match
finds nothing. That's untested, and it could be a second exploit route.

## Probe protocol (play by hand, turn preview ON)

Preview mode is the cheap way to test: a correction comes back without being scored, so you see the mapping and can
clear the box and try another garble on the same prompt.

For each prompt, try these against an answer you know is valid (e.g. a prompt where "kangaroo" fits):

| # | Kind | Example input | Result (correction / accepted / rejected) |
|---|------|---------------|-------------------------------------------|
| 1 | one typo | kangaroa | |
| 2 | two typos | kangarra | |
| 3 | dropped letters | kngro | |
| 4 | prefix only | kanga | |
| 5 | shorter prefix | kan | |
| 6 | phonetic | cangaru | |
| 7 | case/space/punct | Kan Garoo! | |
| 8 | extra word | the kangaroo | |
| 9 | alias / nickname | roo | |
| 10 | wrong but close to a different valid answer | ? | |

What each pattern would mean:

- 1-2 pass, 3-5 fail: edit-distance with a small threshold. A model has to know the answer but may misspell it.
- 4-5 pass: prefix matching. Big win: a model only needs the first few letters of a rare answer.
- 6 passes: phonetic (Soundex/Metaphone-like). Spelling barely matters.
- 9 passes: aliases do the work, and they're in the reveal data, so training data should include them.
- Note when two valid answers are near each other: which one does it pick? A model can aim garbles at the rarer one.
