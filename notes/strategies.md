# Guessing strategies, measured (Dive #125, emulator-scored)

Run with `analysis/strategies.py --n 800 --tries 40`. Raw numbers are in `analysis/strategies-results.json`.

**Game rule simulated:** you submit guesses in order until one is accepted (a rejection is free), with 40 tries per
round. `E[game]` is the expected round score under that rule. `+preview` adds the correction-preview rule: a corrected
(non-exact) guess can be declined if it maps to a basic answer, while an exact answer is scored at once. Saying the
obvious answer scores about 28 on every prompt.

| Prompt (pool size) | Best strategy | E[game] | +preview | Basic answers |
|---|---|---|---|---|
| Animal, 7+ letters (73k, almost all fossil/Latin genera) | char-model nonsense / Latin genus splice | 85 | 85 | 29 |
| One-word food ending in O (3.1k) | diminutive splice (`-illo`, `-etto`, `-inho`) | 84 | 84 | 27 |
| Retired player (25k) | name spam: common first name × common surname | 85 | 85 | 28 |
| City ending -ville/-burg (3.3k) | suffix splice / direction prefix (`North X`) | 82 | 85 | 18 |
| White-inside fruit/veg (453) | char-model nonsense | 70 | 74 | 26 |
| Game character (288) | char-model nonsense | 61 | 65 | 29 |
| Talking-animal movie (4.5k, long multi-word titles) | sequel/extension of a known franchise | 43 | 43 | 29 |

## Patterns found in the answer lists

- **Latin genera dominate the animal prompt.** Of 73k answers, the top endings are -ella (4k), -ites (3k), -eras (2.7k),
  -inus, -urus, -spis, -odon and -chus. Almost any well-formed fake genus lands on a real one (48% acceptance), and
  every hit is a Deep Cut.
- **Foreign diminutives fill the "-o" food prompt.** The top endings are -ello, -illo, -etto, -rito, -inho and -cino.
  5% of answers have non-ASCII characters.
- **A direction word upgrades a common answer.** `Nashville` scores 10, but `East Nashville` is a separate 85-point
  answer. East, West, South and New all work for cities, and every hit is 85.
- **A descriptor upgrades a common answer.** Examples: Abalone mushroom, Aerial yam, African custard apple, Chinese X,
  Bush X. 105 of the fruit/veg answers extend a basic answer this way. Scores rise with word count: 1 word averages
  68, 4 words average 85.
- **Franchise extensions work for movies.** Examples: Scooby-Doo spin-offs, "Alvin and the Chipmunks Meet …", "… 2",
  "… the Movie", "(1998)". The catch is that an extension often snaps back to the base franchise, which is a common
  answer.
- **Name spam works because player pools are huge.** 25k retired players, 11% with accented names (La Liga), and many
  disambiguated with "(born 1982)". "Greg Smith" and "Paul Long" exist and score 85.

## What this means for the model

1. **Strategy depends on the prompt.** Big pools with regular word shapes (Latin genera, -o foods, -ville towns,
   person names) are best attacked with well-shaped nonsense. Small or irregular pools (games, fruit, film titles) need
   real knowledge. The model has to tell which kind of prompt it's facing, which needs many prompts in training.
2. **A rejection costs nothing, so more guesses win.** Low per-guess precision (2–20%) still gives a high round score
   with 40 tries. Accuracy is the wrong thing to optimise; the round score under the retry rule is the right one.
3. **Never say the obvious answer, except as a last resort.** It scores about 28 everywhere and ends the round.
4. **Preview turns garbles into a safe probe.** An exact answer is committed immediately, but a garble shows its
   correction first, so a bot can see what it would map to and decline basics. That's worth +2–7 points where
   corrections often land on common answers.

**Caveat:** the char-model generators are trained on the prompt's own answer list, so they overstate what a model can
do on a prompt it has never seen. On a new prompt, the model's equivalent is its pretraining knowledge of how names in
that domain look (Latin taxonomy, Italian food words, -ville towns). That transfers well for genera and names, and
less well for film titles.
