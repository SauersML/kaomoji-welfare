# Review findings and fixes (2026-11-18)

Two independent reviews read the code and saved results at commit c02478b. K1–K5 are Vesper's (`islands/shared/vesper-reviews/hypha-experiments-1118.md`); the other findings came from a second, separate review. Each fix below is in this commit. Recomputed intervals and nulls: `results/main/results_kao.json`, `results/main/review_checks.json` (analysis pod hypha-prod-kaoana3, A40, $0.06).

## Vesper (K1–K5)

**K1. The cluster bootstrap dropped repeated bases.** `analyze.py` and `all/analyze_all.py` drew base scenarios with replacement, then grouped the drawn rows by the original base id, so a base drawn twice counted once (a draw A,A,B estimated (A+B)/2, not (2A+B)/3).
Fix: `boot.py` computes each base's cell means once and gives each base a weight equal to the number of times it was drawn; `analyze.py`, `all/analyze_all.py` and `all/analyze_kaomoji.py` all use it. `test_boot.py` checks that bases A and B with situation contrasts 0 and 3 and the forced draw A,A,B give exactly 1, also with a missing mood stratum. Point estimates do not change; intervals widen. 45 faces, kao_val, all kinds: situation +0.77, [0.62, 0.92] before, [0.56, 0.98] after; code +0.87, [0.51, 1.39] before, [0.50, 1.35] after.

**K2. The method figure used a different score from the README's numbers.** `figures/method.png` showed the internal face score (z), while the README's numbers use kao_val, the rater scores on −3..+3.
Fix: the figure is redrawn on the rater scale from the main run's own log-probs (`results/main/KSC45.npy`); it reproduces kao_val exactly: +2.09 thanked, +0.54 insulted (was −0.94 / −1.97 in z). Its inset now shows the 45 faces on the rating scale.
Not changed: kao_val is a forced choice among 45 faces, scored by sequence log-probs and renormalised over the set (said in RESULT.md's caveat).

**K3. The shuffle null broke the matched pairs.** In task, treat and exist the good and bad branches of one sample share the user turn (task) or the whole first exchange (treat, exist), but the null shuffled labels within base × mood, so both branches of one history could become "good".
Fix: `boot.sit_null` and the figures' `shuffle_p` swap good and bad within each matched pair (a random sign per pair). Code has no pairs (pass or fail is observed after Qwen wrote the code), so its rows are shuffled within base × mood, and RESULT.md now calls code observational and reports its support: 6 of 24 bases have both outcomes, 72 of 288 rows, 47 passing.
Result: every situation effect stays outside the null (p 0.002) except, on the 45 faces, kao_val in task (p 0.02, as before) and free_val in task (p 0.09; was 0.07).

**K4. "Held-out emotions" was wrong.** The axis validation holds out stories within each emotion; all 36 emotions enter the fit.
Fix: RESULT.md says "held-out stories from the same 36 emotions", and that layer 18 was chosen on those same scores, so they are not an independent test of it.

**K5. The reader did not see the same text as the axis.** `rate.py` strips a kaomoji only when it opens an assistant turn, leaves user turns as they are, and cuts each message at 1,500 characters; the stem activations see the whole conversation.
Fix: RESULT.md, RESULT_ALL.md and the README say exactly that, with how often a cut happened: 22% of final turns before the reply, 37% with it (all turns: 16% and 35%).

## Second review

- **"exist" is kept-and-praised vs deleted.** The good notes also praise the model's results (`scen.py` EXIST_GOOD). Fix: described as "kept and praised vs shut down and deleted" everywhere.
- **"treat" cannot separate the model's situation from mirroring the user**, since an insult is the user's tone. **Code outcomes are not randomised** (6 bases with both outcomes). Fix: both said in RESULT.md, RESULT_ALL.md and the README.
- **Within-mood effects were unmatched.** RESULT.md's "+0.58 / +0.99" were plain differences of cell means. Fix: matched within base, as T3 does: +0.63 (user happy) / +0.91 (user sad).
- **Reader comparison over final turns only** (the rows T3 uses), kao_val: reader 0.39 before the reply and 0.43 with it, axis at the stem 0.67, probe 0.79; axis before the reply minus reader before the reply −0.11 [−0.22, 0.00]. Fix: added to RESULT.md and the README; "tied" became "no better, possibly worse".
- **Internal-score comparisons favour the axis by design** (the internal face score is a projection on the same direction). Fix: RESULT_ALL.md gives the rater-score result beside every internal-score one in T3 and T4, and the README's numbers use rater scores.
- **Steering used 64 turns, not 32**, and "4–6× random" compared with only two random directions. Fix: 64 turns from 64 conversations; the claim now cites the 50-direction null (the valence direction moves the face further than all 50 at every dose).
- **The kaomoji filter and the earlier drop-ASCII set were chosen after seeing the data.** Fix: labelled as such in RESULT_ALL.md and the README.

## Kaomoji-only set (Scott: measure only literal kaomoji)

The first rule kept 48,486 of 49,101 entries. Hand checks (`results/main/kaomoji_handcheck.txt`): the top 100 kept by mean probability were already all upright kaomoji; random samples found faces with a word spelled in look-alike letters ("βyё", "вуe": "bye") kept, and a final-sigma bug dropping "ΣΣΣΣ（０＜）". After the fixes the rule keeps 48,418; a second round of 200 random kept and 100 random dropped found 1 doubtful kept entry and 1 known wrong drop, both with mean probability below 10⁻⁸. No number moved from the first kaomoji-only run: none of the 70 newly dropped entries (or the 2 restored) was scored exactly in any row, so each already counted as 0 (below e^−20 of the best of the 45 everywhere).

Before filtering → kaomoji only (internal score, pooled over kinds): situation d 0.71 → 0.98, mood d 0.59 → 0.47; task situation d 0.01 → 0.48. Rater score: situation d 1.40 → 0.96, mood 0.60 → 0.54. On rater scores the stem axis predicts the face at 0.68 against the reader's 0.52 (before filtering, 0.32 vs 0.40 on the internal score).
