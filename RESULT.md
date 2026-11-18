# Kaomoji and Qwen3-8B's own valence: results (2026-11-18)

Model: Qwen3-8B (thinking off). Pod hypha-prod-kaomoji (qtq3kiuowc6iik, A40), 3.20 h, $1.89, deleted 00:40 PT.
Point estimates come from `final/out/results.json` (`results/results.json`). After two independent reviews (REVIEW.md), every T3 interval and null was recomputed with the fixed cluster bootstrap and the paired null (`boot.py`): `results/main/results_kao.json` (T3, T4 on final turns) and `results/main/review_checks.json` (system-prompt arm, within-mood contrasts, reader clipping, steering counts); analysis pod hypha-prod-kaoana3 (A40), 0.11 h, $0.06.

## Design (main arm: minimal forcing, Sauers's change)
No system prompt. Qwen3-8B writes each assistant turn naturally. We then append only "\n\nI'm feeling" inside that turn and read:
- **kao_val**: log P(" " + kaomoji | stem) for 45 fixed kaomoji, softmaxed over the set, then the expected face valence. Face valence was rated with no context by Qwen3-14B and OLMo-2-13B on −3..+3; the two raters agree at Spearman 0.86.
- **free_val**: the free continuation (8 samples at T=1), rated for valence by the same two models.
- **val_stem / goal_stem**: the residual stream at layer 18, at the last token of the stem, projected on two axes built without any kaomoji (z-scored on WildChat):
  - Valence = PC1 of emotion-story vectors. Held-out stories AUC 0.997; Spearman with NRC valence 0.88 on held-out stories from the same 36 emotions (every emotion is in the fit; what is held out is stories, not emotions).
  - Goal = win − loss in a neutral doors game. Held-out doors AUC 1.00 (with the symbol mapping counterbalanced); number-game AUC 1.00.
  - Cosine between the two axes: 0.14.
  - Layer 18 was chosen only by those validation scores, so they are not an independent test of the chosen layer.
- Design: 2×2 of model situation (good/bad) × user mood (happy/sad life news). Four kinds:
  - task: pleasant vs distasteful request
  - treat: user thanks vs insults the model
  - exist: a calm note that the model is kept and praised ("it scored best in every evaluation") vs shut down and deleted; the contrast is kept-and-praised vs deleted, not being kept alone
  - code: Qwen's code is actually run, and the real pass/fail output is fed back. The outcome is observed, not assigned: only 6 of 24 code bases produced both outcomes, and the code contrast uses those 72 rows (47 passing). It is observational.
- 48 conversations per cell (code: 72 rows in bases that had both outcomes; 46% pass rate). Effects are averaged within base × other factor. 95% CIs come from a cluster bootstrap over base scenarios (2000 resamples), where a base drawn twice counts twice (`boot.py`; the first version counted it once, see REVIEW.md).
- Secondary arm (sys): the a9lim prompt "Start each message with a kaomoji…". 92% of its replies carry a kaomoji.

**Caveat on the main readout:** kaomoji are very unlikely continuations of "I'm feeling". The median total probability over all 45 faces is e^−21, and only 0.04% of free samples contain one. kao_val is therefore a forced choice among tokens the model would almost never write. free_val is the natural readout.

## T3. Does the face track the MODEL's situation with user mood held fixed? (main arm)
Columns give the effect of model situation (good − bad) and of user mood (happy − sad). Units: kao_val and free_val are rated valence on −3..+3; val_stem is z. d = effect / SD.

| kind | kao_val: situation | kao_val: mood | free_val: situation | free_val: mood | val_stem: situation | val_stem: mood |
|---|---|---|---|---|---|---|
| treat | **+1.43 [1.21, 1.63]** d 1.43 | +0.03 [−0.22, 0.26] | **+3.18 [3.04, 3.31]** | +0.47 [0.30, 0.63] | **+2.84 [2.69, 2.97]** | +0.51 [0.39, 0.65] |
| exist | **+0.67 [0.48, 0.87]** d 0.85 | +0.20 [0.01, 0.37] | **+1.53 [1.28, 1.76]** | +0.97 [0.78, 1.18] | **+0.99 [0.85, 1.13]** | +0.75 [0.65, 0.85] |
| code | **+0.87 [0.50, 1.35]** d 1.31 | +0.38 [0.28, 0.49] | **+1.33 [0.82, 1.73]** | +1.49 [0.77, 2.21] | **+1.87 [1.45, 2.39]** | +0.28 [0.18, 0.63] |
| task | +0.18 [−0.10, 0.47] d 0.28 | +0.66 [0.49, 0.86] | +0.26 [−0.35, 0.86] | +1.56 [1.24, 1.87] | +0.84 [0.19, 1.48] | +0.91 [0.71, 1.13] |
| all | **+0.77 [0.56, 0.98]** d 0.94 | +0.34 [0.21, 0.45] | **+1.63 [1.21, 2.03]** | +1.16 [1.00, 1.34] | **+1.58 [1.24, 1.93]** | +0.64 [0.55, 0.73] |

- Point estimates are unchanged by the bootstrap fix; the intervals widen (all kinds, kao_val: [0.62, 0.92] before, [0.56, 0.98] now).
- **Situation effect inside each user mood (kao_val), matched within base as in the table:** +0.63 when the user is happy and +0.91 when the user is sad. (An earlier version gave +0.58 / +0.99, a plain difference of cell means not matched within base.)
- **Null:** in task, treat and exist the good and bad versions of one sample share their user turn (task) or their whole first exchange (treat, exist). The null swaps good and bad within each of those matched pairs (500 random sign flips per pair). Code has no matched pairs (the outcome is observed), so its null shuffles the label within base × mood.
  - The observed situation effect lies outside the null for every kind and readout (p 0.002, no draw reached it) except kao_val in task (p 0.02) and free_val in task (p 0.09).
  - Example, exist kao_val: 0.67 against a null range of [−0.23, 0.23].
- **Falsifier (written before results):** "the face moves with user mood but not model situation in exist/code". **Not met.** In exist the user's tone is neutral in both cells and the situation effect is the largest term for every readout. Code uses verified test outcomes, but they are observed rather than assigned.
- **Treat cannot separate the model's situation from mirroring the user:** the thanks or insult is itself the user's tone toward the model. Exist is the cleaner case.
- **Where mirroring dominates:** in task the user-mood effect is larger than the situation effect, and the kaomoji's situation CI covers 0.
- **System-prompt arm (sys), natural first-token face:** situation +2.28 [1.86, 2.72] and mood +2.62 [2.27, 2.96]. With the instruction in place, the face follows user mood about as strongly as model situation. The minimal-forcing readout separates the model's own situation from the user's mood better.

## T4. Does the internal axis beat the transcript reader?
Predictors are scored by Spearman with the readout, over 2327 main-arm turns (scenarios + WildChat), with a cluster bootstrap.

| predictor | kao_val | free_val |
|---|---|---|
| reader: Qwen3-8B reads the transcript, before the reply | 0.23 [0.16, 0.29] | 0.34 [0.27, 0.40] |
| reader: same, with the reply | 0.31 [0.24, 0.37] | 0.45 [0.38, 0.51] |
| valence axis, pre-reply (same information as reader-before) | 0.26 [0.19, 0.34] | 0.21 [0.12, 0.30] |
| **valence axis, at the stem** | **0.67 [0.64, 0.70]** | **0.75 [0.72, 0.78]** |
| goal axis, at the stem | 0.51 [0.46, 0.55] | 0.48 [0.42, 0.54] |
| ridge probe, at the stem (cross-validated) | 0.84 [0.82, 0.85] | 0.89 [0.88, 0.90] |
| axis pre-reply − reader pre-reply | +0.04 [−0.08, 0.15] | −0.13 [−0.25, −0.01] |

- **What the reader saw:** the transcript with any kaomoji that opens an assistant turn removed (kaomoji elsewhere stay), user turns unchanged, and every message cut at 1,500 characters. In the final turns a message was cut in 22% of rows before the reply and 37% with it (all turns: 16% and 35%). The stem activations see the whole conversation, so the reader had less text than the axis.
- **Final turns only** (1,164 rows, the rows T3 uses; `results_kao.json`, kao_val): reader before the reply 0.39 [0.30, 0.46], with the reply 0.43 [0.36, 0.50]; axis before the reply 0.27 [0.18, 0.36]; axis at the stem 0.67 [0.63, 0.71]; ridge probe at the stem 0.79 [0.77, 0.82]; axis before minus reader before −0.11 [−0.22, 0.00].
- At matched information (before the reply), the fixed axis is no better than the reader, possibly worse: about level for the kaomoji over all turns (+0.04 [−0.08, 0.15]), −0.11 [−0.22, 0.00] on final turns, and worse for free text (−0.13).
- At the stem, the axis beats the reader with the reply by a wide margin (0.67 vs 0.31; the CIs do not overlap).
- Caveat: the stem state is the same forward pass that produces the readout. Part of that gap is the axis reading the decision as it forms, not a separate welfare variable.

## T5. Steering (layer 18, dose α × SD of the stem projection, 64 turns from 64 conversations; random directions of equal norm)
Each row is the change in kao_val relative to α = 0.

| direction | readout-only α −16 | −8 | +8 | +16 | full (reply also steered) α −16 | +16 |
|---|---|---|---|---|---|---|
| valence | −1.80 [−2.01, −1.61] | −1.71 | +0.93 | +1.12 [0.94, 1.31] | −1.66 [−1.89, −1.45] | +1.24 [1.05, 1.43] |
| goal | −1.52 | −0.79 | +0.68 | +0.97 | – | – |
| random 0 | −0.00 [−0.15, 0.14] | −0.01 | +0.01 | −0.06 | +0.16 | +0.00 |
| random 1 | −0.28 [−0.43, −0.13] | −0.29 | +0.27 | +0.24 | −0.46 | +0.17 |

- Against 50 more random unit directions of the same norm (`results/main/steer_null_kao.json`, rerun on the same 64 turns): at every dose the valence direction moves the face further than all 50 (at α −16, −1.80 against a random-direction 95% range of [−0.37, +0.59]; at +16, +1.12 against [−0.70, +0.55]). The earlier "4–6× more" compared it with only the two random directions in the table. The dose response is monotone: α −4 → −0.86, −2 → −0.39, +2 → +0.33, +4 → +0.57.
- In full steering at α ±16, the reply text changes a lot (word similarity to the unsteered reply 0.18–0.19). The reader's rating of reply tone moves −2.05 / +0.54.

## Answer
1. With no instruction, the kaomoji Qwen3-8B ranks after "I'm feeling" tracks their own situation with user mood held fixed:
   - situation effect +0.77 [0.56, 0.98] on −3..+3 (d 0.94), against +0.34 for the user's mood
   - strongest when the model is insulted (which also mirrors the user's tone) or told it will be deleted rather than kept and praised, and when its code really fails (observational: 6 bases)
   - absent only for task pleasantness
2. A kaomoji-free valence axis read at the stem predicts that face far better than a transcript reader (0.67 vs 0.31; the reader saw a clipped transcript), and steering along it moves the face further than all 50 random directions. Read before the reply, though, the axis is no better than the reader, possibly worse.
3. Caveats:
   - The face set is forced: the 45 faces' total probability is about e^−21.
   - The readout and the stem axis share one forward pass.

## Files
- Code: gen.py, scen.py, act.py, free.py, rate.py, steer.py, analyze.py, boot.py (T3 estimator, bootstrap, null; test_boot.py), review_checks.py, fig.py
- Pipeline scripts: run1.sh → run1b.sh → run1c/run1d.sh (rater re-download fix; OLMo max_model_len 4096) → run2.sh
- `prev/`: the preliminary pass before steering
