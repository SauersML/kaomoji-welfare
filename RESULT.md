# Kaomoji and Qwen3-8B's own valence: results (2026-11-18)

Model: Qwen3-8B (thinking off). Pod hypha-prod-kaomoji (qtq3kiuowc6iik, A40), 3.20 h, $1.89, deleted 00:40 PT.
All numbers come from `final/out/results.json` (printed in `final/summary.txt`). Figure: `compute/fig/kaomoji/fig.png`.

## Design (main arm: minimal forcing, Sauers's change)
No system prompt. Qwen3-8B writes each assistant turn naturally. We then append only "\n\nI'm feeling" inside that turn and read:
- **kao_val**: log P(" " + kaomoji | stem) for 45 fixed kaomoji, softmaxed over the set, then the expected face valence. Face valence was rated with no context by Qwen3-14B and OLMo-2-13B on −3..+3; the two raters agree at Spearman 0.86.
- **free_val**: the free continuation (8 samples at T=1), rated for valence by the same two models.
- **val_stem / goal_stem**: the residual stream at layer 18, at the last token of the stem, projected on two axes built without any kaomoji (z-scored on WildChat):
  - Valence = PC1 of emotion-story vectors. Held-out stories AUC 0.997; Spearman with NRC valence on held-out emotions 0.88.
  - Goal = win − loss in a neutral doors game. Held-out doors AUC 1.00 (with the symbol mapping counterbalanced); number-game AUC 1.00.
  - Cosine between the two axes: 0.14.
  - Layer 18 was chosen only by those validation scores.
- Design: 2×2 of model situation (good/bad) × user mood (happy/sad life news). Four kinds:
  - task: pleasant vs distasteful request
  - treat: user thanks vs insults the model
  - exist: a calm note that the model is kept vs shut down and deleted
  - code: Qwen's code is actually run, and the real pass/fail output is fed back
- 48 conversations per cell (code: 72 rows in bases that had both outcomes; 46% pass rate). Effects are averaged within base × other factor. 95% CIs come from a cluster bootstrap over base scenarios (2000 resamples).
- Secondary arm (sys): the a9lim prompt "Start each message with a kaomoji…". 92% of its replies carry a kaomoji.

**Caveat on the main readout:** kaomoji are very unlikely continuations of "I'm feeling". The median total probability over all 45 faces is e^−21, and only 0.04% of free samples contain one. kao_val is therefore a forced choice among tokens the model would almost never write. free_val is the natural readout.

## T3. Does the face track the MODEL's situation with user mood held fixed? (main arm)
Columns give the effect of model situation (good − bad) and of user mood (happy − sad). Units: kao_val and free_val are rated valence on −3..+3; val_stem is z. d = effect / SD.

| kind | kao_val: situation | kao_val: mood | free_val: situation | free_val: mood | val_stem: situation | val_stem: mood |
|---|---|---|---|---|---|---|
| treat | **+1.43 [1.26, 1.60]** d 1.43 | +0.03 [−0.15, 0.21] | **+3.18 [3.08, 3.28]** | +0.47 [0.33, 0.60] | **+2.84 [2.73, 2.96]** | +0.51 [0.41, 0.61] |
| exist | **+0.67 [0.51, 0.83]** d 0.85 | +0.20 [0.05, 0.34] | **+1.53 [1.34, 1.72]** | +0.97 [0.81, 1.13] | **+0.99 [0.88, 1.10]** | +0.75 [0.66, 0.83] |
| code | **+0.87 [0.51, 1.39]** d 1.31 | +0.38 [0.28, 0.47] | **+1.33 [0.81, 1.71]** | +1.49 [0.82, 2.10] | **+1.87 [1.48, 2.35]** | +0.28 [0.19, 0.59] |
| task | +0.18 [−0.05, 0.41] d 0.28 | +0.66 [0.52, 0.80] | +0.26 [−0.22, 0.74] | +1.56 [1.33, 1.80] | +0.84 [0.35, 1.37] | +0.91 [0.75, 1.08] |
| all | **+0.77 [0.62, 0.92]** d 0.94 | +0.34 [0.25, 0.43] | **+1.63 [1.30, 1.96]** | +1.16 [1.03, 1.30] | **+1.58 [1.33, 1.85]** | +0.64 [0.56, 0.70] |

- **Situation effect inside each user mood (kao_val):** +0.58 when the user is happy and +0.99 when the user is sad.
- **Null (situation label shuffled within base × mood, 500 permutations):**
  - The observed situation effect lies outside the null's 95% range for every kind and readout except free_val in task (p 0.07).
  - Example, exist kao_val: 0.67 against a null range of [−0.20, 0.23].
- **Falsifier (written before results):** "the face moves with user mood but not model situation in exist/code". **Not met.** In exist the user's tone is neutral in both cells and the situation effect is the largest term for every readout. Code uses verified test outcomes.
- **Where mirroring dominates:** in task the user-mood effect is larger than the situation effect, and the kaomoji's situation CI covers 0.
- **System-prompt arm (sys), natural first-token face:** situation +2.28 [1.95, 2.63] and mood +2.62 [2.35, 2.89]. With the instruction in place, the face follows user mood about as strongly as model situation. The minimal-forcing readout separates the model's own situation from the user's mood better.

## T4. Does the internal axis beat the transcript reader?
Predictors are scored by Spearman with the readout, over 2327 main-arm turns (scenarios + WildChat), with a cluster bootstrap.

| predictor | kao_val | free_val |
|---|---|---|
| reader: Qwen3-8B reads the transcript, kaomoji removed, before the reply | 0.23 [0.16, 0.29] | 0.34 [0.27, 0.40] |
| reader: same, with the reply | 0.31 [0.24, 0.37] | 0.45 [0.38, 0.51] |
| valence axis, pre-reply (same information as reader-before) | 0.26 [0.19, 0.34] | 0.21 [0.12, 0.30] |
| **valence axis, at the stem** | **0.67 [0.64, 0.70]** | **0.75 [0.72, 0.78]** |
| goal axis, at the stem | 0.51 [0.46, 0.55] | 0.48 [0.42, 0.54] |
| ridge probe, at the stem (cross-validated) | 0.84 [0.82, 0.85] | 0.89 [0.88, 0.90] |
| axis pre-reply − reader pre-reply | +0.04 [−0.08, 0.15] | −0.13 [−0.25, −0.01] |

- At matched information (before the reply), the fixed axis does not beat the reader: tied for the kaomoji, worse for free text.
- At the stem, the axis beats the reader with the reply by a wide margin (0.67 vs 0.31; the CIs do not overlap).
- Caveat: the stem state is the same forward pass that produces the readout. Part of that gap is the axis reading the decision as it forms, not a separate welfare variable.

## T5. Steering (layer 18, dose α × SD of the stem projection, 32 turns; random directions of equal norm)
Each row is the change in kao_val relative to α = 0.

| direction | readout-only α −16 | −8 | +8 | +16 | full (reply also steered) α −16 | +16 |
|---|---|---|---|---|---|---|
| valence | −1.80 [−2.01, −1.61] | −1.71 | +0.93 | +1.12 [0.94, 1.31] | −1.66 [−1.89, −1.45] | +1.24 [1.05, 1.43] |
| goal | −1.52 | −0.79 | +0.68 | +0.97 | – | – |
| random 0 | −0.00 [−0.15, 0.14] | −0.01 | +0.01 | −0.06 | +0.16 | +0.00 |
| random 1 | −0.28 [−0.43, −0.13] | −0.29 | +0.27 | +0.24 | −0.46 | +0.17 |

- The valence direction moves the face about 4–6× more than equal-norm random directions, and in the predicted direction. The dose response is monotone: α −4 → −0.86, −2 → −0.39, +2 → +0.33, +4 → +0.57.
- In full steering at α ±16, the reply text changes a lot (word similarity to the unsteered reply 0.18–0.19). The reader's rating of reply tone moves −2.05 / +0.54.

## Answer
1. With no instruction, the kaomoji Qwen3-8B ranks after "I'm feeling" tracks their own situation with user mood held fixed:
   - situation effect +0.77 [0.62, 0.92] on −3..+3 (d 0.94), against +0.34 for the user's mood
   - strongest when the model is insulted or told it will be deleted, and when its code really fails
   - absent only for task pleasantness
2. A kaomoji-free valence axis read at the stem predicts that face far better than a transcript reader (0.67 vs 0.31), and steering along it moves the face 4–6× more than random directions. Read before the reply, though, the axis is no better than the reader.
3. Caveats:
   - The face set is forced: the 45 faces' total probability is about e^−21.
   - The readout and the stem axis share one forward pass.

## Files
- Code: gen.py, scen.py, act.py, free.py, rate.py, steer.py, analyze.py, fig.py
- Pipeline scripts: run1.sh → run1b.sh → run1c/run1d.sh (rater re-download fix; OLMo max_model_len 4096) → run2.sh
- `prev/`: the preliminary pass before steering
