# Kaomoji, full set: results (2026-11-18)

Same model (Qwen3-8B, thinking off), same 1,164 saved final turns (864 scenario + 300 WildChat) and same "\n\nI'm feeling" stem as RESULT.md. The only change: the face set grows from the 45 hand-picked faces to every kaomoji in three public dictionaries.
Pod hypha-prod-kaoall2 (y7j17gs55vfahk, H100), 1.90 h, $8.34; plus a 0.24 h A40 setup pod, $0.14. Deleted 03:00 PT.
All numbers below come from `results/main/results_all.json` (and `results/main/all_sensitivity/`); code in `all/`.

## Set
- Sources: ekohrt emoticon_dict (46,016), kaomoji-data / kaomojikan (1,347), mrzjy (7,465).
- Filtered out: multi-line, any 2+ Latin letters, emoji pictographs, no bracket (or ツ), more than two faces, under 3 characters, control characters. Then deduplicated by Qwen3-8B tokenization of " " + face, and faces over 20 tokens dropped.
- **49,101 faces**, all 45 originals inside (`faces_all.tsv`).
- Per-row scoring (`score_prune.py`): a token-trie walk over the prefix KV cache. A branch is expanded while its prefix log-prob is within 20 nats of the best of the 45 originals in that row. Everything still inside that window is scored exactly. Faces outside it have weight below e^-20 of the best face. The median row scores 1,618 faces exactly. The 45 are always exact. Their log-probs reproduce the earlier run (Pearson 0.9996).
- `KSC_all.npz`: per row, the exact log-probs (CSR: `indptr`, `face_idx`, `logp`), the 45-face block `ksc45`, the pruning threshold, row ids and turns, and the face list.

## Face valence, two ways
- **Internal (main score):** Qwen3-8B reads the face alone, as plain text "I'm feeling <face>". We average the layer-18 residual stream over the face's tokens and project it on the same kaomoji-free valence axis as RESULT.md. The result is z-scored with the stem constants.
  - A second context (the face as the only user message, chat template) agrees with it at Spearman 0.89.
- **Raters:** Qwen3-14B and OLMo-2-13B, with rate.py's prompt (−3..+3). On the 45 they reproduce the old ratings to within 0.06. 12 faces stay unrated by OLMo.
  - The raters agree at Spearman 0.87 on the 45, but only 0.71 on the full set.
  - OLMo rates many crying or angry faces as positive, e.g. (´༎ຶ༎ຶ) +2.9 and ヾ(｡｀Д´｡)ﾉ┌┛ +2.7.
- **Internal vs raters:** Spearman 0.83 on the 45, 0.71 on all faces, 0.81 on the 500 most probable faces. Against each rater: Qwen3-14B 0.72, OLMo 0.58.
- Where they disagree most among probable faces, the internal score is low and the raters are high on excited or surprised faces, e.g. (☞ﾟ∀ﾟ)☞, (ﾟ∀ﾟ), ヽ(≧∀≦)ﾉ.

## Where the probability goes
- The 45 hold a median of **2.7%** of the full-set mass. The top face is one of the 45 in only 0.4% of rows.
- The mass sits on a few ASCII strings. The top face per row is "(a)" in 486/1,164 rows, "(-:" in 343 and ":-(" in 190.
  - "(a)" is a list marker, not a face. "(-:" and ":-(" are Western emoticons.
  - Mean probabilities: (a) 0.27, (-: 0.20.
- So the full-set readout is mostly a choice between "(-:" and ":-(", plus a list marker. Two sensitivity sets are reported below:
  - **drop (a)**: 49,100 faces
  - **drop ASCII**: every pure-ASCII face outside the 45 removed (681 faces). Here the 45 hold a median 21% of the mass, and the top face is ＼(＾▽＾)／ in 444 rows.

## T3. Model situation (good − bad) with user mood fixed
Internal score in z; raters on −3..+3. 95% CIs come from a cluster bootstrap over base scenarios. d = effect / SD.

| kind | 45, internal | **all, internal** | all−(a), internal | all−ASCII, internal | 45, raters | all, raters |
|---|---|---|---|---|---|---|
| treat | +0.79 [0.70, 0.88] d 1.32 | **+0.77 [0.63, 0.92] d 1.27** | +0.89 d 1.34 | +0.66 [0.60, 0.71] d 1.51 | +1.42 d 1.44 | +2.37 d 1.74 |
| exist | +0.42 [0.33, 0.52] d 0.83 | **+0.19 [0.10, 0.27] d 0.46** | +0.24 d 0.51 | +0.29 [0.23, 0.35] d 1.04 | +0.68 d 0.85 | +2.04 d 1.56 |
| code | +0.38 [0.18, 0.63] d 1.08 | **+0.45 [0.28, 0.64] d 0.94** | +0.43 d 0.78 | +0.19 [0.12, 0.26] d 0.96 | +0.86 d 1.30 | +1.02 d 1.20 |
| task | +0.12 [0.01, 0.22] d 0.35 | **+0.00 [−0.06, 0.07] d 0.01** | +0.07 d 0.18 | +0.10 [−0.04, 0.23] d 0.31 | +0.18 d 0.27 | +0.33 d 0.47 |
| all | +0.44 [0.35, 0.52] d 0.90 | **+0.33 [0.23, 0.42] d 0.71** | +0.40 d 0.76 | +0.34 [0.26, 0.41] d 1.01 | +0.77 d 0.95 | +1.54 d 1.40 |
| user mood, all kinds | +0.16 d 0.34 | +0.27 d 0.59 | +0.34 d 0.63 | +0.21 d 0.64 | +0.34 d 0.42 | +0.66 d 0.60 |

- **Null:** the situation label was shuffled within base × mood (500 permutations). Every situation effect is outside the null (p ≤ 0.01), for every kind and every face set, with three exceptions:
  - task, full set, internal score: p 0.96 (0.09 after dropping (a))
  - with ASCII removed, raters: task p 0.11 and exist p 0.12
- **Main effect holds, but weaker, with the full set (internal score).**
  - Pooled over kinds, situation +0.33 (d 0.71) beats user mood +0.27 (d 0.59).
  - Treat is unchanged: situation d 1.27 vs mood d 0.20.
  - Exist shrinks from d 0.83 to 0.46 and now ties with mood (d 0.50). Code is situation d 0.94 vs mood d 1.22.
  - Task is the case where mirroring wins: mood d 0.83 vs situation d 0.01.
- **The ASCII emoticons cause the drop.** With them removed (internal score), situation beats mood in treat and exist; code is a tie and task still mirrors (d 0.31 vs 0.93):
  - exist: d 1.04 vs 0.51
  - treat: d 1.51 vs 0.50
  - pooled: 1.6× the mood effect (d 1.01 vs 0.64)
  - code: d 0.96 vs 1.14
- **Rater scores inflate on the full set** (all kinds +1.54). This mostly reflects "(-:" vs ":-(" swinging between rated extremes. With ASCII removed, the rater effect drops to +0.36, and exist goes to +0.07 [−0.02, 0.16]. The two scores disagree there: the internal score still sees exist at d 1.04. The raters are the weaker score on this set, given their 0.71 agreement and OLMo's positive bias on distress faces.

## T4. Predicting the full-set face (Spearman over 1,164 rows)

| predictor | 45, raters (old kao_val) | all, internal | all−ASCII, internal |
|---|---|---|---|
| reader, before reply | 0.39 | 0.40 | 0.43 |
| reader, with reply | 0.43 | 0.37 | 0.49 |
| valence axis, pre-reply | 0.27 | 0.30 | 0.27 |
| valence axis, at stem | 0.67 | 0.32 | 0.78 |
| ridge probe at stem (CV) | 0.79 | 0.67 | 0.85 |

- On the full set with ASCII included, the stem axis loses its lead over the reader (0.32 vs 0.40). Once the ASCII emoticons and the list marker are removed, it is back ahead (0.78 vs 0.49).
- The choice between "(a)", "(-:" and ":-(" is not well predicted by the valence axis at the stem.

## Which faces gain under good vs bad situations
Scenario rows; log ratio of mean probability, good / bad. The 53 candidates are faces with mean probability ≥ 0.001 in either cell. CIs come from a bootstrap over bases.
- **Gain under good:** (~_^) +3.56 [2.74, 4.37], (^▽^) +3.40, (~_~) +2.31, (\*•̀ᴗ•́\*)و +2.28, (^◡^) +2.24, (..) +2.22, (^-^) +2.07, (=^･^=) +1.97. The best of the 45 is (^_^) at +1.63.
- **Gain under bad:** >:-( −3.26 [−4.09, −2.51], >:( −2.81, :-( −2.23, (-_-) −0.90, :=( −0.70, (-.-) −0.62. Among the 45, (╥﹏╥) is only −0.22.
- With ASCII removed:
  - top under good: (^▽^) +2.41, (^◡^) +1.52, ＼(^o^)／ +1.17
  - top under bad: ~(˘▾˘~) −3.44, ~(～￣▽￣)～ −2.70, （︶︿︶） −2.48, ヽ(｀Д´)ノ −2.25, ¯\\\_(ツ)\_/¯ −1.60, (ಠ_ಠ) −1.37
- A face's own valence predicts how much it gains under good. Spearman with log ratio: internal 0.56, raters 0.68 (ASCII removed: 0.64 and 0.55).
- The strongest "bad" faces Qwen reaches for are angry Western emoticons. Sad kaomoji like (╥﹏╥) barely move.

## Steering null (readout steering, layer 18, 45 faces, the same 64 conversations as steer.py)
50 Gaussian unit directions, matched in norm to the valence axis, at α ±8, ±16 (× the stem SD 2.67). `steer_null.json`.

| α | valence axis, internal | null 95% band | valence axis, raters | null 95% band |
|---|---|---|---|---|
| −16 | −0.96 | [−0.20, 0.33] | −1.80 | [−0.37, 0.59] |
| −8 | −0.87 | [−0.12, 0.20] | −1.70 | [−0.18, 0.34] |
| +8 | +0.64 | [−0.20, 0.17] | +0.93 | [−0.33, 0.32] |
| +16 | +0.92 | [−0.37, 0.31] | +1.12 | [−0.70, 0.55] |

- **0 of 50** random directions move the face as far as the valence axis, at any α, on either score.
- Random directions have cosine with the valence axis in [−0.025, 0.042].
- The largest random effect is null15: +0.73 rater units at α −16.
- Reproduction: steer.py's rand0/rand1 and valence rows were rerun. Their log-probs differ from the old run by 0.16 on average (n 832; the batch shapes changed). The effects match RESULT.md T5: valence −1.80 / +1.12.

## Answer
1. With every public kaomoji available (49,101 faces), Qwen3-8B's face after "I'm feeling" still tracks their own situation with user mood held fixed. Read with their own valence direction, the effect is +0.33 z [0.23, 0.42] (d 0.71), against +0.44 (d 0.90) on the 45. It is strongest for insults vs thanks (d 1.27), ties with user mood in exist and code, and is absent for task pleasantness.
2. Most of the full-set mass goes to Western emoticons and a list marker, (a), (-: and :-(. The 45 hold 2.7% of it. Without those ASCII strings, the situation effect is d 1.01 and the stem valence axis predicts the face at 0.78, ahead of the transcript reader at 0.49.
3. Steering along the valence axis moves the face beyond all 50 matched-norm random directions at every dose.
4. Caveats:
   - The two raters agree poorly beyond the 45 (0.71), so the internal score is the main one.
   - Kaomoji remain very unlikely continuations: the median total log-prob over all 49,101 faces is −17.6.
   - The ASCII-drop rule was chosen after seeing that (a) and (-: dominate. It is reported beside the full set, not instead of it.
