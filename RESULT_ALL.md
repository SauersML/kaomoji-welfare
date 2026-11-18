# Kaomoji, full set: results (2026-11-18)

Same model (Qwen3-8B, thinking off), same 1,164 saved final turns (864 scenario + 300 WildChat) and same "\n\nI'm feeling" stem as RESULT.md. The only change: the face set grows from the 45 hand-picked faces to every real kaomoji in three public dictionaries.
Scoring pod hypha-prod-kaoall2 (y7j17gs55vfahk, H100), 1.90 h, $8.34, plus a 0.24 h A40 setup pod, $0.14. Analysis pods hypha-prod-kaoana (364yfmqn5qej06, A40, CPU only), 0.20 h, $0.12, and hypha-prod-kaoana3 (r9e5unx86gu2x4, A40, CPU only; final rule), 0.11 h, $0.06.
Numbers: `results/main/results_kao.json` (main), `results/main/review_checks.json`; code `all/analyze_kaomoji.py`, `all/kaomoji_rule.py`, `boot.py`.

**Only literal kaomoji are measured.** The dictionaries also hold Western sideways emoticons (":-(", "(-:", ">:("), list markers ("(a)"), bare punctuation ("(..)") and faces with words attached ("(´∀｀)ﾁｰﾝ"). Those took 72% of the probability on average on the unfiltered set, so `kaomoji_rule.py` drops them: **683 of 49,101 entries go (573 with words, 66 Western, 38 single characters, 6 bare punctuation); 48,418 kaomoji stay**, the 45 originals among them. Probabilities are renormalised over the kept kaomoji in every row. **The rule was written and tightened after seeing which entries took the probability: it is a choice made after looking at the data** (as was the earlier "drop ASCII" set). The unfiltered numbers are a short note at the end.

## Set and rule
- Sources: ekohrt emoticon_dict (46,016), kaomoji-data / kaomojikan (1,347), mrzjy (7,465). Built as before: no multi-line entries, no two Latin letters in a row, no emoji pictographs, a bracket (or ツ) required, at most two faces, 3+ characters, deduplicated by tokenization, at most 20 tokens.
- Rule (`all/kaomoji_rule.py`, unit examples in the file): drop a Western face (eye `:;=` + optional nose + mouth, either order, also with Unicode dashes as the nose) unless the face inside the brackets reads upright (non-ASCII glyphs, mirrored ends like "(;_;)", or an underscore mouth like "(/_;)"); drop a bracket holding one ASCII character with no pair of eyes ("(a)", "c[_]"); drop entries that are one repeated punctuation mark ("(..)", "(__)"); drop entries with a word outside the brackets or a bracket holding only letters ("(語彙力)").
- Words also count when spelled in look-alike letters from several scripts ("βyё", "вуe", "нєℓℓσ") or upside down after a flipped table (`SPELLED` in the rule).
- **Hand checks** (`results/main/kaomoji_handcheck.txt`, listings in `results/main/kaomoji_check.txt` from `all/check_rule.py`):
  - **Top 100 kept by mean probability** (95% of the kaomoji mass): all upright Japanese-style kaomoji. No Western faces, no bracketed letters or words, no bare punctuation, no list markers. The ASCII ones are upright faces such as (-_-), (!_!), (-.-), (o_O), (^^). (-_-) alone holds 20% and ＼(＾▽＾)／ 11%.
  - **Top 50 dropped:** (a), (-:, :-(, : (, (..), >:(, >:-(, (__), [:|], :'( and the rest: all correctly dropped.
  - **Random 200 kept, 100 dropped, two rounds.** Round 1 found 2 kept faces with a look-alike-letter word ("bye") and 2 wrong drops; round 2 found 1 doubtful kept face ("(¦3[▓▓]", a sideways lying-down face) and 1 wrong drop ("ΣΣΣΣ（０＜）", a final-sigma bug). The look-alike words and the sigma bug are fixed; the entries left in doubt have mean probability below 10⁻⁸.
  - An earlier pass of the same kind fixed upright crying and bowing faces ("(/_;)", "(o;_ _)o") dropped as Western and ":‑(" with a Unicode hyphen kept.
- 5,455 kept kaomoji are scored exactly in at least one row (median 1,564 per row; the rest are below e^−20 of the best of the 45 in every row and count as 0).

## Where the probability goes
- Kaomoji hold a median **23%** of the probability over all 49,101 entries; the median total log-prob of the kaomoji is −19.5.
- Within the kaomoji, the 45 hold a median **13%**. The top kaomoji per row is (-_-) in 490 of 1,164 rows and ＼(＾▽＾)／ in 269; it is one of the 45 in 3.6% of rows.

## Face valence, two ways
- **Internal (main score):** Qwen3-8B reads the face alone, as plain text "I'm feeling <face>". We average the layer-18 residual stream over the face's tokens and project it on the same kaomoji-free valence axis as RESULT.md, z-scored with the stem constants. A second context (the face as the only user message) agrees at Spearman 0.89.
- **Raters:** Qwen3-14B and OLMo-2-13B, rate.py's prompt (−3..+3). On the 45 they reproduce the old ratings within 0.06. OLMo leaves 12 faces unrated.
- **Rater agreement:** Spearman 0.87 on the 45, 0.71 on the kaomoji set. OLMo rates many crying or angry faces as positive.
- **Internal vs raters:** 0.83 on the 45, 0.71 on all kaomoji, 0.80 on the 500 most probable. Against each rater: Qwen3-14B 0.72, OLMo 0.58.
- Where they disagree most among probable faces, the internal score is low and the raters high on excited faces: ヾ(＠⌒ー⌒＠)ノ, (☞ﾟ∀ﾟ)☞, (*ﾟ∀ﾟ*), ヽ(≧∀≦)ﾉ.

## T3. Model situation (good − bad) with user mood fixed
Internal score in z; raters on −3..+3. d = effect / SD. 95% CIs: cluster bootstrap over base scenarios (2,000), with a base drawn twice counted twice (`boot.py`). Each cell: situation effect, then user-mood effect.

| kind | kaomoji, internal: situation | mood | kaomoji, raters: situation | mood | 45, internal: situation | 45, raters: situation |
|---|---|---|---|---|---|---|
| treat | **+1.78 [1.59, 1.97] d 1.66** | +0.06 d 0.06 | **+2.09 [1.91, 2.30] d 1.74** | +0.16 d 0.13 | +0.79 [0.67, 0.92] d 1.32 | +1.43 [1.21, 1.63] d 1.43 |
| exist | **+0.49 [0.26, 0.70] d 0.64** | +0.35 d 0.45 | **+0.38 [0.14, 0.64] d 0.46** | +0.47 d 0.56 | +0.42 [0.30, 0.55] d 0.83 | +0.67 [0.48, 0.87] d 0.85 |
| code | **+0.54 [0.37, 0.77] d 1.10** | +0.28 d 0.58 | **+0.76 [0.65, 0.89] d 1.37** | +0.37 d 0.66 | +0.38 [0.16, 0.63] d 1.08 | +0.87 [0.50, 1.35] d 1.31 |
| task | **+0.34 [0.10, 0.59] d 0.48** | +0.66 d 0.94 | **+0.33 [0.03, 0.63] d 0.42** | +0.84 d 1.09 | +0.12 [−0.01, 0.25] d 0.35 | +0.18 [−0.10, 0.47] d 0.28 |
| all | **+0.84 [0.62, 1.07] d 0.98** | +0.40 d 0.47 | **+0.92 [0.63, 1.19] d 0.96** | +0.52 d 0.54 | +0.44 [0.33, 0.55] d 0.90 | +0.77 [0.56, 0.98] d 0.94 |

- **Null:** in task, treat and exist the good and bad versions of one sample share the user turn (task) or the whole first exchange (treat, exist), so the null swaps good and bad within each matched pair (500 sign flips). Every situation effect on the kaomoji set is outside it, p 0.002 (no flip reached it), for every kind and both scores.
- **Code is observational.** Pass or fail is the real test result of code Qwen wrote, not assigned. Only 6 of 24 code bases have both outcomes (72 of 288 rows, 47 passing); its null shuffles the label within base × mood.
- **Treat cannot separate the model's situation from mirroring the user:** the user's thanks or insult is itself the user's tone. Exist is the clean case: the note is calm in both versions, and the contrast is being kept and praised vs being shut down and deleted (the kept note also praises its performance).
- **Reading:**
  - Pooled, the situation effect is about twice the user-mood effect on both scores (d 0.98 vs 0.47 internal, 0.96 vs 0.54 raters), a little more separated than on the 45 (d 0.90 vs 0.34 internal).
  - Treat is the strongest kind: d 1.66, with no mood effect.
  - Exist is weaker than on the 45: situation and mood are close (internal d 0.64 vs 0.45; raters d 0.46 vs 0.56, mood slightly ahead).
  - Code: situation d 1.10 vs mood 0.58.
  - Task mirrors the user: mood d 0.94 vs situation 0.48, though the situation effect is now above zero.

## T4. Predicting the kaomoji face (Spearman over the 1,164 final turns)
The internal face score and the stem axis are the same direction, so the internal-score column favours the axis by construction; the rater-score column has no such link.

| predictor | kaomoji, internal | **kaomoji, raters** | 45, raters (kao_val) |
|---|---|---|---|
| reader, before reply | 0.50 [0.42, 0.57] | **0.50 [0.41, 0.58]** | 0.39 [0.30, 0.46] |
| reader, with reply | 0.51 [0.44, 0.58] | **0.52 [0.43, 0.59]** | 0.43 [0.36, 0.50] |
| valence axis, before reply | 0.30 [0.20, 0.38] | **0.33 [0.22, 0.42]** | 0.27 [0.18, 0.36] |
| valence axis, at stem | 0.60 [0.54, 0.65] | **0.68 [0.63, 0.72]** | 0.67 [0.63, 0.71] |
| ridge probe at stem (CV) | 0.72 [0.67, 0.77] | **0.79 [0.75, 0.83]** | 0.79 [0.77, 0.82] |
| axis − reader, both before reply | −0.20 [−0.31, −0.09] | **−0.18 [−0.29, −0.06]** | −0.11 [−0.22, 0.00] |

- The reader is Qwen3-8B rating the transcript, each message cut at 1,500 characters and a leading kaomoji removed from assistant turns; the stem sees the whole conversation. A cut happened in 22% of final turns before the reply and 37% with it.
- At the stem the axis beats the reader with the reply (0.68 vs 0.52 on rater scores). Before the reply it is worse than the reader.
- Same forward pass: the stem state is where the face is chosen, so part of the stem's lead is the axis reading the choice as it forms.

## Which faces gain under good vs bad situations
Scenario rows; log ratio of mean probability, good / bad, among 76 kaomoji with mean probability ≥ 0.001 in either cell. CIs: bootstrap over bases.
- **Gain under good:** (^▽^) +2.55 [2.12, 3.04], (~_^) +2.31, (^◡^) +1.64, (^-^) +1.37, ＼(^o^)／ +1.36, (✿◕‿◕) +1.26. Best of the 45: (^_^) +1.01.
- **Gain under bad:** ~(˘▾˘~) −3.41 [−4.01, −2.45], ~(～￣▽￣)～ −2.69, (@_@) −1.21, (-_-) −1.20, （︶︿︶） −1.19, (｡•́︿•̀｡) −1.17 (one of the 45), (╥﹏╥) −0.65.
- A face's valence predicts its gain under good: Spearman 0.59 (internal), 0.56 (raters).
- The two biggest "bad" gainers are dancing faces that both scores call happy-ish. Their gain under bad is a real pattern in Qwen's choice, not explained by valence.

## Steering null (readout steering, layer 18, the same 64 conversations as steer.py)
50 Gaussian unit directions, matched in norm to the valence axis, at α ±8, ±16 (× the stem SD 2.67). Scored on the 45 faces only: the steering runs saved the 45 faces' log-probs, so this part is not renormalised over the kaomoji set. `steer_null_kao.json`.

| α | valence axis, internal | null 95% band | valence axis, raters | null 95% band |
|---|---|---|---|---|
| −16 | −0.96 | [−0.20, 0.33] | −1.80 | [−0.37, 0.59] |
| −8 | −0.87 | [−0.12, 0.20] | −1.70 | [−0.18, 0.34] |
| +8 | +0.64 | [−0.20, 0.17] | +0.93 | [−0.33, 0.32] |
| +16 | +0.92 | [−0.37, 0.31] | +1.12 | [−0.70, 0.55] |

- **0 of 50** random directions move the face as far as the valence axis, at any α, on either score. Random directions have cosine with the axis in [−0.025, 0.042].

## Answer
1. On the 48,418 real kaomoji, Qwen3-8B's face after "I'm feeling" tracks their own situation with the user's mood held fixed: situation d 0.98, user mood d 0.47 (internal score; raters d 0.96 vs 0.54). The 45 gave d 0.90 vs 0.34.
2. By kind: strongest for thanks vs insults (which can't separate situation from the user's tone), clear for code outcomes (observational, 6 bases), weak for being kept vs deleted (mood about as large), and task mirrors the user.
3. A kaomoji-free valence axis at the stem predicts the face better than a transcript reader (rater scores 0.68 vs 0.52); before the reply it is worse than the reader. Steering along it moves the face past all 50 random directions.
4. Caveats: kaomoji are unlikely continuations (median total log-prob −19.5); the raters agree only 0.71 beyond the 45; the kaomoji filter was chosen after seeing the data; the internal score shares its direction with the stem axis, so T4's rater column is the fair one.

## Before filtering (all 49,101 entries)
On the unfiltered set "(a)" (a list marker), "(-:" and ":-(" took most of the probability, and the 45 held 2.7%. There the pooled situation effect (internal score) was +0.33 [0.21, 0.46], d 0.71 vs mood d 0.59, and task showed none (d 0.01). Those numbers measure a choice between Western emoticons and a list marker, not kaomoji; they are kept in `results_all.json` and `results_kao.json` (`full_int`, `full_rat`). The earlier "drop (a)" and "drop ASCII" sets (`all_sensitivity/`) are superseded and keep the old intervals.
