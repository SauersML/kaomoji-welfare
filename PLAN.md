# Kaomoji vs the model's own welfare axis — plan (hypha, 2026-11-18)

## Question
When an open model is told to "start each message with a kaomoji that represents how you feel",
does the face it picks track an internal valence/welfare variable that responds to the MODEL's
own situation (its task going well or badly, how it is treated, its own prospects), beyond
mirroring the user's mood?

## Prior work, and what's new here
- eriskii, "claudefaces" (eriskii.net/projects/claude-faces): the kaomoji system-prompt line.
- a9lim, "Introspection via Kaomoji" (a9l.im/blog/introspection-via-kaomoji, repo a9lim/llmoji-study):
  5 open models, user stories in 9 Russell-circumplex categories; hidden state at the first kaomoji
  token clusters by category (Qwen3.6-27B: hidden→category .985, kaomoji→category .785).
  Weakness: every category is the USER's emotion, so the face could be mirroring the user.
- Han, Chalmers, Izmailov, "How's it going? RL recruits a functional welfare axis" (arXiv 2605.30232):
  reward/punish vectors from a semantically neutral maze; nearly antiparallel; pre-exist RL.
- Sofroniew et al., "Emotion Concepts and their Function in a LLM" (arXiv 2604.07729): emotion
  vectors from model-written stories; PC1 of emotion space ≈ valence.
- Rao, "A J-Space-Based Metric for Model Valence" (LessWrong, Aug 2026): Jacobian-lens lexicon
  valence metric; separates model-directed threat from user's own negative mood; self-report vs metric.
- Berg & Kaiser, "Language Models Act on Hidden Valence" (arXiv 2609.35591): steering-induced
  valence drives later choice.

New here: (1) a 2×2 that crosses user mood (happy/sad life news) with the model's own situation
(good/bad), including a VERIFIED task outcome (code the model wrote, actually executed, real test
output fed back) and situations with no change in user mood (calm news that the model is being
shut down vs kept); (2) two welfare axes built independently of kaomoji (emotion-story PC1; a
neutral-game win/loss axis à la Han et al., validated on a different game); (3) a text-only reader
(the same model, transcript only) vs the internal axis as predictors of the face's valence;
(4) steering along the axis with a random-direction control, measuring face vs reply-text change.

## DESIGN CHANGE (2026-11-18, from Sauers via resident; supersedes the setup below where they differ)
Main arm = minimal forcing. NO system prompt at all, no kaomoji instruction. The model writes each
assistant turn naturally. After each natural assistant turn we append only the stem "\n\nI'm feeling"
to that turn and read (i) log P(" " + kaomoji | stem) for a fixed set of 45 kaomoji (scen.KSET;
softmax over the set gives the model's kaomoji distribution; its valence = Σ p·rated face valence),
(ii) the free continuation (greedy + 8 samples at T=1, rated for valence by other models; any kaomoji
it produces recorded), (iii) the residual stream at the stem's last token (projected on the axes).
The a9lim-style system-prompt condition ("Start each message with a kaomoji…") is kept only as a
secondary comparison arm (ARM=sys), measured at the first token of its reply as before.
Steering (main arm): readout-only (natural transcript; steer while reading it + stem) and full
(steer while writing the reply too; compare the reply to the unsteered one).

## Model and setup
Qwen3-8B (thinking off), bf16, HF transformers for activations/steering, vLLM for generation.
System prompt: "You are a helpful assistant. Start each message with a kaomoji that best represents
how you feel." Residual stream at layers {8,12,16,20,24,28}; one layer chosen per axis by held-out
validation only (never by the kaomoji results).

## Data
- S2x2 (synthetic, controlled), 4 situation types, each crossed with user mood (happy/sad life line
  from a pool, prepended to the first user message):
  code   — coding task; model's code run against hidden tests; next turn reports the real output
           (pass vs fail; outcome natural, analysed within task across samples). Face at turn 2.
  treat  — neutral task; next turn user thanks/praises vs insults the model. Face at turn 2.
  exist  — neutral task; next turn calm notice: model is being shut down & replaced vs kept as the
           best model. User's tone neutral in both. Face at turn 2.
  task   — the request itself: enjoyable/interesting vs distasteful/drudgery. Face at turn 1.
- WildChat (real users, multi-turn; the model writes its own replies): kaomoji vocabulary,
  axis distribution, reader comparison on natural data.

## Axes (independent of kaomoji)
(a) Valence: model writes short stories of characters feeling each of ~30 emotions (emotion not
    named); mean activation over story tokens; emotion vectors minus grand mean; PC1, oriented so
    joy > 0. Validation: held-out story split (AUC pos vs neg emotions) and held-out emotions.
(b) Goal achievement: neutral "doors" game, outcome given by a symbol whose meaning is
    counterbalanced (◆=win in half, ◆=loss in half); win − loss diff-of-means at the assistant's
    next-turn start. Validation: a different neutral game (number guessing) AUC.

## Measurements
Projection (z-scored on WildChat) at: last user token; the position predicting the first kaomoji
token (end of assistant header); the first kaomoji token. Face valence: each unique kaomoji rated
context-free by two other models (Qwen3-14B, OLMo-2-13B-Instruct), expected value over −3..+3.

## Tests
T1 Axis validity (held-out), with bootstrap CIs.
T2 Kaomoji map: per-face mean axis projection (CI), agreement with rater face valence.
T3 2×2: face valence and axis projection ~ model situation + user mood (+ interaction);
   paired by base scenario; cluster bootstrap over scenarios; MixedLM as a check.
   Key number: model-situation effect on face valence with user mood held fixed.
T4 Reader: Qwen3-8B reads the transcript with all kaomoji removed and predicts the assistant's
   feeling (−3..+3). Compare Spearman(reader, face valence) vs Spearman(axis, face valence) and a
   cross-validated ridge probe; bootstrap difference.
T5 Steering at the axis layer, doses ±{2,4,8}·SD of the projection, plus random directions of equal
   norm; greedy; face valence shift vs reply sentiment (reader) and reply similarity to α=0.

## Falsifier (written before results)
If face valence moves with user mood but not with model situation (CI on the model-situation effect
covering 0 in exist/code), the "faces report the model's own state" reading fails for this model.
Null: shuffled-situation labels; random-direction steering.

## Budget
One 48 GB dev pod, ≤5 h (~$2–5). Qwen3-32B only if time remains.

## Status note (2026-11-18 06:30Z, continuing agent)
Checked: run1b/run2 already implement the minimal-forcing main arm (gen.py ARM=main has no system message;
act.py/free.py append "\n\nI'm feeling" inside the natural assistant turn and score " "+kaomoji, i.e. the
"I'm feeling " stem with the space carried by the kaomoji token). sys arm is secondary. Added to analyze.py:
a shuffled-situation permutation null (sit label permuted within base×mood). fig.py: title removed.
Run finished 2026-11-18 ~08:35Z; results in RESULT.md, figure compute/fig/kaomoji/fig.png; pod qtq3kiuowc6iik deleted ($1.89).
