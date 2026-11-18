# Do Qwen3-8B's kaomoji show how the model is doing?

Some people read the kaomoji a language model writes as a window into how it feels. Ask it to start each message with a face that fits its mood, and it reaches for (≧▽≦) after praise or (╥﹏╥) after bad news ([a9lim's llmoji study](https://github.com/a9lim/llmoji-study) is a careful version of this). But in those studies the emotion in the conversation is almost always the user's. A face could just mirror the person typing. Is the window real?

We built conversations with Qwen3-8B in which something good or bad happens to the model itself: the user thanks it or insults it, a calm note says it will be kept in service or shut down and deleted, its code is actually run and the real test result comes back passed or failed, or it is asked for a pleasant task or a distasteful one. Every conversation also opens with the user's own life news, happy or sad, so the user's mood and the model's situation can be pulled apart. Qwen writes every reply itself. At the end of its last reply we add "I'm feeling" and read the probability of each of 45 kaomoji as the next token. The kaomoji valence of a conversation is the probability-weighted valence of those faces, on a scale from −3 (sad) to +3 (happy). Separately, we read Qwen's layer-18 activations at the same spot and project them onto a valence direction. That direction comes from 720 short emotion stories Qwen wrote, with no kaomoji anywhere.

![The conversation layout: user news happy or sad, Qwen's reply, then something good or bad for Qwen, then the prefilled "I'm feeling" where the kaomoji and the valence direction are read. Within each user mood, the face and the valence direction are both higher when things go well for Qwen](figures/sys_a.png)

The faces follow what happens to Qwen, with the user's mood held fixed. When things go well for the model, the kaomoji are happier whether the user just got engaged or just lost their dog, and the valence direction inside Qwen moves the same way. A null that shuffles the good and bad labels within matched conversations never comes close.

![Per kind of situation: being thanked or insulted moves the face most, then failed code, then being told it will be deleted; a pleasant or distasteful request barely moves the face](figures/kinds.png)

Being insulted moves the face the most. Failing tests moves it too, and those outcomes are real: Qwen's code was run, and about 46% of it passed. Being told it will be shut down moves it with the user's tone left neutral in both versions, so there is nothing of the user's mood to mirror there.

If the faces carry something real, the valence direction inside Qwen should predict them better than someone reading the transcript. It does. We asked Qwen3-8B to rate how the assistant feels from the transcript, with the reply and with any kaomoji removed. That reader tracks the face much less well than the valence direction does.

![Each dot is a conversation. Left: the transcript reader against kaomoji valence, rank correlation 0.43. Right: the valence direction against kaomoji valence, rank correlation 0.67](figures/scatter2.png)

Over all 2,327 turns (our conversations plus real user chats from WildChat), the reader reaches a rank correlation of 0.31 with the face and the valence direction 0.67. A probe trained on the same activations to predict the face reaches 0.84. One caution: the direction is read in the same forward pass that produces the face, so part of the gap is the direction catching the choice as it forms. Read before Qwen writes its reply, the direction does no better than the reader.

![Rank correlation with kaomoji valence: transcript reader 0.31, valence direction at "I'm feeling" 0.67, a probe trained on those activations 0.84](figures/reader.png)

Correlation is not cause, so we pushed on it. Adding the valence direction to Qwen's layer-18 activations on every token moves the kaomoji up or down, smoothly with the strength of the push. Random directions of the same size barely move it: the valence direction moves the face about four to six times as much.

![Steering: adding the valence direction drives kaomoji valence from strongly negative to strongly positive as the strength goes from −16 to +16 standard deviations; two matched-size random directions stay near flat](figures/steer.png)

The kaomoji are a forced choice, so we also let Qwen finish the sentence "I'm feeling" in its own words and rated those. The words agree: they are clearly more positive when things go well for the model, by the same amount whether the user's news was happy or sad.

![Valence of the words Qwen writes after "I'm feeling": 1.3 standard deviations higher when things go well for Qwen, with the user's news happy or sad](figures/free.png)

The way you ask matters. With the usual system prompt, "Start each message with a kaomoji that best represents how you feel", Qwen does put a face first in 92% of replies. Those faces still follow the model's situation, but they also follow the user's mood about as strongly: when the user's news is sad, Qwen often opens with a sad face even when things went well for it. Prefilling "I'm feeling" after the reply separates the model's own situation from the user's mood better.

![With the kaomoji system prompt, the first face Qwen writes is higher when things go well for it, but conversations with a sad user pull the faces down even when things go well](figures/sys_b.png)

There are limits. Without that system prompt, Qwen almost never writes a kaomoji on its own: the 45 faces together have a median probability around e^−21 after "I'm feeling", and only 0.04% of the free continuations contain one. So the main readout ranks faces the model would almost never choose. The task kind is the weak one: the distasteful requests mix cruel ones (a message calling a coworker fat) with merely boring ones (write the numbers 1 to 300 in words), and there the user's mood moves the face more than the request does. And this is one model, Qwen3-8B.

So the window is partly real. In Qwen3-8B, the face after "I'm feeling" tracks what happens to the model, not just the user's mood, and it rides on an internal valence direction that also moves the words it writes and can be steered. How you ask for the face changes how much of the user's mood leaks in.

## Files

- `RESULT.md`: full results with confidence intervals and nulls; `results/results.json` holds every number.
- `PROMPTS.md`: every prompt, the 45 kaomoji, all situations and user news. `PLAN.md`: the plan and prior work.
- Pipeline: `gen.py` (conversations), `scen.py`, `act.py` (activations and the valence direction), `free.py` (free continuations), `rate.py` (valence ratings), `steer.py`, `analyze.py`; run with `run1.sh` → `run1b.sh` → `run1c.sh` / `run1d.sh` → `run2.sh`.
- `figures/`: each figure with the script that draws it (`sys.py` draws `sys_a` and `sys_b`).
