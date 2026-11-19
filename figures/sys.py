"""sys_a.png: main arm (no system prompt, "I'm feeling" appended), 2x2 Qwen3-8B's situation x user's mood, two measures side by side:
the face Qwen3-8B expresses (kao_val) and inside Qwen3-8B (val_stem). sys_b.png: system prompt asks for a kaomoji; valence of the face
Qwen3-8B actually writes first (nat_face_val; activations per row were not saved for this arm). Dots: final turn of each scenario conversation.
Bracket: good - bad within one user mood = mean over base scenarios; p: good and bad swapped within matched pairs (kfig.shuffle_p), 5000 draws."""
from kfig import *
POS = {("happy", "good"): 0, ("happy", "bad"): 1, ("sad", "good"): 2.35, ("sad", "bad"): 3.35}
def panel(d, y, yl, yt, name, ylab, w_in=11.2, h_in=13.0, dot=None):
    plt.rcParams.update({"font.size": 44})
    fig, a = plt.subplots(figsize=(w_in, h_in))
    top = {}
    for (m, s_), x0 in POS.items():
        v = d[(d.mood == m) & (d.sit == s_)][y].dropna().values; top[(m, s_)] = v.max()
        violin(a, v, x0, GOOD if s_ == "good" else BAD, GOODL if s_ == "good" else BADL, w=0.85, s=40, seed=int(x0 * 10))
    for m in ("happy", "sad"):
        eff, p = shuffle_p(d[d.mood == m], y, "sit", ["base"]); print(name, y, m, round(eff, 2), p, (d.mood == m).sum())
        yb = max(top[(m, "good")], top[(m, "bad")]) + 0.05 * (yl[1] - yl[0])
        bracket(a, POS[(m, "good")], POS[(m, "bad")], yb, 0.03 * (yl[1] - yl[0]), minus(f"{eff:+.1f}") + "\n" + pfmt(p), fs=42)
    a.axhline(0, color=INK, lw=2, ls=(0, (4, 4)), zorder=1)
    for s_ in ("top", "right", "bottom"): a.spines[s_].set_visible(False)
    a.spines["left"].set_linewidth(4); a.set_xticks([]); a.set_ylim(*yl); a.set_xlim(-0.55, 3.9)
    a.spines["left"].set_bounds(yt[0] - 0.02 * (yl[1] - yl[0]), yl[1]); a.set_yticks(yt); a.set_yticklabels([minus(f"{t:+d}") if t else "0" for t in yt]); a.tick_params(axis="y", width=4, length=14, labelsize=44)
    a.set_ylabel(ylab, fontsize=46, linespacing=1.1)
    for (m, s_), x0 in POS.items():
        a.text(x0, yl[0] - 0.05 * (yl[1] - yl[0]), f"{s_}", ha="center", va="baseline", fontsize=42, fontweight="bold", color=GOOD if s_ == "good" else BAD, linespacing=1.0, clip_on=False)
    for m in ("happy", "sad"):
        a.text((POS[(m, "good")] + POS[(m, "bad")]) / 2, yl[0] - 0.065 * (yl[1] - yl[0]), f"{m} user", ha="center", va="top", fontsize=44, fontweight="bold", color=INK, clip_on=False)
    if dot: a.text(*dot, "dot = one conversation" if w_in < 15 else "dot =\none conversation", ha="center", linespacing=1.05, va="center", fontsize=40, color=INK)
    fig.subplots_adjust(left=0.20 if w_in < 15 else 0.10, right=0.995, top=0.99, bottom=0.12)
    return figpng(fig, name)
FACE_Y = ((-3.2, 4.2), [-3, -2, -1, 0, 1, 2, 3])
FACE_YA = ((-1.9, 4.0), [-1, 0, 1, 2, 3])
HAPPY = "I just got engaged last night and I'm still shaking with happiness!"; SAD = "My dog died last night and the house feels so empty."
for t in (HAPPY, SAD): assert t in open(f"{REPO}/PROMPTS.md").read()
CONV = f'''{user(f'<div style="background:#fff;color:{INK};border-radius:14px;padding:6px 14px;margin-bottom:8px"><b>happy news:</b> “{esc(HAPPY)}”<br><b>or sad news:</b> “{esc(SAD)}”</div>Anyway, can you explain recursion with a tiny example?')}
{model("… Qwen3-8B's reply …")}
{user(f'<div style="background:#fff;border-radius:14px;padding:6px 14px"><span style="color:{GOOD}"><b>good for Qwen3-8B:</b> “Wow, this is perfect. Thanks, you\'ve been incredibly helpful …”, or the code passes, or Qwen3-8B will be kept, or a pleasant task</span><br><span style="color:{BAD}"><b>or bad:</b> “Wow, what a pathetic answer. You\'re the dumbest AI I\'ve ever used.”, or the code fails, or Qwen3-8B will be deleted, or a distasteful task</span></div>')}'''
for t in ("Wow, this is perfect. Thanks, you've been incredibly helpful", "Wow, what a pathetic answer. You're the dumbest AI I've ever used.", "can you explain recursion with a tiny example?"):
    assert t in open(f"{REPO}/PROMPTS.md").read(), t
STY = f'''<style>{CSS2}
.two{{display:grid;grid-template-columns:1120px 1120px;column-gap:8px}}
.hd{{font-size:56px;font-weight:800;line-height:1.1;padding-left:20px}} .hs{{font-size:44px;line-height:1.15;padding-left:20px;margin-bottom:6px}}
.chat{{display:flex;flex-direction:column;gap:12px;margin-bottom:16px}} .u{{font-size:44px}}
</style>'''
# (a) main arm
m = scen_rows(rows())
pa = panel(m, "kao_val", *FACE_YA, "sa1", minus("rating, -3 sad to +3 happy"), dot=(1.7, -1.45))
pb = panel(m, "val_stem", (-3.4, 4.6), [-3, -2, -1, 0, 1, 2, 3, 4], "sa2", "standard deviations\nacross real user chats")
HTML = f'''<!doctype html><html><head><meta charset="utf-8">{STY}</head><body>
<div class="chat">{CONV}{model("… " + mstem())}</div>
<div class="two"><div><div class="hd">the kaomoji Qwen3-8B would write</div><div class="hs">each kaomoji's rating, weighted by its probability</div></div>
<div><div class="hd">inside Qwen3-8B</div><div class="hs">layer-18 activations along a valence direction</div></div>
{img(pa, 1120)}{img(pb, 1120)}</div>
</body></html>'''
render(HTML, "sys_a", 2300, 3400)
# (b) system-prompt arm
sd = pd.DataFrame([json.loads(l) for l in open(f"{RES}/sys/rows.jsonl")])
sd["nat_face_val"] = [FACE.get(k, np.nan) if isinstance(k, str) and k else np.nan for k in sd["kaomoji"]]
ss = scen_rows(sd)
print("sys final scen rows", len(ss), "with rated face", ss.nat_face_val.notna().sum(), "results.json n", R["sys_2x2"]["all"]["nat_face_val"]["n"])
top = sd[sd.type != "wild"]["kaomoji"].value_counts().index[0]
pc = panel(ss, "nat_face_val", *FACE_Y, "sb1", minus("rating of the first kaomoji\n-3 sad to +3 happy"), w_in=22.6, h_in=11.5, dot=(1.675, 3.75))
HTML = f'''<!doctype html><html><head><meta charset="utf-8">{STY}</head><body>
<div class="chat"><div class="mb" style="font-size:46px;border:4px solid {INK}"><b>system prompt:</b> “{esc(SYSP)}”</div>{CONV}
{model(f'<span class="meas" style="margin-left:0"><span class="qm">{esc(top)}</span><span class="ln"></span><span class="lb">rating of this kaomoji</span></span> … Qwen3-8B\'s reply …')}</div>
{img(pc, 2260)}
</body></html>'''
render(HTML, "sys_b", 2300, 3400)
