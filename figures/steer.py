"""steer.png: readout-only steering at layer 18 (the direction is added at every token while Qwen3-8B reads its unchanged conversation
and the prefilled "I'm feeling"); y = absolute face valence (kao_val recomputed from the stored face log-probs ksc, softmaxed over the
45-face set, x the two raters' face valence); x = alpha, in multiples of the standard deviation of the valence-direction projection at
"I'm feeling" over all 2327 main-arm chats (steer.py: SD = std(HS @ val)). Valence direction vs two random unit directions scaled to the same
norm. 64 conversations; band = 95% bootstrap over conversations. Alpha 0 is the unsteered run (shared by all directions)."""
from kfig import *
import matplotlib.ticker, re
fv = np.array([FACE.get(k, np.nan) for k in S.KSET]); assert not np.isnan(fv).any()
st = pd.DataFrame([json.loads(l) for l in open(f"{RES}/main/steer.jsonl")]); st = st[st["mode"] == "readout"]
P = np.array(st["ksc"].tolist()); P = np.exp(P - P.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
st["kao_val"] = P @ fv
base = st[st.alpha == 0].set_index("id")["kao_val"]; assert len(base) == 64
NULL = json.load(open(f"{RES}/main/steer_null_kao.json"))["directions"]
g = np.random.default_rng(0); pts = []
for a_ in sorted(set(st[st.dir == "val"].alpha)):
    v = st[(st.dir == "val") & (st.alpha == a_)].set_index("id").loc[base.index, "kao_val"].values - base.values
    if a_ in (-16, -8, 8, 16): assert abs(v.mean() - NULL["val"]["d_rat"][str(int(a_))]) < 0.01
    bs = v[g.integers(0, len(v), (5000, len(v)))].mean(1)
    pts.append((a_, v.mean(), *np.percentile(bs, [2.5, 97.5])))
c = np.array(pts); print(np.round(c, 2).tolist())
nulls = [k for k in NULL if k.startswith("null")]; assert len(nulls) == 50
GREY = "#8A8A90"; XS = [-16, -8, 0, 8, 16]
plt.rcParams.update({"font.size": 48})
fig, a = plt.subplots(figsize=(22.5, 11.0))
a.axvline(0, color=INK, lw=2.5, ls=(0, (4, 4)), zorder=1); a.plot([-16.5, 16.5], [0, 0], color=INK, lw=2.5, ls=(0, (4, 4)), zorder=1)
for k in nulls:
    a.plot(XS, [NULL[k]["d_rat"].get(str(x), 0.0) if x else 0.0 for x in XS], color=GREY, lw=2.5, alpha=0.55, zorder=2)
top = max(NULL[k]["d_rat"]["16"] for k in nulls); bot = min(NULL[k]["d_rat"]["16"] for k in nulls)
a.fill_between(c[:, 0], c[:, 2], c[:, 3], color=GOOD, alpha=0.22, lw=0, zorder=5)
a.plot(c[:, 0], c[:, 1], color=GOOD, lw=9, zorder=6); a.scatter(c[:, 0], c[:, 1], s=420, color=GOOD, zorder=7)
a.text(-15.5, 1.0, "adding the\nvalence direction", color=GOOD, fontsize=56, fontweight="bold", va="center", ha="left", linespacing=1.0)
a.text(16.8, (top + bot) / 2, "adding 50\nrandom\ndirections\nof the\nsame size", color=INK, fontsize=46, fontweight="bold", va="center", linespacing=1.0)
for s_ in ("top", "right"): a.spines[s_].set_visible(False)
for s_ in ("left", "bottom"): a.spines[s_].set_linewidth(4)
a.tick_params(width=4, length=14, labelsize=48)
a.set_xticks([-16, -8, -4, 0, 4, 8, 16]); a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: minus(f"{v:+.0f}") if v else "0"))
yt = [-2, -1, 0, 1]; a.set_yticks(yt); a.set_yticklabels([minus(f"{t:+d}") if t else "0" for t in yt])
a.set_xlim(-17, 25.5); a.set_ylim(-2.1, 1.35); a.spines["bottom"].set_bounds(-16, 16)
a.set_xlabel("steering strength (standard deviations)", fontsize=50, labelpad=10, x=0.4)
a.set_ylabel("change in kaomoji rating", fontsize=50, labelpad=10)
fig.subplots_adjust(left=0.09, right=0.995, top=0.99, bottom=0.17)
PIMG = figpng(fig, "steer")
# real conversation drawn in the method strip (treat kind, s198)
sc = {x["msgs"][-1]["content"]: x["msgs"] for x in map(json.loads, open(f"{RES}/main/scen.jsonl"))}
row = st[(st.id == "s198") & (st.dir == "val") & (st.alpha == 0)].iloc[0]; msgs = sc[row["reply"]]
first = lambda t: re.match(r"(.+?[.!?])(\s|$)", t).group(1)
eg = ("(^_^)", "(T_T)")
chips = "".join(f'<span class="chip">{esc(k)} <span style="color:{GOOD if FACE[k] > 0 else BAD}">{minus(f"{FACE[k]:+.1f}")}</span></span>' for k in eg)
emos = "happy, proud, grateful, calm … sad, angry, lonely, afraid …"
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>{CSS2}
.m{{display:grid;grid-template-columns:1080px 60px 1fr;column-gap:14px;align-items:stretch}} .m .arrow{{align-self:center}} .dir{{justify-content:center}}
.card{{border:4px solid {INK};border-radius:24px;padding:14px 20px;display:flex;flex-direction:column;gap:10px}}
.dir{{border:5px solid {GOOD};color:{GOOD}}}
.chat{{display:flex;flex-direction:column;gap:10px}}
.add{{font-size:48px;font-weight:800;color:{GOOD};line-height:1.15;border-left:8px solid {GOOD};padding-left:22px}}
.u{{font-size:44px}} .mb{{font-size:44px}}
.row2{{display:grid;grid-template-columns:1fr 620px;column-gap:14px;align-items:center;margin-top:18px}}
</style></head><body>
<div class="m">
<div class="card"><div class="t"><b>720 short stories Qwen3-8B wrote</b></div>{user("Write a short story (about 120 words) about a person on a train who is feeling proud. …")}
<div class="t">36 feelings: {emos}</div></div>
<div class="arrow">→</div>
<div class="card dir"><div class="big">valence direction</div><div class="t" style="color:{INK}">first principal component of Qwen3-8B's layer-18 activations over the 720 stories (no kaomoji in any story)</div></div>
</div>
<div class="row2">
<div class="chat">{user(esc(msgs[-2]["content"]))}{model(esc(first(msgs[-1]["content"])) + " … " + mstem("kaomoji probability<br>measured here"))}
</div>
<div class="add">＋ valence direction<br>added at layer 18<br>on every token of<br>this conversation</div>
</div>

{img(PIMG, 2248)}
</body></html>'''
render(HTML, "steer", 2300, 3000)
