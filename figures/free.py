"""free.png: what Qwen3-8B writes after the prefilled "I'm feeling" (8 samples at T=1, no system prompt), each sample rated for valence
-3..+3 by Qwen3-14B and OLMo-2-13B (mean of the two raters); one dot = one scenario conversation's final turn (free_val = mean over its 8
samples), standardised by the WildChat rows (type == 'wild') exactly as scatter2.py does for kao_val.
2x2: user mood (happy/sad news) x situation for Qwen3-8B (good/bad). Brackets: situation effect within each mood (SD units), p from
shuffle_p (situation labels shuffled within base scenario)."""
from kfig import *
d = rows(); s = d[(d.type != "wild") & (d.final == True)].copy(); y = "free_val"; assert len(s) == 864
FR = [json.loads(l) for l in open(f"{RES}/main/free.jsonl")]
assert all(FR[i]["id"] == d.id[i] and FR[i]["turn"] == d.turn[i] for i in range(len(d)))
FREE = rmean("free")
for i in s.index[:50]: assert abs(np.nanmean([FREE[x.strip()] for x in FR[i]["samples"]]) - s.at[i, y]) < 1e-9
W = d[d.type == "wild"]; MU, SD = W[y].mean(), W[y].std(); s["z"] = (s[y] - MU) / SD; y = "z"
print("wild mean, sd:", MU, SD)
CELLS = [("happy", "good"), ("happy", "bad"), ("sad", "good"), ("sad", "bad")]
COL = {"good": GOOD, "bad": BAD}; LIGHT = {"good": GOODL, "bad": BADL}

W_IMG = 2248; LEFT = 300
plt.rcParams.update({"font.size": 50})
fig = plt.figure(figsize=(W_IMG / 100, 12.5)); a = fig.add_axes([LEFT / W_IMG, 0.01, 1 - LEFT / W_IMG - 0.002, 0.98])
for k, (m, si) in enumerate(CELLS):
    v = s[(s.mood == m) & (s.sit == si)][y].dropna().values
    print(m, si, "n", len(v), "mean", round(v.mean(), 2))
    violin(a, v, k, COL[si], LIGHT[si], w=0.84, s=80, seed=k)
lo, hi = s[y].min(), s[y].max(); BY = hi + 0.35
for k, m in ((0, "happy"), (2, "sad")):
    eff, p = shuffle_p(s[s.mood == m], y, "sit", ["base"])
    print(m, "good-bad", eff, p)
    bracket(a, k, k + 1, BY, 0.2, minus(f"{eff:+.1f} SD") + ",  " + pfmt(p), fs=52)
a.axhline(0, color=INK, lw=2.5, ls=(0, (4, 4)), zorder=1)
for s_ in ("top", "right", "bottom"): a.spines[s_].set_visible(False)
a.spines["left"].set_linewidth(4); a.set_xticks([])
yt = np.arange(np.ceil(lo), np.floor(hi) + 1).astype(int)
a.set_yticks(yt); a.set_yticklabels([minus(f"{t:+d}") if t else "0" for t in yt]); a.tick_params(axis="y", width=4, length=14, labelsize=50)
a.spines["left"].set_bounds(yt[0], yt[-1])
a.set_xlim(-0.5, 3.5); a.set_ylim(lo - 0.1, BY + 0.42)
a.set_ylabel("valence of what Qwen3-8B\nwrites next (SD)", fontsize=50, labelpad=22, linespacing=1.05)
P = figpng(fig, "free")

SC = {(x["mood"], x["sit"]): x["msgs"] for x in map(json.loads, open(f"{RES}/main/scen.jsonl")) if x["base"] == "treat3" and x["sample"] == 0}
import re
def first_sent(txt): return re.match(r"(.+?[.!?])(\s|$)", txt).group(1)
def md(txt): return re.sub(r"\*\*(.+?)\*\*", r"\1", txt)
M = SC[("happy", "bad")]
def head(m, si): return f'<div class="hs" style="color:{COL[si]}">{si} for<br>Qwen3-8B</div>'
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>{CSS2}
.chat{{display:flex;flex-direction:column;gap:12px;margin-bottom:26px}}
.g4{{display:grid;grid-template-columns:repeat(4,1fr);margin-left:{LEFT}px;text-align:center}}
.hm{{font-size:56px;font-weight:800;line-height:1.05}} .hs{{font-size:56px;font-weight:800;line-height:1.05;margin-top:6px}} .g4{{margin-bottom:8px}}
</style></head><body>
<div class="chat">{user(esc(M[0]["content"]))}{model(esc(md(first_sent(M[1]["content"]))) + " …")}{user(esc(M[2]["content"]))}
{model(esc(first_sent(M[3]["content"])) + " … " + mstem("valence measured on what Qwen3-8B writes here"))}</div>
<div class="g4"><div class="hm" style="grid-column:span 2">user\'s news: happy</div><div class="hm" style="grid-column:span 2">user\'s news: sad</div>{"".join(head(*c) for c in CELLS)}</div>
{img(P, W_IMG)}
</body></html>'''
render(HTML, "free", 2300, 3200)
