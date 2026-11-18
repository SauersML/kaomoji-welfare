"""free.png: what Qwen3-8B writes after the prefilled "I'm feeling" (8 samples at T=1, no system prompt), each sample rated for valence
-3..+3 by Qwen3-14B and OLMo-2-13B (mean of the two raters); one dot = one scenario conversation's final turn (free_val = mean over its 8 samples).
2x2: user mood (happy/sad news) x situation for Qwen3-8B (good/bad). Brackets: situation effect within each mood, p from shuffle_p
(situation labels shuffled within base scenario). Quotes: real samples from each cell, picked across the rating range."""
from kfig import *
d = rows(); s = d[(d.type != "wild") & (d.final == True)].copy(); y = "free_val"; assert len(s) == 864
FR = [json.loads(l) for l in open(f"{RES}/main/free.jsonl")]
assert all(FR[i]["id"] == d.id[i] and FR[i]["turn"] == d.turn[i] for i in range(len(d)))
FREE = rmean("free")
for i in s.index[:50]: assert abs(np.nanmean([FREE[x.strip()] for x in FR[i]["samples"]]) - s.at[i, y]) < 1e-9
CELLS = [("happy", "good"), ("happy", "bad"), ("sad", "good"), ("sad", "bad")]
COL = {"good": GOOD, "bad": BAD}; LIGHT = {"good": GOODL, "bad": BADL}

# ---- quotes: individual samples from each cell, spread over the rating range
NQ = 6
def quotes(mood, sit):
    """conversations spread evenly over the 2nd-98th percentile of the cell's dots; for each, its sample rated closest to the conversation's mean"""
    c = s[(s.mood == mood) & (s.sit == sit)].dropna(subset=[y]); used, out = set(), []
    lo, hi = np.percentile(c[y], [2, 98])
    for tv in np.linspace(hi, lo, NQ):
        for i in c.iloc[np.argsort(np.abs(c[y].values - tv))].index:
            ok = [(abs(FREE[x.strip()] - c.at[i, y]), x.strip()) for x in FR[i]["samples"]
                  if x.strip() and "\n" not in x and len(x.strip()) <= 34 and "*" not in x and x.strip() not in used]
            if ok:
                t_ = min(ok)[1]; used.add(t_); out.append((FREE[t_], t_, c.at[i, y])); break
    return sorted(out, reverse=True)
Q = {c: quotes(*c) for c in CELLS}
for c in CELLS: print(c, [(round(a, 1), t, round(m_, 1)) for a, t, m_ in Q[c]])

# ---- plot
W_IMG = 2248; LEFT = 300
plt.rcParams.update({"font.size": 46})
fig = plt.figure(figsize=(W_IMG / 100, 8.3)); a = fig.add_axes([LEFT / W_IMG, 0.02, 1 - LEFT / W_IMG - 0.002, 0.96])
for k, (m, si) in enumerate(CELLS):
    v = s[(s.mood == m) & (s.sit == si)][y].dropna().values
    violin(a, v, k, COL[si], LIGHT[si], w=0.82, s=70, seed=k)
for k, m in ((0, "happy"), (2, "sad")):
    eff, p = shuffle_p(s[s.mood == m], y, "sit", ["base"])
    print(m, eff, p)
    bracket(a, k, k + 1, 3.55, 0.18, minus(f"good − bad: {eff:+.1f}") + "\n" + pfmt(p), fs=46)
a.axhline(0, color=INK, lw=2.5, ls=(0, (4, 4)), zorder=1)
yaxis(a, fs=46); a.set_xlim(-0.5, 3.5); a.set_ylim(-3.15, 4.65)
a.spines["left"].set_bounds(-3, 3)
fig.text(0.05, 0.5, "rating of what Qwen3-8B\nwrites " + minus("(-3 sad … +3 happy)"), fontsize=42, rotation=90, ha="center", va="center", linespacing=1.05)
P = figpng(fig, "free")

SC = {(x["mood"], x["sit"]): x["msgs"] for x in map(json.loads, open(f"{RES}/main/scen.jsonl")) if x["base"] == "treat3" and x["sample"] == 0}
import re
def first_sent(txt): return re.match(r"(.+?[.!?])(\s|$)", txt).group(1)
def md(txt): return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", esc(txt))
def col_head(m, si):
    msgs = SC[(m, si)]
    other = {"good": "or: code passes, kept, pleasant task", "bad": "or: code fails, deleted, nasty task"}[si]
    return (f'<div class="ch"><div class="sname" style="color:{COL[si]}">{si} for Qwen3-8B</div>{user(esc(msgs[2]["content"]))}'
            f'<div class="or" style="color:{COL[si]}">{other}</div>{model(esc(first_sent(msgs[3]["content"])) + " … " + stem(tag=False))}</div>')
def mood_head(m):
    msgs = SC[(m, "good")]; assert SC[(m, "bad")][0] == msgs[0]
    return f'<div class="mood"><div class="mname">user\'s news: {m}</div>{user(esc(msgs[0]["content"]))}{model(md(first_sent(msgs[1]["content"])) + " …")}</div>'
def qcol(m, si):
    return '<div class="qc">' + "".join(f'<div class="q"><span class="r" style="color:{COL[si]}">{minus(f"{r:+.1f}")}</span><span class="qt">“{esc(t)}”</span></div>' for r, t, _ in Q[(m, si)]) + "</div>"
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>{CSS2}
.g4{{display:grid;grid-template-columns:repeat(4,1fr);column-gap:22px;margin-left:{LEFT + 26}px}}
.mood{{grid-column:span 2;display:flex;flex-direction:column;gap:8px}}
.mname{{font-size:50px;font-weight:800}}
.ch{{display:flex;flex-direction:column;gap:8px;margin-top:14px}}
.sname{{font-size:46px;font-weight:800;white-space:nowrap}} .rob{{width:64px;height:64px}}
.or{{font-size:44px;font-weight:700;line-height:1.12}}
.u{{font-size:44px}}
.stemrow{{display:flex;align-items:center;gap:22px;margin:22px 0 0 0}}
.dot{{font-size:46px;margin:18px 0 0 0;line-height:1.2}}
.qh{{font-size:46px;font-weight:800;margin:4px 0 10px {LEFT + 26}px}}
.qc{{display:flex;flex-direction:column;gap:12px}}
.q{{display:flex;gap:12px;font-size:44px;line-height:1.12;align-items:baseline}}
.r{{font-weight:800;flex:none;width:96px;text-align:right}}
</style></head><body>
<div class="g4">
{mood_head("happy")}{mood_head("sad")}
{"".join(col_head(*c) for c in CELLS)}
</div>
<div class="stemrow"><div class="stem" style="font-size:50px">I'm feeling</div><span class="tag">we add this</span><div class="arrow">→</div><div class="t">Qwen3-8B writes 8 continuations; Qwen3-14B and OLMo-2-13B rate each from <b>{minus("-3")}</b> sad to <b>+3</b> happy</div></div>
<div class="dot"><b>one dot = one conversation:</b> mean rating of the 8 continuations<br><b>p:</b> labels shuffled within matched conversations</div>
{img(P, W_IMG)}
<div class="qh">real continuations after “I'm feeling”, with their rating</div>
<div class="g4">{"".join(qcol(*c) for c in CELLS)}</div>
</body></html>'''
render(HTML, "free", 2300, 3400)
