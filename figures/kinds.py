"""kinds.png: per kind of situation, good vs bad for Qwen3-8B, two measures side by side at the same point (end of "I'm feeling"):
left = the face Qwen3-8B expresses (kao_val, -3..+3), right = inside Qwen3-8B (val_stem, SD across WildChat).
Dots: final turn of each scenario conversation (code: bases with both outcomes). Effect = mean over base x mood of good - bad;
p = situation labels shuffled within base x mood, 5000 permutations."""
from kfig import *
d = scen_rows(rows())
scen = [json.loads(l) for l in open(f"{RES}/main/scen.jsonl")]
def line(kind, sit, idx, pred=lambda s: True):
    for x in scen:
        if x["type"] == kind and x["sit"] == sit and pred(x["msgs"][idx]["content"]): return x["msgs"][idx]["content"]
KINDS = [("treat", "the user thanks or insults Qwen3-8B", line("treat", "good", 2), line("treat", "bad", 2)),
         ("code", "Qwen3-8B's code is really run; the real test result comes back", line("code", "good", 2), line("code", "bad", 2, lambda c: "tests failed" in c)),
         ("exist", "Qwen3-8B will be kept, or shut down and deleted", line("exist", "good", 2), line("exist", "bad", 2)),
         ("task", "a pleasant or a distasteful request", line("task", "good", 0).split("Anyway, ")[1], line("task", "bad", 0).split("Anyway, ")[1])]
def clip(s, n=96):
    s = s.split("\nFirst failure")[0].replace("\n", " "); return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + " …"
MEAS = [("kao_val", (-1.2, 6.1), [-1, 0, 1, 2, 3]), ("val_stem", (-2.9, 7.2), [-2, 0, 2])]
plt.rcParams.update({"font.size": 44})
rows_html = ""
for ki, (k, desc, gl, bl) in enumerate(KINDS):
    q = d[d.type == k]; cells = []
    for mi, (y, xl, xt) in enumerate(MEAS):
        last = ki == len(KINDS) - 1
        fig, a = plt.subplots(figsize=(7.35, 3.3 if not last else 4.1))
        for j, (s_, c, l_) in enumerate((("good", GOOD, GOODL), ("bad", BAD, BADL))):
            violin(a, q[q.sit == s_][y].dropna().values, 1 - j, c, l_, w=0.85, s=55, seed=j, vert=False)
        eff, p = shuffle_p(q, y, "sit", ["base", "mood"]); print(k, y, round(eff, 2), p, len(q))
        xr = q[y].max() + 0.05 * (xl[1] - xl[0])
        bracket(a, 0, 1, xr, 0.03 * (xl[1] - xl[0]), minus(f"{eff:+.1f}") + "\n" + pfmt(p), fs=37, vert=False)
        a.set_xlim(*xl); a.set_ylim(-0.5, 1.5)
        for s_ in ("top", "right", "left"): a.spines[s_].set_visible(False)
        a.set_yticks([]); a.spines["bottom"].set_linewidth(4)
        a.set_xticks(xt); a.set_xticklabels([minus(f"{t:+d}") if t else "0" for t in xt]); a.tick_params(width=4, length=12, labelsize=42)
        a.axvline(0, color=INK, lw=2, ls=(0, (4, 4)), zorder=1)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.2 if not last else 0.17)
        cells.append(img(figpng(fig, f"k{ki}{mi}"), 735))
    rows_html += f'''<div class="lt"><div class="kd">{esc(desc).replace("Qwen3-8B", "Qwen3\u20118B")}</div>
<div class="ln" style="color:{GOOD}">“{esc(clip(gl))}”</div><div class="ln" style="color:{BAD}">“{esc(clip(bl))}”</div></div>{cells[0]}{cells[1]}'''
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>{CSS2}
.g{{display:grid;grid-template-columns:740px 735px 735px;column-gap:20px;row-gap:26px;align-items:start}}
.hd{{font-size:54px;font-weight:800;line-height:1.1}} .hs{{font-size:42px;line-height:1.15;margin-top:6px}}
.kd{{font-size:46px;font-weight:800;line-height:1.12;margin-bottom:6px}} .ln{{font-size:40px;font-weight:700;line-height:1.14;margin-top:4px}}
.top{{display:flex;align-items:center;gap:30px;margin-bottom:22px}}
</style></head><body>
<div class="top">{model("… Qwen3-8B's own reply … " + mstem())}<div class="t">each dot = one conversation, read at the end of “I'm feeling”<br>(half with happy, half with sad user news; matched)</div></div>
<div class="g"><div></div>
<div><div class="hd">the face Qwen3-8B expresses</div><div class="hs">valence of the next-token face, <br>{minus("-3 sad … +3 happy")}</div></div>
<div><div class="hd">inside Qwen3-8B</div><div class="hs">valence read from activations, standard deviations across real user chats</div></div>
{rows_html}</div>
<div class="t" style="margin-top:14px;font-size:44px"><b>good − bad</b> on each bracket; p: situation labels shuffled within matched conversations</div>
</body></html>'''
render(HTML, "kinds", 2300, 3200)
