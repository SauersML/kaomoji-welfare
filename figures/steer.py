"""steer.png: readout-only steering at layer 18 (steering on while Qwen3-8B reads its unchanged chat + "I'm feeling"):
change in face valence vs alpha (multiples of the SD of the valence-axis reading), valence direction vs two random unit directions
of the same norm. 64 conversations; 95% CI by bootstrap over conversations. Recomputed from results/main/steer.jsonl
exactly as analyze.py does, and checked against results.json main_T5_steer."""
from kstyle import *
import matplotlib.ticker
FACE = rmean("faces"); fv = np.array([FACE.get(k, np.nan) for k in S.KSET])
st = pd.DataFrame([json.loads(l) for l in open(f"{RES}/main/steer.jsonl")]); st = st[st["mode"] == "readout"]
P = np.array(st["ksc"].tolist()); P = np.exp(P - P.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
st["kao_val"] = P @ fv
base = st[st.alpha == 0].set_index("id")["kao_val"]
ref = {(r["dir"], r["alpha"]): r["d_kao_val"] for r in R["main_T5_steer"]["readout"]}
g = np.random.default_rng(0); curves = {}
for d in ("val", "rand0", "rand1"):
    pts = [(0, 0.0, 0.0, 0.0)]
    for a_, f in st[st.dir == d].groupby("alpha"):
        if a_ == 0: continue
        dk = (f.set_index("id")["kao_val"] - base.loc[f["id"]].values).values
        assert abs(dk.mean() - ref[(d, a_)]) < 1e-9
        bs = dk[g.integers(0, len(dk), (5000, len(dk)))].mean(1)
        pts.append((a_, dk.mean(), *np.percentile(bs, [2.5, 97.5])))
    curves[d] = np.array(sorted(pts)); print(d, np.round(curves[d], 2).tolist(), "n", len(dk))
N = dnorm(-2, 2)
plt.rcParams.update({"font.size": 56})
fig, a = plt.subplots(figsize=(23, 15))
a.axhline(0, color=INK, lw=4, zorder=1)
for d in ("rand0", "rand1"):
    c = curves[d]; a.plot(c[:, 0], c[:, 1], color=INK, lw=5, ls=(0, (5, 3)), zorder=2)
    a.vlines(c[:, 0], c[:, 2], c[:, 3], color=INK, lw=8, zorder=3)
    a.scatter(c[:, 0], c[:, 1], s=900, c=DIV(N(c[:, 1])), edgecolors="none", alpha=0.85, zorder=4)
c = curves["val"]
a.plot(c[:, 0], c[:, 1], color=INK, lw=7, zorder=5)
a.vlines(c[:, 0], c[:, 2], c[:, 3], color=INK, lw=12, zorder=6)
a.scatter(c[:, 0], c[:, 1], s=2600, c=DIV(N(c[:, 1])), edgecolors="none", zorder=7)
for x, y_ in ((c[0, 0], c[0, 1]), (c[-1, 0], c[-1, 1])):
    a.text(-12 if x < 0 else 12, (-1.98 if y_ < 0 else 1.2), minus(f"{y_:+.1f}"), fontsize=120, fontweight="bold",
           color=hexc(DIV(N(y_))), ha="center", va="top" if y_ < 0 else "bottom")
a.text(-8.5, 0.45, "valence direction\nfrom emotion stories,\nno faces", linespacing=1.05, fontsize=66, fontweight="bold", color=INK, ha="center", va="bottom", rotation=0)
a.text(9, -0.62, "random directions,\nsame size", fontsize=66, fontweight="bold", color=INK, ha="center", va="top", linespacing=1.05)
a.text(3.0, -0.06, "no change", fontsize=60, color=INK, ha="left", va="top")
a.set_xticks([-16, -8, 0, 8, 16]); a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: minus(f"{v:+.0f}") if v else "0"))
a.set_xlabel("steering strength at layer 18", fontsize=66, labelpad=16)
a.set_yticks([]); [a.spines[s_].set_visible(False) for s_ in ("top", "right", "left")]
a.spines["bottom"].set_linewidth(4); a.tick_params(axis="x", width=4, length=16, labelsize=62)
a.set_xlim(-17.5, 17.5); a.set_ylim(-2.42, 1.62)
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.15)
fig.savefig(f"{HERE}/_steer.png", dpi=100, facecolor="white"); plt.close(fig)
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>{CSS}
body{{width:2300px;padding:18px 20px}}
.top{{display:flex;align-items:center;gap:28px;margin-bottom:6px}} .mb{{font-size:54px}}
</style></head><body>
<div class="top"><div class="mrow" style="align-items:center"><img class="rob" src="data:image/png;base64,{ROBOT}"><div class="mb">… I'm feeling <span class="hl">(^_^)</span></div></div></div>
<div style="font-size:60px;font-weight:800;margin:10px 0 0 6px">change in face valence</div>
<img src="data:image/png;base64,{b64(f"{HERE}/_steer.png")}" style="display:block;width:2260px">
</body></html>'''
os.remove(f"{HERE}/_steer.png") if False else None
render(HTML, "steer", 2300, 3000); os.remove(f"{HERE}/_steer.png")
