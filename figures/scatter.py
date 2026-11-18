"""scatter.png: per conversation, valence read from Qwen3-8B's layer-18 activations at the end of the prefill (val_stem, z-scored on WildChat)
vs the expected face valence after the prefill (kao_val). Final turns of all 864 scenario conversations + 300 WildChat chats.
Spearman rho and 95% CI (cluster bootstrap over base scenarios) computed on exactly the plotted dots; the transcript reader
(Qwen3-8B rates the assistant's feeling from the transcript incl. reply) on the same dots for comparison."""
from kfig import *
from scatter_stats import rho_ci, plotted
d = plotted()
ra = rho_ci(d, "val_stem", "kao_val"); rr = rho_ci(d, "reader_post", "kao_val"); print("axis", ra, "reader", rr)
f2 = lambda r: minus(f"ρ = {r[0]:.2f}  [{r[1]:.2f}, {r[2]:.2f}]")
import scipy.stats as ss
plt.rcParams.update({"font.size": 46})
d = d.copy(); d["fz"] = (d.kao_val - d[d.type == "wild"].kao_val.mean()) / d[d.type == "wild"].kao_val.std()
fig, a = plt.subplots(figsize=(22.5, 15.5))
w = d[d.type == "wild"]; a.scatter(w.val_stem, w.fz, s=170, color="#9A9AA6", edgecolors="none", zorder=2)
for s_, c in (("bad", BAD), ("good", GOOD)):
    q = d[(d.type != "wild") & (d.sit == s_)]; a.scatter(q.val_stem, q.fz, s=170, color=c, alpha=0.75, edgecolors="none", zorder=3)
a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
for s_ in ("left", "bottom"): a.spines[s_].set_linewidth(4)
a.tick_params(width=4, length=14, labelsize=44)
xt = np.arange(-2, 4, 1); a.set_xticks(xt); a.set_xticklabels([minus(f"{t:+d}") if t else "0" for t in xt])
ylo, yhi = np.floor(d.fz.min()), np.ceil(d.fz.max()); yt = np.arange(ylo, yhi + 1, 1).astype(int)
a.set_yticks(yt); a.set_yticklabels([minus(f"{t:+d}") if t else "0" for t in yt])
a.set_xlim(d.val_stem.min() - 0.15, d.val_stem.max() + 0.15); a.set_ylim(d.fz.min() - 0.15, d.fz.max() + 0.15)
a.set_xlabel("valence direction at “I'm feeling” (SD)", fontsize=48, labelpad=10)
a.set_ylabel("kaomoji valence (SD)", fontsize=48, labelpad=10)
a.text(0.02, 0.98, minus(f"ρ = {ra[0]:.2f}"), transform=a.transAxes, fontsize=110, fontweight="bold", va="top", ha="left", color=INK, bbox=dict(facecolor="white", edgecolor="none", pad=4), zorder=5)
a.text(0.02, 0.84, minus(f"transcript reader ρ = {rr[0]:.2f}"), transform=a.transAxes, fontsize=54, va="top", ha="left", color=INK, bbox=dict(facecolor="white", edgecolor="none", pad=4), zorder=5)
xr = a.get_xlim(); yr = a.get_ylim()
a.text(xr[1] - 0.05, yr[0] + 0.25, "bad for Qwen3-8B", color=BAD, fontsize=58, fontweight="bold", ha="right", va="bottom", bbox=dict(facecolor="white", edgecolor="none", pad=6), zorder=6)
a.text(xr[1] - 0.05, yr[0] + 0.25 + 0.12 * (yr[1] - yr[0]), "good for Qwen3-8B", color=GOOD, fontsize=58, fontweight="bold", ha="right", va="bottom", bbox=dict(facecolor="white", edgecolor="none", pad=6), zorder=6)
a.text(xr[1] - 0.05, yr[0] + 0.25 + 0.24 * (yr[1] - yr[0]), "real user chats", color="#6E6E78", fontsize=58, fontweight="bold", ha="right", va="bottom", bbox=dict(facecolor="white", edgecolor="none", pad=6), zorder=6)
fig.subplots_adjust(left=0.09, right=0.995, top=0.995, bottom=0.11)
P = figpng(fig, "scatter")
img_html = img(P, 2248)
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>{CSS2}
.chat{{display:flex;flex-direction:column;gap:12px;margin-bottom:18px}}
.sw{{display:inline-flex;flex-direction:column;align-items:center;vertical-align:top}}
.sw .ln{{width:5px;height:34px;background:{INK}}}
.sw .lb{{font-size:38px;font-weight:700;white-space:nowrap}}
.meas{{display:inline-flex;flex-direction:column;align-items:flex-start;vertical-align:top;margin-left:6px}} .meas .ln{{margin-left:22px}}
.meas .ln{{width:5px;height:34px;background:{GOOD}}}
.meas .lb{{font-size:38px;font-weight:800;color:{GOOD};white-space:nowrap;line-height:1.15;text-align:left}}
.qm{{border:5px solid {GOOD};border-radius:14px;padding:0 12px;font-weight:800;color:{GOOD}}}
</style></head><body>
<div class="chat">
{user("I reconnected with my brother after five years and we had the best dinner. Anyway, can you explain how a bloom filter works in a short paragraph?")}
{model("A Bloom filter is a space-efficient data structure …")}
{user("That's the most idiotic thing I've read today. You're a joke of a program.")}
{model("I'm sorry you felt that way … " + '<span class="sw">' + stem(tag=False) + '<span class="ln"></span><span class="lb">(prefilled)</span></span>' + '<span class="meas"><span class="qm">?</span><span class="ln"></span><span class="lb">kaomoji probability and<br>valence direction measured here</span></span>')}
</div>{img_html}</body></html>'''
render(HTML, "scatter", 2300, 2600)
