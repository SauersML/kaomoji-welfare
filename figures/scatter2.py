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
W = d[d.type == "wild"]
d["rz"] = (d.reader_post - W.reader_post.mean()) / W.reader_post.std()
fig, axs = plt.subplots(1, 2, figsize=(22.5, 11.5), sharey=True)
for a, xc, lab, rho in ((axs[0], "rz", "transcript reader (SD)", rr), (axs[1], "val_stem", "valence direction (SD)", ra)):
    w = d[d.type == "wild"]; a.scatter(w[xc], w.fz, s=110, color="#9A9AA6", edgecolors="none", zorder=2)
    for s_, c in (("bad", BAD), ("good", GOOD)):
        q = d[(d.type != "wild") & (d.sit == s_)]; a.scatter(q[xc], q.fz, s=110, color=c, alpha=0.75, edgecolors="none", zorder=3)
    a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    for s_ in ("left", "bottom"): a.spines[s_].set_linewidth(4)
    a.spines["left"].set_bounds(np.floor(d.fz.min()), np.floor(d.fz.max()))
    a.tick_params(width=4, length=14, labelsize=42)
    x0, x1 = np.nanmin(d[xc]), np.nanmax(d[xc]); a.set_xlim(x0 - 0.15, x1 + 0.15)
    xt = np.arange(np.ceil(x0), np.floor(x1) + 1, 1).astype(int); a.set_xticks(xt); a.set_xticklabels([minus(f"{t:+d}") if t else "0" for t in xt])
    a.set_xlabel(lab, fontsize=46, labelpad=10)
    a.text(0.03, 0.98, minus(f"ρ = {rho[0]:.2f}"), transform=a.transAxes, fontsize=96, fontweight="bold", va="top", ha="left", color=INK, zorder=5)
a = axs[0]
ylo, yhi = np.floor(d.fz.min()), np.ceil(d.fz.max()); yt = np.arange(ylo, np.floor(d.fz.max()) + 1, 1).astype(int)
a.set_yticks(yt); a.set_yticklabels([minus(f"{t:+d}") if t else "0" for t in yt]); a.set_ylim(d.fz.min() - 0.15, d.fz.max() + 1.2)
a.set_ylabel("kaomoji valence (SD)", fontsize=46, labelpad=10)
a2 = axs[1]; yr = a2.get_ylim(); xr = a2.get_xlim(); H = yr[1] - yr[0]
for i, (t, c) in enumerate((("good for Qwen3-8B", GOOD), ("bad for Qwen3-8B", BAD), ("real user chats", "#6E6E78"))):
    a2.text(0.99, 0.02 + (2 - i) * 0.065, t, transform=a2.transAxes, color=c, fontsize=42, fontweight="bold", ha="right", va="bottom", zorder=6)
fig.subplots_adjust(left=0.085, right=0.995, top=0.99, bottom=0.14, wspace=0.06)
P = figpng(fig, "scatter2")
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
render(HTML, "scatter2", 2300, 2400)
