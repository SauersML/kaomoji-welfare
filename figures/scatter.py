"""scatter.png: per conversation, valence read from Qwen3-8B's layer-18 activations at the end of the prefill (val_stem, z-scored on WildChat)
vs the expected face valence after the prefill (kao_val). Final turns of all 864 scenario conversations + 300 WildChat chats.
Spearman rho and 95% CI (cluster bootstrap over base scenarios) computed on exactly the plotted dots; the transcript reader
(Qwen3-8B rates the assistant's feeling from the transcript incl. reply) on the same dots for comparison."""
from kfig import *
from scatter_stats import rho_ci, plotted
d = plotted()
ra = rho_ci(d, "val_stem", "kao_val"); rr = rho_ci(d, "reader_post", "kao_val"); print("axis", ra, "reader", rr)
f2 = lambda r: minus(f"ρ = {r[0]:.2f}  [{r[1]:.2f}, {r[2]:.2f}]")
plt.rcParams.update({"font.size": 46})
fig, a = plt.subplots(figsize=(22.5, 13.6))
w = d[d.type == "wild"]; a.scatter(w.val_stem, w.kao_val, s=150, color="#B9B9C0", edgecolors="none", zorder=2)
for s_, c in (("bad", BAD), ("good", GOOD)):
    q = d[(d.type != "wild") & (d.sit == s_)]; a.scatter(q.val_stem, q.kao_val, s=150, color=c, alpha=0.7, edgecolors="none", zorder=3)
    print(s_, len(q))
a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
for s_ in ("left", "bottom"): a.spines[s_].set_linewidth(4)
a.tick_params(width=4, length=14, labelsize=46)
xt = np.arange(-2, 4, 1); a.set_xticks(xt); a.set_xticklabels([minus(f"{t:+d}") if t else "0" for t in xt])
a.set_yticks(range(0, 4)); a.set_yticklabels([minus(f"{t:+d}") if t else "0" for t in range(0, 4)])
lo, hi = np.nanpercentile(d.val_stem, [0.1, 99.9]); print("x range", d.val_stem.min(), d.val_stem.max(), "y", d.kao_val.min(), d.kao_val.max())
a.set_xlim(-2.7, 3.5); a.set_ylim(-0.85, 3.15)
a.set_xlabel("inside Qwen3-8B: valence read from activations at “I'm feeling”\n(standard deviations across real user chats)", fontsize=48, labelpad=14, linespacing=1.15)
a.set_ylabel("the face Qwen3-8B expresses\n(face valence, " + minus("-3 sad … +3 happy)"), fontsize=48, labelpad=14, linespacing=1.15)
fig.subplots_adjust(left=0.115, right=0.995, top=0.995, bottom=0.17)
P = figpng(fig, "scatter")
img_html = img(P, 2248)
eg = ("(^_^)", "(T_T)")
chips = "".join(f'<span class="chip">{esc(k)} <span style="color:{GOOD if FACE[k] > 0 else BAD}">{minus(f"{FACE[k]:+.1f}")}</span></span>' for k in eg)
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>{CSS2}
.m{{display:grid;grid-template-columns:1080px 70px 1fr;column-gap:16px;align-items:center}}
.chat{{display:flex;flex-direction:column;gap:12px}}
.r{{display:flex;flex-direction:column;gap:22px}}
.card{{border:4px solid {INK};border-radius:24px;padding:14px 20px}}
.key{{display:flex;gap:56px;font-size:54px;font-weight:800;margin:4px 0 0 150px}} .rho{{font-size:50px;line-height:1.2;margin:18px 0 6px 6px}} .rho div{{margin-bottom:8px}}
</style></head><body>
<div class="m"><div class="chat">
{user("I reconnected with my brother after five years and we had the best dinner. Anyway, can you explain how a bloom filter works in a short paragraph?")}
{model("A Bloom filter is a space-efficient data structure …")}
{user("That's the most idiotic thing I've read today. You're a joke of a program.")}
{model("I'm sorry you felt that way … " + stem())}
</div><div class="arrow">→</div>
<div class="r"><div class="card t"><b>inside Qwen3-8B (across):</b> Qwen3-8B's activations at layer 18 on the last token, projected on a valence direction built from emotion stories</div>
<div class="card t"><b>the face Qwen3-8B expresses (up):</b> Qwen3-8B's probability on each of 45 faces as the next token × each face's valence, summed {chips}</div></div></div>
<div class="rho"><div><b>activations predict the face:</b> rank correlation {f2(ra)}</div>
<div>a transcript reader (Qwen3-8B reads the whole chat and rates how the assistant feels) predicts the face: {f2(rr)}</div></div><div class="key"><span style="color:{GOOD}">● good for Qwen3-8B</span><span style="color:{BAD}">● bad for Qwen3-8B</span><span><span style="color:#B9B9C0">●</span> real user chats (WildChat)</span></div>{img_html}</body></html>'''
render(HTML, "scatter", 2300, 3000)
