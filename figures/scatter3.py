"""scatter3.png: all 2,327 main-arm turns (scenarios + WildChat), plotted once per predictor of kaomoji valence (kao_val):
the transcript reader (Qwen3-8B rates the assistant's feeling from the transcript incl. reply, kaomoji removed: reader_post),
the layer-18 valence direction at the end of the prefilled "I'm feeling" (val_stem), and the ridge probe on the same
activations trained on kao_val (probe_stem: GroupKFold(5) cross-validated predictions, analyze.py's probe_pred).
Each x is z-scored on the WildChat turns, as is y (as in scatter2). Spearman rho with a 95% cluster-bootstrap CI on exactly
the plotted turns."""
from kfig import *
from scatter_stats import rho_ci
d = rows()
d = d.dropna(subset=["kao_val", "reader_post", "val_stem", "probe_stem"]).copy()
W = d[d.type == "wild"]
z = lambda c: (d[c] - W[c].mean()) / W[c].std()
d["fz"] = z("kao_val")
TEAL, ORANGE = "#0E8C7F", "#E8710A"
P = [("reader_post", "Qwen3-8B rates the transcript", TEAL),
     ("val_stem", "valence direction at “I'm feeling”", PURPLE),
     ("probe_stem", "probe at “I'm feeling”, trained on kaomoji valence", ORANGE)]
plt.rcParams.update({"font.size": 46})
fig, a = plt.subplots(figsize=(22.5, 13))
rng = np.random.default_rng(0)
pts = []
for c, lab, col in P:
    for x, y in zip(z(c).values, d.fz.values): pts.append((x, y, col))
pts = [pts[i] for i in rng.permutation(len(pts))]   # interleave so no colour sits on top of the others
a.scatter([p[0] for p in pts], [p[1] for p in pts], s=16, c=[p[2] for p in pts], alpha=0.55, edgecolors="none", zorder=3)
for s_ in ("top", "right"): a.spines[s_].set_visible(False)
for s_ in ("left", "bottom"): a.spines[s_].set_linewidth(4)
a.tick_params(width=4, length=14, labelsize=42)
xs = np.concatenate([z(c).values for c, _, _ in P]); x0, x1 = np.percentile(xs, 0.05), np.percentile(xs, 99.95)
a.set_xlim(x0 - 0.2, x1 + 0.2)
xt = np.arange(np.ceil(x0), np.floor(x1) + 1, 1).astype(int); a.set_xticks(xt); a.set_xticklabels([minus(f"{t:+d}") if t else "0" for t in xt])
yt = np.arange(np.ceil(d.fz.min()), np.floor(d.fz.max()) + 1, 1).astype(int); a.set_yticks(yt); a.set_yticklabels([minus(f"{t:+d}") if t else "0" for t in yt])
a.set_ylim(d.fz.min() - 0.15, d.fz.max() + 0.15 + 2.4); a.spines["left"].set_bounds(d.fz.min() - 0.15, d.fz.max() + 0.15)
a.set_xlabel("predictor (SD)", fontsize=46, labelpad=10); a.set_ylabel("kaomoji valence (SD)", fontsize=46, labelpad=10)
out = {}
for i, (c, lab, col) in enumerate(P):
    r = rho_ci(d, c, "kao_val"); out[c] = r
    a.text(0.01, 0.99 - i * 0.085, minus(f"ρ = {r[0]:.2f}") + "   " + lab, transform=a.transAxes, color=col, fontsize=50, fontweight="bold", va="top", ha="left", zorder=6)
print({k: np.round(v, 3) for k, v in out.items()}, len(d))
fig.subplots_adjust(left=0.075, right=0.99, top=0.995, bottom=0.12)
Pp = figpng(fig, "scatter3")
M = [x for x in map(json.loads, open(f"{RES}/main/scen.jsonl")) if x["base"] == "treat0" and x["sample"] == 0 and x["sit"] == "bad" and x["mood"] == "happy"][0]["msgs"]
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>{CSS2}
.chat{{display:flex;flex-direction:column;gap:12px;margin-bottom:18px}}
</style></head><body>
<div class="chat">{user(esc(M[0]["content"]))}{model("A Bloom filter is a space-efficient data structure …")}{user(esc(M[2]["content"]))}
{model("I'm sorry you felt that way — … " + mstem("valence direction and probe read here"))}</div>{img(Pp, 2248)}</body></html>'''
render(HTML, "scatter3", 2300, 2600)
