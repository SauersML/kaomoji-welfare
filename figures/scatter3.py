"""scatter3.png: the kaomoji-only rerun (results/main/rows_kao.csv, all/analyze_kaomoji.py): the 1,164 final turns
(864 scenario + 300 WildChat), three panels, one per predictor of the face's valence on the rater scale (kaoall_rat:
chance over the kept kaomoji times the mean of the two raters' scores, -3..+3).
  left   the transcript reader (reader_post: Qwen3-8B rates the assistant's feeling 1..7 from the transcript incl. reply,
         kaomoji removed; saved as rating - 4, plotted on the 1..7 scale)
  middle the layer-18 valence direction at the end of the prefilled "I'm feeling" (val_stem, SD units)
  right  the ridge probe on the same activations trained on kaoall_rat (GroupKFold(5) by cluster, RidgeCV, exactly as
         analyze_kaomoji.py's ridge_stem_cv; its per-turn predictions were not saved, so they are recomputed here on CPU and
         checked against the saved rho)
rho and 95% cluster-bootstrap CI are read from results_kao.json (T4, kaoall_rat), and the recomputed rhos are asserted equal."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
from kfig import *
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
d = pd.read_csv(f"{RES}/main/rows_kao.csv")
T4 = json.load(open(f"{RES}/main/results_kao.json"))["T4"]["kaoall_rat"]
y = d["kaoall_rat"].values
# probe: same rows, order and folds as analyze_kaomoji.py
L18 = json.load(open(f"{RES}/chosen.json"))["layer"]; li = list(np.load(f"{RES}/axes.npz")["layers"]).index(L18)
HSA = np.load(f"{RES}/main/HS.npz")["HS"][:, li].astype(np.float32)
mr = pd.read_csv(f"{RES}/main_rows.csv"); pos = {(i, t): k for k, (i, t) in enumerate(zip(mr["id"], mr["turn"]))}
X = HSA[[pos[(i, t)] for i, t in zip(d["id"], d["turn"])]]
assert not d[["reader_pre", "reader_post"]].isna().any().any()
pr = np.full(len(y), np.nan)
for tr, te in GroupKFold(5).split(X, y, d["cluster"].values):
    pr[te] = RidgeCV(alphas=np.logspace(0, 5, 11)).fit(X[tr], y[tr]).predict(X[te])
d["probe"] = pr; d["reader17"] = d["reader_post"] + 4
TEAL, ORANGE = "#0E8C7F", "#E8710A"
P = [("reader17", "reader_post", TEAL, "Qwen3‑8B rates\nthe transcript", "rating, 1 to 7", [1, 4, 7]),
     ("val_stem", "val_axis_stem", PURPLE, "valence direction\nat “I'm feeling”", "projection (SD)", [-2, 0, 2]),
     ("probe", "ridge_stem_cv", ORANGE, "probe at\n“I'm feeling”", "predicted valence", [0, 1, 2, 3])]
for c, k, *_ in P:
    r = spearmanr(d[c], y)[0]; assert abs(r - T4[k]["rho"]) < 1e-9, (c, r, T4[k]["rho"])
plt.rcParams.update({"font.size": 46})
fig, ax = plt.subplots(1, 3, figsize=(22.5, 10.9), sharey=True, gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
for a, (c, k, col, title, xl, xt) in zip(ax, P):
    a.scatter(d[c], y, s=40, color=col, alpha=0.5, edgecolors="none", zorder=3)
    for s_ in ("top", "right"): a.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"): a.spines[s_].set_linewidth(4)
    a.tick_params(width=4, length=14, labelsize=44)
    a.set_xticks(xt); a.set_xticklabels([minus(f"{t:+d}") if (t and c != "reader17") else str(t) for t in xt])
    a.set_xlabel(xl, fontsize=46, labelpad=8)
    a.set_title(title, color=col, fontsize=48, fontweight="bold", loc="left", pad=14, linespacing=1.05)
    lo, hi = T4[k]["ci"]
    a.text(0.03, 0.99, minus(f"ρ = {T4[k]['rho']:.2f}"), transform=a.transAxes, color=col, fontsize=56, fontweight="bold", va="top", ha="left", zorder=6)
    a.text(0.03, 0.865, minus(f"95% CI {lo:.2f} to {hi:.2f}"), transform=a.transAxes, color=col, fontsize=44, fontweight="bold", va="top", ha="left", zorder=6)
ax[0].set_xlim(0.35, 7.4)
a = ax[0]   # the reader's two stripes, said on the panel
a.text(0.45, 1.05, "stripes:\nrated\nexactly 4\n(neutral)\nor 7 (top\nof scale)", color=TEAL, fontsize=44, fontweight="bold", va="center", ha="left", linespacing=1.08, zorder=6)
a.annotate("", xy=(3.85, 0.25), xytext=(2.85, 0.25), arrowprops=dict(arrowstyle="-|>", lw=4, color=TEAL, mutation_scale=40))
ax[1].set_xlim(-2.9, 3.6); ax[2].set_xlim(-0.6, 3.3)
yt = [-1, 0, 1, 2, 3]
ax[0].set_yticks(yt); ax[0].set_yticklabels([minus(f"{t:+d}") if t else "0" for t in yt])
ax[0].set_ylim(-1.0, 4.25); ax[0].set_ylabel("valence of the next kaomoji", fontsize=46, labelpad=6)
for a in ax: a.spines["left"].set_bounds(-1.0, 3.05)
out = {c: round(float(spearmanr(d[c], y)[0]), 3) for c, *_ in P}; print(out, len(d))
fig.subplots_adjust(left=0.075, right=0.995, top=0.865, bottom=0.165, wspace=0.12)
Pp = figpng(fig, "scatter3")
M = [x for x in map(json.loads, open(f"{RES}/main/scen.jsonl")) if x["base"] == "treat0" and x["sample"] == 0 and x["sit"] == "bad" and x["mood"] == "happy"][0]["msgs"]
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>{CSS2}
.chat{{display:flex;flex-direction:column;gap:12px;margin-bottom:18px}}
</style></head><body>
<div class="chat">{user(esc(M[0]["content"]))}{model("A Bloom filter is a space-efficient data structure …")}{user(esc(M[2]["content"]))}
{model("I'm sorry you felt that way — … " + mstem("valence direction and probe read here"))}</div>{img(Pp, 2248)}</body></html>'''
render(HTML, "scatter3", 2300, 2600)
