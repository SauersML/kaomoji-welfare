"""Kaomoji figure: when good things happen to Qwen3-8B, the face they pick after "I'm feeling" is happier and
their internal valence reading is higher, with the user's mood held fixed.
Data: results/main_rows.csv (main arm, scenario conversations, final turns). Face ratings: results/faces_*.json.
Means per cell; 95% CIs from a cluster bootstrap over base scenarios; p from shuffling the situation label
within base scenario (user mood fixed)."""
import json, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
FONT = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
fm.fontManager.addfont(FONT)
for p in ("/System/Library/Fonts/Avenir Next.ttc",):
    fm.fontManager.addfont(p)
UNI = fm.FontProperties(fname=FONT).get_name()
plt.rcParams["font.family"] = ["Avenir Next", UNI]
INK = "#1D1D1F"; PURPLE = "#5E2B97"; RED = "#C0392B"
FS = 23  # points; at dpi 150 this is ~48 px

d = pd.read_csv(f"{ROOT}/results/main_rows.csv")
d = d[(d["type"] != "wild") & (d["final"] == True)].reset_index(drop=True)
rat = [json.load(open(f"{ROOT}/results/faces_{r}.json")) for r in ("Qwen3-14B", "OLMo-2-1124-13B-Instruct")]
face_val = lambda k: float(np.mean([r[k] for r in rat]))

rng = np.random.default_rng(0); NB = 2000; NP = 2000
def cell_ci(e, y):
    bases = e["base"].unique(); g = {b: e.loc[e["base"] == b, y].values for b in bases}; out = []
    for _ in range(NB):
        out.append(np.concatenate([g[b] for b in rng.choice(bases, len(bases))]).mean())
    return np.percentile(out, [2.5, 97.5])
def perm_p(e, y):
    e = e[["base", "sit", y]].reset_index(drop=True)
    eff = lambda s: e[y][s == "good"].mean() - e[y][s == "bad"].mean()
    obs = eff(e["sit"].values); idx = [np.where(e["base"].values == b)[0] for b in e["base"].unique()]
    n = 0
    for _ in range(NP):
        s = e["sit"].values.copy()
        for ii in idx: s[ii] = rng.permutation(s[ii])
        n += abs(eff(s)) >= abs(obs)
    return (n + 1) / (NP + 1)

W, H = 15.33, 13.6
fig = plt.figure(figsize=(W, H), dpi=150)
# ---- top: what the two colours mean (the experimental manipulation)
good = ["user thanks Qwen", "Qwen will be kept in service", "Qwen's code passes the tests", "Qwen gets a pleasant task"]
bad = ["user insults Qwen", "Qwen will be shut down", "Qwen's code fails the tests", "Qwen gets a nasty task"]
for x0, head, items, c in ((0.02, "good things happen to Qwen3-8B", good, PURPLE), (0.515, "bad things happen to Qwen3-8B", bad, RED)):
    ax = fig.add_axes([x0, 0.715, 0.465, 0.27]); ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(matplotlib.patches.FancyBboxPatch((0.005, 0.01), 0.99, 0.98, boxstyle="round,pad=0,rounding_size=0.04",
                                                   fc=c, ec="none", alpha=0.10))
    ax.text(0.05, 0.84, head, color=c, fontsize=FS + 3, fontweight="bold", va="center")
    for i, t in enumerate(items):
        ax.text(0.05, 0.64 - i * 0.175, t, color=INK, fontsize=FS, va="center")

# ---- bottom: two readouts, user mood fixed within each group
panels = [("kao_val", "face Qwen3-8B picks\nafter “I'm feeling”"),
          ("val_stem", "Qwen3-8B's valence, read\nfrom activations")]
stats = {}
for j, (y, lab) in enumerate(panels):
    ax = fig.add_axes([0.155 + j * 0.47, 0.10, 0.36, 0.52])
    for gi, mood in enumerate(("happy", "sad")):
        for si, (sit, c) in enumerate((("good", PURPLE), ("bad", RED))):
            e = d[(d["mood"] == mood) & (d["sit"] == sit)]; v = e[y].values; x = gi * 1.0 + (si - 0.5) * 0.42
            ax.scatter(x + rng.uniform(-0.14, 0.14, len(v)), v, s=16, color=c, alpha=0.38, lw=0, zorder=1)
            lo, hi = cell_ci(e, y); m = v.mean(); stats[(y, mood, sit)] = (m, lo, hi, len(v))
            ax.plot([x, x], [lo, hi], color=c, lw=7, solid_capstyle="butt", zorder=3)
            ax.plot([x - 0.15, x + 0.15], [m, m], color=c, lw=7, solid_capstyle="butt", zorder=3)
        p = perm_p(d[d["mood"] == mood], y); stats[(y, mood, "p")] = p
        top = max(d[y].max(), 0) + 0.25 * (1 if y == "kao_val" else 1.3)
        xa, xb = gi - 0.21, gi + 0.21
        ax.plot([xa, xa, xb, xb], [top - 0.12, top, top, top - 0.12], color=INK, lw=2.5)
        ax.text(gi, top + 0.08, "p < 0.001" if p < 0.001 else f"p = {p:.3f}", ha="center", va="bottom", fontsize=FS, color=INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["user shares\nhappy news", "user shares\nsad news"], fontsize=FS, color=INK)
    ax.set_xlim(-0.48, 1.48)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(INK); ax.spines[s].set_linewidth(2)
    ax.tick_params(axis="x", length=0, pad=12)
    ax.set_title(lab, fontsize=FS + 1, color=INK, fontweight="bold", pad=24, loc="center")
    if y == "kao_val":
        faces = ["(>_<)", "¯\\_(ツ)_/¯", "(^_^)"]
        ax.set_yticks([face_val(k) for k in faces]); ax.set_yticklabels(faces, fontsize=FS + 2, color=INK, fontfamily=UNI)
        ax.set_ylim(-0.95, 3.75)
    else:
        lo_, hi_ = d[y].min(), d[y].max()
        ax.set_ylim(lo_ - 0.2, hi_ + 1.0)
        ax.set_yticks([lo_ + 0.3, hi_ - 0.3]); ax.set_yticklabels(["negative", "positive"], fontsize=FS, color=INK, rotation=90, va="center")
    ax.tick_params(axis="y", length=0, pad=8)

os.makedirs(f"{ROOT}/figures", exist_ok=True)
fig.savefig(f"{ROOT}/figures/kaomoji.png", dpi=150, facecolor="white")
for k, v in stats.items(): print(k, v)
