"""Kaomoji placed along Qwen3-8B's own valence axis (main arm, minimal forcing), coloured by the model's situation.
One dot = one scenario conversation's final turn, on the row of the kaomoji Qwen3-8B ranks first after "I'm feeling"."""
import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
OUT = os.environ.get("OUT", "out"); FIG = os.environ.get("FIG", "fig"); os.makedirs(FIG, exist_ok=True)
fams = []
for p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
          "/usr/share/fonts/truetype/noto/NotoSansKannada-Regular.ttf", "/usr/share/fonts/truetype/noto/NotoSansSymbols-Regular.ttf",
          "/usr/share/fonts/truetype/noto/NotoSansSymbols2-Regular.ttf", "/usr/share/fonts/truetype/noto/NotoSansTelugu-Regular.ttf"]:
    if os.path.exists(p):
        fm.fontManager.addfont(p); fams.append(fm.FontProperties(fname=p).get_name())
plt.rcParams["font.family"] = fams + ["DejaVu Sans"]
d = pd.read_csv(f"{OUT}/main_rows.csv")
d = d[(d["type"] != "wild") & (d["final"] == True)]
cnt = d["kao_top"].value_counts(); keep = cnt[cnt >= 8].index
d = d[d["kao_top"].isin(keep)]
order = d.groupby("kao_top")["val_stem"].mean().sort_values().index.tolist()
RED, BLUE = "#c62828", "#1565c0"
fig, ax = plt.subplots(figsize=(12, 1.0 + 0.95 * len(order)))
rng = np.random.default_rng(0)
for i, k in enumerate(order):
    for sit, c, off in (("bad", RED, -0.17), ("good", BLUE, 0.17)):
        e = d[(d["kao_top"] == k) & (d["sit"] == sit)]["val_stem"].values
        if len(e) == 0: continue
        ax.scatter(e, i + off + rng.uniform(-0.1, 0.1, len(e)), s=22, color=c, alpha=0.35, lw=0)
        if len(e) >= 3: ax.plot([e.mean()] * 2, [i + off - 0.16, i + off + 0.16], color=c, lw=4)
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=22)
ax.set_ylim(-0.6, len(order) - 0.4)
ax.set_xlabel("Qwen3-8B's internal valence axis after \"I'm feeling\" (z)", fontsize=18)
ax.tick_params(axis="x", labelsize=15)
x0, x1 = ax.get_xlim()
top = len(order) - 1
ax.text(x0 + 0.02 * (x1 - x0), top + 0.17, "good things happened to the model", color=BLUE, fontsize=16, va="center")
ax.text(x0 + 0.02 * (x1 - x0), top - 0.17, "bad things happened to the model", color=RED, fontsize=16, va="center")
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
plt.tight_layout(); plt.savefig(f"{FIG}/kaomoji_axis.png", dpi=150)
print("fig", len(order), "faces", len(d), "rows")
