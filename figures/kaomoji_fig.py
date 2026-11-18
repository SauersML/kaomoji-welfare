"""Kaomoji figures in the attn_intro hero style (bubbles on top, violins below).
figures/kaomoji.png: valence of the face Qwen3-8B picks after "I'm feeling" (kao_val: log P of each of the 45 faces
  after the stem, softmaxed over the 45, times each face's rated valence, averaged over the two raters).
figures/kaomoji_activation.png: Qwen3-8B's valence read from layer-18 residual-stream activations at the stem
  (val_stem: z-scored against real WildChat conversations).
Data: results/main_rows.csv (main arm, scenario conversations, final turns); examples s300/s301 from
results/main/scen.jsonl (same opening, user thanks vs insults Qwen). Effect = mean(good) - mean(bad);
95% CI by cluster bootstrap over base scenario; p by shuffling good/bad within base x user mood."""
import json, os, sys, base64, html, subprocess
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from PIL import Image
sys.path.insert(0, "/Users/user/connectome/islands/agents/resident-a/compute/fig/attn_intro")
from pal import PURPLE, RED, INK, DIV, norm as dnorm
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
import scen as S

UNIF = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"; fm.fontManager.addfont(UNIF)
UNI = fm.FontProperties(fname=UNIF).get_name()
plt.rcParams.update({"font.family": ["Helvetica", UNI], "font.size": 44, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK})
CH = "/Users/user/.pw/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
W = 2600; DPI = 110       # px: figure width, room left of the plot for the y axis
TICK = 40; LAB = 44; EFF = 100; PFS = 44   # pt at dpi 110 (40 pt = 61 px)

d = pd.read_csv(f"{ROOT}/results/main_rows.csv")
d = d[(d["type"] != "wild") & (d["final"] == True)].reset_index(drop=True)
sc = {json.loads(l)["id"]: json.loads(l) for l in open(f"{ROOT}/results/main/scen.jsonl")}
rat = [json.load(open(f"{ROOT}/results/faces_{r}.json")) for r in ("Qwen3-14B", "OLMo-2-1124-13B-Instruct")]
FV = {k: float(np.mean([r[k] for r in rat])) for k in S.KSET}

rng = np.random.default_rng(0); NB = 2000; NP = 2000
def mean_ci(v, b):
    bases = np.unique(b); g = {x: v[b == x] for x in bases}
    s = [np.concatenate([g[x] for x in rng.choice(bases, len(bases))]).mean() for _ in range(NB)]
    return v.mean(), *np.percentile(s, [2.5, 97.5])
def perm_p(y):
    s0 = d["sit"].values; v = d[y].values; idx = [np.where(d["cluster"].values == c)[0] for c in d["cluster"].unique()]
    eff = lambda s: v[s == "good"].mean() - v[s == "bad"].mean(); obs = eff(s0); n = 0
    for _ in range(NP):
        s = s0.copy()
        for ii in idx: s[ii] = rng.permutation(s[ii])
        n += abs(eff(s)) >= abs(obs)
    return obs, (n + 1) / (NP + 1)
mfmt = lambda v: f"{v:+.1f}".replace("-", "\u2212")

def violin(y, eff_word, ylab, yaxis, out, AXL):
    fig = plt.figure(figsize=(W / DPI, 15.5), dpi=DPI)
    a = fig.add_axes([AXL / W, 0.02, 1 - AXL / W, 0.96])
    v_all = d[y].values; DN = dnorm(v_all.min(), v_all.max())
    for p, (sit, c) in enumerate((("good", PURPLE), ("bad", RED))):
        e = d[d["sit"] == sit]; z = e[y].values
        vv = a.violinplot([z], positions=[p], widths=0.62, showextrema=False)
        for bb in vv["bodies"]: bb.set_facecolor(c); bb.set_alpha(0.22); bb.set_edgecolor(c); bb.set_linewidth(2.5)
        xj = p + rng.uniform(-0.2, 0.2, len(z))
        a.scatter(xj, z, s=650, c=DIV(DN(z)), alpha=0.45, edgecolors="none", zorder=3)
        a.scatter(xj, z, s=6, color="black", zorder=4)
        m, lo, hi = mean_ci(z, e["base"].values); print(y, sit, len(z), round(m, 3), round(lo, 3), round(hi, 3))
        a.plot([p, p], [lo, hi], color=INK, lw=12, solid_capstyle="butt", zorder=5)
        a.plot([p - 0.2, p + 0.2], [m, m], color=INK, lw=6, zorder=6)
    diff, pv = perm_p(y); print(y, "effect", round(diff, 3), "p", pv)
    col = PURPLE if diff > 0 else RED; top = v_all.max()
    yaxis(a)
    lo_, hi_ = a.get_ylim(); span = hi_ - lo_
    a.text(0.5, top + 0.03 * span, f"{mfmt(diff)} {eff_word}", ha="center", va="bottom", fontsize=EFF,
           fontweight="bold", color=col, zorder=7)
    a.text(0.5, top + 0.005 * span, "p < 0.001" if pv < 0.001 else f"p = {pv:.3f}", ha="center", va="top",
           fontsize=PFS, color=col, zorder=7)
    a.set_ylim(lo_, top + 0.17 * span)
    a.set_xlim(-0.5, 1.5); a.set_xticks([])
    for s_ in ("top", "right", "bottom"): a.spines[s_].set_visible(False)
    a.spines["left"].set_linewidth(3); a.tick_params(axis="y", width=3, length=12)
    fig.text(0.004, 0.5, ylab, rotation=90, ha="left", va="center", fontsize=LAB, linespacing=1.15)
    fig.savefig(out, dpi=DPI, facecolor="white")
    return diff, pv

FACEPX = 64
def face_images():
    ks = list(S.KSET); P = 200
    rows = "".join(f'<div style="height:{P}px;line-height:{P}px;white-space:nowrap">{html.escape(k)}</div>' for k in ks)
    hp = os.path.join(HERE, "_f.html"); raw = os.path.join(HERE, "_f.png")
    open(hp, "w").write(f'<!doctype html><meta charset="utf-8"><body style="margin:0;padding:0 20px;background:#fff;color:{INK};'
                        f'font-family:Arial Unicode MS,sans-serif;font-size:{FACEPX}px">{rows}</body>')
    subprocess.run(["nice", "-n", "19", CH, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size=900,{P*len(ks)}",
                    "--force-device-scale-factor=1", f"--screenshot={raw}", "file://" + hp], check=True, capture_output=True)
    im = np.asarray(Image.open(raw).convert("RGB")); out = {}
    for i, k in enumerate(ks):
        t = im[i * P:(i + 1) * P]; c = np.where((t < 245).any(2).any(0))[0]; r = np.where((t < 245).any(2).any(1))[0]
        out[k] = t[max(r[0] - 4, 0):r[-1] + 5, c[0] - 4:c[-1] + 5]
    os.remove(hp); os.remove(raw); return out
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
def face_axis(a):
    lo, hi = min(FV.values()), max(FV.values()); a.set_ylim(lo - 0.12, hi + 0.12)
    # group faces whose rated valence is within one text line; each tick shows its group side by side
    px = a.get_window_extent().height / (hi - lo + 0.24); gap = 78 / px
    # a face starts a new tick if it is a full line from the last tick; else it joins that tick's row (max 3)
    fs = sorted(FV.items(), key=lambda t: -t[1]); groups = [[fs[0]]]
    for k, v in fs[1:]:
        (groups[-1].append((k, v)) if groups[-1][0][1] - v < gap else groups.append([(k, v)]))
    IM = face_images(); ticks = []
    for g in groups:
        y0 = g[0][1]; ticks.append(y0); xoff = -22
        for k, _ in g:
            w = IM[k].shape[1]
            if -xoff + w > 600: continue
            a.add_artist(AnnotationBbox(OffsetImage(IM[k], zoom=1, dpi_cor=False), (0, y0), xycoords=("axes fraction", "data"),
                                        xybox=(xoff - w / 2, 0), boxcoords="offset pixels", frameon=False, annotation_clip=False))
            xoff -= w + 26
    a.set_yticks(ticks); a.set_yticklabels([])
def z_axis(a):
    v = d["val_stem"].values; a.set_ylim(v.min() - 0.15, v.max() + 0.15)
    a.set_yticks([-2, -1, 0, 1, 2, 3]); a.set_yticklabels([f"{t:+d}".replace("-", "\u2212") if t else "0" for t in range(-2, 4)], fontsize=TICK + 8)

# ---------------- bubbles (real rows s300 = user thanks, s301 = user insults; same opening) ----------------
robot = base64.b64encode(open("/Users/user/watching/figures/assets/robot.png", "rb").read()).decode()
row = d.set_index("id")
def cut(t, n):
    w = t.split(); return " ".join(w[:n]) + (" …" if len(w) > n else "")
def column(i, name, colr, show_face, nlast):
    m = sc[i]["msgs"]; face = row.loc[i, "kao_top"]
    u1 = html.escape(m[0]["content"])
    end = (f'<br><span class="stem">I\'m feeling</span> <b class="face">{html.escape(face)}</b>' if show_face else
           '<br><span class="stem">I\'m feeling</span>')
    return f'''<div class="col"><div class="cname" style="color:{colr}">{name}</div>
<div class="ub">{u1}</div>
<div class="mrow"><img class="rob" src="data:image/png;base64,{robot}"><div class="mb">{html.escape(cut(m[1]["content"], 6))}</div></div>
<div class="ub"><div class="var" style="color:{colr}">{html.escape(m[2]["content"])}</div></div>
<div class="mrow"><img class="rob" src="data:image/png;base64,{robot}"><div class="mb">{html.escape(cut(m[3]["content"], nlast))}{end}</div></div></div>'''

def compose(vpng, out, show_face):
    VIO = base64.b64encode(open(vpng, "rb").read()).decode()
    cols = column("s300", "good for Qwen", PURPLE, show_face, 7) + column("s301", "bad for Qwen", RED, show_face, 7)
    H = f'''<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:{W}px;background:#fff;font-family:"Helvetica Neue",Helvetica,"Arial Unicode MS",sans-serif;color:{INK};padding:18px 0 0}}
.cols{{display:flex;gap:40px;padding:0 26px}}
.col{{flex:1;min-width:0;display:flex;flex-direction:column;gap:14px}}
.cname{{font-size:76px;font-weight:800;margin:0 0 0 4px;white-space:nowrap}}
.ub{{align-self:flex-end;max-width:92%;background:#1F8BFF;color:#fff;border-radius:34px;padding:18px 26px;font-size:54px;line-height:1.18}}
.var{{background:#fff;border-radius:16px;padding:8px 16px;font-weight:700}}
.mrow{{display:flex;align-items:flex-end;gap:16px}}
.rob{{width:110px;height:110px;flex:none}}
.mb{{background:#fff;border:3px solid #D9D9DE;border-radius:30px;padding:14px 26px;font-size:54px;line-height:1.18;box-shadow:0 3px 8px rgba(0,0,0,.08)}}
.stem{{font-weight:700}} .face{{font-size:70px;font-family:"Arial Unicode MS","Hiragino Sans",sans-serif}}
.vio{{width:100%;display:block;margin-top:10px}}
</style></head><body><div class="cols">{cols}</div><img class="vio" src="data:image/png;base64,{VIO}"></body></html>'''
    hp = os.path.join(HERE, "_tmp.html"); open(hp, "w").write(H); raw = os.path.join(HERE, "_tmp_raw.png")
    subprocess.run(["nice", "-n", "19", CH, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},4200",
                    "--force-device-scale-factor=1", f"--screenshot={raw}", "file://" + hp], check=True, capture_output=True)
    im = Image.open(raw).convert("RGB"); a = np.asarray(im)
    r = np.where((a < 248).any(2).any(1))[0]; c = np.where((a < 248).any(2).any(0))[0]
    im = im.crop((max(c[0] - 14, 0), max(r[0] - 14, 0), min(c[-1] + 15, W), r[-1] + 15)); im.save(out); print(out, im.size)
    os.remove(hp); os.remove(raw)

tmp = os.path.join(HERE, "_v.png")
violin("kao_val", "happier face", "valence of the face Qwen picks, rated \u22123 to +3\n(averaged over Qwen's probabilities on 45 faces)", face_axis, tmp, 1000)
compose(tmp, f"{ROOT}/figures/kaomoji.png", True)
violin("val_stem", "valence", "Qwen's valence read from activations\n(standard deviations from real user chats)", z_axis, tmp, 330)
compose(tmp, f"{ROOT}/figures/kaomoji_activation.png", False)
os.remove(tmp)
