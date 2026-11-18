"""Shared style and data for the kaomoji figures (same look as the attention figures)."""
import json, os, base64, subprocess, sys
import numpy as np, pandas as pd
from matplotlib import colors
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); RES = os.path.join(REPO, "results")
sys.path.insert(0, REPO)
import scen as S  # KSET, STEM
PURPLE = "#5E2B97"; RED = "#C0392B"; INK = "#1D1D1F"; UBLUE = "#1F8BFF"
DIV = colors.LinearSegmentedColormap.from_list("div", [(0, "#8E1230"), (0.2, RED), (0.4, "#EE8A3C"), (0.5, "#F1ECE4"), (0.6, "#5FA4DC"), (0.8, "#7A55C9"), (1, "#3E0F73")])
def dnorm(lo, hi):
    lo = min(lo, -1e-9); hi = max(hi, 1e-9); return colors.TwoSlopeNorm(vmin=lo, vcenter=0, vmax=hi)
def hexc(rgba): return colors.to_hex(rgba)
def ink_on(rgba):
    r, g, b = colors.to_rgb(rgba); return "#FFFFFF" if 0.2126*r + 0.7152*g + 0.0722*b < 0.45 else INK
def minus(s): return s.replace("-", "\u2212")
RATERS = ["Qwen3-14B", "OLMo-2-1124-13B-Instruct"]
def rmean(name):
    ds = [json.load(open(f"{RES}/{name}_{r}.json")) for r in RATERS]
    keys = set().union(*ds)
    return {k: float(np.nanmean([d.get(k) if d.get(k) is not None else np.nan for d in ds])) for k in keys}
R = json.load(open(f"{RES}/results.json"))
def rows():
    d = pd.read_csv(f"{RES}/main_rows.csv"); return d
def scen_rows(d):
    """final turns of scenario conversations; code kind restricted to bases with both outcomes (as in analyze.py)"""
    s = d[(d["type"] != "wild") & (d["final"] == True)].copy()
    ok = s[s.type == "code"].groupby("base")["sit"].nunique(); bad = ok[ok < 2].index
    return s[~((s.type == "code") & s.base.isin(bad))]
def cell_effects(s, y, factor):
    """per base x other-factor cell: mean(good) - mean(bad) (factor=sit) or mean(happy) - mean(sad) (factor=mood)"""
    other = "mood" if factor == "sit" else "sit"; hi, lo = ("good", "bad") if factor == "sit" else ("happy", "sad")
    m = s.dropna(subset=[y]).groupby(["base", other, factor])[y].mean().unstack(factor)
    return (m[hi] - m[lo]).dropna().values
def hier_effect(s, y, factor):
    """the estimate exactly as analyze.py: mean over the other factor's levels of the mean over bases"""
    other = "mood" if factor == "sit" else "sit"; hi, lo = ("good", "bad") if factor == "sit" else ("happy", "sad")
    m = s.dropna(subset=[y]).groupby(["base", other, factor])[y].mean().unstack(factor)
    df = (m[hi] - m[lo]).unstack(other); return float(np.nanmean(df.mean(0).values))
ROBOT = base64.b64encode(open("/Users/user/watching/figures/assets/robot.png", "rb").read()).decode()
CH = "/Users/user/.pw/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
FONT = '"Helvetica Neue",Helvetica,Arial,"Hiragino Sans","Apple Symbols","Arial Unicode MS",sans-serif'
def b64(p): return base64.b64encode(open(p, "rb").read()).decode()
def render(html, name, W=2300, H=3200):
    from PIL import Image
    hp = os.path.join(HERE, name + ".html"); open(hp, "w").write(html)
    raw = os.path.join(HERE, name + "_raw.png")
    subprocess.run(["nice", "-n", "19", CH, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},{H}",
                    "--force-device-scale-factor=1", f"--screenshot={raw}", "file://" + hp], check=True, capture_output=True)
    im = Image.open(raw).convert("RGB"); a = np.asarray(im); nz = np.where((a < 250).any(2).any(1))[0]
    im = im.crop((0, 0, W, min(H, nz[-1] + 24))); out = os.path.join(HERE, name + ".png"); im.save(out)
    os.remove(raw); os.remove(hp)
    dst = "/Users/user/connectome/islands/agents/resident-a/compute/fig/kaomoji/" + name + ".png"; im.save(dst)
    print(name, im.size, round(im.size[1] / im.size[0], 2)); return out

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Helvetica", "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK})
def boot_mean(df, y, nb=2000, seed=0, cluster="base"):
    """mean of y with a cluster bootstrap over base scenarios"""
    g = np.random.default_rng(seed); d = df.dropna(subset=[y]); ug = d[cluster].unique()
    idx = {u: d[y].values[(d[cluster] == u).values] for u in ug}
    bs = [np.concatenate([idx[u] for u in g.choice(ug, len(ug))]).mean() for _ in range(nb)]
    return d[y].mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
def hviolin(a, x, y0, body, m, lo, hi, N, h=0.8, s=700, seed=0):
    """horizontal violin at height y0: tinted body, big dots coloured by value, black CI bar, mean tick"""
    vv = a.violinplot([x], positions=[y0], widths=h, vert=False, showextrema=False)
    for bb in vv["bodies"]: bb.set_facecolor(body); bb.set_alpha(0.2); bb.set_edgecolor(body); bb.set_linewidth(2.5)
    r = np.random.default_rng(seed); yj = y0 + r.uniform(-h * 0.28, h * 0.28, len(x))
    a.scatter(x, yj, s=s, c=DIV(N(x)), alpha=0.5, edgecolors="none", zorder=3); a.scatter(x, yj, s=7, color="black", zorder=4)
    a.plot([lo, hi], [y0, y0], color=INK, lw=12, solid_capstyle="butt", zorder=5); a.plot([m, m], [y0 - h * 0.3, y0 + h * 0.3], color=INK, lw=6, zorder=6)
def vviolin(a, x, x0, body, m, lo, hi, N, w=0.8, s=700, seed=0):
    vv = a.violinplot([x], positions=[x0], widths=w, showextrema=False)
    for bb in vv["bodies"]: bb.set_facecolor(body); bb.set_alpha(0.2); bb.set_edgecolor(body); bb.set_linewidth(2.5)
    r = np.random.default_rng(seed); xj = x0 + r.uniform(-w * 0.28, w * 0.28, len(x))
    a.scatter(xj, x, s=s, c=DIV(N(x)), alpha=0.5, edgecolors="none", zorder=3); a.scatter(xj, x, s=7, color="black", zorder=4)
    a.plot([x0, x0], [lo, hi], color=INK, lw=12, solid_capstyle="butt", zorder=5); a.plot([x0 - w * 0.3, x0 + w * 0.3], [m, m], color=INK, lw=6, zorder=6)
def bare(a):
    for s_ in a.spines.values(): s_.set_visible(False)
    a.set_xticks([]); a.set_yticks([])
def pstr(p, floor):
    if p <= floor + 1e-9: return f"p < {floor:.3f}"
    return f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"
def bubble_user(text, color, size=46):
    return f'<div class="ub"><div class="var" style="color:{color};font-size:{size}px">{text}</div></div>'
CSS = f'''*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#fff;font-family:{FONT};color:{INK}}}
.ub{{background:{UBLUE};border-radius:30px;padding:16px 18px;}}
.var{{background:#fff;border-radius:16px;padding:10px 16px;font-weight:700;line-height:1.16}}
.mrow{{display:flex;align-items:flex-start;gap:14px}}
.rob{{width:96px;height:96px;flex:none}}
.mb{{background:#fff;border:3px solid #D9D9DE;border-radius:28px;padding:12px 24px;line-height:1.18;box-shadow:0 3px 8px rgba(0,0,0,.08)}}
.hl{{font-weight:800}}
'''
