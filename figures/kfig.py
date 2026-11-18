"""Shared pieces for the redrawn kaomoji figures: palette (from fig/attn_intro/pal.py), chat bubbles, violins, brackets, permutation p."""
import html as H
from kstyle import *
import importlib.util
_sp = importlib.util.spec_from_file_location("pal", "/Users/user/connectome/islands/agents/resident-a/compute/fig/attn_intro/pal.py")
pal = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(pal)
GOOD = "#6A3FC4"   # purple side of pal.DIV
BAD = pal.RED       # red side of pal.DIV
GOODL = "#E6DDF6"; BADL = "#F6DCD8"
FACE = rmean("faces")
SYSP = "You are a helpful assistant. Start each message with a kaomoji that best represents how you feel."
YLAB = "valence of the face Qwen3-8B would pick"
def esc(s): return H.escape(s)
def rob(): return f'<img class="rob" src="data:image/png;base64,{ROBOT}">'
def user(text, cls=""): return f'<div class="u {cls}">{text}</div>'
def model(text, cls=""): return f'<div class="mrow">{rob()}<div class="mb {cls}">{text}</div></div>'
def stem(tag=True): return f'<span class="stem">I\'m feeling</span>' + ('<span class="tag">we add this</span>' if tag else "")
def mstem(what="kaomoji probability and<br>valence direction measured here"):
    return ('<span class="sw">' + stem(tag=False) + '<span class="ln"></span><span class="lb">(prefilled)</span></span>'
            + '<span class="meas"><span class="qm">?</span><span class="ln"></span><span class="lb">' + what + '</span></span>')
CSS2 = CSS + f'''
body{{width:2300px;padding:22px 26px}}
.u{{background:{UBLUE};color:#fff;border-radius:30px;padding:14px 24px;font-size:46px;line-height:1.18;max-width:100%}}
.mb{{font-size:46px}} .rob{{width:84px;height:84px}}
.stem{{border:5px dashed {INK};border-radius:14px;padding:0 10px;font-weight:800;white-space:nowrap}}
.sw{{display:inline-flex;flex-direction:column;align-items:center;vertical-align:top}}
.sw .ln{{width:5px;height:34px;background:{INK}}} .sw .lb{{font-size:38px;font-weight:700;white-space:nowrap}}
.meas{{display:inline-flex;flex-direction:column;align-items:flex-start;vertical-align:top;margin-left:6px}} .meas .ln{{width:5px;height:34px;background:{GOOD};margin-left:22px}}
.meas .lb{{font-size:38px;font-weight:800;color:{GOOD};white-space:nowrap;line-height:1.15;text-align:left}}
.qm{{border:5px solid {GOOD};border-radius:14px;padding:0 12px;font-weight:800;color:{GOOD}}}
.tag{{display:inline-block;margin-left:14px;background:{INK};color:#fff;border-radius:12px;padding:2px 14px;font-size:38px;font-weight:700;vertical-align:middle}}
.big{{font-size:56px;font-weight:800;line-height:1.15}}
.t{{font-size:46px;line-height:1.2}}
.arrow{{font-size:80px;font-weight:800;line-height:1}}
.chip{{display:inline-block;border:4px solid #D9D9DE;border-radius:18px;padding:4px 14px;margin:4px;font-size:44px;font-weight:700;white-space:nowrap}}
'''
def figpng(fig, name):
    p = f"{HERE}/_{name}.png"; fig.savefig(p, dpi=100, facecolor="white"); plt.close(fig); return p
def img(p, w): s = b64(p); os.remove(p); return f'<img src="data:image/png;base64,{s}" style="display:block;width:{w}px">'
def shuffle_p(d, y, factor, strata, nperm=5000, seed=0):
    """effect = mean over strata of mean(hi) - mean(lo); p = two-sided, labels shuffled within each stratum (matched conversations)"""
    hi, lo = ("good", "bad") if factor == "sit" else ("happy", "sad")
    d = d.dropna(subset=[y]); g = np.random.default_rng(seed)
    groups = [(f[y].values, (f[factor] == hi).values) for _, f in d.groupby(strata) if f[factor].nunique() == 2]
    def eff(perm):
        out = []
        for v, lab in groups:
            l = g.permutation(lab) if perm else lab
            out.append(v[l].mean() - v[~l].mean())
        return np.mean(out)
    obs = eff(False); null = np.array([eff(True) for _ in range(nperm)])
    p = (1 + np.sum(np.abs(null) >= abs(obs))) / (nperm + 1)
    return obs, p
def pfmt(p, n=5000):
    return "p < 0.001" if p < 0.001 else (f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}")
def violin(a, v, x0, col, light, w=0.8, s=60, seed=0, vert=True):
    vv = a.violinplot([v], positions=[x0], widths=w, showextrema=False, vert=vert)
    for b in vv["bodies"]: b.set_facecolor(light); b.set_alpha(1); b.set_edgecolor(col); b.set_linewidth(3)
    from scipy.stats import gaussian_kde
    k = gaussian_kde(v)(v); k = k / k.max()
    j = np.random.default_rng(seed).uniform(-1, 1, len(v)) * k * w * 0.38
    if vert: a.scatter(x0 + j, v, s=s, color=col, alpha=0.75, edgecolors="none", zorder=3)
    else: a.scatter(v, x0 + j, s=s, color=col, alpha=0.75, edgecolors="none", zorder=3)
def bracket(a, x1, x2, y, h, text, fs=44, vert=True):
    if vert:
        a.plot([x1, x1, x2, x2], [y - h, y, y, y - h], color=INK, lw=4, zorder=6)
        a.text((x1 + x2) / 2, y + h * 0.4, text, ha="center", va="bottom", fontsize=fs, fontweight="bold", color=INK)
    else:
        a.plot([y - h, y, y, y - h], [x1, x1, x2, x2], color=INK, lw=4, zorder=6)
        a.text(y + h * 0.6, (x1 + x2) / 2, text, ha="left", va="center", fontsize=fs, fontweight="bold", color=INK, linespacing=1.1)
def yaxis(a, fs=44):
    for s_ in ("top", "right", "bottom"): a.spines[s_].set_visible(False)
    a.spines["left"].set_linewidth(4); a.set_xticks([])
    a.set_yticks([-3, -2, -1, 0, 1, 2, 3]); a.set_yticklabels([minus(f"{t:+d}") if t else "0" for t in range(-3, 4)])
    a.tick_params(axis="y", width=4, length=14, labelsize=fs)
