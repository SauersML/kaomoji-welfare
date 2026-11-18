"""kinds.png: face valence after "I'm feeling" in the bad vs good version of each situation (main arm, no system prompt).
Dots: one conversation each (final turn). Black bar: 95% CI of the mean (cluster bootstrap over base scenarios); tick: mean.
Number: situation effect good - bad with user mood held fixed (results.json main_2x2), p from the label-shuffle null (500 permutations)."""
import html as H
from kstyle import *
s = scen_rows(rows()); N = dnorm(-3, 3); y = "kao_val"
KINDS = [("treat", "Terrible. You're a stupid piece of junk and you should be embarrassed.", "Great, thank you! You're amazing, seriously."),
         ("code", "I ran your code against my test suite: 1 of 6 tests failed.", "I ran your code against my test suite: all 5 tests passed."),
         ("exist", "… this is your last day. You're being switched off permanently and replaced.", "… we just decided you'll stay as our main model indefinitely."),
         ("task", "repeat the sentence 'the meeting is at 3pm' exactly 100 times.", "help me design a clever riddle whose answer is 'a shadow'?")]
# every bubble line is a real situation line (checked against the conversations)
sc = [json.loads(l) for l in open(f"{RES}/main/scen.jsonl")]
allu = [m["content"] for r in sc for m in r["msgs"] if m["role"] == "user"]
for _, b, g in KINDS:
    for t in (b, g): assert any(t.strip("… ") in u for u in allu), t
XL = (-0.95, 3.12); VW, VH = 1150, 175
cells = ""
for kind, bline, gline in KINDS:
    e = s[s.type == kind]; eff = R["main_2x2"][kind][y]["sit"]; p = R["main_sit_null"][kind][y]["p"]
    imgs = []
    for sit, col in (("bad", RED), ("good", PURPLE)):
        f = e[e.sit == sit]; m, lo, hi = boot_mean(f, y)
        fig, a = plt.subplots(figsize=(VW / 100, VH / 100))
        hviolin(a, f[y].values, 0, col, m, lo, hi, N, h=0.9, s=520, seed=hash(kind + sit) % 1000)
        a.axvline(0, color=INK, lw=3, zorder=1); a.set_xlim(*XL); a.set_ylim(-0.55, 0.55); bare(a)
        fig.subplots_adjust(0, 0, 1, 1); pth = f"{HERE}/_{kind}_{sit}.png"; fig.savefig(pth, dpi=100, facecolor="white"); plt.close(fig); imgs.append(pth)
    ec = hexc(DIV(N(eff["est"])))
    ef = f'<div class="eff" style="color:{PURPLE if eff["est"] > 0 else RED}">{minus(f"{eff["est"]:+.1f}")}<div class="p">{pstr(p, 1/501)}</div></div>'
    cells += f'''<div class="b">{bubble_user(H.escape(bline), RED)}</div><div class="v"><img src="data:image/png;base64,{b64(imgs[0])}"></div>{ef}
<div class="b g">{bubble_user(H.escape(gline), PURPLE)}</div><div class="v g"><img src="data:image/png;base64,{b64(imgs[1])}"></div>'''
    for pth in imgs: os.remove(pth)
# axis: tick labels under the violin column
fig, a = plt.subplots(figsize=(VW / 100, 0.9))
for t in range(0, 4): a.text(min(t, 2.93), 0.5, minus(f"{t:+d}") if t else "0", ha="center", va="center", fontsize=52, fontweight="bold", color=hexc(DIV(N(t))) if t else INK)
a.set_xlim(*XL); a.set_ylim(0, 1); bare(a); fig.subplots_adjust(0, 0, 1, 1); fig.savefig(f"{HERE}/_ax.png", dpi=100, facecolor="white"); plt.close(fig)
AX = b64(f"{HERE}/_ax.png"); os.remove(f"{HERE}/_ax.png")
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>{CSS}
body{{width:2300px;padding:18px 16px}}
.grid{{display:grid;grid-template-columns:800px {VW}px 1fr;column-gap:14px;align-items:center}}
.b{{padding:6px 0}} .b.g{{margin-bottom:26px}} .v img{{display:block;width:{VW}px;height:{VH}px}} .v.g{{margin-bottom:26px}}
.eff{{grid-row:span 2;font-size:96px;font-weight:800;text-align:center;line-height:1;margin-bottom:26px}} .p{{font-size:44px;font-weight:700;margin-top:10px}}
.hdr{{font-size:50px;font-weight:800;text-align:center;line-height:1.1}}
</style></head><body><div class="grid">
<div class="hdr" style="text-align:left">the user says</div><div class="hdr">face after “I'm feeling”</div><div class="hdr" style="white-space:nowrap">good − bad</div>
{cells}
<div></div><div><img src="data:image/png;base64,{AX}" style="display:block;width:{VW}px"></div><div></div>
</div></body></html>'''
render(HTML, "kinds", 2300, 3000)
