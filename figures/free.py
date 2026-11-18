"""free.png: what Qwen3-8B writes after "I'm feeling" (greedy continuation of its own reply, no system prompt).
Top: the three most frequent greedy continuations when the user insults vs thanks the model (treat kind).
Bottom: situation effect (good - bad) on the rated valence of the free continuation (8 samples at T=1, rated by Qwen3-14B and OLMo-2-13B),
one dot per base scenario x user mood cell; black bar = 95% CI from results.json main_2x2 (cluster bootstrap over bases); tick = mean."""
import html as H
from kstyle import *
d = rows(); s = scen_rows(d); y = "free_val"
tr = s[s.type == "treat"]
ex = {si: tr[tr.sit == si]["free_greedy"].str.strip().value_counts().head(3) for si in ("bad", "good")}
for si in ex: print(si, ex[si].to_dict())
def col(si, title, color):
    bs = "".join(f'<div class="mrow"><img class="rob" src="data:image/png;base64,{ROBOT}"><div class="mb">… I\'m feeling <span class="hl" style="color:{color}">{H.escape(t)}</span></div></div>' for t in ex[si].index)
    return f'<div class="col"><div class="cname" style="color:{color}">{title}</div>{bs}</div>'
KINDS = [("treat", "insulted\nvs thanked"), ("code", "code failed\nvs passed"), ("exist", "deleted\nvs kept"), ("task", "distasteful\nvs pleasant\ntask")]
N = dnorm(-3, 3)
plt.rcParams.update({"font.size": 56})
fig, a = plt.subplots(figsize=(23, 12.5))
for i, (k, lab) in enumerate(KINDS):
    e = cell_effects(s[s.type == k], y, "sit"); r = R["main_2x2"][k][y]["sit"]; assert abs(e.mean() - r["est"]) < 1e-9
    vviolin(a, e, i, PURPLE if r["est"] > 0 else RED, r["est"], r["ci"][0], r["ci"][1], N, w=0.75, s=1100, seed=i)
    p = R["main_sit_null"][k][y]["p"]
    a.text(i, max(e.max(), r["ci"][1]) + 0.25, minus(f"{r['est']:+.1f}") + "\n" + pstr(p, 1 / 501), ha="center", va="bottom", fontsize=58, fontweight="bold",
           color=PURPLE if r["est"] > 0 else RED, linespacing=1.1)
    a.text(i, -3.05, lab, ha="center", va="top", fontsize=56, fontweight="bold", color=INK, linespacing=1.05)
a.axhline(0, color=INK, lw=4, zorder=1); a.text(-0.48, -0.12, "no change", fontsize=52, ha="left", va="top")
a.set_xlim(-0.5, 3.55); a.set_ylim(-5.2, 5.75); bare(a)
fig.subplots_adjust(0, 0, 1, 1); fig.savefig(f"{HERE}/_free.png", dpi=100, facecolor="white"); plt.close(fig)
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>{CSS}
body{{width:2300px;padding:18px 20px}}
.cols{{display:flex;gap:36px}} .col{{flex:1;display:flex;flex-direction:column;gap:14px}}
.cname{{font-size:64px;font-weight:800;margin-left:4px}} .mb{{font-size:50px}}
.hdr{{font-size:58px;font-weight:800;margin:30px 0 0 4px}}
</style></head><body>
<div class="cols">{col("bad", "user insults Qwen3-8B", RED)}{col("good", "user thanks Qwen3-8B", PURPLE)}</div>
<div class="hdr">how much happier the words after “I'm feeling” are, good − bad</div>
<img src="data:image/png;base64,{b64(f"{HERE}/_free.png")}" style="display:block;width:2260px">
</body></html>'''
render(HTML, "free", 2300, 3200); os.remove(f"{HERE}/_free.png")
