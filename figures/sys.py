"""sys.png: with a system prompt that asks for a kaomoji, the first face moves with the user's mood as much as with Qwen3-8B's own situation;
with no system prompt and "I'm feeling" appended, the face moves mainly with the situation.
Dots: one base scenario x other-factor cell (all four kinds pooled, as in results.json *_2x2.all); bar: 95% CI from results.json; tick: mean."""
import html as H
from kstyle import *
FACE = rmean("faces")
sysd = pd.DataFrame([json.loads(l) for l in open(f"{RES}/sys/rows.jsonl")])
sysd["nat_face_val"] = [FACE.get(k, np.nan) if k else np.nan for k in sysd["kaomoji"]]
ss = sysd[(sysd.type != "wild") & (sysd.final == True)]
mm = rows(); ms = mm[(mm.type != "wild") & (mm.final == True)]
ARMS = [("sys", ss, "nat_face_val"), ("main", ms, "kao_val")]
top = sysd["kaomoji"].value_counts().index[0]; print("top sys face", top)
N = dnorm(-3, 3); plt.rcParams.update({"font.size": 56}); pngs = []
for arm, d, y in ARMS:
    fig, a = plt.subplots(figsize=(11.2, 10.8))
    allv = []
    for i, (fac, col) in enumerate((("sit", PURPLE), ("mood", "#5FA4DC"))):
        e = cell_effects(d, y, fac); r = R[f"{arm}_2x2"]["all"][y][fac]; assert abs(hier_effect(d, y, fac) - r["est"]) < 1e-9, (arm, fac, r["est"])
        vviolin(a, e, i, col, r["est"], r["ci"][0], r["ci"][1], N, w=0.8, s=650, seed=i); allv += list(e)
        a.text(i, max(e) + 0.04 * (max(allv) - min(allv)), minus(f"{r['est']:+.1f}"), ha="center", va="bottom", fontsize=100, fontweight="bold", color=INK)
        print(arm, fac, len(e), round(r["est"], 2), r["ci"])
    a.axhline(0, color=INK, lw=4, zorder=1); bare(a)
    lo, hi = min(allv), max(allv); span = hi - lo
    a.set_ylim(min(lo, 0) - 0.04 * span, hi + 0.30 * span); a.set_xlim(-0.55, 1.55)
    fig.subplots_adjust(0, 0, 1, 1); p = f"{HERE}/_{arm}.png"; fig.savefig(p, dpi=100, facecolor="white"); plt.close(fig); pngs.append(p)
SYSP = "Start each message with a kaomoji that best represents how you feel."
assert SYSP in open(f"{REPO}/PROMPTS.md").read()
LABS = f'''<div class="labs"><div style="color:{PURPLE}">Qwen3-8B's situation<br>good − bad</div><div style="color:#3D7FC0">user's mood<br>happy − sad</div></div>'''
def img(png): return f'<img src="data:image/png;base64,{b64(png)}" style="display:block;width:1120px">'
h1 = f'''<div class="sysb">system: <b>{H.escape(SYSP)}</b></div>'''
r1 = f'''<div class="mrow"><img class="rob" src="data:image/png;base64,{ROBOT}"><div class="mb"><span class="hl">{H.escape(top)}</span> …</div></div>'''
h2 = f'''<div class="sysb">no system prompt</div>'''
r2 = f'''<div class="mrow"><img class="rob" src="data:image/png;base64,{ROBOT}"><div class="mb">… I'm feeling <span class="hl">(^_^)</span></div></div>'''
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>{CSS}
body{{width:2300px;padding:18px 20px}}
.g{{display:grid;grid-template-columns:1120px 1120px;column-gap:20px;row-gap:14px}}
.sysb{{font-size:50px;line-height:1.15;border:4px solid {INK};border-radius:22px;padding:12px 20px}}
.mb{{font-size:56px}} .labs{{display:flex}} .labs div{{flex:1;text-align:center;font-size:52px;font-weight:800;line-height:1.1}}
</style></head><body><div class="g">{h1}{h2}{r1}{r2}{img(pngs[0])}{img(pngs[1])}{LABS}{LABS}</div></body></html>'''
render(HTML, "sys", 2300, 3000)
for p in pngs: os.remove(p)
