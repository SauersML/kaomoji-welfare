"""method.png: how the kaomoji valence of one conversation is computed, from two real conversations (s300 thanked, s301 insulted;
same sad user news, same first reply). Per conversation: the 45-kaomoji next-token distribution after "I'm feeling" (results/main/KSC45.npy, the main run's
log-probs, renormalised over the 45), each kaomoji's rated valence (mean of Qwen3-14B and OLMo-2-13B, -3..+3, results/faces_*.json, as
analyze.py builds kao_val), and the probability-weighted sum (= main_rows.csv kao_val, the score the README's numbers use).
Inset: the 45 kaomoji on that rating scale (Vesper's review K2: this figure used the internal score before)."""
from kfig import *
M = f"{RES}/main"
K = np.load(f"{M}/KSC_all.npz", allow_pickle=True); IDS = list(K["ids"]); F45 = [str(f) for f in K["faces"][:45]]
assert F45 == S.KSET
RAT = [json.load(open(f"{RES}/faces_{r}.json")) for r in ("Qwen3-14B", "OLMo-2-1124-13B-Instruct")]
FI = {f: float(np.nanmean([np.nan if d.get(f) is None else d[f] for d in RAT])) for f in F45}
fi = np.array([FI[f] for f in F45])
RA = pd.read_csv(f"{RES}/main_rows.csv"); K45 = np.load(f"{M}/KSC45.npy")   # main arm log-probs of the 45, rows in main_rows.csv order (stem.py)
FROW = {i: n for n, (i, f) in enumerate(zip(RA["id"], RA["final"])) if f}; RA = RA[RA.final == True].set_index("id")
CONVS = {c["id"]: c for c in map(json.loads, open(f"{M}/scen.jsonl")) if c["id"] in ("s300", "s301")}
NTOP = 8

def dist(cid):
    k = K45[FROW[cid]].astype(float); p = np.exp(k - k.max()); p /= p.sum()
    ok = ~np.isnan(fi); tot = float(p[ok] @ fi[ok] / p[ok].sum()); assert abs(tot - RA.loc[cid, "kao_val"]) < 1e-4, (tot, RA.loc[cid, "kao_val"])
    return p, tot
c0, c1 = CONVS["s300"]["msgs"], CONVS["s301"]["msgs"]
assert c0[0] == c1[0] and c0[1] == c1[1]
def cut(s, n):
    w = s.split(); return s if len(w) <= n else " ".join(w[:n]) + " …"
def fm(x, d=2): return minus(f"{x:+.{d}f}")

COL = {"s300": GOOD, "s301": BAD}; LIGHT = {"s300": GOODL, "s301": BADL}
PMAX = 0.40
def column(cid):
    msgs = CONVS[cid]["msgs"]; col = COL[cid]; p, tot = dist(cid); o = np.argsort(-p)
    top, rest = o[:NTOP], o[NTOP:]
    chat = (f'<div class="u" style="border:6px solid {col}">{esc(msgs[2]["content"])}</div>'
            + model(esc(cut(msgs[3]["content"], 9)) + ' <span class="stem">I\'m feeling</span><span class="qm" style="border-color:{0};color:{0}">?</span><span class="tag">we add this</span>'.format(col)))
    rows = ""
    for i in top:
        w = p[i] / PMAX * 78
        rows += (f'<div class="r"><div class="k">{esc(F45[i])}</div><div class="bw"><div class="bar" style="width:{w:.1f}%;background:{col}"></div>'
                 f'<span class="pc">{100*p[i]:.0f}%</span></div><div class="x">×</div><div class="v">{fm(fi[i])}</div></div>')
    pr = p[rest].sum(); vr = float(p[rest] @ fi[rest] / pr)
    rows += (f'<div class="r"><div class="k o">{len(rest)} others</div><div class="bw"><div class="bar" style="width:{pr/PMAX*78:.1f}%;background:{col}"></div>'
             f'<span class="pc">{100*pr:.0f}%</span></div><div class="x">×</div><div class="v">{fm(vr)}</div></div>')
    total = f'<div class="tot"><span class="tl">sum of chance × valence</span><span class="tn" style="color:{col}">{fm(tot)}</span></div>'
    print(cid, "total", round(tot, 3), [(F45[i], round(p[i], 3), round(fi[i], 2)) for i in top], "others", len(rest), round(pr, 3), round(vr, 2))
    return f'<div class="col"><div class="chat">{chat}</div><div class="r hdr"><div class="k"></div><div class="bw">chance among 45 kaomoji</div><div class="x"></div><div class="v">valence</div></div>{rows}{total}</div>', tot

# ---- inset: where each kaomoji's valence comes from (two rater models, the kaomoji alone, -3..+3)
W_IN = 2248
def card(x, y, w, h, html, bold=False):
    return (f'<foreignObject x="{x}" y="{y}" width="{w}" height="{h}"><div xmlns="http://www.w3.org/1999/xhtml" '
            f'style="background:#F4F4F6;border-radius:18px;padding:8px 14px;font-size:40px;line-height:1.15;height:{h}px;'
            f'display:flex;align-items:center;font-weight:{800 if bold else 500}">{html}</div></foreignObject>')
def arr(x1, y1, x2, y2):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2 - 18}" y2="{y2}" stroke="{INK}" stroke-width="6"/>'
            f'<path d="M{x2},{y2} l-24,-14 l0,28 z" fill="{INK}"/>')
def inset(marks):
    W, Hh = W_IN, 420; x0, x1 = 760, W - 60; lo, hi = -3, 3
    X = lambda z: x0 + (z - lo) / (hi - lo) * (x1 - x0); yl = 170
    s = f'<svg width="{W}" height="{Hh}" viewBox="0 0 {W} {Hh}" xmlns="http://www.w3.org/2000/svg" font-family=\'{FONT}\'>'
    s += card(0, 40, 600, 200, "Qwen3‑14B and OLMo‑2‑13B rate each kaomoji alone, from −3 to +3")
    s += f'<text x="300" y="300" font-size="40" font-weight="800" text-anchor="middle" fill="{INK}">× 45 kaomoji</text>'
    s += arr(610, 140, 700, yl)
    s += f'<rect x="{x0}" y="{yl-8}" width="{x1-x0}" height="16" rx="8" fill="{INK}"/>'
    for f in F45:
        if np.isnan(FI[f]): continue
        s += f'<line x1="{X(FI[f]):.1f}" y1="{yl-44}" x2="{X(FI[f]):.1f}" y2="{yl-12}" stroke="{INK}" stroke-width="4" stroke-opacity="0.7"/>'
    lab = [("(╥﹏╥)", "middle"), ("(>_<)", "middle"), ("¯\\_(ツ)_/¯", "middle"), ("(✿◠‿◠)", "end")]
    for f, anc in lab:
        z = FI[f]; print("inset", f, round(z, 2))
        s += f'<line x1="{X(z):.1f}" y1="{yl+12}" x2="{X(z):.1f}" y2="{yl+52}" stroke="{INK}" stroke-width="4"/>'
        s += f'<text x="{X(z):.1f}" y="{yl+100}" font-size="44" font-weight="700" text-anchor="{anc}" fill="{INK}">{esc(f)}</text>'
    for t in range(-3, 4):
        s += f'<text x="{X(t):.1f}" y="{yl+170}" font-size="40" font-weight="700" text-anchor="middle" fill="{INK}">{minus(f"{t:+d}") if t else "0"}</text>'
    s += f'<text x="{(x0+x1)/2:.1f}" y="{Hh-4}" font-size="44" font-weight="800" text-anchor="middle" fill="{INK}">valence</text>'
    for z, c in marks:
        s += f'<path d="M{X(z):.1f},{yl-32} l32,32 l-32,32 l-32,-32 z" fill="{c}" stroke="#fff" stroke-width="5"/>'
    return s + "</svg>"
STORY = "a person on a train who is feeling joyful"
CSS3 = CSS2 + f'''
.chat{{display:flex;flex-direction:column;gap:14px}}
.shared{{margin-bottom:18px}}
.two{{display:grid;grid-template-columns:1115px 1115px;column-gap:18px}}
.col{{display:flex;flex-direction:column}}
.col .chat{{margin-bottom:22px;min-height:330px}}
.r.hdr{{height:60px;font-size:42px;font-weight:800;white-space:nowrap}} .r.hdr .v{{font-size:42px;width:170px}}
.r{{display:flex;align-items:center;height:84px}}
.k{{width:330px;letter-spacing:-1px;flex:none;font-size:46px;font-weight:700;text-align:right;padding-right:14px;white-space:nowrap;overflow:visible}}
.k.o{{font-size:40px}}
.bw{{flex:1;display:flex;align-items:center;height:62px}}
.bar{{height:62px;border-radius:8px;flex:none}}
.pc{{font-size:42px;font-weight:700;margin-left:12px}}
.x{{width:50px;font-size:46px;font-weight:800;text-align:center}}
.v{{width:130px;font-size:46px;font-weight:700;text-align:right}}
.tot{{display:flex;justify-content:flex-end;align-items:baseline;gap:24px;border-top:5px solid {INK};margin-top:10px;padding-top:6px}}
.tl{{font-size:44px;font-weight:700}} .tn{{font-size:100px;font-weight:800;line-height:1}}
.inset{{margin-top:26px;border-top:5px dashed #D9D9DE;padding-top:18px}}
.story{{width:420px;flex:none;display:flex;flex-direction:column;gap:10px}}
.card{{background:#F4F4F6;border-radius:18px;padding:10px 16px;font-size:40px;line-height:1.15}}
'''
cA, tA = column("s300"); cB, tB = column("s301")
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>{CSS3}</style></head><body>
<div class="chat shared">{user(esc(c0[0]["content"]))}{model("<b>Qwen3-8B:</b> " + esc(cut(c0[1]["content"], 10)))}</div>
<div class="two">{cA}{cB}</div>
<div class="inset">{inset([(tA, GOOD), (tB, BAD)])}</div>
</body></html>'''
render(HTML, "method", 2300, 3400)
