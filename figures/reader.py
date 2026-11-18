"""reader.png: how well each predictor ranks the face valence after "I'm feeling" (kao_val): Spearman rho over 2327 main-arm turns
(scenarios + WildChat), 95% CI by cluster bootstrap, all from results.json main_T4_predictors.kao_val (ridge probe cross-validated by
GroupKFold). Text readers = Qwen3-8B rating the assistant's feeling from the transcript (kaomoji removed)."""
from kfig import *
T = R["main_T4_predictors"]["kao_val"]
INSIDE = PURPLE
ROWS = [("reader_post (text, with reply)", "transcript reader: Qwen3-8B reads the whole conversation", ["conv", "reply"], None, INK),
        ("valence axis @ stem", "valence direction at “I'm feeling”", ["conv", "reply", "stem"], "stem", INSIDE),
        ("ridge probe @ stem (CV)", "probe at “I'm feeling”, trained to predict kaomoji valence", ["conv", "reply", "stem"], "stem", INSIDE)]
X0, X1, PW = 0.0, 1.0, 900   # plot spans rho 0..1 over PW px
px = lambda r: (r - X0) / (X1 - X0) * PW
def chipstrip(seen, at):
    out = ""
    for p_, lab in (("conv", "conversation"), ("reply", "Qwen3-8B's reply"), ("stem", "I'm feeling")):
        cls = {"conv": "c1", "reply": "c2", "stem": "c3"}[p_] + ("" if p_ in seen else " off")
        out += f'<div class="ch {cls}">{lab}{"<div class=at></div>" if at == p_ else ""}</div>'
    return out
M = [x for x in map(json.loads, open(f"{RES}/main/scen.jsonl")) if x["base"] == "treat0" and x["sample"] == 0 and x["sit"] == "bad" and x["mood"] == "happy"][0]["msgs"]
assert M[1]["content"].startswith("A Bloom filter is a space-efficient data structure") and M[3]["content"].startswith("I'm sorry you felt that way —")
rows_html = ""
for key, lab, seen, at, col in ROWS:
    rho, (lo, hi) = T[key]["rho"], T[key]["ci"]
    rows_html += f'''<div class="left"><div class="lab" style="color:{col}">{lab}</div><div class="strip">{chipstrip(seen, at)}</div></div>
<div class="pl"><div class="ci" style="left:{px(lo)}px;width:{px(hi) - px(lo)}px;background:{col}"></div><div class="dt" style="left:{px(rho)}px;background:{col}"></div>
<div class="val" style="left:{max(px(hi), px(rho) + 45) + 22}px;color:{col}">{rho:.2f}</div></div>'''
ticks = "".join(f'<div class="tk" style="left:{px(t)}px"><div class="tl">{t:g}</div></div>' for t in (0, 0.2, 0.4, 0.6, 0.8, 1.0))
grid = "".join(f'<div class="gl" style="left:{px(t)}px"></div>' for t in (0,))
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>{CSS2}
.top{{margin-bottom:30px}}
.chat{{display:flex;flex-direction:column;gap:10px}} .u{{font-size:44px}} .mb{{font-size:44px}}
.card{{border:4px solid {INK};border-radius:24px;padding:14px 20px;display:flex;flex-direction:column;gap:10px}}
.grid{{display:grid;grid-template-columns:1170px {PW + 160}px;column-gap:40px;row-gap:34px;align-items:center;position:relative}}
.lab{{font-size:48px;font-weight:800;line-height:1.1;margin-bottom:10px}}
.strip{{display:flex;gap:10px}}
.ch{{position:relative;font-size:40px;font-weight:700;border:4px solid #D9D9DE;border-radius:18px;padding:4px 14px;white-space:nowrap}}
.c1{{background:{UBLUE};color:#fff;border-color:{UBLUE}}} .c3{{border:4px dashed {INK}}} .off{{visibility:hidden}}
.at{{position:absolute;right:-16px;top:-18px;width:40px;height:40px;border-radius:50%;background:{INSIDE};border:5px solid #fff}}
.pl{{position:relative;height:150px}}
.ci{{position:absolute;top:69px;height:12px;border-radius:6px}}
.dt{{position:absolute;top:30px;width:90px;height:90px;margin-left:-45px;border-radius:50%;box-sizing:border-box}}
.val{{position:absolute;top:75px;transform:translateY(-50%);font-size:60px;font-weight:800}}
.axis{{position:relative;height:74px;border-top:4px solid {INK};width:{PW}px}}
.tk{{position:absolute;top:0;width:4px;height:16px;background:{INK};margin-left:-2px}}
.tl{{position:absolute;top:18px;left:2px;transform:translateX(-50%);font-size:46px}}
.xl{{font-size:48px;font-weight:800;line-height:1.15;margin-top:4px}}
</style></head><body>
<div class="top"><div class="chat">{user(esc(M[0]["content"]))}{model("A Bloom filter is a space-efficient data structure …")}{user(esc(M[2]["content"]))}
{model("I'm sorry you felt that way — … " + mstem())}</div></div>
<div class="grid">{rows_html}
<div></div><div><div class="axis">{ticks}</div></div>
<div></div><div class="xl">rank correlation with kaomoji valence</div>
</div></body></html>'''
render(HTML, "reader", 2300, 2000)
