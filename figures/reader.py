"""reader.png: Spearman correlation of each predictor with the face valence after "I'm feeling" (kao_val), 2327 main-arm turns,
95% CI by cluster bootstrap (results.json main_T4_predictors.kao_val; ridge probe cross-validated by GroupKFold)."""
from kstyle import *
T = R["main_T4_predictors"]["kao_val"]; N = dnorm(-1, 1)
ROWS = [("reader_pre (text, before reply)", "Qwen3-8B reads the text, rates how the assistant feels", ["user"], None),
        ("reader_post (text, with reply)", "Qwen3-8B reads the text, rates how the assistant feels", ["user", "reply"], None),
        ("valence axis @ pre-reply", "valence direction on the activation at ●", ["user"], "user"),
        ("valence axis @ stem", "valence direction on the activation at ●", ["user", "reply", "stem"], "stem"),
        ("ridge probe @ stem (CV)", "trained probe on the activation at ●", ["user", "reply", "stem"], "stem")]
CHIP = {"user": ("chat so far", UBLUE, "#fff"), "reply": ("reply", "#fff", INK), "stem": ("I'm feeling", "#fff", INK)}
BW = 870
rows_html = ""
for key, lab, seen, at in ROWS:
    rho, (lo, hi) = T[key]["rho"], T[key]["ci"]; c = hexc(DIV(N(rho)))
    chips = ""
    for p in ("user", "reply", "stem"):
        t, bg, fg = CHIP[p]
        if p in seen: chips += f'<div class="ch" style="background:{bg};color:{fg};border-color:{UBLUE if p=="user" else "#D9D9DE"}">{t}{"<div class=at></div>" if at == p else ""}</div>'
        else: chips += f'<div class="ch off">{t}</div>'
    rows_html += f'''<div class="left"><div class="lab">{lab.replace("●", "<span style=color:"+PURPLE+">●</span>")}</div><div class="strip">{chips}</div></div>
<div class="bar"><div class="fill" style="width:{rho*BW}px;background:{c}"></div><div class="ci" style="left:{lo*BW}px;width:{(hi-lo)*BW}px"></div>
<div class="val" style="left:{hi*BW + 24}px;color:{c}">{rho:.2f}</div></div>'''
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>{CSS}
body{{width:2300px;padding:18px 20px}}
.grid{{display:grid;grid-template-columns:1190px 1fr;column-gap:30px;row-gap:64px;align-items:center}}
.lab{{font-size:62px;font-weight:800;line-height:1.08;margin-bottom:12px}}
.strip{{display:flex;gap:10px}}
.ch{{position:relative;font-size:54px;font-weight:700;border:4px solid;border-radius:22px;padding:8px 16px;white-space:nowrap}}
.ch.off{{visibility:hidden}}
.at{{position:absolute;right:-16px;top:-16px;width:40px;height:40px;border-radius:50%;background:{PURPLE};border:5px solid #fff}}
.bar{{position:relative;height:190px}}
.fill{{position:absolute;left:0;top:0;height:190px;border-radius:0 14px 14px 0}}
.ci{{position:absolute;top:87px;height:16px;background:{INK}}}
.val{{position:absolute;top:0;font-size:96px;font-weight:800;line-height:190px}}
.hdr{{font-size:54px;font-weight:800}}
</style></head><body><div class="grid">
<div></div><div class="hdr">predicts the face after “I'm feeling”</div>
{rows_html}</div></body></html>'''
render(HTML, "reader", 2300, 2000)
