"""faces.png: the 45 kaomoji scored after "I'm feeling", placed at their rated valence (mean of Qwen3-14B and OLMo-2-13B, each rating the face alone on -3..+3)."""
import json, html
from kstyle import *
FACE = rmean("faces")
items = sorted([(k, FACE[k]) for k in S.KSET], key=lambda t: -t[1])
assert len(items) == 45 and not any(np.isnan(v) for _, v in items)
N = dnorm(-3, 3)
W = 2300; PX = 330; TOP = 50; AXW = 170; LO = -2.5
chips = ""
for k, v in items:
    c = DIV(N(v)); chips += f'<div class="c" data-v="{v:.4f}" style="background:{hexc(c)};color:{ink_on(c)}">{html.escape(k)}</div>'
ticks = ""
for t in range(3, -3, -1):
    c = DIV(N(t)); col = hexc(c) if abs(t) >= 1 else INK
    ticks += f'<div class="t" style="top:{TOP + (3 - t) * PX}px;color:{col}">{minus(f"{t:+d}") if t else "0"}</div>'
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:{W}px;background:#fff;font-family:{FONT};color:{INK}}}
#wrap{{position:relative;width:{W}px;height:{TOP + (3 - LO) * PX + 60}px}}
.c{{position:absolute;font-size:86px;line-height:1;padding:10px 14px;border-radius:24px;white-space:nowrap;font-weight:600}}
.t{{position:absolute;left:0;width:{AXW - 40}px;text-align:right;font-size:84px;font-weight:800;transform:translateY(-50%)}}
.ax{{position:absolute;left:{AXW - 20}px;top:{TOP}px;width:8px;height:{(3 - LO) * PX}px;border-radius:4px;background:linear-gradient(to bottom,{",".join(hexc(DIV(x)) for x in np.linspace(1, N(LO), 11))})}}
</style></head><body><div id="wrap"><div class="ax"></div>{ticks}{chips}</div>
<script>
const W={W}, PX={PX}, TOP={TOP}, X0={AXW + 10}, GAP=12;
const placed=[]; let maxShift=0, maxB=0;
for (const el of document.querySelectorAll('.c')) {{
  const v=parseFloat(el.dataset.v), w=el.offsetWidth, h=el.offsetHeight;
  let y=TOP+(3-v)*PX-h/2; if (y<0) y=0;
  const y0=y;
  while (true) {{
    // leftmost x at this y that clears every placed chip
    let x=X0, moved=true;
    while (moved) {{ moved=false;
      for (const p of placed) if (y<p.y+p.h+GAP/2 && y+h>p.y-GAP/2 && x<p.x+p.w+GAP && x+w>p.x-GAP) {{ x=p.x+p.w+GAP; moved=true; }} }}
    if (x+w<=W-10) {{ el.style.left=x+'px'; el.style.top=y+'px'; placed.push({{x,y,w,h}}); break; }}
    y+=4;
  }}
  maxShift=Math.max(maxShift,(y-y0)/PX); maxB=Math.max(maxB,y+h);
}}
document.body.insertAdjacentHTML('beforeend','<div id=sh style="position:absolute;left:-9999px">'+maxShift.toFixed(2)+'</div>');
document.getElementById('wrap').style.height=Math.max(maxB,TOP+(3-({LO}))*PX+20)+'px';
</script></body></html>'''
render(HTML, "faces", W, 2700)
