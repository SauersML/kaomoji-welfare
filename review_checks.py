"""Numbers for REVIEW.md that the analysis scripts do not print -> out_kao/review_checks.json.
Run from the analysis folder (main_rows.csv, main/{scen,wild}.jsonl, steer.jsonl, out_kao/rows_kao.csv)."""
import json
import numpy as np, pandas as pd
import boot

R = {}
m = pd.read_csv("main_rows.csv"); sc = m[(m["final"] == True) & (m["type"] != "wild")]
# situation effect within each user mood, kao_val (45 faces, raters): unmatched (difference of cell means over all rows)
# vs matched within base (good - bad per base, then averaged over bases), as T3 does
for mo in ("happy", "sad"):
    e = sc[sc["mood"] == mo]
    um = e[e.sit == "good"]["kao_val"].mean() - e[e.sit == "bad"]["kao_val"].mean()
    bases, M = boot.cell_means(e, "kao_val"); col = 0 if mo == "happy" else 2
    mt = np.nanmean(M[:, col] - M[:, col + 1])
    R[f"sit_within_{mo}"] = dict(unmatched=float(um), matched_within_base=float(mt))
# reader's input: each message clipped at 1,500 characters (rate.py transcript); share of final turns where a clip happened
recs = {json.loads(l)["id"]: json.loads(l) for f in ("scen.jsonl", "wild.jsonl") for l in open(f"main/{f}")}
def clipped(x, with_reply):
    msgs = recs[x["id"]]["msgs"]; ai = [i for i, q in enumerate(msgs) if q["role"] == "assistant"][int(x["turn"]) - 1]
    upto = msgs[:ai + 1] if with_reply else msgs[:ai]
    return any(len(q["content"]) > 1500 for q in upto if q["role"] != "system")
for nm, rows in (("final_turns", m[m["final"] == True]), ("all_turns", m)):
    R[f"reader_clip_share_{nm}"] = dict(before_reply=float(np.mean([clipped(x, False) for x in rows.to_dict("records")])),
                                        with_reply=float(np.mean([clipped(x, True) for x in rows.to_dict("records")])), n=int(len(rows)))
# steering: how many conversations / turns
st = [json.loads(l) for l in open("steer.jsonl")]
ro = [x for x in st if x.get("mode") == "readout" and x["dir"] == "val"]
R["steer_readout_val"] = dict(n_rows=len(ro), alphas=sorted({x["alpha"] for x in ro}), n_turns=len({(x["id"], x["turn"]) for x in ro}),
                              n_conversations=len({x["id"] for x in ro}))
# system-prompt arm: first face Qwen writes (nat_face_val, mean of the two raters as in analyze.py), T3 with the fixed
# bootstrap and paired null; final scenario turns, code restricted to bases with both outcomes
import os
rng = np.random.default_rng(0)
if os.path.exists("sys/rows.jsonl"):
    FA = [json.load(open(f"faces_{r}.json")) for r in ("Qwen3-14B", "OLMo-2-1124-13B-Instruct")]
    face = lambda k: np.nanmean([d.get(k) if d.get(k) is not None else np.nan for d in FA]) if k else np.nan
    sy = pd.DataFrame([json.loads(l) for l in open("sys/rows.jsonl")])
    sy["nat_face_val"] = [face(k) for k in sy["kaomoji"]]
    ss = sy[(sy["type"] != "wild") & (sy["final"] == True)]
    out = {}
    for typ in ("task", "treat", "exist", "code", "all"):
        e = ss if typ == "all" else ss[ss["type"] == typ]
        if typ == "code":
            ok = e.groupby("base")["sit"].nunique(); e = e[e["base"].isin(ok[ok == 2].index)]
        out[typ] = dict(effects=boot.effects(e, "nat_face_val", rng), null=boot.sit_null(e, "nat_face_val", rng))
    R["sys_nat_face_val"] = out; R["sys_kaomoji_rate"] = float(sy["kaomoji"].notna().mean())
json.dump(R, open("out_kao/review_checks.json", "w"), indent=1)
print(json.dumps(R, indent=1))
