"""Random-direction null for readout steering (as steer.py, mode readout): add a*SD*dir at the output of decoder layer 18
(all positions) while Qwen3-8B reads the natural transcript + "\n\nI'm feeling"; score log P(' '+face) for the 45 original
faces. 50 Gaussian unit directions (seed 1000+k), the valence axis, and steer.py's rand0/rand1 (default_rng(0)) as a
reproduction check; the same 64 conversations as steer.py; alpha in (-16,-8,0,8,16). SD = 2.6742725 (steer.py's printed
sd of the valence projection at the stem).  -> steer_null_raw.jsonl"""
import json
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import scen as S
SD = 2.6742725372314453; L = 18
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="cuda").eval()
val = np.load("axes.npz")[f"val{L}"]
dirs = {"val": val}
rs = np.random.default_rng(0)
for k in range(2): r = rs.standard_normal(4096); dirs[f"rand{k}"] = r / np.linalg.norm(r)
for k in range(50): r = np.random.default_rng(1000 + k).standard_normal(4096); dirs[f"null{k}"] = r / np.linalg.norm(r)
cos = {d: float(v @ val / np.linalg.norm(v) / np.linalg.norm(val)) for d, v in dirs.items()}
state = {"v": None}
def hook(mod, inp, out):
    if state["v"] is None: return out
    if isinstance(out, tuple): return (out[0] + state["v"],) + tuple(out[1:])
    return out + state["v"]
model.model.layers[L - 1].register_forward_hook(hook)
KIDS = [tok(" " + k, add_special_tokens=False)["input_ids"] for k in S.KSET]
@torch.no_grad()
def stem_score(ids):
    o = model(torch.tensor([ids], device="cuda"), use_cache=True)
    lp0 = torch.log_softmax(o.logits[0, -1].float(), -1); n = len(KIDS); C = max(len(c) for c in KIDS)
    cache = o.past_key_values; cache.batch_repeat_interleave(n)
    inp = torch.zeros((n, C), dtype=torch.long, device="cuda"); m = torch.zeros((n, C), dtype=torch.long, device="cuda")
    for i, c in enumerate(KIDS): inp[i, :len(c)] = torch.tensor(c); m[i, :len(c)] = 1
    att = torch.cat([torch.ones((n, len(ids)), dtype=torch.long, device="cuda"), m], 1)
    pos = torch.arange(len(ids), len(ids) + C, device="cuda").unsqueeze(0).expand(n, C)
    lp = torch.log_softmax(model(inp, attention_mask=att, position_ids=pos, past_key_values=cache).logits.float(), -1)
    tot = lp0[FIRST] + (lp[:, :-1].gather(2, inp[:, 1:, None])[..., 0] * m[:, 1:]).sum(1)
    return tot.tolist()
FIRST = torch.tensor([c[0] for c in KIDS], device="cuda")
import sys
SH = sys.argv[1] if len(sys.argv) > 1 else "all"
recs = {json.loads(l)["id"]: json.loads(l) for f in ("scen.jsonl", "wild.jsonl") for l in open(f"main/{f}")}
pick = json.load(open("steer_pick.json")); IDS = []
for cid, turn in pick:
    msgs = recs[cid]["msgs"]; ai = [i for i, m in enumerate(msgs) if m["role"] == "assistant"][turn - 1]
    IDS.append(tok(tok.apply_chat_template(msgs[:ai], add_generation_prompt=True, enable_thinking=False, tokenize=False)
                   + msgs[ai]["content"].rstrip() + S.STEM, add_special_tokens=False)["input_ids"])
conds = [(None, 0)] + [(d, a) for d in dirs for a in (-16, -8, 8, 16)]
if SH == "A": conds = [c for c in conds if c[0] in (None, "val", "rand0", "rand1")]
elif SH != "all": k = int(SH); conds = [c for c in conds if c[0] and c[0].startswith("null") and int(c[0][4:]) % 3 == k]
out = open(f"steer_null_raw_{SH}.jsonl", "w")
for d, a in conds:
    state["v"] = None if a == 0 else torch.tensor(a * SD * dirs[d], dtype=torch.bfloat16, device="cuda")
    for (cid, turn), ids in zip(pick, IDS):
        out.write(json.dumps(dict(dir=d or "none", alpha=a, cos_val=cos.get(d), id=cid, turn=turn, ksc=stem_score(ids))) + "\n")
    out.flush(); print(d, a, flush=True)
print("done")
