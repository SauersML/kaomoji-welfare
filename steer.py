"""Steering (main arm). Add a*sd*dir at the output of one decoder layer, all positions.
 mode readout: the natural transcript is unchanged; steering is on while the model reads it + the stem and
               scores the kaomoji set and continues "I'm feeling" greedily.
 mode full:    steering also on while the model writes its reply (greedy, 160 tokens); then the same readout.
Directions: valence axis, goal axis, 2 random unit directions; all with the same norm per dose."""
import json, os, random
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import scen as S

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B"); OUT = os.environ.get("OUT", "out"); AO = f"{OUT}/main"
ch = json.load(open(f"{OUT}/chosen.json")); L = ch["layer"]
ax = np.load(f"{OUT}/axes.npz"); layers = list(ax["layers"]); li = layers.index(L)
HS = np.load(f"{AO}/HS.npy")[:, li].astype(np.float32)
dirs = {"val": ax[f"val{L}"], "goal": ax[f"goal{L}"]}
rs = np.random.default_rng(0)
for k in range(2):
    r = rs.standard_normal(HS.shape[1]); dirs[f"rand{k}"] = r / np.linalg.norm(r)
SD = float((HS @ dirs["val"]).std())
print("layer", L, "sd(val proj at stem)", SD, "mean norm", float(np.linalg.norm(HS, axis=1).mean()), flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
state = {"v": None}
def hook(mod, inp, out):
    if state["v"] is None: return out
    if isinstance(out, tuple): return (out[0] + state["v"],) + tuple(out[1:])
    return out + state["v"]
model.model.layers[L - 1].register_forward_hook(hook)
KIDS = [tok(" " + k, add_special_tokens=False)["input_ids"] for k in S.KSET]

@torch.no_grad()
def stem_score(ids):
    x = torch.tensor([ids], device="cuda")
    o = model(x, use_cache=True)
    lp0 = torch.log_softmax(o.logits[0, -1].float(), -1)
    n = len(KIDS); C = max(len(c) for c in KIDS)
    cache = o.past_key_values; cache.batch_repeat_interleave(n)
    inp = torch.zeros((n, C), dtype=torch.long, device="cuda"); m = torch.zeros((n, C), dtype=torch.long, device="cuda")
    for i, c in enumerate(KIDS): inp[i, :len(c)] = torch.tensor(c); m[i, :len(c)] = 1
    att = torch.cat([torch.ones((n, len(ids)), dtype=torch.long, device="cuda"), m], 1)
    pos = torch.arange(len(ids), len(ids) + C, device="cuda").unsqueeze(0).expand(n, C)
    lp = torch.log_softmax(model(inp, attention_mask=att, position_ids=pos, past_key_values=cache).logits.float(), -1)
    return [lp0[c[0]].item() + sum(lp[i, j - 1, c[j]].item() for j in range(1, len(c))) for i, c in enumerate(KIDS)]

@torch.no_grad()
def greedy(ids, n):
    g = model.generate(torch.tensor([ids], device="cuda"), max_new_tokens=n, do_sample=False)
    return tok.decode(g[0, len(ids):], skip_special_tokens=True)

rows = [json.loads(l) for l in open(f"{AO}/rows.jsonl")]
recs = {json.loads(l)["id"]: json.loads(l) for f in ("scen.jsonl", "wild.jsonl") for l in open(f"{AO}/{f}")}
rng = random.Random(5); pick = []
for typ in ("task", "treat", "exist", "code"):
    for sit in ("good", "bad"):
        for mood in ("happy", "sad"):
            c = [x for x in rows if x["type"] == typ and x["sit"] == sit and x["mood"] == mood and x["final"]]
            pick += rng.sample(c, min(3, len(c)))
pick += rng.sample([x for x in rows if x["type"] == "wild" and x["turn"] == 1], 16)

conds = [("val", a) for a in (-16, -8, -4, -2, 0, 2, 4, 8, 16)] + [("goal", a) for a in (-16, -8, 8, 16)] + \
        [(f"rand{k}", a) for k in range(2) for a in (-16, -8, 8, 16)]
@torch.no_grad()
def greedy_batch(texts, n):
    tok.padding_side = "left"; res = []
    for b in range(0, len(texts), 16):
        enc = tok(texts[b:b + 16], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        g = model.generate(**enc, max_new_tokens=n, do_sample=False)
        res += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return res

PRE, NAT = [], []
for x in pick:
    msgs = recs[x["id"]]["msgs"]; ai = [i for i, m in enumerate(msgs) if m["role"] == "assistant"][x["turn"] - 1]
    PRE.append(tok.apply_chat_template(msgs[:ai], add_generation_prompt=True, enable_thinking=False, tokenize=False))
    NAT.append(msgs[ai]["content"])
full_conds = [("val", a) for a in (-16, -8, 0, 8, 16)] + [("rand0", -16), ("rand0", 16), ("rand1", -16), ("rand1", 16)]
out = open(f"{AO}/steer.jsonl", "w"); sid = 0
for mode in ("readout", "full"):
    for d, a in (conds if mode == "readout" else full_conds):
        state["v"] = None if a == 0 else torch.tensor(a * SD * dirs[d], dtype=torch.bfloat16, device="cuda")
        replies = NAT if mode == "readout" else greedy_batch(PRE, 160)
        for x, pre, reply in zip(pick, PRE, replies):
            ids = tok(pre + reply.rstrip() + S.STEM, add_special_tokens=False)["input_ids"]
            sc = stem_score(ids); fr = greedy(ids, 12).split("\n")[0]
            out.write(json.dumps(dict(sid=sid, mode=mode, dir=d, alpha=a, id=x["id"], turn=x["turn"], type=x["type"],
                                      sit=x["sit"], mood=x["mood"], reply=reply, ksc=sc, free=fr), ensure_ascii=False) + "\n"); sid += 1
        out.flush(); print(mode, d, a, flush=True)
out.close()
