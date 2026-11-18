"""log P(' ' + face | prefix + '\n\nI'm feeling') for every face, at the end of Qwen3-8B's last reply in every saved
main-arm conversation. Prefix KV computed once; faces packed as a token trie into one sequence with a 4D mask
(each trie node sees the prefix + its own ancestors), in chunks; cache cropped back to the prefix between chunks.
Also saves the layer-18 residual stream at the stem's last token and at the pre-reply position.
  python score_all.py faces.json OUTDIR [shard nshard] [--check]"""
import json, os, sys, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import scen as S

faces = [x["face"] for x in json.load(open(sys.argv[1]))]; OUTD = sys.argv[2]; os.makedirs(OUTD, exist_ok=True)
shard, nsh = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
CHECK = "--check" in sys.argv
MODEL = "Qwen/Qwen3-8B"; LAYER = 18; QCH = int(os.environ.get("QCH", "12288"))
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=getattr(torch, os.environ.get("DT", "bfloat16")), device_map="cuda", attn_implementation="sdpa").eval()
def ptmpl(msgs): return tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False)
FIDS = [tok(" " + f, add_special_tokens=False)["input_ids"] for f in faces]

# --- trie over tokens 0..n-2 of each face (the node for prefix c[:j+1] predicts c[j+1])
order = sorted(range(len(faces)), key=lambda i: FIDS[i])
chunks = []  # each: (node_tok, node_parent(local idx or -1), node_depth, [(face, [node idx predicting c[1..]])])
cur = None
def new_chunk(): return dict(tok=[], par=[], dep=[], key={}, faces=[])
cur = new_chunk()
for i in order:
    c = FIDS[i]
    need = sum(1 for j in range(len(c) - 1) if tuple(c[:j + 1]) not in cur["key"])
    if len(cur["tok"]) + need > QCH: chunks.append(cur); cur = new_chunk()
    nodes = []
    for j in range(len(c) - 1):
        k = tuple(c[:j + 1])
        if k not in cur["key"]:
            cur["key"][k] = len(cur["tok"]); cur["tok"].append(c[j]); cur["par"].append(cur["key"][k[:-1]] if j > 0 else -1); cur["dep"].append(j)
        nodes.append(cur["key"][k])
    cur["faces"].append((i, nodes))
chunks.append(cur)
for ch in chunks:
    n = len(ch["tok"]); anc = np.zeros((n, n), dtype=bool)
    for a in range(n):
        anc[a, a] = True; p = ch["par"][a]
        if p >= 0: anc[a] |= anc[p]
    ch["anc"] = torch.tensor(anc); ch["tok_t"] = torch.tensor(ch["tok"]); ch["dep_t"] = torch.tensor(ch["dep"])
    fi, tgt, nd = [], [], []
    for i, nodes in ch["faces"]:
        for j, a in enumerate(nodes): fi.append(i); nd.append(a); tgt.append(FIDS[i][j + 1])
    ch["fi"] = torch.tensor(fi, dtype=torch.long); ch["nd"] = torch.tensor(nd, dtype=torch.long); ch["tg"] = torch.tensor(tgt, dtype=torch.long)
print("faces", len(faces), "trie nodes", sum(len(c["tok"]) for c in chunks), "chunks", len(chunks), "naive", sum(len(c) - 1 for c in FIDS), flush=True)
FIRST = torch.tensor([c[0] for c in FIDS], device="cuda")

@torch.no_grad()
def score(ids, pre_pos):
    P = len(ids)
    o = model(torch.tensor([ids], device="cuda"), use_cache=True, output_hidden_states=True)
    hs = o.hidden_states[LAYER][0]; h = torch.stack([hs[-1], hs[pre_pos]]).float().cpu().numpy()
    lp0 = torch.log_softmax(o.logits[0, -1].float(), -1)
    cache = o.past_key_values; del o
    sc = lp0[FIRST].clone()
    for ch in chunks:
        n = len(ch["tok"])
        m = torch.ones((n, P + n), dtype=torch.bool, device="cuda"); m[:, P:] = ch["anc"].cuda()
        out = model(ch["tok_t"].cuda()[None], attention_mask=m[None, None], position_ids=(P + ch["dep_t"].cuda())[None],
                    past_key_values=cache, cache_position=torch.arange(P, P + n, device="cuda"), use_cache=True)
        lp = torch.log_softmax(out.logits[0].float(), -1)
        sc.index_add_(0, ch["fi"].cuda(), lp[ch["nd"].cuda(), ch["tg"].cuda()])
        cache.crop(-n); del out, lp
    return sc.cpu().numpy(), h

@torch.no_grad()
def score_naive(ids, idx):  # the original act.py method, for a check
    ks = [FIDS[i] for i in idx]
    x = torch.tensor([ids], device="cuda"); o = model(x, use_cache=True)
    lp0 = torch.log_softmax(o.logits[0, -1].float(), -1); n = len(ks); C = max(len(c) for c in ks)
    cache = o.past_key_values; cache.batch_repeat_interleave(n)
    inp = torch.full((n, C), tok.pad_token_id or 0, device="cuda"); mm = torch.zeros((n, C), dtype=torch.long, device="cuda")
    for i, c in enumerate(ks): inp[i, :len(c)] = torch.tensor(c); mm[i, :len(c)] = 1
    att = torch.cat([torch.ones((n, len(ids)), dtype=torch.long, device="cuda"), mm], 1)
    pos = torch.arange(len(ids), len(ids) + C, device="cuda").unsqueeze(0).expand(n, C)
    lp = torch.log_softmax(model(inp, attention_mask=att, position_ids=pos, past_key_values=cache).logits.float(), -1)
    return np.array([lp0[c[0]].item() + sum(lp[i, j - 1, c[j]].item() for j in range(1, len(c))) for i, c in enumerate(ks)])

recs = [json.loads(l) for f in ("scen.jsonl", "wild.jsonl") for l in open(f"main/{f}")]
recs = recs[shard::nsh]
if CHECK: recs = recs[:3] + [r for r in recs if r["type"] == "wild"][:2]
SC, HH, meta = [], [], []
t0 = time.time()
for n_, r in enumerate(recs):
    msgs = r["msgs"]; ai = [i for i, m in enumerate(msgs) if m["role"] == "assistant"][-1]
    pre = tok(ptmpl(msgs[:ai]), add_special_tokens=False)["input_ids"]
    ids = tok(ptmpl(msgs[:ai]) + msgs[ai]["content"].rstrip() + S.STEM, add_special_tokens=False)["input_ids"]
    pre_ok = ids[:len(pre)] == pre
    sc, h = score(ids, len(pre) - 1)
    SC.append(sc.astype(np.float32)); HH.append(h.astype(np.float16))
    meta.append(dict(id=r["id"], turn=ai and sum(m["role"] == "assistant" for m in msgs[:ai + 1]), P=len(ids), pre_ok=pre_ok))
    if CHECK:
        idx = list(range(45)) + list(range(45, len(faces), max(1, len(faces) // 40)))[:40]
        nv = score_naive(ids, idx)
        print("check", r["id"], len(ids), "max|trie-naive|", float(np.abs(nv - sc[idx]).max()), "median", float(np.median(np.abs(nv - sc[idx]))), flush=True)
    if n_ % 20 == 0: print(n_, len(recs), round(time.time() - t0), "s", flush=True)
np.save(f"{OUTD}/SC_{shard}.npy", np.stack(SC)); np.save(f"{OUTD}/H_{shard}.npy", np.stack(HH))
json.dump(meta, open(f"{OUTD}/meta_{shard}.json", "w"))
print("done", len(recs), round(time.time() - t0), flush=True)
