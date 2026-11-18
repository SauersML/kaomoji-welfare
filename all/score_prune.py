"""Exact-where-it-matters version of score_all.py: level-by-level over the token trie of all faces, appending each level
to the prefix KV cache (4D mask: each node sees the prefix + its ancestors). A subtree is expanded only while its
prefix log-prob is >= (best exact score among the 45 original faces in this row) - MARGIN; since P(face) <= P(prefix),
a pruned face's log-prob is below that bound and its softmax weight relative to the best face is < e^-MARGIN.
The 45 original faces are always scored exactly. Output per row: log-prob (exact, or the upper bound for pruned faces)
and a flag exact. Also saves layer-18 residual at the stem's last token and the pre-reply position.
  python score_prune.py faces.json OUTDIR [shard nshard] [--check]"""
import json, os, sys, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import scen as S
faces = [x["face"] for x in json.load(open(sys.argv[1]))]; OUTD = sys.argv[2]; os.makedirs(OUTD, exist_ok=True)
shard, nsh = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 and not sys.argv[3].startswith("-") else (0, 1)
CHECK = "--check" in sys.argv; MARGIN = float(os.environ.get("MARGIN", "20")); LAYER = 18; NMAX = int(os.environ.get("NMAX", "8192"))
MODEL = "Qwen/Qwen3-8B"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=getattr(torch, os.environ.get("DT", "bfloat16")), device_map="cuda", attn_implementation="sdpa").eval()
def ptmpl(msgs): return tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False)
FIDS = [tok(" " + f, add_special_tokens=False)["input_ids"] for f in faces]
K45 = set(range(45)); assert faces[:45] == S.KSET
# trie: node = token prefix of length >= 1; children; faces ending at node
node_of = {}; tokn, par, dep = [], [], []; kids = []; ends = []; force = []
def get(k):
    if k in node_of: return node_of[k]
    p = get(k[:-1]) if len(k) > 1 else -1
    i = len(tokn); node_of[k] = i; tokn.append(k[-1]); par.append(p); dep.append(len(k) - 1); kids.append([]); ends.append([]); force.append(False)
    if p >= 0: kids[p].append(i)
    return i
for fi, c in enumerate(FIDS):
    n = get(tuple(c)); ends[n].append(fi)
    if fi in K45:
        for j in range(1, len(c) + 1): force[node_of[tuple(c[:j])]] = True
roots = [i for i in range(len(tokn)) if par[i] < 0]
tokn_t = torch.tensor(tokn, device="cuda"); force_t = torch.tensor(force, device="cuda")
print("faces", len(faces), "trie nodes", len(tokn), "roots", len(roots), flush=True)

@torch.no_grad()
def score(ids, pre_pos):
    P = len(ids)
    o = model(torch.tensor([ids], device="cuda"), use_cache=True, output_hidden_states=True)
    hs = o.hidden_states[LAYER][0]; h = torch.stack([hs[-1], hs[pre_pos]]).float().cpu().numpy()
    lp0 = torch.log_softmax(o.logits[0, -1].float(), -1); cache = o.past_key_values; del o
    lpn = np.full(len(tokn), np.nan)        # log-prob of the token path to node (exact when computed)
    lpn_r = lp0[torch.tensor([tokn[r] for r in roots], device="cuda")].cpu().numpy()
    for r, v in zip(roots, lpn_r): lpn[r] = v
    # exact KSET first: expand forced nodes level by level along with others; threshold from KSET needs KSET done first,
    # so do two passes: pass A expands only forced nodes; pass B expands the rest with the threshold.
    slot = {}; added = 0
    def expand(frontier_fn, start):
        nonlocal added
        front = start
        while front:
            for b in range(0, len(front), NMAX):
                blk = front[b:b + NMAX]; n = len(blk); base = P + added
                ri, cj = [], []
                for qi, nd in enumerate(blk):
                    a = par[nd]
                    while a >= 0: ri.append(qi); cj.append(P + slot[a]); a = par[a]
                    ri.append(qi); cj.append(base + qi)
                m = torch.zeros((n, base + n), dtype=torch.bool, device="cuda"); m[:, :P] = True
                m[torch.tensor(ri, device="cuda"), torch.tensor(cj, device="cuda")] = True
                pos = torch.tensor([P + dep[nd] for nd in blk], device="cuda")
                out = model(tokn_t[torch.tensor(blk, device="cuda")][None], attention_mask=m[None, None], position_ids=pos[None],
                            past_key_values=cache, cache_position=torch.arange(base, base + n, device="cuda"), use_cache=True)
                lp = torch.log_softmax(out.logits[0].float(), -1); del out
                ci, cn, ct = [], [], []
                for qi, nd in enumerate(blk):
                    slot[nd] = added + qi
                    for c in kids[nd]: ci.append(qi); cn.append(c); ct.append(tokn[c])
                added += n
                if cn:
                    v = (lp[torch.tensor(ci, device="cuda"), torch.tensor(ct, device="cuda")]).cpu().numpy()
                    for c, x, qi in zip(cn, v, ci): lpn[c] = lpn[blk[qi]] + x
                del lp
            front = [c for nd in front for c in kids[nd] if frontier_fn(c)]
            front = [c for c in front if c not in slot]
        return
    # nodes are expanded (run through the model) only if they have children worth scoring
    expand(lambda c: force[c] and kids[c], [r for r in roots if force[r] and kids[r]])
    best = max(lpn[node_of[tuple(FIDS[i])]] for i in K45); thr = best - MARGIN
    ok = lambda c: kids[c] and lpn[c] >= thr
    expand(ok, [r for r in roots if ok(r) and r not in slot] + [c for nd in list(slot) for c in kids[nd] if ok(c) and c not in slot])
    cache.crop(-added)
    sc = np.empty(len(faces)); ex = np.zeros(len(faces), bool)
    for fi, c in enumerate(FIDS):
        nd = node_of[tuple(c)]
        if not np.isnan(lpn[nd]): sc[fi] = lpn[nd]; ex[fi] = True
        else:  # deepest computed ancestor gives an upper bound
            a = par[nd]
            while np.isnan(lpn[a]): a = par[a]
            sc[fi] = lpn[a]
    return sc, ex, h, added, thr

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

recs = [json.loads(l) for f in ("scen.jsonl", "wild.jsonl") for l in open(f"main/{f}")][shard::nsh]
if CHECK: recs = recs[:2] + [r for r in recs if r["type"] == "wild"][:2]
SC, EX, HH, meta = [], [], [], []; t0 = time.time()
for n_, r in enumerate(recs):
    msgs = r["msgs"]; ai = [i for i, m in enumerate(msgs) if m["role"] == "assistant"][-1]
    pre = tok(ptmpl(msgs[:ai]), add_special_tokens=False)["input_ids"]
    ids = tok(ptmpl(msgs[:ai]) + msgs[ai]["content"].rstrip() + S.STEM, add_special_tokens=False)["input_ids"]
    t1 = time.time(); sc, ex, h, added, thr = score(ids, len(pre) - 1)
    SC.append(sc.astype(np.float32)); EX.append(ex); HH.append(h.astype(np.float16))
    meta.append(dict(id=r["id"], n_assistant=sum(m["role"] == "assistant" for m in msgs), P=len(ids), pre_ok=ids[:len(pre)] == pre,
                     nodes_run=added, thr=float(thr), n_exact=int(ex.sum())))
    if CHECK:
        idx = list(range(45)) + [i for i in np.where(ex)[0] if i >= 45][::max(1, int(ex.sum()) // 60)][:60]
        nv = score_naive(ids, idx); print("check max|prune-naive|", float(np.abs(nv - sc[idx]).max()), "median", float(np.median(np.abs(nv - sc[idx]))), flush=True)
    if CHECK or n_ % 20 == 0:
        print(n_, len(recs), r["id"], "P", len(ids), "nodes", added, "exact", int(ex.sum()), "row s", round(time.time() - t1, 2), "total", round(time.time() - t0), flush=True)
np.save(f"{OUTD}/SC_{shard}.npy", np.stack(SC)); np.save(f"{OUTD}/EX_{shard}.npy", np.stack(EX)); np.save(f"{OUTD}/H_{shard}.npy", np.stack(HH))
json.dump(meta, open(f"{OUTD}/meta_{shard}.json", "w")); print("done", len(recs), round(time.time() - t0), flush=True)
