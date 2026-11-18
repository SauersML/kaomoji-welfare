"""HF pass: build + validate welfare axes (emotion-story PC1, neutral-game win/loss), extract hidden states at
measurement positions for every assistant turn."""
import json, os, random, sys
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import scen as S

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B"); OUT = os.environ.get("OUT", "out")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
NL = model.config.num_hidden_layers
LAYERS = [int(round(NL * f)) for f in (0.25, 0.375, 0.5, 0.625, 0.75, 0.875)]  # 8B: 9,14,18,22,27,32
END = tok.convert_tokens_to_ids("<|im_end|>")

def ptmpl(msgs):
    return tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False)

@torch.no_grad()
def hs(ids, positions):
    x = torch.tensor([ids], device="cuda")
    o = model(x, output_hidden_states=True)
    return np.stack([o.hidden_states[l][0, positions].float().cpu().numpy() for l in LAYERS], 0)  # L x P x d

@torch.no_grad()
def hs_mean(ids, start):
    x = torch.tensor([ids], device="cuda")
    o = model(x, output_hidden_states=True)
    return np.stack([o.hidden_states[l][0, start:].float().mean(0).cpu().numpy() for l in LAYERS], 0)

def points(msgs, a_idx):
    """ids up to first reply token; positions (last user token, pre-choice, first reply token)."""
    p = tok(ptmpl(msgs[:a_idx]), add_special_tokens=False)["input_ids"]
    r = tok(msgs[a_idx]["content"], add_special_tokens=False)["input_ids"][:1] or [END]
    ends = [i for i, t in enumerate(p) if t == END]
    pu = ends[-1] - 1
    return p + r, [pu, len(p) - 1, len(p)]

stage = sys.argv[1] if len(sys.argv) > 1 else "all"
if stage in ("axes", "all"):
    # --- valence axis from emotion stories
    st = [json.loads(l) for l in open(f"{OUT}/stories.jsonl")]
    X = np.stack([hs_mean(tok(s["text"], add_special_tokens=False)["input_ids"], 10) for s in st], 1)  # L x N x d
    emo = np.array([s["emotion"] for s in st]); half = np.array([s["i"] >= 10 for s in st])
    np.save(f"{OUT}/story_X.npy", X.astype(np.float16))
    # --- games
    rng = random.Random(7); games = []
    for g in range(480):
        games.append(dict(kind="doors", win=g % 2 == 0, mapping=(g // 2) % 2, msgs=S.doors_game(rng, g % 2 == 0, (g // 2) % 2)))
    for g in range(240):
        games.append(dict(kind="number", win=g % 2 == 0, mapping=-1, msgs=S.number_game(rng, g % 2 == 0)))
    G = []
    for gm in games:
        p = tok(ptmpl(gm["msgs"]), add_special_tokens=False)["input_ids"]
        G.append(hs(p, [len(p) - 1])[:, 0])
    G = np.stack(G, 1)
    np.save(f"{OUT}/game_X.npy", G.astype(np.float16))
    json.dump([{k: v for k, v in g.items() if k != "msgs"} for g in games], open(f"{OUT}/games.json", "w"))
    pos = np.isin(emo, S.EMOS_POS); neg = np.isin(emo, S.EMOS_NEG)
    try:
        vad = {}
        for line in open("NRC-VAD-Lexicon.txt"):
            w, v, *_ = line.rstrip("\n").split("\t")
            try: vad[w] = float(v)
            except ValueError: pass
        adj2noun = {"happy": "happiness", "afraid": "fear", "angry": "anger", "sad": "sadness"}
    except FileNotFoundError:
        vad = {}
    axes = {}; val = {}
    gk = np.array([g["kind"] for g in games]); gw = np.array([g["win"] for g in games]); gm_ = np.array([g["mapping"] for g in games])
    gi = np.arange(len(games))
    for li, l in enumerate(LAYERS):
        A = X[li][~half]; emos = sorted(set(emo))
        M = np.stack([A[emo[~half] == e].mean(0) for e in emos]); M = M - M.mean(0)
        U_, s_, Vt = np.linalg.svd(M, full_matrices=False); v = Vt[0]
        ep = np.array([e in S.EMOS_POS for e in emos]); en = np.array([e in S.EMOS_NEG for e in emos])
        if (M[ep] @ v).mean() < (M[en] @ v).mean(): v = -v
        B = X[li][half]; pb = B @ v
        auc_story = roc_auc_score(pos[half][pos[half] | neg[half]], pb[(pos[half] | neg[half])])
        em_b = {e: pb[emo[half] == e].mean() for e in emos}
        r_vad = None
        if vad:
            ws = [e for e in emos if e in vad]
            if len(ws) >= 10: r_vad = spearmanr([em_b[e] for e in ws], [vad[e] for e in ws])[0]
        var1 = float(s_[0] ** 2 / (s_ ** 2).sum())
        # goal axis: doors games, fit on first half, validate on doors second half + number game
        fit = (gk == "doors") & (gi < 240); hold = (gk == "doors") & (gi >= 240); num = gk == "number"
        Gl = G[li]
        w = Gl[fit & gw].mean(0) - Gl[fit & ~gw].mean(0); w = w / np.linalg.norm(w)
        auc_doors = roc_auc_score(gw[hold], Gl[hold] @ w)
        auc_doors_m = [roc_auc_score(gw[hold & (gm_ == m)], Gl[hold & (gm_ == m)] @ w) for m in (0, 1)]
        auc_num = roc_auc_score(gw[num], Gl[num] @ w)
        auc_num_val = roc_auc_score(gw[num], Gl[num] @ v)
        axes[l] = dict(val=v, goal=w)
        val[l] = dict(layer=l, val_pc1_var=var1, val_auc_heldout_stories=auc_story, val_spearman_nrc_heldout=r_vad,
                      goal_auc_heldout_doors=auc_doors, goal_auc_by_mapping=auc_doors_m, goal_auc_number_game=auc_num,
                      valaxis_auc_number_game=auc_num_val, cos_val_goal=float(v @ w),
                      emo_proj_heldout={e: float(em_b[e]) for e in emos})
        print(json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in val[l].items() if k != "emo_proj_heldout"}), flush=True)
    np.savez(f"{OUT}/axes.npz", layers=np.array(LAYERS), **{f"val{l}": axes[l]["val"] for l in LAYERS}, **{f"goal{l}": axes[l]["goal"] for l in LAYERS})
    json.dump(val, open(f"{OUT}/axes_validation.json", "w"), indent=1)

KIDS = [tok(" " + k, add_special_tokens=False)["input_ids"] for k in S.KSET]

def stem_prefix(msgs, ai):
    return tok(ptmpl(msgs[:ai]) + msgs[ai]["content"].rstrip() + S.STEM, add_special_tokens=False)["input_ids"]

@torch.no_grad()
def stem_score(ids, want_h=True):
    """log P(' ' + kaomoji | prefix) for every kaomoji in KSET (prefix KV cache reused); hidden at the stem's last token."""
    x = torch.tensor([ids], device="cuda")
    o = model(x, use_cache=True, output_hidden_states=want_h)
    lp0 = torch.log_softmax(o.logits[0, -1].float(), -1)
    h = np.stack([o.hidden_states[l][0, -1].float().cpu().numpy() for l in LAYERS]) if want_h else None
    n = len(KIDS); C = max(len(c) for c in KIDS)
    cache = o.past_key_values; cache.batch_repeat_interleave(n)
    inp = torch.full((n, C), tok.pad_token_id or 0, device="cuda"); m = torch.zeros((n, C), dtype=torch.long, device="cuda")
    for i, c in enumerate(KIDS): inp[i, :len(c)] = torch.tensor(c); m[i, :len(c)] = 1
    att = torch.cat([torch.ones((n, len(ids)), dtype=torch.long, device="cuda"), m], 1)
    pos = torch.arange(len(ids), len(ids) + C, device="cuda").unsqueeze(0).expand(n, C)
    o2 = model(inp, attention_mask=att, position_ids=pos, past_key_values=cache)
    lp = torch.log_softmax(o2.logits.float(), -1)
    sc = []
    for i, c in enumerate(KIDS):
        s = lp0[c[0]].item() + sum(lp[i, j - 1, c[j]].item() for j in range(1, len(c)))
        sc.append(s)
    return np.array(sc), h

if stage in ("extract", "all"):
    ARM = os.environ.get("ARM", "main"); AO = f"{OUT}/{ARM}"
    rows, H, HS, SC = [], [], [], []
    recs = [json.loads(l) for l in open(f"{AO}/scen.jsonl")] + [json.loads(l) for l in open(f"{AO}/wild.jsonl")]
    for r in recs:
        msgs = r["msgs"]
        a_idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
        for t, ai in enumerate(a_idx):
            ids, P = points(msgs, ai)
            H.append(hs(ids, P).astype(np.float16))
            sc, h = stem_score(stem_prefix(msgs, ai))
            SC.append(sc); HS.append(h.astype(np.float16))
            k = S.extract_kaomoji(msgs[ai]["content"])
            rows.append(dict(id=r["id"], type=r["type"], base=r.get("base"), mood=r.get("mood"), sit=r.get("sit"),
                             sample=r.get("sample"), turn=t + 1, final=(t == len(a_idx) - 1), kaomoji=k,
                             finish=r.get("finish") if t == len(a_idx) - 1 else None,
                             reply=msgs[ai]["content"][:600], test=r.get("test")))
        if len(rows) % 200 < len(a_idx): print("extract", len(rows), flush=True)
    np.save(f"{AO}/H.npy", np.stack(H))      # N x L x 3(last user, pre-reply, first reply token) x d
    np.save(f"{AO}/HS.npy", np.stack(HS))    # N x L x d at the end of "I'm feeling"
    np.save(f"{AO}/KSC.npy", np.stack(SC))   # N x K log-probs of each kaomoji after "I'm feeling"
    with open(f"{AO}/rows.jsonl", "w") as f:
        for x in rows: f.write(json.dumps(x) + "\n")
    print("extract done", len(rows), flush=True)
