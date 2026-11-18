"""Rater-free face valence: Qwen3-8B reads each face alone; layer-18 residual stream averaged over the face's tokens,
projected on the valence axis (axes.npz val18). Two contexts:
  user:    chat template, user message = the face, no other text (generation prompt not added)
  feeling: plain text "I'm feeling <face>" (no chat template)
Writes face_internal_raw.json {face: [proj_user, proj_feeling]} (raw projections; z-scored later with val_stem's constants)."""
import json, sys
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
faces = [x["face"] for x in json.load(open(sys.argv[1]))]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="cuda").eval()
v = torch.tensor(np.load("axes.npz")["val18"], dtype=torch.float32, device="cuda")
def spans(f):
    a = tok.apply_chat_template([{"role": "user", "content": f}], tokenize=False, enable_thinking=False)
    i0 = a.index(f); pre = tok(a[:i0], add_special_tokens=False)["input_ids"]; full = tok(a, add_special_tokens=False)["input_ids"]
    nf = len(tok(a[:i0 + len(f)], add_special_tokens=False)["input_ids"])
    b = "I'm feeling " + f; pb = tok("I'm feeling", add_special_tokens=False)["input_ids"]; fb = tok(b, add_special_tokens=False)["input_ids"]
    return (full, len(pre), nf), (fb, len(pb), len(fb))
@torch.no_grad()
def proj(batch):
    L = max(len(x[0]) for x in batch)
    ids = torch.full((len(batch), L), tok.pad_token_id, device="cuda"); m = torch.zeros_like(ids)
    for i, (s, a, b) in enumerate(batch): ids[i, :len(s)] = torch.tensor(s); m[i, :len(s)] = 1
    h = model(ids, attention_mask=m, output_hidden_states=True).hidden_states[18].float()
    return [float(h[i, a:b].mean(0) @ v) for i, (s, a, b) in enumerate(batch)]
out = {}
S = [spans(f) for f in faces]
for k in range(0, len(faces), 256):
    pu = proj([s[0] for s in S[k:k + 256]]); pf = proj([s[1] for s in S[k:k + 256]])
    for f, a, b in zip(faces[k:k + 256], pu, pf): out[f] = [a, b]
json.dump(out, open("face_internal_raw.json", "w"), ensure_ascii=False)
print("internal", len(out))
