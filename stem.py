"""Run from a folder holding act.py, scen.py, chosen.json, axes.npz, main_rows.csv and main/{scen,wild,rows}.jsonl (copied from results/). Re-extract only the "I'm feeling" stem hidden states (and kaomoji log-probs) for the saved main-arm conversations,
using act.py's own stem_prefix/stem_score (stage 'none': no axes, no regeneration)."""
import sys, os, json, numpy as np
OUTD = sys.argv[1]; sys.argv = [sys.argv[0], "none"]
import act as A, scen as S
recs = [json.loads(l) for l in open("main/scen.jsonl")] + [json.loads(l) for l in open("main/wild.jsonl")]
saved = [json.loads(l) for l in open("main/rows.jsonl")]
HS, SC, n = [], [], 0
for r in recs:
    msgs = r["msgs"]; a_idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    for t, ai in enumerate(a_idx):
        s = saved[n]; assert s["id"] == r["id"] and s["turn"] == t + 1, (n, s["id"], r["id"])
        ids = A.stem_prefix(msgs, ai)
        try:
            sc, h = A.stem_score(ids) if SC is not None else (None, None)
        except Exception as e:
            print("stem_score failed, hidden states only:", repr(e)[:300], flush=True); SC = None; sc, h = None, None
        if h is None:
            with A.torch.no_grad():
                o = A.model(A.torch.tensor([ids], device="cuda"), output_hidden_states=True)
            h = np.stack([o.hidden_states[l][0, -1].float().cpu().numpy() for l in A.LAYERS])
        if SC is not None: SC.append(sc)
        HS.append(h.astype(np.float16)); n += 1
        if n % 200 == 0: print("stem", n, flush=True)
assert n == len(saved)
os.makedirs(OUTD, exist_ok=True)
np.save(f"{OUTD}/HS.npy", np.stack(HS)); (np.save(f"{OUTD}/KSC.npy", np.stack(SC)) if SC else None)
print("stem done", n, "layers", A.LAYERS, flush=True)
