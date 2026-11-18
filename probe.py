"""Run from a folder holding act.py, scen.py, chosen.json, axes.npz, main_rows.csv and main/{scen,wild,rows}.jsonl (copied from results/). analyze.py's probe_pred (GroupKFold(5), RidgeCV alphas logspace(0,5,11)) on the re-extracted stem activations."""
import sys, json, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
import scen as S
OUTD = sys.argv[1]
L = json.load(open("chosen.json"))["layer"]; ax = np.load("axes.npz"); li = list(ax["layers"]).index(L); VAL = ax[f"val{L}"]
import os
HS = np.load(f"{OUTD}/HS.npy")[:, li].astype(np.float32); ksc = np.load(f"{OUTD}/KSC.npy") if os.path.exists(f"{OUTD}/KSC.npy") else None
rows = pd.read_csv("main_rows.csv")
assert len(rows) == len(HS)
# kao_val recomputed from the new log-probs must match the saved column (same face ratings via fv implied by saved kao_val)
if ksc is not None:
    p = np.exp(ksc - ksc.max(1, keepdims=True)); p /= p.sum(1, keepdims=True); top = [S.KSET[i] for i in p.argmax(1)]
else: top = [None] * len(rows)
w = (rows["type"] == "wild").values
pr = HS @ VAL; val_stem_new = (pr - pr[w].mean()) / pr[w].std()
out = dict(layer=int(L), kao_top_match=float(np.mean(np.array(top) == rows["kao_top"].values)),
           val_stem_corr_with_saved=float(np.corrcoef(val_stem_new, rows["val_stem"].values)[0, 1]),
           val_stem_maxabsdiff=float(np.abs(val_stem_new - rows["val_stem"].values).max()))
def probe_pred(X, y, groups):
    pr = np.full(len(y), np.nan); gk = GroupKFold(5)
    for tr, te in gk.split(X, y, groups):
        m = RidgeCV(alphas=np.logspace(0, 5, 11)).fit(X[tr], y[tr]); pr[te] = m.predict(X[te])
    return pr
rng = np.random.default_rng(0)
def boot_ci(f, groups, nb=2000):
    ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}; vals = []
    for _ in range(nb):
        s = rng.choice(ug, len(ug)); vals.append(f(np.concatenate([idx[g] for g in s])))
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]
d = rows.reset_index().dropna(subset=["kao_val", "reader_pre", "reader_post"])
y = d["kao_val"].values; g = d["cluster"].values; ii = d["index"].values
pp = probe_pred(HS[ii], y, g)
probe = np.full(len(rows), np.nan); probe[ii] = pp
for nm, v in (("probe_stem", pp), ("val_stem_new", val_stem_new[ii]), ("val_stem_saved", d["val_stem"].values), ("reader_pre", d["reader_pre"].values)):
    f = lambda jj, v=v: spearmanr(v[jj], y[jj])[0]
    out[nm] = dict(rho=float(f(np.arange(len(y)))), ci=boot_ci(f, g))
out["n"] = int(len(y))
pd.DataFrame({"probe_stem": probe, "val_stem_new": val_stem_new}).to_csv(f"{OUTD}/probe_stem.csv", index=False)
json.dump(out, open(f"{OUTD}/probe_check.json", "w"), indent=1); print(json.dumps(out, indent=1))
