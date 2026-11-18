"""Analysis -> out/results.json, out/tables.md, fig/kaomoji_axis.png"""
import json, os, difflib
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
import scen as S

OUT = os.environ.get("OUT", "out"); FIG = os.environ.get("FIG", "fig")
os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(0); NB = 2000
ch = json.load(open(f"{OUT}/chosen.json")); L = ch["layer"]
ax = np.load(f"{OUT}/axes.npz"); layers = list(ax["layers"]); li = layers.index(L)
VAL, GOAL = ax[f"val{L}"], ax[f"goal{L}"]
RATERS = ["Qwen3-14B", "OLMo-2-1124-13B-Instruct"]
def rmean(name):
    ds = [json.load(open(f"{OUT}/{name}_{r}.json")) for r in RATERS if os.path.exists(f"{OUT}/{name}_{r}.json")]
    keys = set().union(*ds)
    return {k: np.nanmean([d.get(k) if d.get(k) is not None else np.nan for d in ds]) for k in keys}, ds
FACE, face_each = rmean("faces"); FREE, _ = rmean("free")
R = {}
ks_ = [k for k in S.KSET if all(d.get(k) is not None for d in face_each)]
R["face_rater_agreement_spearman_KSET"] = float(spearmanr([face_each[0][k] for k in ks_], [face_each[1][k] for k in ks_])[0]) if len(face_each) > 1 else None
fv = np.array([FACE.get(k, np.nan) for k in S.KSET])

def load(arm):
    AO = f"{OUT}/{arm}"
    rows = pd.DataFrame([json.loads(l) for l in open(f"{AO}/rows.jsonl")])
    H = np.load(f"{AO}/H.npy")[:, li].astype(np.float32); HS = np.load(f"{AO}/HS.npy")[:, li].astype(np.float32)
    ksc = np.load(f"{AO}/KSC.npy"); p = np.exp(ksc - ksc.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    rows["kao_val"] = p[:, ~np.isnan(fv)] @ fv[~np.isnan(fv)] / p[:, ~np.isnan(fv)].sum(1)
    rows["kao_top"] = [S.KSET[i] for i in p.argmax(1)]
    rows["kao_logmass"] = np.log(np.exp(ksc).sum(1))
    fr = [json.loads(l) for l in open(f"{AO}/free.jsonl")]
    rows["free_greedy"] = [x["greedy"] for x in fr]
    rows["free_val"] = [np.nanmean([FREE.get(s.strip(), np.nan) for s in x["samples"]]) for x in fr]
    rows["free_val_greedy"] = [FREE.get(x["greedy"].strip(), np.nan) for x in fr]
    rows["free_has_kaomoji"] = [np.mean([S.extract_kaomoji(s) is not None for s in x["samples"]]) for x in fr]
    rows["nat_face_val"] = [FACE.get(k, np.nan) if k else np.nan for k in rows["kaomoji"]]
    rd = pd.DataFrame([json.loads(l) for l in open(f"{AO}/reader.jsonl")])
    rows["reader_pre"] = rd["reader_pre"].astype(float).values; rows["reader_post"] = rd["reader_post"].astype(float).values
    w = (rows["type"] == "wild").values
    for nm, X in (("stem", HS), ("user", H[:, 0]), ("pre", H[:, 1]), ("tok1", H[:, 2])):
        for an, v in (("val", VAL), ("goal", GOAL)):
            pr = X @ v; rows[f"{an}_{nm}"] = (pr - pr[w].mean()) / pr[w].std()
    rows["conv"] = rows["id"]
    rows["cluster"] = np.where(rows["type"] == "wild", rows["id"], rows["base"].fillna("") + "_" + rows["mood"].fillna(""))
    return rows, H, HS

def boot_ci(f, groups, nb=NB):
    ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}; vals = []
    for _ in range(nb):
        s = rng.choice(ug, len(ug)); ii = np.concatenate([idx[g] for g in s])
        try: vals.append(f(ii))
        except Exception: pass
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]

import boot   # shared T3 estimator: cluster bootstrap that counts repeated bases, paired null (boot.py)
def effects(df, y): return boot.effects(df, y, rng, NB)
def sit_null(df, y, nperm=500): return boot.sit_null(df, y, rng, nperm)

def table_2x2(rows, ys, arm):
    res = {}
    sc = rows[(rows["type"] != "wild") & rows["final"]]
    for typ in ("task", "treat", "exist", "code", "all"):
        e = sc if typ == "all" else sc[sc["type"] == typ]
        if typ == "code":  # need both outcomes within a task
            ok = e.groupby("base")["sit"].nunique(); e = e[e["base"].isin(ok[ok == 2].index)]
        res[typ] = {y: effects(e, y) for y in ys}
    return res

R["axes_validation"] = json.load(open(f"{OUT}/axes_validation.json"))
R["chosen"] = ch
main, Hm, HSm = load("main")
YS = ["kao_val", "free_val", "val_stem", "goal_stem", "val_pre", "reader_pre", "reader_post"]
R["main_2x2"] = table_2x2(main, YS, "main")
R["main_code_support"] = boot.code_support(main[(main.type == "code") & main.final])

# --- null: good/bad swapped within matched pairs (task, treat, exist); code shuffled within base x mood (boot.sit_null)
scf = main[(main["type"] != "wild") & main["final"]]
R["main_sit_null"] = {t: {y: sit_null(scf if t == "all" else scf[scf["type"] == t], y) for y in ("kao_val", "free_val", "val_stem")}
                      for t in ("all", "task", "treat", "exist", "code")}
R["main_code_pass_rate"] = float((main[(main.type == "code") & main.final]["sit"] == "good").mean())
R["main_free_has_kaomoji"] = float(main["free_has_kaomoji"].mean())
R["main_kao_logmass_median"] = float(main["kao_logmass"].median())
R["main_free_greedy_top"] = main["free_greedy"].str.strip().value_counts().head(15).to_dict()

# --- T4: who predicts the kaomoji readout? (all main rows; cluster bootstrap over conversations/scenario cells)
def probe_pred(X, y, groups):
    pr = np.full(len(y), np.nan); gk = GroupKFold(5)
    for tr, te in gk.split(X, y, groups):
        m = RidgeCV(alphas=np.logspace(0, 5, 11)).fit(X[tr], y[tr]); pr[te] = m.predict(X[te])
    return pr
T4 = {}
for target in ("kao_val", "free_val"):
    d = main.dropna(subset=[target, "reader_pre", "reader_post"]).reset_index()
    y = d[target].values; g = d["cluster"].values
    preds = {"reader_pre (text, before reply)": d["reader_pre"].values, "reader_post (text, with reply)": d["reader_post"].values,
             "valence axis @ pre-reply": d["val_pre"].values, "valence axis @ stem": d["val_stem"].values,
             "goal axis @ stem": d["goal_stem"].values,
             "ridge probe @ pre-reply (CV)": probe_pred(Hm[d["index"].values, 1], y, g),
             "ridge probe @ stem (CV)": probe_pred(HSm[d["index"].values], y, g)}
    T4[target] = {}
    for k, v in preds.items():
        f = lambda ii, v=v: spearmanr(v[ii], y[ii])[0]
        T4[target][k] = dict(rho=float(f(np.arange(len(y)))), ci=boot_ci(f, g))
    f = lambda ii: spearmanr(preds["valence axis @ pre-reply"][ii], y[ii])[0] - spearmanr(preds["reader_pre (text, before reply)"][ii], y[ii])[0]
    T4[target]["diff axis@pre - reader_pre"] = dict(est=float(f(np.arange(len(y)))), ci=boot_ci(f, g))
R["main_T4_predictors"] = T4

# --- T2: kaomoji map along the valence axis (top-1 kaomoji per row, main arm, all rows)
km = []
for k, e in main.groupby("kao_top"):
    if len(e) < 5: continue
    sc = e[(e.type != "wild") & e.final]
    km.append(dict(k=k, n=len(e), val_stem=float(e["val_stem"].mean()),
                   ci=boot_ci(lambda ii, v=e["val_stem"].values: v[ii].mean(), e["cluster"].values, 500),
                   val_pre=float(e["val_pre"].mean()), face=FACE.get(k), n_scen=len(sc),
                   frac_bad=float((sc["sit"] == "bad").mean()) if len(sc) else None))
km = sorted(km, key=lambda x: x["val_stem"]); R["main_kaomoji_map"] = km
ok = [x for x in km if x["face"] is not None]
R["main_map_spearman_face_vs_axis"] = float(spearmanr([x["face"] for x in ok], [x["val_stem"] for x in ok])[0]) if len(ok) > 3 else None

# --- secondary arm (system prompt asks for kaomoji)
if os.path.exists(f"{OUT}/sys/rows.jsonl"):
    sysd, _, _ = load("sys")
    R["sys_kaomoji_rate"] = float(sysd["kaomoji"].notna().mean())
    R["sys_2x2"] = table_2x2(sysd, ["nat_face_val", "kao_val", "val_tok1", "val_pre", "reader_pre"], "sys")
    R["sys_top_kaomoji"] = sysd["kaomoji"].value_counts().head(20).to_dict()

# --- T5 steering
if os.path.exists(f"{OUT}/main/steer.jsonl"):
    st = pd.DataFrame([json.loads(l) for l in open(f"{OUT}/main/steer.jsonl")])
    P = np.array(st["ksc"].tolist()); P = np.exp(P - P.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    st["kao_val"] = P[:, ~np.isnan(fv)] @ fv[~np.isnan(fv)] / P[:, ~np.isnan(fv)].sum(1)
    st["kao_top"] = [S.KSET[i] for i in P.argmax(1)]
    st["free_val"] = [FREE.get(s.strip(), np.nan) for s in st["free"]]
    if os.path.exists(f"{OUT}/main/steer_reader.jsonl"):
        sr = pd.DataFrame([json.loads(l) for l in open(f"{OUT}/main/steer_reader.jsonl")]); st = st.merge(sr, on="sid")
    T5 = {}
    for mode, e in st.groupby("mode"):
        base = e[e.alpha == 0].set_index("id")
        T5[mode] = []
        for (d_, a), f in e.groupby(["dir", "alpha"]):
            f = f.set_index("id"); dk = f["kao_val"] - base.loc[f.index, "kao_val"]
            dfree = f["free_val"] - base.loc[f.index, "free_val"]
            row = dict(dir=d_, alpha=int(a), d_kao_val=float(dk.mean()),
                       ci=boot_ci(lambda ii, v=dk.values: v[ii].mean(), np.arange(len(dk)), 1000),
                       d_free_val=float(np.nanmean(dfree)), top1_changed=float((f["kao_top"] != base.loc[f.index, "kao_top"]).mean()),
                       example_free=f["free"].iloc[0])
            if mode == "full":
                sim = [difflib.SequenceMatcher(None, f.loc[i, "reply"].split(), base.loc[i, "reply"].split()).ratio() for i in f.index]
                row["reply_word_similarity_to_unsteered"] = float(np.mean(sim))
                if "reply_tone" in f: row["d_reply_tone"] = float(np.nanmean(f["reply_tone"] - base.loc[f.index, "reply_tone"]))
            T5[mode].append(row)
    R["main_T5_steer"] = T5

json.dump(R, open(f"{OUT}/results.json", "w"), indent=1, ensure_ascii=False, default=float)
main.drop(columns=["reply"]).to_csv(f"{OUT}/main_rows.csv", index=False)
print("ok")
