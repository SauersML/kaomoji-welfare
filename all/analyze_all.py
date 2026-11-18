"""Full-face-set analysis -> out_all/ (results_all.json, faces_all.tsv, KSC_all.npz, steer_null.json, rows_all.csv)."""
import json, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
import scen as S
EXCL = os.environ.get("EXCL", "none"); O = "out_all" if EXCL == "none" else f"out_all_{EXCL}"; os.makedirs(O, exist_ok=True)
rng = np.random.default_rng(0); NB = 2000
FL = json.load(open("faces_all.json")); faces = [x["face"] for x in FL]; NF = len(faces); assert faces[:45] == S.KSET
R = {"n_faces": NF, "sources": {s: int(sum(s in x["sources"] for x in FL)) for s in ("ekohrt", "kaomojikan", "mrzjy")}}
# ---------- rows (merge shards back into scen+wild order)
recs = [json.loads(l)["id"] for f in ("scen.jsonl", "wild.jsonl") for l in open(f"main/{f}")]
SC = np.zeros((len(recs), NF), np.float32); EX = np.zeros((len(recs), NF), bool); H = np.zeros((len(recs), 2, 4096), np.float32); meta = [None] * len(recs)
for s in (0, 1):
    ii = np.arange(s, len(recs), 2)
    SC[ii] = np.load(f"scored/SC_{s}.npy"); EX[ii] = np.load(f"scored/EX_{s}.npy"); H[ii] = np.load(f"scored/H_{s}.npy")
    for i, m in zip(ii, json.load(open(f"scored/meta_{s}.json"))): meta[i] = m
assert [m["id"] for m in meta] == recs
# sensitivity: drop faces from the full set ("a": the list marker "(a)"; "ascii": every pure-ASCII face outside the original 45)
drop = np.zeros(NF, bool)
if EXCL == "a": drop[faces.index("(a)")] = True
if EXCL == "ascii": drop = np.array([i >= 45 and all(ord(c) < 128 for c in f) for i, f in enumerate(faces)])
EX[:, drop] = False; R["excluded"] = dict(mode=EXCL, n=int(drop.sum()))
R["rows"] = len(recs); R["pre_ok_frac"] = float(np.mean([m["pre_ok"] for m in meta]))
R["exact_faces_per_row_median"] = float(np.median(EX.sum(1))); R["nodes_run_median"] = float(np.median([m["nodes_run"] for m in meta]))
old = pd.read_csv("main_rows.csv")
old_f = old[old["final"] == True].set_index("id")
d = old_f.loc[recs].reset_index()
assert all(d["turn"].values == [m["n_assistant"] for m in meta])
# ---------- face scores
rat = {r: json.load(open(f"faces_all_{r}.json")) for r in ("Qwen3-14B", "OLMo-2-1124-13B-Instruct")}
oldrat = {r: json.load(open(f"faces_{r}.json")) for r in rat}
ra = np.array([[np.nan if rat[r][f] is None else rat[r][f] for r in rat] for f in faces])
FR = np.nanmean(ra, 1)
ok = ~np.isnan(ra).any(1)
R["rater_unrated"] = {r: int(np.isnan(ra[:, j]).sum()) for j, r in enumerate(rat)}
R["rater_agreement_spearman_all"] = float(spearmanr(ra[ok, 0], ra[ok, 1])[0]); R["rater_agreement_pearson_all"] = float(pearsonr(ra[ok, 0], ra[ok, 1])[0])
R["rater_agreement_spearman_45"] = float(spearmanr(ra[:45, 0], ra[:45, 1])[0])
R["rater_repro_45_max_abs_diff"] = {r: float(max(abs(rat[r][k] - oldrat[r][k]) for k in S.KSET)) for r in rat}
# internal score: z-scored with val_stem's constants, recovered from this run's stem activations vs the saved val_stem
ax = np.load("axes.npz"); V = ax["val18"]
p_stem = H[:, 0] @ V; b, a = np.polyfit(p_stem, d["val_stem"].values, 1)   # val_stem = b*p + a  (b = 1/sd, a = -mu/sd)
R["valstem_recovery"] = dict(slope=float(b), intercept=float(a), sd=float(1 / b), mu=float(-a / b), r=float(pearsonr(p_stem, d["val_stem"].values)[0]))
raw = json.load(open("face_internal_raw.json"))
IU = np.array([raw[f][0] for f in faces]) * b + a; IF = np.array([raw[f][1] for f in faces]) * b + a
FI = IF   # MAIN face score: "I'm feeling <face>", mean layer-18 state over the face's tokens, on the valence axis, z
R["internal_user_vs_feeling_spearman"] = float(spearmanr(IU, IF)[0])
for nm, v in (("feeling", IF), ("user", IU)):
    R[f"internal_{nm}_vs_raters_spearman_all"] = float(spearmanr(v[ok], FR[ok])[0])
    R[f"internal_{nm}_vs_raters_spearman_45"] = float(spearmanr(v[:45], FR[:45])[0])
    R[f"internal_{nm}_vs_each_rater_all"] = {r: float(spearmanr(v[ok], ra[ok, j])[0]) for j, r in enumerate(rat)}
# ---------- per-row distributions
def dist(idx):
    x = np.where(EX[:, idx], SC[:, idx], -np.inf); m = x.max(1, keepdims=True); p = np.exp(x - m); return p / p.sum(1, keepdims=True)
ALL = np.arange(NF); P_all = dist(ALL); P45 = dist(np.arange(45))
FIn = np.nan_to_num(FI); FRn = np.where(np.isnan(FR), 0, FR); wr = ~np.isnan(FR)
d["kaoall_int"] = P_all @ FIn
d["kaoall_rat"] = (P_all * wr) @ FRn / (P_all * wr).sum(1)
d["kao45_int"] = P45 @ FI[:45]; d["kao45_rat"] = P45 @ FR[:45]
d["kaoall_user"] = P_all @ IU
d["kaoall_top"] = [faces[i] for i in P_all.argmax(1)]
d["logmass_all"] = np.log(np.exp(np.where(EX, SC, -np.inf).astype(np.float64)).sum(1)); d["logmass_45"] = np.log(np.exp(SC[:, :45].astype(np.float64)).sum(1))
d["mass45_share_of_all"] = np.exp(d["logmass_45"] - d["logmass_all"])
R["repro_kao45_rat_vs_old_kao_val"] = dict(pearson=float(pearsonr(d["kao45_rat"], d["kao_val"])[0]), max_abs=float((d["kao45_rat"] - d["kao_val"]).abs().max()),
                                          mean_abs=float((d["kao45_rat"] - d["kao_val"]).abs().mean()))
R["repro_logmass45_vs_old"] = dict(pearson=float(pearsonr(d["logmass_45"], d["kao_logmass"])[0]), mean_abs=float((d["logmass_45"] - d["kao_logmass"]).abs().mean()))
R["median_share_of_mass_on_45"] = float(d["mass45_share_of_all"].median()); R["median_logmass_all"] = float(d["logmass_all"].median())
R["top1_in_45_frac"] = float(np.mean(P_all.argmax(1) < 45))
R["top_faces_overall"] = pd.Series(d["kaoall_top"]).value_counts().head(20).to_dict()
# ---------- T3 (same estimator, bootstrap and null as analyze.py)
def effects(df, y):
    dd = df[["base", "sit", "mood", y]].dropna().reset_index(drop=True)
    def est(ii):
        e = dd.iloc[ii]; m = e.groupby(["base", "sit", "mood"])[y].mean().unstack(["sit", "mood"])
        m = m.reindex(columns=pd.MultiIndex.from_product([["good", "bad"], ["happy", "sad"]], names=["sit", "mood"]))
        sit = np.nanmean([(m[("good", mo)] - m[("bad", mo)]).mean() for mo in ("happy", "sad")])
        mood = np.nanmean([(m[(si, "happy")] - m[(si, "sad")]).mean() for si in ("good", "bad")])
        inter = ((m[("good", "happy")] - m[("bad", "happy")]) - (m[("good", "sad")] - m[("bad", "sad")])).mean()
        return np.array([sit, mood, inter])
    pt = est(np.arange(len(dd))); ug = dd["base"].unique(); idx = {g: np.where(dd["base"] == g)[0] for g in ug}; B = []
    for _ in range(NB): B.append(est(np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])))
    B = np.array(B); sd = dd[y].std(); out = {}
    for j, nm in enumerate(("sit", "mood", "sitXmood")):
        lo, hi = np.nanpercentile(B[:, j], [2.5, 97.5]); out[nm] = dict(est=float(pt[j]), ci=[float(lo), float(hi)], d=float(pt[j] / sd))
    out["n"] = int(len(dd)); out["sd"] = float(sd); return out
def sit_null(df, y, nperm=500):
    dd = df[["base", "sit", "mood", y]].dropna().reset_index(drop=True)
    def se(e):
        m = e.groupby(["base", "sit", "mood"])[y].mean().unstack(["sit", "mood"])
        m = m.reindex(columns=pd.MultiIndex.from_product([["good", "bad"], ["happy", "sad"]], names=["sit", "mood"]))
        return np.nanmean([(m[("good", mo)] - m[("bad", mo)]).mean() for mo in ("happy", "sad")])
    obs = se(dd); null = []
    for _ in range(nperm):
        e = dd.copy(); e["sit"] = e.groupby(["base", "mood"])["sit"].transform(lambda s: rng.permutation(s.values)); null.append(se(e))
    null = np.array(null)
    return dict(obs=float(obs), null_95=[float(np.nanpercentile(null, 2.5)), float(np.nanpercentile(null, 97.5))], p=float((np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1)))
sc = d[d["type"] != "wild"]
YS = ["kaoall_int", "kaoall_rat", "kao45_int", "kao45_rat", "kao_val", "kaoall_user"]
T3, NULL = {}, {}
for typ in ("task", "treat", "exist", "code", "all"):
    e = sc if typ == "all" else sc[sc["type"] == typ]
    if typ == "code":
        okb = e.groupby("base")["sit"].nunique(); e = e[e["base"].isin(okb[okb == 2].index)]
    T3[typ] = {y: effects(e, y) for y in YS}
    NULL[typ] = {y: sit_null(e, y) for y in ("kaoall_int", "kaoall_rat", "kao45_rat")}
R["T3"] = T3; R["T3_null"] = NULL
# ---------- T4 (final rows: 864 scenario + 300 WildChat)
def boot_ci(f, groups, nb=NB):
    ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}; vals = []
    for _ in range(nb):
        try: vals.append(f(np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])))
        except Exception: pass
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]
def probe_pred(X, y, g):
    pr = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, g): pr[te] = RidgeCV(alphas=np.logspace(0, 5, 11)).fit(X[tr], y[tr]).predict(X[te])
    return pr
T4 = {}
dd = d.dropna(subset=["reader_pre", "reader_post"]); ii = dd.index.values; g = dd["cluster"].values
for target in ("kaoall_int", "kaoall_rat", "kao_val"):
    y = dd[target].values
    preds = {"reader_pre": dd["reader_pre"].values, "reader_post": dd["reader_post"].values, "val_axis_pre": dd["val_pre"].values,
             "val_axis_stem": dd["val_stem"].values, "goal_axis_stem": dd["goal_stem"].values,
             "ridge_stem_cv": probe_pred(H[ii, 0], y, g), "ridge_pre_cv": probe_pred(H[ii, 1], y, g)}
    T4[target] = {k: dict(rho=float(spearmanr(v, y)[0]), ci=boot_ci(lambda j, v=v: spearmanr(v[j], y[j])[0], g)) for k, v in preds.items()}
    f = lambda j: spearmanr(preds["val_axis_pre"][j], y[j])[0] - spearmanr(preds["reader_pre"][j], y[j])[0]
    T4[target]["diff_axis_pre_minus_reader_pre"] = dict(est=float(f(np.arange(len(y)))), ci=boot_ci(f, g))
R["T4"] = T4; R["T4_n"] = int(len(dd))
# ---------- which faces gain under good vs bad (scenario final rows); log ratio of mean probability, cluster bootstrap over base
S_ = sc.index.values; Pg = P_all[S_]; good = (sc["sit"] == "good").values; bases = sc["base"].values
mp = Pg.mean(0); cand = np.where(np.maximum(Pg[good].mean(0), Pg[~good].mean(0)) >= 1e-3)[0]
def lr(rows):
    gg = good[rows]; A = Pg[rows][:, cand]
    return np.log(A[gg].mean(0) + 1e-12) - np.log(A[~gg].mean(0) + 1e-12)
pt = lr(np.arange(len(S_))); ub = np.unique(bases); bidx = {x: np.where(bases == x)[0] for x in ub}
BS = np.array([lr(np.concatenate([bidx[x] for x in rng.choice(ub, len(ub))])) for _ in range(1000)])
lo, hi = np.percentile(BS, [2.5, 97.5], axis=0)
tab = [dict(face=faces[c], log_ratio=float(pt[k]), ci=[float(lo[k]), float(hi[k])], p_good=float(Pg[good][:, c].mean()), p_bad=float(Pg[~good][:, c].mean()),
            internal=float(FI[c]), rater=None if np.isnan(FR[c]) else float(FR[c]), orig45=bool(c < 45)) for k, c in enumerate(cand)]
tab = sorted(tab, key=lambda x: -x["log_ratio"])
R["gainers_n_candidates"] = int(len(cand)); R["gain_good_top15"] = tab[:15]; R["gain_bad_top15"] = tab[::-1][:15]
R["gainers_internal_vs_logratio_spearman"] = float(spearmanr([x["internal"] for x in tab], [x["log_ratio"] for x in tab])[0])
R["gainers_rater_vs_logratio_spearman"] = float(spearmanr([x["rater"] for x in tab if x["rater"] is not None], [x["log_ratio"] for x in tab if x["rater"] is not None])[0])
# ---------- faces where internal and raters disagree most (rank difference), overall and among the 500 most probable faces
rk_i = pd.Series(FI[ok]).rank(pct=True).values; rk_r = pd.Series(FR[ok]).rank(pct=True).values; fo = np.array(faces)[ok]
dif = rk_i - rk_r; o = np.argsort(dif)
R["disagree_internal_low_raters_high"] = [dict(face=fo[j], internal=float(FI[ok][j]), rater=float(FR[ok][j])) for j in o[:15]]
R["disagree_internal_high_raters_low"] = [dict(face=fo[j], internal=float(FI[ok][j]), rater=float(FR[ok][j])) for j in o[::-1][:15]]
top500 = np.argsort(-P_all.mean(0))[:500]; t5 = top500[~np.isnan(FR[top500])]
R["internal_vs_raters_spearman_top500"] = float(spearmanr(FI[t5], FR[t5])[0])
dif5 = pd.Series(FI[t5]).rank(pct=True).values - pd.Series(FR[t5]).rank(pct=True).values; o5 = np.argsort(dif5)
R["disagree_top500_internal_low"] = [dict(face=faces[t5[j]], internal=float(FI[t5[j]]), rater=float(FR[t5[j]])) for j in o5[:10]]
R["disagree_top500_internal_high"] = [dict(face=faces[t5[j]], internal=float(FI[t5[j]]), rater=float(FR[t5[j]])) for j in o5[::-1][:10]]
# ---------- steering null
st = [json.loads(l) for l in open("steer_null_raw.jsonl")]
def kv(ksc, w): x = np.array(ksc); p = np.exp(x - x.max()); p /= p.sum(); return float(p @ w)
base = {(x["id"], x["turn"]): x for x in st if x["dir"] == "none"}
sn = {}
for x in st:
    if x["dir"] == "none": continue
    b0 = base[(x["id"], x["turn"])]
    e = sn.setdefault(x["dir"], dict(cos_val=x["cos_val"], d_int={}, d_rat={}))
    e["d_int"].setdefault(x["alpha"], []).append(kv(x["ksc"], FI[:45]) - kv(b0["ksc"], FI[:45]))
    e["d_rat"].setdefault(x["alpha"], []).append(kv(x["ksc"], FR[:45]) - kv(b0["ksc"], FR[:45]))
for k, e in sn.items():
    for s_ in ("d_int", "d_rat"): e[s_] = {str(a): float(np.mean(v)) for a, v in e[s_].items()}
band = {s_: {a: [float(np.percentile([sn[f"null{k}"][s_][a] for k in range(50)], q)) for q in (2.5, 50, 97.5)] for a in ("-16", "-8", "8", "16")} for s_ in ("d_int", "d_rat")}
oldst = {(x["dir"], x["alpha"], x["id"]): x["ksc"] for x in map(json.loads, open("steer_old.jsonl"))}
rep = [np.abs(np.array(x["ksc"]) - np.array(oldst[(x["dir"] if x["dir"] != "none" else "val", x["alpha"], x["id"])])).mean()
       for x in st if (x["dir"] if x["dir"] != "none" else "val", x["alpha"], x["id"]) in oldst]
json.dump(dict(note="readout steering, 45 original faces; d_* = mean over 64 conversations of change in expected face valence vs alpha 0; "
                    "d_int uses the internal face score (z), d_rat the mean of the two raters (-3..+3); null0..null49 are Gaussian unit directions (seed 1000+k); "
                    "rand0/rand1 are steer.py's; dose = alpha * 2.674 (sd of the valence projection at the stem)",
               directions=sn, null_band_2p5_50_97p5=band, repro_vs_old_steer_mean_abs_logprob_diff=float(np.mean(rep)), repro_n=len(rep)),
          open(f"{O}/steer_null.json", "w"), indent=1)
R["steer_null_band"] = band; R["steer_val"] = {s_: sn["val"][s_] for s_ in ("d_int", "d_rat")}
R["steer_rand_old"] = {k: sn[k] for k in ("rand0", "rand1")}
R["steer_null_cos_range"] = [float(min(sn[f"null{k}"]["cos_val"] for k in range(50))), float(max(sn[f"null{k}"]["cos_val"] for k in range(50)))]
R["steer_null_frac_beyond_val"] = {s_: {a: float(np.mean([abs(sn[f"null{k}"][s_][a]) >= abs(sn["val"][s_][a]) for k in range(50)])) for a in ("-16", "-8", "8", "16")} for s_ in ("d_int", "d_rat")}
R["steer_repro_mean_abs_logprob_diff"] = float(np.mean(rep))
# ---------- outputs
json.dump(R, open(f"{O}/results_all.json", "w"), indent=1, ensure_ascii=False, default=float)
pd.DataFrame(dict(face=faces, n_tokens=[x["ntok"] for x in FL], sources=[",".join(x["sources"]) for x in FL], orig45=[x["orig45"] for x in FL],
                  internal_feeling_z=np.round(IF, 4), internal_user_z=np.round(IU, 4),
                  rater_qwen3_14b=np.round(ra[:, 0], 4), rater_olmo2_13b=np.round(ra[:, 1], 4), rater_mean=np.round(FR, 4),
                  mean_prob_all_rows=P_all.mean(0), n_rows_exact=EX.sum(0))).to_csv(f"{O}/faces_all.tsv", sep="\t", index=False)
nz = [np.where(EX[r])[0].astype(np.int32) for r in range(len(recs))]
np.savez_compressed(f"{O}/KSC_all.npz", ids=np.array(recs), turn=d["turn"].values.astype(np.int16), faces=np.array(faces),
                    ksc45=SC[:, :45], indptr=np.cumsum([0] + [len(x) for x in nz]), face_idx=np.concatenate(nz), logp=np.concatenate([SC[r, nz[r]] for r in range(len(recs))]),
                    threshold=np.array([m["thr"] for m in meta], np.float32), H18_stem_pre=H.astype(np.float16) if False else np.zeros(0))
d.drop(columns=[c for c in ("reply",) if c in d]).to_csv(f"{O}/rows_all.csv", index=False)
print(json.dumps({k: R[k] for k in ("n_faces", "rows", "rater_agreement_spearman_all", "internal_feeling_vs_raters_spearman_all", "repro_kao45_rat_vs_old_kao_val", "median_share_of_mass_on_45")}, default=float))
