"""Kaomoji-only analysis of the full face set -> out_kao/ (results_kao.json, rows_kao.csv, steer_null_kao.json).
Same estimators as analyze_all.py, run on the faces kaomoji_rule.py keeps; probabilities renormalized over those faces only.
Reads the saved per-row log-probs (results/main/KSC_all.npz) instead of the pod's dense arrays; the valence-axis z constants
come from results_all.json. T4's ridge probe uses the re-extracted stem activations (results/main/HS.npz, stem.py); the
pre-reply activations were not saved, so ridge_pre is not recomputed.
Run from a folder holding scen.py, boot.py, kaomoji_rule.py, rows_all.csv, faces_all.json, KSC_all.npz, results_all.json, main_rows.csv, HS.npz,
axes.npz, chosen.json, the rater files, face_internal_raw.json, steer_null_raw.jsonl, steer_old.jsonl and main/{scen,wild}.jsonl."""
import json, os, time
T0 = time.time()
def step(x): print(f"[{time.time() - T0:6.1f} s] {x}", flush=True)
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
import scen as S
O = "out_kao"; os.makedirs(O, exist_ok=True)
from kaomoji_rule import why_not
rng = np.random.default_rng(0); NB = 2000
FL = json.load(open("faces_all.json")); faces = [x["face"] for x in FL]; NF = len(faces); assert faces[:45] == S.KSET
R = {"n_faces": NF, "sources": {s: int(sum(s in x["sources"] for x in FL)) for s in ("ekohrt", "kaomojikan", "mrzjy")}}
# ---------- rows (merge shards back into scen+wild order)
recs = [json.loads(l)["id"] for f in ("scen.jsonl", "wild.jsonl") for l in open(f"main/{f}")]
Z = np.load("KSC_all.npz"); assert list(Z["ids"]) == recs and list(Z["faces"]) == faces
# keep only literal kaomoji (kaomoji_rule.py); everything else is removed before renormalizing
reason = [why_not(f) for f in faces]; drop = np.array([r is not None for r in reason]); assert not drop[:45].any()
FIDX, LP, IP = Z["face_idx"], Z["logp"].astype(np.float64), Z["indptr"]
LM_ENTRIES = np.array([np.logaddexp.reduce(LP[IP[r]:IP[r + 1]]) for r in range(len(recs))])   # every entry, before the rule
# compact columns: kept faces scored exactly in at least one row (a face never scored exactly has weight < e^-20 of the best
# of the 45 in every row and counts as 0, as in analyze_all.py); the 45 come first, so column j < 45 is original face j
COLS = np.array(sorted(set(np.unique(FIDX).tolist()) - set(np.where(drop)[0].tolist()))); assert (COLS[:45] == np.arange(45)).all()
cpos = np.full(NF, -1); cpos[COLS] = np.arange(len(COLS)); NC = len(COLS)
SC = np.full((len(recs), NC), -np.inf, np.float32); EX = np.zeros((len(recs), NC), bool)
for r in range(len(recs)):
    fi, lp = FIDX[IP[r]:IP[r + 1]], Z["logp"][IP[r]:IP[r + 1]]; k = cpos[fi] >= 0
    SC[r, cpos[fi[k]]] = lp[k]; EX[r, cpos[fi[k]]] = True
assert EX[:, :45].all() and np.allclose(SC[:, :45], Z["ksc45"])
meta = [dict(id=i, n_assistant=int(t), thr=float(th)) for i, t, th in zip(recs, Z["turn"], Z["threshold"])]
R["kaomoji_columns_scored"] = int(NC)
R["kaomoji_rule"] = dict(n_entries=NF, n_kept=int((~drop).sum()), n_dropped=int(drop.sum()),
                         dropped_by_reason={k: int(sum(r == k for r in reason)) for k in ("western", "single", "punctuation", "word")})
R["rows"] = len(recs); R["exact_faces_per_row_median"] = float(np.median(EX.sum(1)))
old = pd.read_csv("main_rows.csv")
old_f = old[old["final"] == True].set_index("id")
d = old_f.loc[recs].reset_index()
assert all(d["turn"].values == [m["n_assistant"] for m in meta])
ra_ = pd.read_csv("rows_all.csv").set_index("id").loc[recs]   # the unfiltered full set (analyze_all.py), for "before filtering"
d["full_int"] = ra_["kaoall_int"].values; d["full_rat"] = ra_["kaoall_rat"].values
step("face scores")
# ---------- face scores
rat = {r: json.load(open(f"faces_all_{r}.json")) for r in ("Qwen3-14B", "OLMo-2-1124-13B-Instruct")}
oldrat = {r: json.load(open(f"faces_{r}.json")) for r in rat}
ra = np.array([[np.nan if rat[r][f] is None else rat[r][f] for r in rat] for f in faces])
FR = np.nanmean(ra, 1)
ok = ~np.isnan(ra).any(1) & ~drop
R["rater_unrated"] = {r: int(np.isnan(ra[:, j]).sum()) for j, r in enumerate(rat)}
R["rater_agreement_spearman_all"] = float(spearmanr(ra[ok, 0], ra[ok, 1])[0]); R["rater_agreement_pearson_all"] = float(pearsonr(ra[ok, 0], ra[ok, 1])[0])
R["rater_agreement_spearman_45"] = float(spearmanr(ra[:45, 0], ra[:45, 1])[0])
R["rater_repro_45_max_abs_diff"] = {r: float(max(abs(rat[r][k] - oldrat[r][k]) for k in S.KSET)) for r in rat}
# internal score: z-scored with val_stem's constants, recovered from this run's stem activations vs the saved val_stem
vr = json.load(open("results_all.json"))["valstem_recovery"]; b, a = vr["slope"], vr["intercept"]   # val_stem = b*projection + a, fitted in analyze_all.py
R["valstem_recovery"] = vr
raw = json.load(open("face_internal_raw.json"))
IU = np.array([raw[f][0] for f in faces]) * b + a; IF = np.array([raw[f][1] for f in faces]) * b + a
FI = IF   # MAIN face score: "I'm feeling <face>", mean layer-18 state over the face's tokens, on the valence axis, z
R["internal_user_vs_feeling_spearman"] = float(spearmanr(IU, IF)[0])
for nm, v in (("feeling", IF), ("user", IU)):
    R[f"internal_{nm}_vs_raters_spearman_all"] = float(spearmanr(v[ok], FR[ok])[0])
    R[f"internal_{nm}_vs_raters_spearman_45"] = float(spearmanr(v[:45], FR[:45])[0])
    R[f"internal_{nm}_vs_each_rater_all"] = {r: float(spearmanr(v[ok], ra[ok, j])[0]) for j, r in enumerate(rat)}
step("per-row distributions")
# ---------- per-row distributions
def dist(idx):
    x = np.where(EX[:, idx], SC[:, idx], -np.inf); m = x.max(1, keepdims=True); p = np.exp(x - m); return p / p.sum(1, keepdims=True)
ALL = np.arange(NC); P_all = dist(ALL); P45 = dist(np.arange(45))
FIn = np.nan_to_num(FI[COLS]); FRn = np.where(np.isnan(FR[COLS]), 0, FR[COLS]); wr = ~np.isnan(FR[COLS])
d["kaoall_int"] = P_all @ FIn
d["kaoall_rat"] = (P_all * wr) @ FRn / (P_all * wr).sum(1)
d["kao45_int"] = P45 @ FI[:45]; d["kao45_rat"] = P45 @ FR[:45]
d["kaoall_user"] = P_all @ IU[COLS]
d["kaoall_top"] = [faces[COLS[i]] for i in P_all.argmax(1)]
d["logmass_all"] = np.log(np.exp(np.where(EX, SC, -np.inf).astype(np.float64)).sum(1))   # kaomoji only
d["logmass_45"] = np.log(np.exp(SC[:, :45].astype(np.float64)).sum(1))
d["kao_share_of_entries"] = np.exp(d["logmass_all"] - LM_ENTRIES)
R["median_kaomoji_share_of_all_entries_mass"] = float(d["kao_share_of_entries"].median())
d["mass45_share_of_all"] = np.exp(d["logmass_45"] - d["logmass_all"])
R["repro_kao45_rat_vs_old_kao_val"] = dict(pearson=float(pearsonr(d["kao45_rat"], d["kao_val"])[0]), max_abs=float((d["kao45_rat"] - d["kao_val"]).abs().max()),
                                          mean_abs=float((d["kao45_rat"] - d["kao_val"]).abs().mean()))
R["repro_logmass45_vs_old"] = dict(pearson=float(pearsonr(d["logmass_45"], d["kao_logmass"])[0]), mean_abs=float((d["logmass_45"] - d["kao_logmass"]).abs().mean()))
R["median_share_of_mass_on_45"] = float(d["mass45_share_of_all"].median()); R["median_logmass_all"] = float(d["logmass_all"].median())
R["top1_in_45_frac"] = float(np.mean(P_all.argmax(1) < 45))
R["top_faces_overall"] = pd.Series(d["kaoall_top"]).value_counts().head(20).to_dict()
step("T3")
# ---------- T3 (same estimator, bootstrap and null as analyze.py)
import boot   # shared T3 estimator: cluster bootstrap that counts repeated bases, paired null (boot.py)
def effects(df, y): return boot.effects(df, y, rng, NB)
def sit_null(df, y, nperm=500): return boot.sit_null(df, y, rng, nperm)
sc = d[d["type"] != "wild"]
YS = ["kaoall_int", "kaoall_rat", "kao45_int", "kao45_rat", "kao_val", "kaoall_user", "full_int", "full_rat", "free_val", "val_stem"]
T3, NULL = {}, {}
for typ in ("task", "treat", "exist", "code", "all"):
    e = sc if typ == "all" else sc[sc["type"] == typ]
    if typ == "code":
        okb = e.groupby("base")["sit"].nunique(); e = e[e["base"].isin(okb[okb == 2].index)]
    T3[typ] = {y: effects(e, y) for y in YS}
    NULL[typ] = {y: sit_null(e, y) for y in ("kaoall_int", "kaoall_rat", "kao45_int", "kao_val", "full_int", "full_rat", "free_val", "val_stem")}
R["T3"] = T3; R["T3_null"] = NULL; R["code_support"] = boot.code_support(sc[sc["type"] == "code"])
step("T4")
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
L18 = json.load(open("chosen.json"))["layer"]; li = list(np.load("axes.npz")["layers"]).index(L18)
HSA = np.load("HS.npz")["HS"][:, li].astype(np.float32)                       # all 2,327 turns, in main_rows.csv order
mr = old.reset_index(drop=True); pos = {(i, t): k for k, (i, t) in enumerate(zip(mr["id"], mr["turn"]))}
HST = HSA[[pos[(i, t)] for i, t in zip(d["id"], d["turn"])]]                   # final turns, in d's order
vchk = HST @ np.load("axes.npz")[f"val{L18}"] * b + a; R["hs_valstem_check_max_abs"] = float(np.abs(vchk - d["val_stem"].values).max())
T4 = {}
dd = d.dropna(subset=["reader_pre", "reader_post"]); ii = dd.index.values; g = dd["cluster"].values
for target in ("kaoall_int", "kaoall_rat", "kao_val", "kao45_int"):
    y = dd[target].values
    preds = {"reader_pre": dd["reader_pre"].values, "reader_post": dd["reader_post"].values, "val_axis_pre": dd["val_pre"].values,
             "val_axis_stem": dd["val_stem"].values, "goal_axis_stem": dd["goal_stem"].values,
             "ridge_stem_cv": probe_pred(HST[ii], y, g)}
    T4[target] = {k: dict(rho=float(spearmanr(v, y)[0]), ci=boot_ci(lambda j, v=v: spearmanr(v[j], y[j])[0], g)) for k, v in preds.items()}
    f = lambda j: spearmanr(preds["val_axis_pre"][j], y[j])[0] - spearmanr(preds["reader_pre"][j], y[j])[0]
    T4[target]["diff_axis_pre_minus_reader_pre"] = dict(est=float(f(np.arange(len(y)))), ci=boot_ci(f, g))
R["T4"] = T4; R["T4_n"] = int(len(dd))
step("which faces gain")
# ---------- which faces gain under good vs bad (scenario final rows); log ratio of mean probability, cluster bootstrap over base
S_ = sc.index.values; Pg = P_all[S_]; good = (sc["sit"] == "good").values; bases = sc["base"].values
mp = Pg.mean(0); cand = np.where(np.maximum(Pg[good].mean(0), Pg[~good].mean(0)) >= 1e-3)[0]
def lr(rows):
    gg = good[rows]; A = Pg[rows][:, cand]
    return np.log(A[gg].mean(0) + 1e-12) - np.log(A[~gg].mean(0) + 1e-12)
pt = lr(np.arange(len(S_))); ub = np.unique(bases); bidx = {x: np.where(bases == x)[0] for x in ub}
BS = np.array([lr(np.concatenate([bidx[x] for x in rng.choice(ub, len(ub))])) for _ in range(1000)])
lo, hi = np.percentile(BS, [2.5, 97.5], axis=0)
tab = [dict(face=faces[COLS[c]], log_ratio=float(pt[k]), ci=[float(lo[k]), float(hi[k])], p_good=float(Pg[good][:, c].mean()), p_bad=float(Pg[~good][:, c].mean()),
            internal=float(FI[COLS[c]]), rater=None if np.isnan(FR[COLS[c]]) else float(FR[COLS[c]]), orig45=bool(c < 45)) for k, c in enumerate(cand)]
tab = sorted(tab, key=lambda x: -x["log_ratio"])
R["gainers_n_candidates"] = int(len(cand)); R["gain_good_top15"] = tab[:15]; R["gain_bad_top15"] = tab[::-1][:15]
R["gainers_internal_vs_logratio_spearman"] = float(spearmanr([x["internal"] for x in tab], [x["log_ratio"] for x in tab])[0])
R["gainers_rater_vs_logratio_spearman"] = float(spearmanr([x["rater"] for x in tab if x["rater"] is not None], [x["log_ratio"] for x in tab if x["rater"] is not None])[0])
step("faces where")
# ---------- faces where internal and raters disagree most (rank difference), overall and among the 500 most probable faces
rk_i = pd.Series(FI[ok]).rank(pct=True).values; rk_r = pd.Series(FR[ok]).rank(pct=True).values; fo = np.array(faces)[ok]
dif = rk_i - rk_r; o = np.argsort(dif)
R["disagree_internal_low_raters_high"] = [dict(face=fo[j], internal=float(FI[ok][j]), rater=float(FR[ok][j])) for j in o[:15]]
R["disagree_internal_high_raters_low"] = [dict(face=fo[j], internal=float(FI[ok][j]), rater=float(FR[ok][j])) for j in o[::-1][:15]]
top500 = COLS[np.argsort(-P_all.mean(0))[:500]]; assert not drop[top500].any(); t5 = top500[~np.isnan(FR[top500])]
R["internal_vs_raters_spearman_top500"] = float(spearmanr(FI[t5], FR[t5])[0])
dif5 = pd.Series(FI[t5]).rank(pct=True).values - pd.Series(FR[t5]).rank(pct=True).values; o5 = np.argsort(dif5)
R["disagree_top500_internal_low"] = [dict(face=faces[t5[j]], internal=float(FI[t5[j]]), rater=float(FR[t5[j]])) for j in o5[:10]]
R["disagree_top500_internal_high"] = [dict(face=faces[t5[j]], internal=float(FI[t5[j]]), rater=float(FR[t5[j]])) for j in o5[::-1][:10]]
step("steering null")
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
          open(f"{O}/steer_null_kao.json", "w"), indent=1)
R["steer_null_band"] = band; R["steer_val"] = {s_: sn["val"][s_] for s_ in ("d_int", "d_rat")}
R["steer_rand_old"] = {k: sn[k] for k in ("rand0", "rand1")}
R["steer_null_cos_range"] = [float(min(sn[f"null{k}"]["cos_val"] for k in range(50))), float(max(sn[f"null{k}"]["cos_val"] for k in range(50)))]
R["steer_null_frac_beyond_val"] = {s_: {a: float(np.mean([abs(sn[f"null{k}"][s_][a]) >= abs(sn["val"][s_][a]) for k in range(50)])) for a in ("-16", "-8", "8", "16")} for s_ in ("d_int", "d_rat")}
R["steer_repro_mean_abs_logprob_diff"] = float(np.mean(rep))
step("outputs")
# ---------- outputs
json.dump(R, open(f"{O}/results_kao.json", "w"), indent=1, ensure_ascii=False, default=float)
d.drop(columns=[c for c in ("reply",) if c in d]).to_csv(f"{O}/rows_kao.csv", index=False)
print(json.dumps({k: R[k] for k in ("kaomoji_rule", "rows", "rater_agreement_spearman_all", "internal_feeling_vs_raters_spearman_all", "repro_kao45_rat_vs_old_kao_val", "median_share_of_mass_on_45")}, default=float))
