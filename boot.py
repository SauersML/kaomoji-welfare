"""T3 estimator, cluster bootstrap and permutation null, shared by analyze.py, all/analyze_all.py, all/analyze_kaomoji.py
and t3_fixed.py.

Situation effect: within each base scenario and user mood, mean score under good minus under bad; averaged over bases,
then over the two moods. Mood effect likewise with the roles swapped. Interaction: (good-bad | happy) - (good-bad | sad).

Bootstrap: base scenarios are drawn with replacement and a base drawn k times counts k times (weight k). The earlier
code expanded the drawn rows but then grouped by the original base id, so a draw A,A,B estimated (A+B)/2 instead of
(2A+B)/3 (Vesper's review, K1). Point estimates are unchanged; only the intervals move.

Null: in task, treat and exist the good and bad branches of one sample share their user turn (task) or whole first
exchange (treat, exist), so the null swaps the two labels within each matched pair (a random sign per pair), keeping the
pairs intact (K3). In code the outcome (tests pass or fail) is observed after the model wrote the code, not assigned, so
there is no matched pair; its null shuffles the label within base x mood, and its effect is observational.
"""
import numpy as np, pandas as pd

CELLS = [("good", "happy"), ("bad", "happy"), ("good", "sad"), ("bad", "sad")]


def cell_means(df, y):
    """bases (array) and M (n_base x 4): mean of y per base in each (sit, mood) cell, nan where the cell is empty."""
    m = df.groupby(["base", "sit", "mood"])[y].mean().unstack(["sit", "mood"])
    m = m.reindex(columns=pd.MultiIndex.from_tuples(CELLS, names=["sit", "mood"]))
    return m.index.values, m.values.astype(float)


def _wmean(x, w):
    ok = ~np.isnan(x)
    return np.sum(w[ok] * x[ok]) / np.sum(w[ok]) if np.sum(w[ok]) > 0 else np.nan


def est_w(M, w):
    """[sit, mood, sitXmood] with base weights w (w = how many times each base was drawn; all ones = the point estimate)."""
    gh, bh, gs, bs = M.T
    with np.errstate(invalid="ignore"):
        sit = np.nanmean([_wmean(gh - bh, w), _wmean(gs - bs, w)])
        mood = np.nanmean([_wmean(gh - gs, w), _wmean(bh - bs, w)])
        inter = _wmean((gh - bh) - (gs - bs), w)
    return np.array([sit, mood, inter])


def draw_weights(draw, n_base):
    """Weights of a bootstrap draw of base positions, e.g. [0, 0, 1] -> [2, 1]."""
    return np.bincount(np.asarray(draw), minlength=n_base).astype(float)


def effects(df, y, rng, nb=2000):
    dd = df[["base", "sit", "mood", y]].dropna().reset_index(drop=True)
    bases, M = cell_means(dd, y); nbase = len(bases)
    pt = est_w(M, np.ones(nbase))
    B = np.array([est_w(M, draw_weights(rng.integers(0, nbase, nbase), nbase)) for _ in range(nb)])
    sd = dd[y].std(); out = {}
    for j, nm in enumerate(("sit", "mood", "sitXmood")):
        lo, hi = np.nanpercentile(B[:, j], [2.5, 97.5])
        out[nm] = dict(est=float(pt[j]), ci=[float(lo), float(hi)], d=float(pt[j] / sd))
    out["n"] = int(len(dd)); out["n_base"] = int(nbase); out["sd"] = float(sd)
    return out


def sit_null(df, y, rng, nperm=500):
    """Rows whose (base, mood, sample) holds exactly one good and one bad row are matched pairs (task, treat, exist):
    each pair's two labels are swapped with probability 1/2. Other rows (code, or a pair broken by a missing score) are
    shuffled within base x mood among themselves, an observational null."""
    dd = df[["base", "sit", "mood", "sample", y]].dropna(subset=[y]).reset_index(drop=True)
    nb_ = dd["base"].nunique(); eff = lambda e: est_w(cell_means(e, y)[1], np.ones(nb_))[0]
    obs = eff(dd)
    key = dd["base"].astype(str) + "|" + dd["mood"].astype(str) + "|" + dd["sample"].astype(str)
    lab = dd.groupby(key)["sit"].transform(lambda s: "".join(sorted(s)))
    pair = (lab == "badgood").values
    codes, uk = pd.factorize(key)
    sit = dd["sit"].values; other = {k: v[~pair[v]] for k, v in dd.groupby(["base", "mood"]).indices.items()}
    other = [v for v in other.values() if len(v) > 1]; null = []
    for _ in range(nperm):
        flip = (rng.random(len(uk)) < 0.5)[codes] & pair
        s2 = np.where(flip, np.where(sit == "good", "bad", "good"), sit)
        for ii in other: s2[ii] = rng.permutation(sit[ii])
        e = dd.copy(); e["sit"] = s2; null.append(eff(e))
    null = np.array(null)
    return dict(obs=float(obs), rows_paired=int(pair.sum()), rows_unpaired=int((~pair).sum()),
                null_mean=float(np.nanmean(null)), null_95=[float(np.nanpercentile(null, 2.5)), float(np.nanpercentile(null, 97.5))],
                p=float((np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1)))


def code_support(df):
    """Code rows used: bases with both outcomes; rows and passing rows."""
    ok = df.groupby("base")["sit"].nunique(); e = df[df["base"].isin(ok[ok == 2].index)]
    return dict(bases_with_both_outcomes=int((ok == 2).sum()), bases_total=int(len(ok)), rows_used=int(len(e)),
                rows_total=int(len(df)), pass_rows_used=int((e["sit"] == "good").sum()))
