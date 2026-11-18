from kstyle import *
from scipy.stats import spearmanr
def rho_ci(d, x, y, nb=2000, seed=0):
    d = d.dropna(subset=[x, y]); g = np.random.default_rng(seed); cl = d["cluster"].values; u = np.unique(cl)
    ix = {c: np.where(cl == c)[0] for c in u}; X, Y = d[x].values, d[y].values
    bs = []
    for _ in range(nb):
        i = np.concatenate([ix[c] for c in g.choice(u, len(u))]); bs.append(spearmanr(X[i], Y[i])[0])
    return spearmanr(X, Y)[0], np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(d)
def plotted():
    d = rows(); return d[(d.final == True)]
if __name__ == "__main__":
    d = plotted(); print(d.type.value_counts().to_dict())
    for x in ("val_stem", "reader_post", "reader_pre"): print(x, np.round(rho_ci(d, x, "kao_val"), 3))
    print(R["main_T4_predictors"]["kao_val"].keys())
