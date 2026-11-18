"""Print the rule's top N kept entries by mean probability (renormalised over kept kaomoji), the top 50 dropped (share of the full set),
and 200 random kept and 100 random dropped entries (seed argv[2]) for hand checking: python all/check_rule.py 100 2 > results/main/kaomoji_check.txt"""
import sys, random, numpy as np, collections
from kaomoji_rule import why_not
Z = np.load(sys.argv[3] if len(sys.argv) > 3 else "results/main/KSC_all.npz"); faces = [str(f) for f in Z["faces"]]
rs = [why_not(f) for f in faces]; drop = np.array([r is not None for r in rs])
print("entries", len(faces), "kept", int((~drop).sum()), dict(collections.Counter(r for r in rs if r)))
FI, LP, IP = Z["face_idx"], Z["logp"].astype(np.float64), Z["indptr"]
mk = np.zeros(len(faces)); md = np.zeros(len(faces)); n = len(IP) - 1
for r in range(n):
    fi, lp = FI[IP[r]:IP[r+1]], LP[IP[r]:IP[r+1]]
    k = ~drop[fi]; p = np.exp(lp[k] - lp[k].max()); mk[fi[k]] += p / p.sum() / n
    q = np.exp(lp - lp.max()); q /= q.sum(); md[fi[~k]] += q[~k] / n
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
o = np.argsort(-mk)[:N]; print(f"TOP{N} KEPT"); [print(f"{i+1}\t{mk[j]:.4f}\t{faces[j]}") for i, j in enumerate(o)]
print("cum", mk[o].sum())
o = np.argsort(-md)[:50]; print("TOP50 DROPPED (mean share in full set)"); [print(f"{i+1}\t{md[j]:.4f}\t{rs[j]}\t{faces[j]}") for i, j in enumerate(o)]
random.seed(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
kept = [f for f, r in zip(faces, rs) if r is None]; dr = [(f, r) for f, r in zip(faces, rs) if r]
print("RANDOM 200 KEPT"); [print(i+1, f) for i, f in enumerate(random.sample(kept, 200))]
print("RANDOM 100 DROPPED"); [print(i+1, r, f) for i, (f, r) in enumerate(random.sample(dr, 100))]
