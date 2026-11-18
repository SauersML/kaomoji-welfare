"""Regression test for the cluster-bootstrap fix (Vesper's review, K1): a base drawn twice must count twice.
Bases A and B have situation contrasts 0 and 3; the draw A,A,B must give (2*0 + 3)/3 = 1, not (0 + 3)/2 = 1.5.
Also with an incomplete stratum: B has no sad rows, so the sad-mood contrast comes from A alone."""
import numpy as np, pandas as pd
from boot import cell_means, est_w, draw_weights


def rows(base, contrast, moods=("happy", "sad")):
    return [dict(base=base, sit=s, mood=m, y=(contrast if s == "good" else 0.0)) for m in moods for s in ("good", "bad")]


def test_repeated_base_counts():
    df = pd.DataFrame(rows("A", 0.0) + rows("B", 3.0))
    bases, M = cell_means(df, "y"); assert list(bases) == ["A", "B"]
    sit = est_w(M, draw_weights([0, 0, 1], 2))[0]
    assert np.isclose(sit, 1.0), sit


def test_incomplete_stratum():
    df = pd.DataFrame(rows("A", 0.0) + rows("B", 3.0, moods=("happy",)))
    bases, M = cell_means(df, "y")
    sit = est_w(M, draw_weights([0, 0, 1], 2))[0]
    assert np.isclose(sit, (1.0 + 0.0) / 2), sit   # happy: (2*0+3)/3 = 1; sad: A only = 0


if __name__ == "__main__":
    test_repeated_base_counts(); test_incomplete_stratum(); print("ok")
