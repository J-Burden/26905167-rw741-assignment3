import json, time, sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
from common import *

KS = [1, 3, 5, 7, 9, 11, 15, 21, 31]
KMAX = max(KS)
NC = 10


class LogDamp:
    def fit(self, X):
        from scipy.stats import skew
        s = skew(X, axis=0, nan_policy="omit")
        self.cols = np.where((X.min(axis=0) >= 0) & (np.abs(s) > 1))[0]
        return self
    def transform(self, X):
        X = X.copy(); X[:, self.cols] = np.log1p(X[:, self.cols]); return X


class CorrFilter:
    def fit(self, X):
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.abs(np.corrcoef(X, rowvar=False))
        r = np.nan_to_num(r)
        drop = set()
        for i in range(r.shape[0]):
            if i in drop: continue
            for j in range(i + 1, r.shape[0]):
                if r[i, j] > 0.95: drop.add(j)
        self.keep = np.array([i for i in range(r.shape[0]) if i not in drop])
        return self
    def transform(self, X):
        return X[:, self.keep]


VARIANTS = {
    # name: (scaler, collapse, log, corr)
    "robust_collapsed":   ("robust",   True,  False, False),
    "standard_collapsed": ("standard", True,  False, False),
    "minmax_collapsed":   ("minmax",   True,  False, False),
    "none_collapsed":     ("none",     True,  False, False),
    "robust_alllevels":   ("robust",   False, False, False),
    "robust_collapsed_log":  ("robust", True, True,  False),
    "robust_collapsed_corr": ("robust", True, False, True),
}
SC = {"robust": RobustScaler, "standard": StandardScaler,
      "minmax": MinMaxScaler}


def prep(Xtr, Xte, scaler, collapse, log, corr):
    col = Collapse({"proto": 10, "state": 5} if collapse else None).fit(Xtr)
    a, b = col.transform(Xtr), col.transform(Xte)
    oh = OneHot().fit(a)
    A, B = oh.transform(a), oh.transform(b)
    nnum = len(oh.num_)
    if log:
        ld = LogDamp().fit(A[:, :nnum])
        A[:, :nnum] = ld.transform(A[:, :nnum]); B[:, :nnum] = ld.transform(B[:, :nnum])
    if corr:
        cf = CorrFilter().fit(A); A, B = cf.transform(A), cf.transform(B)
    if scaler != "none":
        s = SC[scaler]().fit(A); A, B = s.transform(A), s.transform(B)
    return A.astype(np.float32), B.astype(np.float32)


def vote(idx, dist, ytr, k, weighting):
    lab = ytr[idx[:, :k]]
    if weighting == "uniform":
        w = np.ones_like(dist[:, :k])
    else:
        d = dist[:, :k]
        with np.errstate(divide="ignore"):
            w = 1.0 / d
        zero = (d == 0)
        rows = zero.any(axis=1)
        w[rows] = zero[rows].astype(float)  # exact matches take the vote
    tot = np.zeros((lab.shape[0], NC))
    for c in range(NC):
        tot[:, c] = (w * (lab == c)).sum(axis=1)
    return tot.argmax(axis=1)


def main(variants):
    df, y = load_clean()
    Xd, Xt, yd, yt = dev_test_split(df, y)
    Xd = Xd.reset_index(drop=True)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    out = f"{WORK}/knn_tuning.json"
    try: res = json.load(open(out))
    except Exception: res = []
    done = {r["variant"] for r in res}
    for v in variants:
        if v in done: continue
        sc, col, lg, cr = VARIANTS[v]
        per = {(k, w): [] for k in KS for w in ("uniform", "distance")}
        t0 = time.time(); nfeat = None
        for f, (tr, te) in enumerate(skf.split(Xd, yd)):
            A, B = prep(Xd.iloc[tr], Xd.iloc[te], sc, col, lg, cr)
            nfeat = A.shape[1]
            nn = NearestNeighbors(n_neighbors=KMAX, algorithm="brute",
                                  n_jobs=2).fit(A)
            dist, idx = nn.kneighbors(B)
            for k in KS:
                for w in ("uniform", "distance"):
                    p = vote(idx, dist, yd[tr], k, w)
                    per[(k, w)].append([f1_score(yd[te], p, average="macro"),
                                        balanced_accuracy_score(yd[te], p),
                                        accuracy_score(yd[te], p)])
            print(v, "fold", f, "%.0fs" % (time.time() - t0), flush=True)
        for (k, w), s in per.items():
            s = np.array(s)
            res.append({"variant": v, "scaler": sc, "collapse": col,
                        "log": lg, "corr": cr, "k": k, "weights": w,
                        "n_features": nfeat,
                        "macro_f1": s[:, 0].mean(), "macro_f1_std": s[:, 0].std(),
                        "bal_acc": s[:, 1].mean(), "accuracy": s[:, 2].mean(),
                        "fold_macro_f1": s[:, 0].tolist()})
        json.dump(res, open(out, "w"), indent=1)
        best = max([r for r in res if r["variant"] == v], key=lambda r: r["macro_f1"])
        print("BEST", v, best["k"], best["weights"], "%.4f" % best["macro_f1"], flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(VARIANTS))
