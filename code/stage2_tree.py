import json, time, itertools
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
from common import *


def run(Xd, yd, folds, params, enc="ordinal", collapse=None):
    s, tr_s, leaves = [], [], []
    for tr, te in folds:
        col = Collapse(collapse).fit(Xd.iloc[tr])
        a, b = col.transform(Xd.iloc[tr]), col.transform(Xd.iloc[te])
        E = (Ordinal() if enc == "ordinal" else OneHot()).fit(a)
        A, B = E.transform(a), E.transform(b)
        m = DecisionTreeClassifier(random_state=SEED, **params).fit(A, yd[tr])
        p = m.predict(B)
        s.append([f1_score(yd[te], p, average="macro"),
                  balanced_accuracy_score(yd[te], p), accuracy_score(yd[te], p)])
        tr_s.append(f1_score(yd[tr], m.predict(A), average="macro"))
        leaves.append(m.get_n_leaves())
    s = np.array(s)
    return {**{k: (v if v is not None else "None") for k, v in params.items()},
            "enc": enc, "collapse": bool(collapse),
            "macro_f1": s[:, 0].mean(), "macro_f1_std": s[:, 0].std(),
            "bal_acc": s[:, 1].mean(), "accuracy": s[:, 2].mean(),
            "train_macro_f1": float(np.mean(tr_s)), "leaves": float(np.mean(leaves)),
            "fold_macro_f1": s[:, 0].tolist()}


def main():
    df, y = load_clean()
    Xd, Xt, yd, yt = dev_test_split(df, y)
    Xd = Xd.reset_index(drop=True)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xd, yd))
    t0 = time.time()
    grid = [dict(criterion=c, max_depth=d, min_samples_leaf=l, class_weight=w)
            for c, d, l, w in itertools.product(
                ["gini", "entropy"], [8, 12, 16, 20, 24, None], [1, 5, 20],
                [None, "balanced"])]
    A = Parallel(n_jobs=2, verbose=5)(delayed(run)(Xd, yd, folds, g) for g in grid)
    json.dump(A, open(f"{WORK}/tree_gridA.json", "w"), indent=1)
    best = max(A, key=lambda r: r["macro_f1"])
    print("A best", best, "%.0fs" % (time.time() - t0), flush=True)
    bp = {k: (None if best[k] == "None" else best[k])
          for k in ["criterion", "max_depth", "min_samples_leaf", "class_weight"]}
    alphas = [0.0, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]
    B = Parallel(n_jobs=2)(delayed(run)(Xd, yd, folds, {**bp, "ccp_alpha": a})
                           for a in alphas)
    json.dump(B, open(f"{WORK}/tree_ccp.json", "w"), indent=1)
    bestB = max(B, key=lambda r: r["macro_f1"])
    print("B best alpha", bestB["ccp_alpha"], "%.4f" % bestB["macro_f1"], flush=True)
    fp = {**bp, "ccp_alpha": bestB["ccp_alpha"]}
    C = Parallel(n_jobs=2)(delayed(run)(Xd, yd, folds, fp, e, c) for e, c in [
        ("ordinal", None), ("ordinal", {"proto": 10, "state": 5}),
        ("onehot", None), ("onehot", {"proto": 10, "state": 5})])
    json.dump(C, open(f"{WORK}/tree_encoding.json", "w"), indent=1)
    for r in C:
        print("C", r["enc"], r["collapse"], "%.4f" % r["macro_f1"], flush=True)
    json.dump(fp, open(f"{WORK}/tree_best.json", "w"), indent=1)


if __name__ == "__main__":
    main()
