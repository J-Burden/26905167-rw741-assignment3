import json, time
import numpy as np
from scipy.stats import chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.metrics import (f1_score, balanced_accuracy_score, accuracy_score,
                             precision_recall_fscore_support, confusion_matrix)
from sklearn.inspection import permutation_importance
from common import *
from stage2_knn import prep, vote

SEEDS = list(range(10))


def metrics(y, p):
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, zero_division=0)
    return {"macro_f1": f1_score(y, p, average="macro"),
            "bal_acc": balanced_accuracy_score(y, p),
            "accuracy": accuracy_score(y, p),
            "precision": pr.tolist(), "recall": rc.tolist(), "f1": f1.tolist()}


def choose_T(curve, tol=0.001):
    m = np.array(curve["mean"])
    for T in [25, 50, 100, 150, 200, 250, 300]:
        if m[-1] - m[T - 1] <= tol: return T
    return 300


def boot(y, p1, p2, B=2000, rng=np.random.default_rng(SEED)):
    n = len(y); d = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        d.append(f1_score(y[i], p1[i], average="macro") -
                 f1_score(y[i], p2[i], average="macro"))
    d = np.array(d)
    obs = f1_score(y, p1, average="macro") - f1_score(y, p2, average="macro")
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return obs, np.percentile(d, [2.5, 97.5]).tolist(), max(p, 1.0 / B)


def mcnemar(y, p1, p2):
    a = (p1 == y); b = (p2 == y)
    n01 = int((a & ~b).sum()); n10 = int((~a & b).sum())
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return n01, n10, stat, float(chi2.sf(stat, 1))


def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.zeros(m); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, (m - r) * ps[i])); adj[i] = run
    return adj.tolist()


def main():
    df, y = load_clean()
    Xd, Xt, yd, yt = dev_test_split(df, y)
    out = {"n_dev": len(yd), "n_test": len(yt)}
    preds = {}

    kt = json.load(open(f"{WORK}/knn_tuning.json"))
    kb = max(kt, key=lambda r: r["macro_f1"])
    t0 = time.time()
    A, B = prep(Xd, Xt, kb["scaler"], kb["collapse"], kb["log"], kb["corr"])
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=kb["k"], algorithm="brute", n_jobs=2).fit(A)
    t1 = time.time(); dist, idx = nn.kneighbors(B)
    p = vote(idx, dist, yd, kb["k"], kb["weights"]); t2 = time.time()
    out["knn"] = {"config": kb, "fit_s": t1 - t0, "predict_s": t2 - t1,
                  "n_features": A.shape[1], **metrics(yt, p),
                  "confusion": confusion_matrix(yt, p).tolist()}
    preds["knn"] = p
    print("KNN %.4f" % out["knn"]["macro_f1"], flush=True)

    def encoded(enc):
        E = (Ordinal() if enc == "ordinal" else OneHot()).fit(Xd)
        names = E.cols_ if enc == "ordinal" else E.num_ + [f"{c}={l}" for c in CAT for l in E.levels_[c]]
        return E.transform(Xd), E.transform(Xt), names

    tp = json.load(open(f"{WORK}/tree_best.json"))
    tenc = max(json.load(open(f"{WORK}/tree_encoding.json")), key=lambda r: r["macro_f1"])
    assert not tenc["collapse"]
    A, B, names = encoded(tenc["enc"]); out["tree_enc"] = tenc["enc"]
    runs = []
    for s in SEEDS:
        t0 = time.time(); m = DecisionTreeClassifier(random_state=s, **tp).fit(A, yd)
        t1 = time.time(); p = m.predict(B); t2 = time.time()
        runs.append({**metrics(yt, p), "fit_s": t1 - t0, "predict_s": t2 - t1,
                     "train_macro_f1": f1_score(yd, m.predict(A), average="macro"),
                     "leaves": int(m.get_n_leaves()), "depth": int(m.get_depth())})
        if s == 0:
            preds["tree"] = p; tree0 = m
            out["tree_confusion"] = confusion_matrix(yt, p).tolist()
            out["tree_importance"] = dict(zip(names, m.feature_importances_.tolist()))
    out["tree"] = {"config": tp, "runs": runs}
    print("tree %.4f" % np.mean([r["macro_f1"] for r in runs]), flush=True)

    rp = json.load(open(f"{WORK}/rf_best.json"))
    rp = {k: (int(v) if isinstance(v, str) and v.isdigit() else v) for k, v in rp.items()}
    cur = json.load(open(f"{WORK}/rf_curve.json"))
    T = choose_T(cur["best"]); out["T"] = T
    renc = json.load(open(f"{WORK}/rf_encoding.json"))
    rbest = max(renc, key=lambda r: r["macro_f1"])
    A, B, names = encoded(rbest["enc"]); out["rf_enc"] = rbest["enc"]
    if rbest["enc"] == "onehot": rp["max_features"] = rbest["max_features"]
    for name, params in [("rf", rp), ("bagging", dict(rp, max_features=None))]:
        runs = []
        for s in SEEDS:
            t0 = time.time()
            m = RandomForestClassifier(n_estimators=T, n_jobs=2, random_state=s,
                                       **params).fit(A, yd)
            t1 = time.time(); p = m.predict(B); t2 = time.time()
            runs.append({**metrics(yt, p), "fit_s": t1 - t0, "predict_s": t2 - t1,
                         "train_macro_f1": f1_score(yd, m.predict(A), average="macro"),
                         "mean_leaves": float(np.mean([e.get_n_leaves() for e in m.estimators_]))})
            if s == 0:
                preds[name] = p
                out[f"{name}_confusion"] = confusion_matrix(yt, p).tolist()
                if name == "rf":
                    out["rf_importance"] = dict(zip(names, m.feature_importances_.tolist()))
                    rng = np.random.default_rng(SEED)
                    sub = rng.choice(len(yt), 8000, replace=False)
                    pi = permutation_importance(m, B[sub], yt[sub], scoring="f1_macro",
                                                n_repeats=3, random_state=SEED, n_jobs=1)
                    out["rf_perm_importance"] = dict(zip(names, pi.importances_mean.tolist()))
            print(name, s, "%.4f" % runs[-1]["macro_f1"], "%.0fs" % (t1 - t0), flush=True)
        out[name] = {"config": {k: str(v) for k, v in params.items()}, "runs": runs}
        json.dump(out, open(f"{WORK}/final.json", "w"), indent=1, default=float)

    pairs = [("rf", "tree"), ("rf", "knn"), ("tree", "knn"), ("rf", "bagging"), ("bagging", "tree")]
    tests = []
    for a, b in pairs:
        obs, ci, pb = boot(yt, preds[a], preds[b])
        n01, n10, st, pm = mcnemar(yt, preds[a], preds[b])
        tests.append({"a": a, "b": b, "delta_macro_f1": obs, "ci95": ci, "p_boot": pb,
                      "n_a_only": n01, "n_b_only": n10, "mcnemar_chi2": st, "p_mcnemar": pm})
    for k in ("p_boot", "p_mcnemar"):
        adj = holm([t[k] for t in tests])
        for t, a in zip(tests, adj): t[k + "_holm"] = a
    out["tests"] = tests
    np.savez(f"{WORK}/test_preds.npz", y=yt, **preds)
    json.dump(out, open(f"{WORK}/final.json", "w"), indent=1, default=float)
    for t in tests: print(t, flush=True)


if __name__ == "__main__":
    main()
