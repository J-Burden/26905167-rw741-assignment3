import json, time, itertools, sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
from common import *

NT = 100


def encode(Xd, tr, te):
    E = Ordinal().fit(Xd.iloc[tr])
    return E.transform(Xd.iloc[tr]), E.transform(Xd.iloc[te])


def diversity(rf, B, yte, rng, n=25):
    P = np.stack([e.predict(B) for e in rf.estimators_]).astype(int)
    strength = float(np.mean([f1_score(yte, p, average="macro") for p in P]))
    acc_i = float(np.mean((P == yte).mean(axis=1)))
    sel = rng.choice(len(P), size=min(n, len(P)), replace=False)
    agr = [np.mean(P[i] == P[j]) for a, i in enumerate(sel) for j in sel[a + 1:]]
    return strength, acc_i, float(np.mean(agr))


def cv(Xd, yd, folds, params, div=False, train_score=False):
    s, extra = [], []
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    for f, (tr, te) in enumerate(folds):
        A, B = encode(Xd, tr, te)
        rf = RandomForestClassifier(n_estimators=params.get("n_estimators", NT),
                                    n_jobs=2, random_state=SEED + f,
                                    **{k: v for k, v in params.items() if k != "n_estimators"})
        rf.fit(A, yd[tr])
        p = rf.predict(B)
        s.append([f1_score(yd[te], p, average="macro"),
                  balanced_accuracy_score(yd[te], p), accuracy_score(yd[te], p)])
        e = {}
        if div:
            e["strength"], e["tree_acc"], e["agreement"] = diversity(rf, B, yd[te], rng)
        if train_score:
            e["train_macro_f1"] = f1_score(yd[tr], rf.predict(A), average="macro")
        e["nodes"] = float(np.mean([t.tree_.node_count for t in rf.estimators_]))
        extra.append(e)
    s = np.array(s)
    r = {**{k: (v if v is not None else "None") for k, v in params.items()},
         "macro_f1": s[:, 0].mean(), "macro_f1_std": s[:, 0].std(),
         "bal_acc": s[:, 1].mean(), "accuracy": s[:, 2].mean(),
         "fold_macro_f1": s[:, 0].tolist(), "seconds": time.time() - t0}
    for k in extra[0]:
        r[k] = float(np.mean([e[k] for e in extra]))
    return r


def resume(path):
    try: return json.load(open(path))
    except Exception: return []


def key(p):
    return json.dumps({k: p[k] for k in sorted(p)}, default=str)


def stage(path, Xd, yd, folds, grid, **kw):
    res = resume(path)
    done = {r["_key"] for r in res}
    for p in grid:
        if key(p) in done: continue
        r = cv(Xd, yd, folds, p, **kw); r["_key"] = key(p)
        res.append(r); json.dump(res, open(path, "w"), indent=1)
        print(path.split("/")[-1], {k: v for k, v in p.items()},
              "%.4f +- %.4f  %.0fs" % (r["macro_f1"], r["macro_f1_std"], r["seconds"]),
              flush=True)
    return res


def curve(Xd, yd, folds, params, T=300):
    grid = list(range(1, T + 1))
    sc = np.zeros((len(folds), T))
    for f, (tr, te) in enumerate(folds):
        A, B = encode(Xd, tr, te)
        rf = RandomForestClassifier(n_estimators=T, n_jobs=2, random_state=SEED + f,
                                    **params).fit(A, yd[tr])
        acc = np.zeros((len(te), 10))
        for t, e in enumerate(rf.estimators_):
            pr = e.predict_proba(B)
            acc[:, e.classes_.astype(int)] += pr
            sc[f, t] = f1_score(yd[te], acc.argmax(axis=1), average="macro")
        print("curve fold", f, "%.4f" % sc[f, -1], flush=True)
    return {"params": {k: str(v) for k, v in params.items()},
            "mean": sc.mean(axis=0).tolist(), "std": sc.std(axis=0).tolist()}


def main(which):
    df, y = load_clean()
    Xd, Xt, yd, yt = dev_test_split(df, y)
    Xd = Xd.reset_index(drop=True)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xd, yd))

    if "A" in which:
        grid = [dict(max_features=m, class_weight=w)
                for w, m in itertools.product([None, "balanced", "balanced_subsample"],
                                              [1, 2, 4, 6, 10, 16, 24, 42])]
        stage(f"{WORK}/rf_stageA.json", Xd, yd, folds, grid, div=True)
    A = resume(f"{WORK}/rf_stageA.json")
    bA = max(A, key=lambda r: r["macro_f1"])
    base = dict(max_features=bA["max_features"],
                class_weight=None if bA["class_weight"] == "None" else bA["class_weight"])
    print("stage A best", base, "%.4f" % bA["macro_f1"], flush=True)

    if "B" in which:
        grid = [dict(base, min_samples_leaf=l) for l in [1, 2, 5, 10, 20, 50]] + \
               [dict(base, max_depth=d) for d in [8, 12, 16, 20, 24, 32]]
        stage(f"{WORK}/rf_stageB.json", Xd, yd, folds, grid, train_score=True)
    Bres = resume(f"{WORK}/rf_stageB.json")
    if Bres:
        bB = max(Bres, key=lambda r: r["macro_f1"])
        for k in ("min_samples_leaf", "max_depth"):
            if k in bB and bB[k] != "None": base[k] = bB[k]
        print("stage B best", base, "%.4f" % bB["macro_f1"], flush=True)
    json.dump(base, open(f"{WORK}/rf_best.json", "w"), indent=1, default=str)

    if "C" in which:
        out = {"best": curve(Xd, yd, folds, base)}
        json.dump(out, open(f"{WORK}/rf_curve.json", "w"))
        bag = dict(base, max_features=None)
        out["bagging"] = curve(Xd, yd, folds, bag)
        json.dump(out, open(f"{WORK}/rf_curve.json", "w"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ABC")
