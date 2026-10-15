import json, time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
from common import *

df, y = load_clean()
Xd, Xt, yd, yt = dev_test_split(df, y)
Xd = Xd.reset_index(drop=True)
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xd, yd))
base = json.load(open(f"{WORK}/rf_best.json"))
base = {k: (int(v) if isinstance(v, str) and v.isdigit() else v) for k, v in base.items()}
res = []
for enc in ["ordinal", "onehot"]:
    s = []; t0 = time.time(); mf = None
    for f, (tr, te) in enumerate(folds):
        E = (Ordinal() if enc == "ordinal" else OneHot()).fit(Xd.iloc[tr])
        A, B = E.transform(Xd.iloc[tr]), E.transform(Xd.iloc[te])
        p = dict(base)
        if isinstance(p["max_features"], int):
            p["max_features"] = max(1, int(round(base["max_features"] / 42 * A.shape[1])))
        mf = p["max_features"]
        rf = RandomForestClassifier(n_estimators=100, n_jobs=2, random_state=SEED + f,
                                    **p).fit(A, yd[tr])
        q = rf.predict(B)
        s.append([f1_score(yd[te], q, average="macro"),
                  balanced_accuracy_score(yd[te], q), accuracy_score(yd[te], q)])
        nf = A.shape[1]
    s = np.array(s)
    res.append({"enc": enc, "n_features": nf, "max_features": mf,
                "macro_f1": s[:, 0].mean(), "macro_f1_std": s[:, 0].std(),
                "bal_acc": s[:, 1].mean(), "accuracy": s[:, 2].mean(),
                "fold_macro_f1": s[:, 0].tolist(), "seconds": time.time() - t0})
    print(res[-1], flush=True)
    json.dump(res, open(f"{WORK}/rf_encoding.json", "w"), indent=1)
