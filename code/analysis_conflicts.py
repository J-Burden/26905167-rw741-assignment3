import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, accuracy_score
from common import *

df, y = load_clean()
feat = list(df.columns)

d = df.copy(); d["y"] = y
g = d.groupby(feat, dropna=False)["y"].nunique().rename("nlab").reset_index()
d = d.merge(g, on=feat, how="left")
conf = (d["nlab"] > 1).to_numpy()
per = {CLASS_NAMES[c]: {"n": int((y == c).sum()), "conflicting": int((conf & (y == c)).sum()),
                        "pct": round(100 * (conf & (y == c)).sum() / (y == c).sum(), 1)}
       for c in range(10)}
json.dump({"total_conflicting": int(conf.sum()), "per_class": per},
          open(f"{WORK}/conflicts.json", "w"), indent=1)
print("conflicting records", conf.sum(), "%.2f%%" % (100 * conf.mean()))

codes = df.groupby(feat, dropna=False).ngroup().to_numpy()
maj = pd.crosstab(codes, y).to_numpy().argmax(axis=1)
pred = maj[codes]
f1 = f1_score(y, pred, average=None)
json.dump({"oracle_macro_f1": f1_score(y, pred, average="macro"),
           "oracle_acc": float((pred == y).mean()),
           "oracle_f1": {CLASS_NAMES[i]: float(f1[i]) for i in range(10)},
           "oracle_recall": {CLASS_NAMES[i]: float(r) for i, r in
                             enumerate(recall_score(y, pred, average=None))}},
          open(f"{WORK}/oracle.json", "w"), indent=1)
print("ceiling macro F1 %.3f" % f1_score(y, pred, average="macro"))

try:
    P = np.load(f"{WORK}/test_preds.npz")
except FileNotFoundError:
    raise SystemExit("run stage4_final.py first for the post-hoc analysis")
Xd, Xt, yd, yt = dev_test_split(df, y)
h = lambda X: pd.util.hash_pandas_object(X[feat], index=False).to_numpy()
lab = {}
for k, l in zip(h(Xd), yd): lab.setdefault(k, set()).add(l)
kt = h(Xt)
seen = np.array([k in lab for k in kt])
own = np.array([(k in lab) and (l in lab[k]) for k, l in zip(kt, yt)])
assert (P["y"] == yt).all()
out = {"n_seen": int(seen.sum()), "n_own_label_seen": int(own.sum()), "models": {}}
for m in ["knn", "tree", "bagging", "rf"]:
    p = P[m]
    out["models"][m] = {"acc_seen": accuracy_score(yt[seen], p[seen]),
                        "macro_f1_rest": f1_score(yt[~seen], p[~seen], average="macro")}
json.dump(out, open(f"{WORK}/conflict_test.json", "w"), indent=1)
print(json.dumps(out, indent=1))
