import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

SRC = os.environ.get("NT_SRC", os.path.join(os.path.dirname(__file__), "..", "data", "networkTraffic.csv"))
WORK = os.environ.get("NT_WORK", os.path.join(os.path.dirname(__file__), "..", "results"))
SEED = 42
CAT = ["proto", "state", "service"]
CLASS_NAMES = ["Normal", "Reconnaissance", "Backdoor", "DoS", "Exploits",
               "Analysis", "Fuzzers", "Worms", "Shellcode", "Generic"]


def load_clean():
    raw = pd.read_csv(SRC, na_values=["?"], low_memory=False)
    df = raw.drop(columns=["id"])                 # identifier: leakage
    df["service"] = df["service"].fillna("none")  # informative absence
    df["is_ftp_login"] = df["is_ftp_login"].clip(upper=1)  # binary repair
    df = df.drop_duplicates().reset_index(drop=True)
    y = df.pop("attack_cat").to_numpy()
    return df, y


def dev_test_split(df, y, test_size=0.2):
    return train_test_split(df, y, test_size=test_size, stratify=y,
                            random_state=SEED)


class Collapse(BaseEstimator, TransformerMixin):

    def __init__(self, top=None):
        self.top = top 

    def fit(self, X, y=None):
        self.keep_ = {}
        for c, n in (self.top or {}).items():
            if n is not None:
                self.keep_[c] = set(X[c].value_counts().nlargest(n).index)
        return self

    def transform(self, X):
        X = X.copy()
        for c, keep in self.keep_.items():
            X[c] = X[c].where(X[c].isin(keep), "other")
        return X


class Ordinal(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.maps_ = {c: {v: i for i, v in enumerate(sorted(X[c].unique()))}
                      for c in CAT}
        self.cols_ = list(X.columns)
        return self

    def transform(self, X):
        X = X.copy()
        for c in CAT:
            X[c] = X[c].map(self.maps_[c]).fillna(-1)
        return X[self.cols_].to_numpy(dtype=np.float32)


class OneHot(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.levels_ = {c: sorted(X[c].unique()) for c in CAT}
        self.num_ = [c for c in X.columns if c not in CAT]
        return self

    def transform(self, X):
        parts = [X[self.num_].to_numpy(dtype=np.float32)]
        for c in CAT:
            v = X[c].to_numpy()
            parts.append(np.stack([(v == l) for l in self.levels_[c]],
                                  axis=1).astype(np.float32))
        return np.hstack(parts)
