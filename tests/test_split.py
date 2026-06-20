import numpy as np
import pandas as pd
from claims_ml.features.split import make_split


def _df(n=200):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "age": rng.integers(18, 80, n),
        "claim_outcome": (rng.random(n) < 0.15).astype(int),
    })


def test_split_sizes_and_no_overlap():
    df = _df()
    X_train, X_test, y_train, y_test = make_split(df, "claim_outcome")
    assert len(X_train) + len(X_test) == len(df)
    assert set(X_train.index).isdisjoint(set(X_test.index))


def test_split_keeps_class_balance():
    df = _df()
    _, _, y_train, y_test = make_split(df, "claim_outcome")
    assert abs(y_train.mean() - y_test.mean()) < 0.05