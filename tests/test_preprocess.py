import numpy as np
import pandas as pd
from claims_ml.data.preprocess import build_from_dataframe, infer_feature_types


def _df():
    return pd.DataFrame({
        "age": [30, 40, np.nan, 50],
        "region": ["North", None, "South", "North"],
        "claim_outcome": [0, 1, 0, 1],
    })


def test_infer_types_separates_columns():
    num, cat = infer_feature_types(_df(), target="claim_outcome")
    assert num == ["age"] and cat == ["region"]


def test_preprocessor_fits_and_has_no_missing():
    df = _df()
    pre = build_from_dataframe(df, target="claim_outcome")
    out = pre.fit_transform(df.drop(columns=["claim_outcome"]))
    assert not np.isnan(out).any() and out.shape[0] == 4