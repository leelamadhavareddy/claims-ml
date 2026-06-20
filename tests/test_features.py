import pandas as pd
from claims_ml.features.build import add_features


def _df():
    return pd.DataFrame({
        "claim_amount": [1000.0, 2000.0],
        "policy_tenure_years": [0, 4],
        "num_prior_claims": [1, 0],
    })


def test_add_features_creates_columns():
    out = add_features(_df())
    for col in ["claim_per_tenure", "prior_per_tenure", "is_new_customer"]:
        assert col in out.columns


def test_add_features_no_divide_by_zero():
    out = add_features(_df())
    assert out["claim_per_tenure"].notna().all()
    assert out.loc[0, "is_new_customer"] == 1