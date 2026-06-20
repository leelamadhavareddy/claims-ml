"""Feature engineering: create new columns that help the model.

Row-wise features (computed from each row alone) are SAFE before the split --
no information leaks between rows.
"""

from __future__ import annotations

import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["claim_per_tenure"] = df["claim_amount"] / (df["policy_tenure_years"] + 1)
    df["prior_per_tenure"] = df["num_prior_claims"] / (df["policy_tenure_years"] + 1)
    df["is_new_customer"] = (df["policy_tenure_years"] <= 1).astype(int)
    return df