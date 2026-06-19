"""Build the reusable preprocessing Pipeline (the heart of Sprint 1).

numeric  -> fill missing with median, then scale
category -> fill missing with "Missing", then one-hot encode
It is a sklearn object, so the SAME cleaning runs in training and serving.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def infer_feature_types(df: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    features = [c for c in df.columns if c != target]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in features if c not in numeric]
    return numeric, categorical


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ], remainder="drop")


def build_from_dataframe(df: pd.DataFrame, target: str) -> ColumnTransformer:
    numeric, categorical = infer_feature_types(df, target)
    return build_preprocessor(numeric, categorical)