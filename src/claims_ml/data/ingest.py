"""Load raw claims data and check it is usable before anything else runs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from claims_ml.config import load_config


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    cfg = load_config()
    csv_path = Path(path) if path else Path(cfg["paths"]["raw"])
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw data not found: {csv_path}. Put claims.csv in data/raw/"
        )
    return pd.read_csv(csv_path)


def validate(df: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
    cfg = load_config()
    target = target or cfg["target"]["column"]
    if df.empty:
        raise ValueError("Dataset is empty.")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' missing from data.")
    classes = set(df[target].dropna().unique())
    if not classes.issubset({0, 1}):
        raise ValueError(f"Target must be binary 0/1, found: {classes}")
    return df


def load_and_validate(path: str | Path | None = None) -> pd.DataFrame:
    return validate(load_raw(path))