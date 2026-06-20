"""Stratified train/test split: keeps the same class balance in train AND test."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from claims_ml.config import load_config


def make_split(df: pd.DataFrame, target: str | None = None):
    cfg = load_config()
    target = target or cfg["target"]["column"]
    seed = cfg["project"]["random_seed"]
    test_size = cfg["split"]["test_size"]
    stratify_on = df[target] if cfg["split"]["stratify"] else None

    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify_on
    )