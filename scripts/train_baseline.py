"""Full Sprint 2 flow: load -> features -> split -> baseline -> metrics."""

from __future__ import annotations

from claims_ml.config import load_config
from claims_ml.data.ingest import load_and_validate
from claims_ml.features.build import add_features
from claims_ml.features.split import make_split
from claims_ml.models.baseline import train_baseline


def main() -> None:
    cfg = load_config()
    target = cfg["target"]["column"]

    df = load_and_validate()
    df = add_features(df)
    X_train, X_test, y_train, y_test = make_split(df, target)

    print(f"train rows: {len(X_train):,} | test rows: {len(X_test):,}")
    print(f"train positive rate: {y_train.mean():.1%}")
    print(f"test  positive rate: {y_test.mean():.1%}  (should match train)")

    _, metrics = train_baseline(X_train, X_test, y_train, y_test, target)
    print("\n=== BASELINE (Logistic Regression) ===")
    for name, value in metrics.items():
        print(f"{name:>8}: {value}")


if __name__ == "__main__":
    main()