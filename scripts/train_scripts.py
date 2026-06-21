"""Sprint 3: train all models, log to MLflow, print the best by PR-AUC.

Run:  python scripts/train_models.py
Then: mlflow ui     (open http://localhost:5000 to see the runs)
"""

from __future__ import annotations

from claims_ml.config import load_config
from claims_ml.data.ingest import load_and_validate
from claims_ml.features.build import add_features
from claims_ml.features.split import make_split
from claims_ml.models.train import train_all


def main() -> None:
    cfg = load_config()
    target = cfg["target"]["column"]

    df = add_features(load_and_validate())
    X_train, X_test, y_train, y_test = make_split(df, target)

    print("Training 4 models, logging each to MLflow...\n")
    results = train_all(X_train, X_test, y_train, y_test, target)

    best = max(results, key=lambda k: results[k]["pr_auc"])
    print(f"\nBest model by PR-AUC: {best}  ({results[best]['pr_auc']})")


if __name__ == "__main__":
    main()