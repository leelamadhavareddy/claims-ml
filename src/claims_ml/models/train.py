"""Train all candidate models and log every run to MLflow.

Each model gets its own MLflow 'run' recording its params, metrics, and the
saved model artifact -- so you can compare runs side by side and reproduce any
of them later.
"""

from __future__ import annotations

import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline

from claims_ml.config import load_config
from claims_ml.data.preprocess import build_from_dataframe
from claims_ml.models.candidates import get_candidate_models
from claims_ml.models.evaluate import compute_metrics


def train_all(X_train, X_test, y_train, y_test, target: str = "claim_outcome") -> dict:
    cfg = load_config()
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    preprocessor = build_from_dataframe(
        X_train.assign(**{target: y_train.values}), target
    )
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / max(pos, 1)
    candidates = get_candidate_models(scale_pos_weight, cfg["project"]["random_seed"])

    results = {}
    for name, clf in candidates.items():
        with mlflow.start_run(run_name=name):
            model = Pipeline([("preprocess", preprocessor), ("clf", clf)])
            model.fit(X_train, y_train)
            metrics = compute_metrics(model, X_test, y_test)

            mlflow.log_param("model", name)
            mlflow.log_params(
                {k: v for k, v in clf.get_params().items() if v is not None}
            )
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, name= "model", serialization_format="cloudpickle")

            results[name] = metrics
            print(f"{name:>14}: {metrics}")
    return results