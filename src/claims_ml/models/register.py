"""Sprint 4: register the best model and promote it to production.

MLflow 3 uses ALIASES, not stages:
  @staging   = candidate being validated
  @champion  = the production model (what serving loads)
"""

from __future__ import annotations

import mlflow
from mlflow import MlflowClient

from claims_ml.config import load_config


def get_best_run(metric: str = "pr_auc"):
    """Return the run with the highest value of `metric` in the experiment."""
    cfg = load_config()
    client = MlflowClient()
    exp = client.get_experiment_by_name(cfg["mlflow"]["experiment_name"])
    if exp is None:
        raise RuntimeError("No experiment found. Run scripts/train_models.py first.")
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No runs found. Run scripts/train_models.py first.")
    return runs[0]


def register_best(metric: str = "pr_auc"):
    """Register the best run's model and set the @staging alias."""
    cfg = load_config()
    name = cfg["model"]["registered_name"]
    client = MlflowClient()

    run = get_best_run(metric)
    model_name = run.data.params.get("model", "unknown")
    score = run.data.metrics.get(metric)

    model_uri = f"runs:/{run.info.run_id}/model"
    version = mlflow.register_model(model_uri=model_uri, name=name)

    client.set_registered_model_alias(name, "staging", version.version)
    print(f"Best run: {model_name}  ({metric}={score})")
    print(f"Registered '{name}' version {version.version} -> alias @staging")
    return version


def promote_to_production(version: int):
    """Move a version to production by giving it the @champion alias."""
    cfg = load_config()
    name = cfg["model"]["registered_name"]
    MlflowClient().set_registered_model_alias(name, "champion", version)
    print(f"Promoted '{name}' version {version} -> alias @champion (production)")


def load_production_model():
    """Load the current production model. Used by the serving app in Sprint 5."""
    cfg = load_config()
    name = cfg["model"]["registered_name"]
    return mlflow.pyfunc.load_model(f"models:/{name}@champion")