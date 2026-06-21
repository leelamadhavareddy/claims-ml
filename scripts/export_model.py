"""Export the champion model from MLflow to a single local file.

The serving app loads this file, so it does NOT need MLflow running.
Run (with the MLflow server up):  python scripts/export_model.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import mlflow

from claims_ml.config import load_config


def main() -> None:
    mlflow.set_tracking_uri("http://localhost:5000")
    cfg = load_config()
    name = cfg["model"]["registered_name"]

    model = mlflow.sklearn.load_model(f"models:/{name}@champion")
    out = Path("models/production")
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.pkl")
    print(f"Exported champion model -> {out / 'model.pkl'}")


if __name__ == "__main__":
    main()