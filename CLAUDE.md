# Claims ML Pipeline

Production ML project predicting insurance claim outcomes
(imbalanced binary classification).

## Conventions
- Python 3.11, `src/` layout, importable package name = `claims_ml`
- Config via YAML in `configs/` — NEVER hardcode paths or hyperparameters
- Preprocessing MUST be a scikit-learn Pipeline so it travels with the model
- `notebooks/` is throwaway exploration only; real logic lives in `src/`
- Every module in `src/` gets a unit test in `tests/`
- Metrics for this imbalanced problem: PR-AUC, recall, F1 (NOT accuracy)

## Stack
Pandas, NumPy, scikit-learn, XGBoost, MLflow, FastAPI, Docker, AWS ECS, Jenkins, Grafana

## Commands
- `make setup`  install deps + pre-commit hooks + dvc init
- `make test`   run pytest
- `make lint`   ruff + black check (no changes)
- `make format` ruff --fix + black (apply changes)

## Sprint map
- S0 foundation (this) | S1 data+EDA | S2 features+split+baseline
- S3 modelling+MLflow | S4 registry+promotion | S5 FastAPI+Docker
- S6 Jenkins+ECR+ECS | S7 Grafana+drift
