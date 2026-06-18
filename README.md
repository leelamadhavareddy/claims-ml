# Claims ML Pipeline

End-to-end MLOps pipeline for insurance claims prediction
(imbalanced binary classification).

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
make setup     # install deps + pre-commit + dvc init
make test      # smoke test should pass
make lint      # ruff + black, should be clean
```

## Layout
```
configs/        YAML config (paths + params, no hardcoding)
data/           raw / interim / processed (DVC-tracked, git-ignored)
notebooks/      throwaway EDA only
src/claims_ml/  the actual package
  data/         ingestion + preprocessing      (Sprint 1)
  features/     feature engineering + split    (Sprint 2)
  models/       training, MLflow, registry     (Sprint 3-4)
  serving/      FastAPI prediction service     (Sprint 5)
tests/          pytest unit tests
```

## Sprint roadmap
| Sprint | Goal |
|--------|------|
| 0 | Foundation: repo, tooling, CI-ready skeleton (done) |
| 1 | Data ingestion + EDA + reusable preprocessing Pipeline |
| 2 | Feature engineering + stratified split + LR baseline |
| 3 | Train LR/DT/RF/XGB, log runs to MLflow |
| 4 | Register best (XGBoost), promote Staging -> Production |
| 5 | FastAPI service + Docker image |
| 6 | Jenkins CI/CD -> ECR -> AWS ECS |
| 7 | Grafana monitoring: drift, latency, failures |
