FROM python:3.11-slim

WORKDIR /app

# Install only what serving needs (lighter than the full dev install)
RUN pip install --no-cache-dir \
    fastapi "uvicorn[standard]" pandas scikit-learn xgboost joblib pyyaml

COPY src ./src
COPY configs ./configs
COPY models/production ./models/production

ENV PYTHONPATH=/app/src
ENV MODEL_PATH=/app/models/production/model.pkl

EXPOSE 8000
CMD ["uvicorn", "claims_ml.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]