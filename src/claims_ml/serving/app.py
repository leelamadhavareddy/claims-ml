"""FastAPI prediction service for the claims model."""

from __future__ import annotations

import os

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from claims_ml.features.build import add_features

MODEL_PATH = os.getenv("MODEL_PATH", "models/production/model.pkl")
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))

app = FastAPI(title="Claims Prediction Service", version="1.0.0")
_model = None


def get_model():
    """Load the model once, then reuse it."""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


class ClaimInput(BaseModel):
    age: int = Field(..., ge=18, le=120)
    policy_tenure_years: int = Field(..., ge=0)
    claim_amount: float = Field(..., ge=0)
    num_prior_claims: int = Field(..., ge=0)
    region: str
    claim_type: str
    vehicle_age: int = Field(..., ge=0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/version")
def version():
    return {"model": "claims-model", "alias": "champion"}


@app.post("/predict")
def predict(claim: ClaimInput):
    df = add_features(pd.DataFrame([claim.model_dump()]))  # re-apply feature eng
    proba = float(get_model().predict_proba(df)[:, 1][0])
    return {"probability": round(proba, 4), "prediction": int(proba >= THRESHOLD)}