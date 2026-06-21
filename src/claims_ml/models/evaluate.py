"""Shared metric computation, so every model is scored the same way."""

from __future__ import annotations

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(model, X_test, y_test) -> dict[str, float]:
    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    return {
        "pr_auc": round(average_precision_score(y_test, proba), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds), 4),
    }