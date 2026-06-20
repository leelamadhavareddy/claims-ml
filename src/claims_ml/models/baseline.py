"""Train a Logistic Regression baseline -- the number every model must beat.

class_weight='balanced' makes LR care about the rare positive class.
We score with PR-AUC, recall, F1 (NOT accuracy, which lies on imbalanced data).
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from claims_ml.data.preprocess import build_from_dataframe


def train_baseline(X_train, X_test, y_train, y_test, target: str = "claim_outcome"):
    preprocessor = build_from_dataframe(
        X_train.assign(**{target: y_train.values}), target
    )
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    metrics = {
        "pr_auc": round(average_precision_score(y_test, proba), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds), 4),
    }
    return model, metrics