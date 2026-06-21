import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from claims_ml.models.evaluate import compute_metrics


def test_compute_metrics_keys():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.random(100)})
    y = (rng.random(100) < 0.3).astype(int)
    model = LogisticRegression().fit(X, y)
    m = compute_metrics(model, X, y)
    assert set(m) == {"pr_auc", "roc_auc", "recall", "f1"}
    assert all(0 <= v <= 1 for v in m.values())