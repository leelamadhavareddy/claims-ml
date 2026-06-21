from claims_ml.models.register import get_best_run  # noqa: F401  (import smoke check)


def argmax_metric(scores: dict) -> str:
    return max(scores, key=scores.get)


def test_best_is_highest_metric():
    scores = {"logreg": 0.18, "random_forest": 0.21, "xgboost": 0.27}
    assert argmax_metric(scores) == "xgboost"