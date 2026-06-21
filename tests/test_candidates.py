from claims_ml.models.candidates import get_candidate_models


def test_four_candidates_present():
    models = get_candidate_models(scale_pos_weight=5.0)
    assert set(models) == {"logreg", "decision_tree", "random_forest", "xgboost"}