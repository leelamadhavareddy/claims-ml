import pandas as pd
import pytest
from claims_ml.data.ingest import validate


def test_validate_passes_on_good_data():
    df = pd.DataFrame({"age": [30, 40], "claim_outcome": [0, 1]})
    assert len(validate(df, target="claim_outcome")) == 2


def test_validate_rejects_missing_target():
    with pytest.raises(ValueError):
        validate(pd.DataFrame({"age": [1]}), target="claim_outcome")


def test_validate_rejects_non_binary_target():
    with pytest.raises(ValueError):
        validate(pd.DataFrame({"claim_outcome": [0, 1, 2]}), target="claim_outcome")