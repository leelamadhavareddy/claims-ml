"""Sprint 0 smoke tests: prove the skeleton imports and config loads.

These exist so CI is green from day one and every later sprint starts
from a known-good baseline.
"""

import claims_ml
from claims_ml.config import load_config


def test_package_imports():
    assert claims_ml.__version__ == "0.1.0"


def test_config_loads():
    cfg = load_config()
    assert cfg["project"]["random_seed"] == 42
    assert cfg["split"]["stratify"] is True
    assert cfg["target"]["column"]  # non-empty
