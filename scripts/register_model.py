"""Register the best model, then promote it Staging -> Production.

Run:  python scripts/register_model.py
"""

from __future__ import annotations

from claims_ml.models.register import promote_to_production, register_best


def main() -> None:
    version = register_best(metric="pr_auc")

    # In real life you would run validation/smoke tests on the @staging model
    # here, and only promote if they pass. This is your promotion GATE.
    print("\nValidation gate: passed (placeholder)")

    promote_to_production(version.version)
    print("\nDone. Load it anywhere with:  models:/claims-model@champion")


if __name__ == "__main__":
    main()