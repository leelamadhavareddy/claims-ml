"""Quick EDA: print a data summary and save figures to reports/figures/."""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from claims_ml.config import load_config
from claims_ml.data.ingest import load_and_validate


def main() -> None:
    cfg = load_config()
    target = cfg["target"]["column"]
    df = load_and_validate()

    print("\n=== SHAPE ===")
    print(f"{df.shape[0]:,} rows x {df.shape[1]} columns")
    print("\n=== MISSING VALUES ===")
    miss = df.isna().sum()
    print(miss[miss > 0].sort_values(ascending=False).to_string() or "none")
    print("\n=== TARGET BALANCE ===")
    print(df[target].value_counts(normalize=True).round(3).to_string())

    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    df[target].value_counts().plot(kind="bar", title="Target balance")
    plt.tight_layout(); plt.savefig(fig_dir / "target_balance.png"); plt.close()
    num = df.select_dtypes("number").drop(columns=[target], errors="ignore")
    num.hist(figsize=(10, 8))
    plt.tight_layout(); plt.savefig(fig_dir / "numeric_distributions.png"); plt.close()
    print(f"\nFigures saved to {fig_dir}/")


if __name__ == "__main__":
    main()