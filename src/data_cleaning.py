
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "daily_adjusted_prices.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clean_prices() -> pd.DataFrame:
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_FILE}\n"
            "Run python .\\src\\download_data.py first."
        )

    prices = pd.read_csv(RAW_FILE, parse_dates=["Date"])

    if "Date" not in prices.columns:
        raise ValueError("The dataset must contain a Date column.")

    prices = prices.sort_values("Date")
    prices = prices.drop_duplicates(subset=["Date"], keep="last")
    prices = prices.set_index("Date")

    # Convert price columns to numeric.
    prices = prices.apply(pd.to_numeric, errors="coerce")

    # Zero or negative prices are invalid for this strategy.
    prices = prices.where(prices > 0)

    # Save the original missing-value counts for the report.
    missing_before = prices.isna().sum()

    # Do not fill missing prices using future observations.
    # Retain dates with at least one available stock price.
    prices = prices.dropna(how="all")

    output_file = PROCESSED_DIR / "cleaned_prices.csv"
    prices.to_csv(output_file, index_label="Date")

    print("DATA CLEANING REPORT")
    print("-" * 45)
    print(f"Rows after cleaning: {len(prices)}")
    print(f"Date range: {prices.index.min()} to {prices.index.max()}")
    print("\nMissing values before cleaning:")
    print(missing_before)
    print("\nMissing values after cleaning:")
    print(prices.isna().sum())
    print(f"\nSaved cleaned data to: {output_file}")

    return prices


if __name__ == "__main__":
    clean_prices()
