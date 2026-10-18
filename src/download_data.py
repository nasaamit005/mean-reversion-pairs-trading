
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["KO", "PEP", "V", "MA", "XOM", "CVX"]

START_DATE = "2016-01-01"
END_DATE = "2026-01-01"  # Exclusive


def download_prices() -> pd.DataFrame:
    print("Downloading daily adjusted stock prices...")

    data = yf.download(
        TICKERS,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        multi_level_index=False,
    )

    if data.empty:
        raise RuntimeError(
            "No data downloaded. Check your internet connection "
            "and Yahoo Finance availability."
        )

    prices = data["Close"]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    prices = prices.reindex(columns=TICKERS)
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "Date"

    prices = prices.sort_index()
    prices = prices.dropna(how="all")

    output_file = RAW_DIR / "daily_adjusted_prices.csv"
    prices.to_csv(output_file, index_label="Date")

    print(f"\nSaved dataset: {output_file}")
    print(f"Rows and columns: {prices.shape}")
    print(f"First date: {prices.index.min()}")
    print(f"Last date: {prices.index.max()}")

    print("\nMissing values by stock:")
    print(prices.isna().sum())

    print("\nFirst five rows:")
    print(prices.head())

    return prices


if __name__ == "__main__":
    download_prices()
