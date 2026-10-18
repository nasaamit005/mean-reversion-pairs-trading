
from itertools import combinations
from pathlib import Path

import pandas as pd
from statsmodels.tsa.stattools import coint

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "cleaned_prices.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Use training data only for initial pair selection.
TRAIN_END = "2022-12-31"
MIN_OBSERVATIONS = 500
SIGNIFICANCE_LEVEL = 0.05


def select_pairs() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Cleaned data not found. Run "
            "python .\\src\\data_cleaning.py first."
        )

    prices = pd.read_csv(DATA_FILE, parse_dates=["Date"])
    prices = prices.set_index("Date").sort_index()

    # Exclude validation and test periods from pair selection.
    train = prices.loc[:TRAIN_END]

    results = []

    for ticker_a, ticker_b in combinations(train.columns, 2):
        pair = train[[ticker_a, ticker_b]].dropna()

        if len(pair) < MIN_OBSERVATIONS:
            continue

        price_corr = pair[ticker_a].corr(pair[ticker_b])

        returns = pair.pct_change().dropna()
        return_corr = returns[ticker_a].corr(returns[ticker_b])

        try:
            test_stat, p_value, critical_values = coint(
                pair[ticker_a],
                pair[ticker_b],
                trend="c",
                autolag="aic",
            )

            results.append({
                "Ticker_A": ticker_a,
                "Ticker_B": ticker_b,
                "Observations": len(pair),
                "Price_Correlation": price_corr,
                "Return_Correlation": return_corr,
                "Cointegration_Statistic": test_stat,
                "Cointegration_P_Value": p_value,
                "Critical_Value_1pct": critical_values[0],
                "Critical_Value_5pct": critical_values[1],
                "Critical_Value_10pct": critical_values[2],
                "Cointegrated_at_5pct": (
                    p_value < SIGNIFICANCE_LEVEL
                ),
            })

        except (ValueError, np.linalg.LinAlgError) as exc:
            print(f"Skipped {ticker_a}/{ticker_b}: {exc}")

    if not results:
        raise RuntimeError(
            "No pairs could be tested. Check the dataset and date range."
        )

    result_df = pd.DataFrame(results)

    result_df = result_df.sort_values(
        "Cointegration_P_Value"
    ).reset_index(drop=True)

    output_file = OUTPUT_DIR / "pair_selection_results.csv"
    result_df.to_csv(output_file, index=False)

    print("\nPAIR SELECTION RESULTS")
    print("=" * 75)
    print(
        result_df[
            [
                "Ticker_A",
                "Ticker_B",
                "Price_Correlation",
                "Return_Correlation",
                "Cointegration_P_Value",
                "Cointegrated_at_5pct",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print(f"\nResults saved to: {output_file}")
    print(
        "\nImportant: This is an initial screening, not proof "
        "of future profitability."
    )

    return result_df


if __name__ == "__main__":
    import numpy as np

    select_pairs()
