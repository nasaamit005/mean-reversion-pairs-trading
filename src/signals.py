
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PRICES_FILE = ROOT / "data" / "processed" / "cleaned_prices.csv"
PAIRS_FILE = ROOT / "data" / "processed" / "pair_selection_results.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END = "2022-12-31"
ROLLING_WINDOW = 60

ENTRY_Z = 2.0
EXIT_Z = 0.5
STOP_Z = 4.0
SIGNIFICANCE_LEVEL = 0.05


def generate_signals() -> pd.DataFrame:
    if not PRICES_FILE.exists():
        raise FileNotFoundError(
            "Run src/data_cleaning.py before generating signals."
        )

    if not PAIRS_FILE.exists():
        raise FileNotFoundError(
            "Run src/pair_selection.py before generating signals."
        )

    prices = pd.read_csv(PRICES_FILE, parse_dates=["Date"])
    prices = prices.set_index("Date").sort_index()

    results = pd.read_csv(PAIRS_FILE)
    results = results.sort_values("Cointegration_P_Value")

    significant = results[
        results["Cointegration_P_Value"] < SIGNIFICANCE_LEVEL
    ]

    if not significant.empty:
        chosen = significant.iloc[0]
        print("Selected the lowest-p-value candidate below 0.05.")
    else:
        chosen = results.iloc[0]
        print(
            "WARNING: No pair passed the 5% cointegration threshold. "
            "Using the lowest-p-value pair for exploratory analysis only."
        )

    ticker_a = chosen["Ticker_A"]
    ticker_b = chosen["Ticker_B"]
    p_value = float(chosen["Cointegration_P_Value"])

    pair = prices[[ticker_a, ticker_b]].dropna()
    train = pair.loc[:TRAIN_END]

    if len(train) < 100:
        raise ValueError(
            "Insufficient training observations to estimate the hedge ratio."
        )

    # Estimate the hedge ratio and intercept using training data only.
    y = train[ticker_a].to_numpy(dtype=float)
    x = train[ticker_b].to_numpy(dtype=float)

    design = np.column_stack([np.ones(len(x)), x])
    intercept, hedge_ratio = np.linalg.lstsq(
        design, y, rcond=None
    )[0]

    if not np.isfinite(hedge_ratio) or hedge_ratio <= 0:
        raise ValueError(
            f"Invalid hedge ratio ({hedge_ratio:.4f}). "
            "Review the selected pair and regression assumptions."
        )

    # Keep the fitted training-period parameters fixed.
    output = pair.copy()
    output["Spread"] = (
        output[ticker_a] - intercept - hedge_ratio * output[ticker_b]
    )

    # Shift rolling statistics by one session so the current spread
    # is not used to calculate its own signal threshold.
    rolling_mean = (
        output["Spread"]
        .rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW)
        .mean()
        .shift(1)
    )

    rolling_std = (
        output["Spread"]
        .rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW)
        .std(ddof=1)
        .shift(1)
    )

    output["Rolling_Mean"] = rolling_mean
    output["Rolling_Std"] = rolling_std.replace(0, np.nan)

    output["Z_Score"] = (
        (output["Spread"] - output["Rolling_Mean"])
        / output["Rolling_Std"]
    )

    # Position: +1 = long spread; -1 = short spread; 0 = flat.
    # Actual execution will be delayed in the backtest.
    position = 0
    positions = []

    for z in output["Z_Score"]:
        if not np.isfinite(z):
            position = 0

        elif position == 0:
            if z <= -ENTRY_Z:
                position = 1
            elif z >= ENTRY_Z:
                position = -1

        elif abs(z) >= STOP_Z:
            position = 0

        elif abs(z) <= EXIT_Z:
            position = 0

        positions.append(position)

    output["Position"] = positions

    # A transition identifies an entry or exit decision.
    previous_position = output["Position"].shift(1).fillna(0)
    output["Signal_Change"] = (
        output["Position"] - previous_position
    )

    output["Ticker_A"] = ticker_a
    output["Ticker_B"] = ticker_b

    output_file = OUTPUT_DIR / "trading_signals.csv"
    output.to_csv(output_file, index_label="Date")

    print("\nSELECTED PAIR")
    print("=" * 45)
    print(f"Asset A: {ticker_a}")
    print(f"Asset B: {ticker_b}")
    print(f"Training end: {TRAIN_END}")
    print(f"Cointegration p-value: {p_value:.6f}")
    print(f"Intercept: {intercept:.6f}")
    print(f"Hedge ratio: {hedge_ratio:.6f}")
    print(f"Rolling window: {ROLLING_WINDOW} sessions")

    print("\nPosition counts:")
    print(output["Position"].value_counts().sort_index())

    print("\nSaved signals to:")
    print(output_file)

    return output


if __name__ == "__main__":
    generate_signals()
