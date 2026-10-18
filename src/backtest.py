
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRICES_FILE = ROOT / "data" / "processed" / "cleaned_prices.csv"
SIGNALS_FILE = ROOT / "data" / "processed" / "trading_signals.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Assumed one-way trading cost per dollar traded.
TRANSACTION_COST_BPS = 5.0
TRADING_DAYS = 252

TRAIN_END = "2022-12-31"
VALIDATION_END = "2024-12-31"


def calculate_performance(returns: pd.Series) -> dict:
    """Calculate performance statistics from daily portfolio returns."""
    returns = returns.dropna()

    if returns.empty:
        return {
            "Total_Return": np.nan,
            "CAGR": np.nan,
            "Annualized_Volatility": np.nan,
            "Sharpe_Ratio": np.nan,
            "Max_Drawdown": np.nan,
        }

    equity = (1 + returns).cumprod()
    years = len(returns) / TRADING_DAYS

    total_return = equity.iloc[-1] - 1

    cagr = (
        equity.iloc[-1] ** (1 / years) - 1
        if years > 0 and equity.iloc[-1] > 0
        else np.nan
    )

    volatility = returns.std(ddof=1) * np.sqrt(TRADING_DAYS)

    sharpe = (
        returns.mean() / returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
        if returns.std(ddof=1) > 0
        else np.nan
    )

    drawdown = equity / equity.cummax() - 1

    return {
        "Total_Return": total_return,
        "CAGR": cagr,
        "Annualized_Volatility": volatility,
        "Sharpe_Ratio": sharpe,
        "Max_Drawdown": drawdown.min(),
    }


def run_backtest() -> pd.DataFrame:
    if not PRICES_FILE.exists():
        raise FileNotFoundError(
            "Cleaned prices not found. Run data_cleaning.py first."
        )

    if not SIGNALS_FILE.exists():
        raise FileNotFoundError(
            "Trading signals not found. Run signals.py first."
        )

    prices = pd.read_csv(PRICES_FILE, parse_dates=["Date"])
    prices = prices.set_index("Date").sort_index()

    signals = pd.read_csv(SIGNALS_FILE, parse_dates=["Date"])
    signals = signals.set_index("Date").sort_index()

    ticker_a = str(signals["Ticker_A"].dropna().iloc[0])
    ticker_b = str(signals["Ticker_B"].dropna().iloc[0])

    pair_prices = prices[[ticker_a, ticker_b]].dropna()
    signals = signals.reindex(pair_prices.index)

    if signals["Position"].isna().any():
        raise ValueError(
            "Signals are missing for some price dates. "
            "Regenerate signals using the current cleaned dataset."
        )

    returns = pair_prices.pct_change(fill_method=None).fillna(0)

    # Reconstruct the fixed training-period hedge ratio.
    train = pair_prices.loc[:TRAIN_END]
    if len(train) < 100:
        raise ValueError("Insufficient training data.")

    x = train[ticker_b].to_numpy(dtype=float)
    y = train[ticker_a].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    _, beta = np.linalg.lstsq(design, y, rcond=None)[0]

    if not np.isfinite(beta) or beta <= 0:
        raise ValueError(f"Invalid hedge ratio: {beta}")

    # Normalize gross exposure to approximately 100%.
    gross_exposure = 1 + abs(beta)
    weight_a = signals["Position"].astype(float) / gross_exposure
    weight_b = -signals["Position"].astype(float) * beta / gross_exposure

    weights = pd.DataFrame(
        {ticker_a: weight_a, ticker_b: weight_b},
        index=pair_prices.index,
    )

    # Use yesterday's position for today's price return.
    executed_weights = weights.shift(1).fillna(0)

    # One-way cost charged on each dollar of absolute weight traded.
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()

    cost_rate = TRANSACTION_COST_BPS / 10_000
    transaction_cost = turnover * cost_rate

    gross_strategy_return = (
        executed_weights * returns
    ).sum(axis=1)

    net_strategy_return = gross_strategy_return - transaction_cost

    # Equal-weight buy-and-hold benchmark, rebalanced conceptually
    # to 50/50 daily. This is a simple reference, not a market index.
    benchmark_return = returns.mean(axis=1)

    results = pd.DataFrame(index=pair_prices.index)
    results.index.name = "Date"

    results["Position"] = signals["Position"].astype(int)
    results["Weight_A"] = executed_weights[ticker_a]
    results["Weight_B"] = executed_weights[ticker_b]
    results["Turnover"] = turnover
    results["Transaction_Cost"] = transaction_cost
    results["Gross_Strategy_Return"] = gross_strategy_return
    results["Strategy_Return"] = net_strategy_return
    results["Benchmark_Return"] = benchmark_return

    results["Strategy_Equity"] = (
        1 + results["Strategy_Return"]
    ).cumprod()

    results["Benchmark_Equity"] = (
        1 + results["Benchmark_Return"]
    ).cumprod()

    results["Strategy_Drawdown"] = (
        results["Strategy_Equity"]
        / results["Strategy_Equity"].cummax() - 1
    )

    results["Benchmark_Drawdown"] = (
        results["Benchmark_Equity"]
        / results["Benchmark_Equity"].cummax() - 1
    )

    output_file = OUTPUT_DIR / "backtest_results.csv"
    results.to_csv(output_file)

    # Report each period independently. These are descriptive metrics;
    # parameters must not be tuned using validation/test results.
    periods = {
        "Training": (None, TRAIN_END),
        "Validation": (TRAIN_END, VALIDATION_END),
        "Test": (VALIDATION_END, None),
    }

    summary = []

    for period_name, (start, end) in periods.items():
        subset = results.loc[start:end].copy()

        # Avoid counting the boundary date in both periods.
        if period_name == "Validation":
            subset = subset.loc[subset.index > pd.Timestamp(TRAIN_END)]
        elif period_name == "Test":
            subset = subset.loc[
                subset.index > pd.Timestamp(VALIDATION_END)
            ]

        strategy_stats = calculate_performance(subset["Strategy_Return"])
        benchmark_stats = calculate_performance(subset["Benchmark_Return"])

        summary.append({
            "Period": period_name,
            "Observations": len(subset),
            **{f"Strategy_{k}": v for k, v in strategy_stats.items()},
            **{f"Benchmark_{k}": v for k, v in benchmark_stats.items()},
            "Total_Transaction_Cost": subset["Transaction_Cost"].sum(),
        })

    summary_df = pd.DataFrame(summary)
    summary_file = OUTPUT_DIR / "performance_summary.csv"
    summary_df.to_csv(summary_file, index=False)

    print("\nBACKTEST COMPLETE")
    print("=" * 70)
    print(f"Pair: {ticker_a} / {ticker_b}")
    print(f"Training hedge ratio: {beta:.6f}")
    print(f"Assumed cost: {TRANSACTION_COST_BPS:.1f} basis points per dollar traded")
    print(f"Saved daily results: {output_file}")
    print(f"Saved summary: {summary_file}")
    print("\nPerformance by period:")
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    return results


if __name__ == "__main__":
    run_backtest()
