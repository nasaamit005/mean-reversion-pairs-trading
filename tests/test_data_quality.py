
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PRICES_FILE = ROOT / "data" / "processed" / "cleaned_prices.csv"
SIGNALS_FILE = ROOT / "data" / "processed" / "trading_signals.csv"
BACKTEST_FILE = ROOT / "outputs" / "backtest_results.csv"
SUMMARY_FILE = ROOT / "outputs" / "performance_summary.csv"
CHART_DIR = ROOT / "reports" / "charts"


def test_cleaned_prices_file_exists():
    assert PRICES_FILE.exists(), f"Missing file: {PRICES_FILE}"


def test_cleaned_prices_quality():
    assert PRICES_FILE.exists(), "Run data cleaning first."

    prices = pd.read_csv(PRICES_FILE, parse_dates=["Date"])

    assert not prices.empty, "Price dataset is empty."
    assert "Date" in prices.columns
    assert prices["Date"].notna().all()
    assert prices["Date"].is_monotonic_increasing
    assert not prices["Date"].duplicated().any()

    price_columns = [c for c in prices.columns if c != "Date"]

    assert len(price_columns) >= 2
    assert prices[price_columns].notna().all().all()
    assert np.isfinite(prices[price_columns].to_numpy(dtype=float)).all()
    assert (prices[price_columns] > 0).all().all()


def test_signals_file_and_positions():
    assert SIGNALS_FILE.exists(), "Trading signals file is missing."

    signals = pd.read_csv(SIGNALS_FILE, parse_dates=["Date"])

    required = {"Date", "Spread", "Z_Score", "Position"}
    assert required.issubset(signals.columns), (
        f"Missing columns: {required - set(signals.columns)}"
    )

    assert not signals.empty
    assert signals["Date"].notna().all()
    assert not signals["Date"].duplicated().any()
    assert signals["Position"].isin([-1, 0, 1]).all()

    z_scores = signals["Z_Score"].dropna()
    assert np.isfinite(z_scores).all()


def test_backtest_results():
    assert BACKTEST_FILE.exists(), "Backtest results are missing."

    results = pd.read_csv(BACKTEST_FILE, parse_dates=["Date"])

    required = {
        "Date",
        "Strategy_Return",
        "Benchmark_Return",
        "Strategy_Equity",
        "Benchmark_Equity",
        "Strategy_Drawdown",
    }

    assert required.issubset(results.columns), (
        f"Missing columns: {required - set(results.columns)}"
    )
    assert not results.empty
    assert results["Date"].notna().all()
    assert not results["Date"].duplicated().any()

    numeric_columns = [
        "Strategy_Return",
        "Benchmark_Return",
        "Strategy_Equity",
        "Benchmark_Equity",
        "Strategy_Drawdown",
    ]

    values = results[numeric_columns].to_numpy(dtype=float)

    assert np.isfinite(values).all()
    assert (results["Strategy_Equity"] > 0).all()
    assert (results["Benchmark_Equity"] > 0).all()
    assert (results["Strategy_Drawdown"] <= 1e-10).all()


def test_performance_summary_exists():
    assert SUMMARY_FILE.exists(), "Performance summary is missing."

    summary = pd.read_csv(SUMMARY_FILE)

    assert not summary.empty
    assert "Period" in summary.columns
    assert {"Training", "Validation", "Test"}.issubset(
        set(summary["Period"])
    )


def test_interactive_html_charts_exist():
    expected_files = [
        "01_normalized_prices.html",
        "02_spread_mean.html",
        "03_zscore_signals.html",
        "04_backtest_performance.html",
        "05_drawdown.html",
        "mean_reversion_dashboard.html",
    ]

    for filename in expected_files:
        path = CHART_DIR / filename

        assert path.exists(), f"Missing chart: {filename}"
        assert path.stat().st_size > 0, f"Empty chart: {filename}"

        content = path.read_text(encoding="utf-8")
        assert "plotly" in content.lower(), (
            f"Plotly content not found in {filename}"
        )
