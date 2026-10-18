# Mean-Reversion Pairs Trading Strategy

A quantitative finance research project that investigates whether two historically related assets exhibit a mean-reverting price relationship. The project covers data preparation, pair screening, hedge-ratio estimation, trading signals, historical backtesting, and interactive visualization.

## Project Overview

**Selected candidate pair:** Visa (V) and Mastercard (MA)

**Primary objective:** Evaluate a statistical pairs-trading strategy using historical price data and out-of-sample testing.

This is a research and backtesting project, not a live trading system or a claim of guaranteed profitability.

## Key Features

- Historical market-data preparation and cleaning
- Correlation and cointegration-based pair screening
- Hedge-ratio estimation
- Spread construction and rolling Z-score calculation
- Rule-based entry, exit, and position signals
- Historical backtesting with assumed transaction costs
- Training, validation, and test-period performance evaluation
- Interactive Plotly charts and HTML dashboard
- Automated data-quality and output-integrity tests

## Project Structure

```text
mean-reversion-pairs-trading/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── outputs/
│   ├── backtest_results.csv
│   └── performance_summary.csv
├── reports/
│   └── charts/
│       ├── 01_normalized_prices.html
│       ├── 02_spread_mean.html
│       ├── 03_zscore_signals.html
│       ├── 04_backtest_performance.html
│       ├── 05_drawdown.html
│       └── mean_reversion_dashboard.html
├── src/
│   ├── data_cleaning.py
│   ├── download_data.py
│   ├── pair_selection.py
│   ├── signals.py
│   ├── backtest.py
│   ├── metrics.py
│   ├── visualizations.py
│   └── run_pipeline.py
├── tests/
│   └── test_data_quality.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Technology Stack

- **Python:** Data processing and quantitative analysis
- **pandas, NumPy:** Data manipulation and numerical calculations
- **SciPy, statsmodels:** Statistical analysis and time-series methods
- **scikit-learn:** Supporting analytical tools, where applicable
- **yfinance:** Historical market-data retrieval
- **Plotly:** Interactive charts and dashboard
- **pytest:** Automated testing
- **Git and GitHub:** Version control and project publication

## Methodology

### 1. Data Preparation

Retrieve historical prices, standardize the dataset, check missing values and duplicates, and prepare the data for statistical analysis.

### 2. Pair Selection

Screen candidate assets using price relationships and statistical tests. Correlation alone does not establish cointegration or a stable long-term relationship.

### 3. Spread and Hedge Ratio

Estimate a hedge ratio using the training sample and construct a spread between the two assets. The precise spread definition and estimation procedure are documented in the implementation.

### 4. Trading Signals

Calculate a rolling Z-score and use predefined thresholds to identify potential deviations from the estimated mean.

A trading signal indicates a research decision, not a guarantee that an order can be filled profitably.

### 5. Backtesting

Evaluate historical strategy performance against an equal-weight buy-and-hold benchmark for the selected assets. The implementation uses an assumed transaction cost of 5 basis points per dollar traded.

### 6. Out-of-Sample Evaluation

Assess training, validation, and test periods separately. Compare returns, volatility, Sharpe ratio, and maximum drawdown rather than relying on cumulative return alone.

## Interactive Visualizations

After running the visualization script, open:

`reports/charts/mean_reversion_dashboard.html`

The dashboard includes:

1. Normalized asset prices
2. Spread and rolling mean
3. Z-score and trading signals
4. Strategy equity versus benchmark
5. Drawdown comparison

Individual charts are available in the same directory.

## How to Run

### 1. Set up Python

Python 3.12 is the environment used during project development.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run the pipeline

From the project root:

```powershell
python .\src\run_pipeline.py
```

The pipeline starts with data cleaning and continues through pair selection, signals, backtesting, and visualizations. The expected input files must exist before each stage runs.

If the data has not yet been downloaded, run the project's data-download script first.

### 3. Run the tests

```powershell
python -m pytest -v
```

### 4. Open the dashboard

```powershell
Invoke-Item .\reports\charts\mean_reversion_dashboard.html
```

## Testing

The current test suite contains six checks covering:

- Cleaned-price file and data quality
- Trading-signal structure and valid positions
- Backtest output integrity
- Performance-summary availability
- Interactive chart generation

The six tests passed during local development. Run the tests again after modifying the code or data.

## Limitations and Risk Considerations

- Historical relationships may break down in the future.
- Cointegration does not guarantee profitable mean reversion.
- Results can be sensitive to pair selection, hedge ratios, and trading thresholds.
- The assumed transaction cost is not a live broker quote.
- Borrowing fees, short-sale constraints, market impact, liquidity, and execution slippage may not be fully modeled.
- Out-of-sample performance must be evaluated independently of model selection.
- The backtest should be reviewed for look-ahead bias, signal timing, and correct execution-price assumptions before drawing investment conclusions.
- No live trading or guaranteed future returns are claimed.

## Future Improvements

- Walk-forward pair selection and hedge-ratio estimation
- Robust tests for structural breaks and parameter stability
- More realistic execution, slippage, and short-borrowing costs
- Sensitivity analysis for entry and exit thresholds
- Statistical significance testing and multiple-testing controls
- Additional benchmarks and risk-adjusted performance measures
- Continuous integration for automated tests

## Disclaimer

This repository is for educational and quantitative research purposes only. It is not investment advice, and the strategy has not been established as suitable for live trading.

## Author

Developed as a quantitative finance portfolio project to demonstrate Python-based financial data analysis, statistical research, backtesting, visualization, and software testing.