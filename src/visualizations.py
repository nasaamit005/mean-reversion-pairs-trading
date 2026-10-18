
from pathlib import Path
import html

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]

PRICES_FILE = ROOT / "data" / "processed" / "cleaned_prices.csv"
SIGNALS_FILE = ROOT / "data" / "processed" / "trading_signals.csv"
BACKTEST_FILE = ROOT / "outputs" / "backtest_results.csv"

CHART_DIR = ROOT / "reports" / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

# Light, professional theme
BG = "#F4F7FB"
PANEL = "#FFFFFF"
TEXT = "#243247"
GRID = "#E2E8F0"
BLUE = "#2563EB"
ORANGE = "#EA580C"
GREEN = "#16A34A"
RED = "#DC2626"
PURPLE = "#7C3AED"
GREY = "#64748B"


def load_data():
    for file in [PRICES_FILE, SIGNALS_FILE, BACKTEST_FILE]:
        if not file.exists():
            raise FileNotFoundError(
                f"Required file missing: {file}\n"
                "Run the data, signals and backtest scripts first."
            )

    prices = pd.read_csv(PRICES_FILE, parse_dates=["Date"])
    prices = prices.set_index("Date").sort_index()

    signals = pd.read_csv(SIGNALS_FILE, parse_dates=["Date"])
    signals = signals.set_index("Date").sort_index()

    backtest = pd.read_csv(BACKTEST_FILE, parse_dates=["Date"])
    backtest = backtest.set_index("Date").sort_index()

    ticker_a = str(signals["Ticker_A"].dropna().iloc[0])
    ticker_b = str(signals["Ticker_B"].dropna().iloc[0])

    return prices, signals, backtest, ticker_a, ticker_b


def style_figure(fig, title, y_title):
    fig.update_layout(
        title=dict(text=title, font=dict(size=21, color=TEXT)),
        template="plotly_white",
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        font=dict(family="Arial, sans-serif", color=TEXT, size=12),
        margin=dict(l=65, r=35, t=85, b=65),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0.85)",
        ),
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor=GRID,
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(title=y_title, showgrid=True, gridcolor=GRID),
        height=540,
    )
    return fig


def save_chart(fig, filename):
    path = CHART_DIR / filename
    fig.write_html(
        str(path),
        include_plotlyjs=True,
        full_html=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "responsive": True,
        },
    )
    print(f"Saved chart: {path}")


def normalized_prices_chart(prices, ticker_a, ticker_b):
    pair = prices[[ticker_a, ticker_b]].dropna()

    if pair.empty:
        raise ValueError("No overlapping prices available for the selected pair.")

    normalized = pair / pair.iloc[0] * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=normalized.index,
        y=normalized[ticker_a],
        mode="lines",
        name=ticker_a,
        line=dict(color=BLUE, width=2.5),
        hovertemplate="%{x|%d %b %Y}<br>" + ticker_a
                      + ": %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=normalized.index,
        y=normalized[ticker_b],
        mode="lines",
        name=ticker_b,
        line=dict(color=ORANGE, width=2.5),
        hovertemplate="%{x|%d %b %Y}<br>" + ticker_b
                      + ": %{y:.2f}<extra></extra>",
    ))

    fig.add_hline(
        y=100, line_dash="dot", line_color=GREY,
        annotation_text="Starting value = 100",
    )
    style_figure(
        fig,
        f"01 · Normalized Prices — {ticker_a} vs {ticker_b}",
        "Normalized price (start = 100)",
    )
    return fig


def spread_chart(signals, ticker_a, ticker_b):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=signals.index,
        y=signals["Spread"],
        mode="lines",
        name="Spread",
        line=dict(color=BLUE, width=1.7),
        hovertemplate="%{x|%d %b %Y}<br>Spread: %{y:.4f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=signals.index,
        y=signals["Rolling_Mean"],
        mode="lines",
        name="Previous 60-day rolling mean",
        line=dict(color=RED, width=2, dash="dash"),
        hovertemplate="Rolling mean: %{y:.4f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=signals.index,
        y=signals["Rolling_Mean"] + 2 * signals["Rolling_Std"],
        mode="lines",
        name="+2 rolling SD",
        line=dict(color=GREY, width=1, dash="dot"),
    ))

    fig.add_trace(go.Scatter(
        x=signals.index,
        y=signals["Rolling_Mean"] - 2 * signals["Rolling_Std"],
        mode="lines",
        name="-2 rolling SD",
        line=dict(color=GREY, width=1, dash="dot"),
        fill="tonexty",
        fillcolor="rgba(100,116,139,0.07)",
    ))

    style_figure(
        fig,
        f"02 · Spread and Rolling Mean — {ticker_a}/{ticker_b}",
        "Spread (price units)",
    )
    return fig


def zscore_chart(signals, ticker_a, ticker_b):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=signals.index,
        y=signals["Z_Score"],
        mode="lines",
        name="Rolling Z-score",
        line=dict(color=BLUE, width=1.8),
        hovertemplate="%{x|%d %b %Y}<br>Z-score: %{y:.3f}<extra></extra>",
    ))

    for value, color, label in [
        (2, RED, "Short-spread entry (+2)"),
        (-2, GREEN, "Long-spread entry (-2)"),
        (0.5, GREY, "Exit (+0.5)"),
        (-0.5, GREY, "Exit (-0.5)"),
        (4, PURPLE, "Stop (+4)"),
        (-4, PURPLE, "Stop (-4)"),
    ]:
        fig.add_hline(
            y=value,
            line_dash="dash",
            line_color=color,
            annotation_text=label,
            annotation_position="top left",
        )

    # Position changes represent signal decisions, not guaranteed fills.
    previous = signals["Position"].shift(1).fillna(0)
    entered_long = (signals["Position"] == 1) & (previous != 1)
    entered_short = (signals["Position"] == -1) & (previous != -1)
    exited = (signals["Position"] == 0) & (previous != 0)

    for mask, name, color, symbol in [
        (entered_long, "Long-spread decision", GREEN, "triangle-up"),
        (entered_short, "Short-spread decision", RED, "triangle-down"),
        (exited, "Exit decision", ORANGE, "x"),
    ]:
        points = signals.loc[mask].dropna(subset=["Z_Score"])
        fig.add_trace(go.Scatter(
            x=points.index,
            y=points["Z_Score"],
            mode="markers",
            name=name,
            marker=dict(color=color, size=10, symbol=symbol),
            hovertemplate=(
                "%{x|%d %b %Y}<br>Z-score: %{y:.3f}<extra>"
                + name + "</extra>"
            ),
        ))

    style_figure(
        fig,
        f"03 · Z-score and Signal Decisions — {ticker_a}/{ticker_b}",
        "Z-score",
    )
    return fig


def equity_chart(backtest, ticker_a, ticker_b):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=backtest.index,
        y=backtest["Strategy_Equity"],
        mode="lines",
        name="Pairs strategy (net of assumed costs)",
        line=dict(color=GREEN, width=2.7),
        hovertemplate="%{x|%d %b %Y}<br>Equity: %{y:.4f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=backtest.index,
        y=backtest["Benchmark_Equity"],
        mode="lines",
        name="Equal-weight buy-and-hold benchmark",
        line=dict(color=BLUE, width=2.2),
        hovertemplate="%{x|%d %b %Y}<br>Equity: %{y:.4f}<extra></extra>",
    ))

    fig.add_hline(y=1, line_dash="dot", line_color=GREY)

    style_figure(
        fig,
        f"04 · Equity Curve — {ticker_a}/{ticker_b}",
        "Growth of initial $1 (before other omitted costs)",
    )
    return fig


def drawdown_chart(backtest, ticker_a, ticker_b):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=backtest.index,
        y=backtest["Strategy_Drawdown"] * 100,
        mode="lines",
        name="Strategy drawdown",
        line=dict(color=RED, width=1.8),
        fill="tozeroy",
        fillcolor="rgba(220,38,38,0.12)",
        hovertemplate="%{x|%d %b %Y}<br>Drawdown: %{y:.2f}%<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=backtest.index,
        y=backtest["Benchmark_Drawdown"] * 100,
        mode="lines",
        name="Benchmark drawdown",
        line=dict(color=BLUE, width=1.6),
        hovertemplate="%{x|%d %b %Y}<br>Drawdown: %{y:.2f}%<extra></extra>",
    ))

    style_figure(
        fig,
        f"05 · Drawdown Analysis — {ticker_a}/{ticker_b}",
        "Drawdown (%)",
    )
    return fig


def create_dashboard(figures, ticker_a, ticker_b):
    sections = []

    for title, fig in figures:
        sections.append(
            "<section><h2>"
            + html.escape(title)
            + "</h2>"
            + fig.to_html(
                full_html=False,
                include_plotlyjs=False,
                config={"displaylogo": False, "responsive": True},
            )
            + "</section>"
        )

    dashboard_html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mean-Reversion Pairs Trading Dashboard</title>
<script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>
<style>
body { margin: 0; background: #F4F7FB; color: #243247;
       font-family: Arial, sans-serif; }
header { padding: 28px 5%; background: white;
         border-bottom: 1px solid #E2E8F0; }
main { max-width: 1450px; margin: auto; padding: 18px; }
section { background: white; padding: 14px; margin: 22px 0;
          border-radius: 12px; box-shadow: 0 3px 14px #24324712; }
h1 { margin-bottom: 8px; }
h2 { font-size: 18px; padding-left: 12px; }
p { color: #64748B; }
</style>
</head>
<body>
<header>
<h1>Mean-Reversion Pairs Trading</h1>
<p>Candidate pair: """ + html.escape(ticker_a) + " / " + html.escape(ticker_b) + """</p>
<p>Research dashboard · Historical simulation, not a live trading system</p>
</header>
<main>
""" + "\n".join(sections) + """
</main>
</body>
</html>
"""

    path = CHART_DIR / "mean_reversion_dashboard.html"
    path.write_text(dashboard_html, encoding="utf-8")
    print(f"Saved dashboard: {path}")


def main():
    prices, signals, backtest, ticker_a, ticker_b = load_data()

    charts = [
        (
            "Normalized Prices",
            normalized_prices_chart(prices, ticker_a, ticker_b),
            "01_normalized_prices.html",
        ),
        (
            "Spread and Rolling Mean",
            spread_chart(signals, ticker_a, ticker_b),
            "02_spread_mean.html",
        ),
        (
            "Z-score and Trading Signals",
            zscore_chart(signals, ticker_a, ticker_b),
            "03_zscore_signals.html",
        ),
        (
            "Strategy Equity Curve",
            equity_chart(backtest, ticker_a, ticker_b),
            "04_backtest_performance.html",
        ),
        (
            "Drawdown Analysis",
            drawdown_chart(backtest, ticker_a, ticker_b),
            "05_drawdown.html",
        ),
    ]

    for title, fig, filename in charts:
        save_chart(fig, filename)

    create_dashboard(
        [(title, fig) for title, fig, _ in charts],
        ticker_a,
        ticker_b,
    )

    print("\nAll interactive charts generated successfully.")
    print(f"Chart directory: {CHART_DIR}")


if __name__ == "__main__":
    main()
