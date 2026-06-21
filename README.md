# Stock Pattern Model

This project is a beginner-friendly, rule-based stock market analysis module. It downloads OHLCV data from `yfinance`, creates candlestick and technical features with `pandas` and `numpy`, detects common chart and candlestick patterns, and summarizes the latest market bias with a confidence score and plain-English explanation.

The project is intentionally focused only on the analysis layer. It does not include a web app, API, frontend, or machine learning model yet. The pattern logic is fully rule-based so it can be reviewed, tested, and later upgraded into a supervised learning workflow when labeled data is available.

## Project Structure

```text
stock-pattern-model/
├── data_loader.py
├── features.py
├── pattern_detector.py
├── model.py
├── main.py
├── requirements.txt
└── README.md
```

## What the Project Does

The model performs these steps:

1. Downloads historical stock data from `yfinance`
2. Cleans the OHLCV dataset
3. Computes candle structure, moving averages, rolling highs/lows, returns, and volatility
4. Detects candlestick and chart patterns using explicit rules
5. Classifies the latest trend
6. Scores recent bullish and bearish signals from the last 5 trading days
7. Returns a structured summary with confidence and explanation

## Installation

Install the required packages:

```bash
cd stock-pattern-model
pip install -r requirements.txt
```

## How to Run

Run the sample analysis script:

```bash
cd stock-pattern-model
python main.py
```

The default watchlist in `main.py` is:

- `AAPL`
- `MSFT`
- `NVDA`
- `TSLA`

You can edit that list directly in `main.py` to analyze other tickers.

## Detected Patterns

The current rule-based detector looks for:

- Bullish Engulfing
- Bearish Engulfing
- Bullish Pin Bar
- Shooting Star
- Inside Bar
- Inside Day Failure
- 20-Day Breakout
- 20-Day Breakdown
- Double Bottom
- Double Top
- Trend classification: Uptrend, Downtrend, or Neutral

## Output Fields

`analyze_stock(symbol)` returns a dictionary with:

- `symbol`: the ticker that was analyzed
- `latest_date`: the most recent trading date in the dataset
- `latest_close`: the latest closing price
- `trend`: the current trend classification
- `detected_patterns`: active patterns found in the last 5 trading days
- `overall_bias`: `Bullish`, `Bearish`, or `Neutral`
- `confidence_score`: a 0 to 100 score based on the weighted rule-based signal strength
- `explanation`: a readable explanation of the latest result

## Pattern Logic Summary

Feature engineering includes:

- Candle body, range, upper wick, and lower wick
- Candle body and wick ratios
- Bullish and bearish candle direction flags
- Daily return
- 20, 50, and 200 day moving averages
- 20 day rolling volume average
- 20 day rolling volatility
- Previous 20 day rolling high and low levels

Scoring includes:

- Bullish patterns add positive points
- Bearish patterns add negative points
- More recent patterns have larger weights
- Trend direction adds extra context to the final score

## Important Note

This project is an analytical educational model, not financial advice. It is designed to demonstrate rule-based technical analysis concepts and should not be used as the sole basis for trading or investment decisions.
