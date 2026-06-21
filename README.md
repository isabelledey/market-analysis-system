# Stock Pattern Model

This project is a beginner-friendly, rule-based intraday stock market analysis module. It downloads 15-minute OHLCV data from `yfinance`, creates candlestick and technical features with `pandas` and `numpy`, detects common intraday chart and candlestick patterns, filters noisy signals, and summarizes the latest market bias with a confidence score and plain-English explanation.

The model now analyzes intraday `15-minute` candles only. It does not analyze daily candles. The project is intentionally focused only on the analysis layer. It does not include a web app, API, frontend, or machine learning model yet. The pattern logic is fully rule-based so it can be reviewed, tested, and later upgraded into a supervised learning workflow when labeled data is available.

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

1. Downloads intraday stock data from `yfinance`
2. Cleans the OHLCV dataset
3. Computes 15-minute bar structure, moving averages, rolling highs/lows, returns, session levels, volatility, and candle significance filters
4. Detects intraday candlestick and breakout patterns using explicit rules
5. Filters weak and conflicting signals so noisy candles do not dominate the result
6. Classifies the latest intraday trend and market state
7. Scores recent bullish and bearish signals from the last 12 candles
8. Returns a structured summary with confidence and explanation

## Intraday Data Limit

`yfinance` intraday history is limited. For `15m` candles, the model uses a default `period="1mo"` and does not attempt to analyze 2 years of data. Intraday data cannot extend beyond the recent Yahoo Finance history window.

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
- Inside Bar Failure
- 20-Bar Breakout
- 20-Bar Breakdown
- Intraday trend classification: Uptrend, Downtrend, or Neutral

Because intraday data is noisy, the model also applies filtering:

- Small candles are ignored unless range or volume is meaningful
- Stronger volume and stronger range are tracked separately
- Same-candle conflicts are resolved with a pattern priority system
- Only the strongest recent patterns influence the final summary

## Output Fields

`analyze_stock(symbol)` returns a dictionary with:

- `symbol`: the ticker that was analyzed
- `latest_datetime`: the most recent 15-minute bar timestamp in the dataset
- `latest_close`: the latest closing price
- `interval`: the candle interval used for the analysis
- `trend`: the current intraday trend classification
- `market_state`: a higher-level label such as `Trending Bullish`, `Choppy`, or `Breakout Attempt`
- `overall_bias`: `Bullish`, `Bearish`, or `Neutral`
- `bullish_score`: the recent weighted bullish score including trend alignment
- `bearish_score`: the recent weighted bearish score including trend alignment
- `total_score`: `bullish_score - bearish_score`
- `confidence_score`: a 0 to 100 score based on the weighted rule-based signal strength
- `top_patterns`: the top 3 recent filtered patterns that most affected the result
- `ignored_patterns_count`: how many lower-priority conflicting patterns were removed
- `explanation`: a readable explanation of the latest result

## Pattern Logic Summary

Feature engineering includes:

- Candle body, range, upper wick, and lower wick
- Candle body and wick ratios
- Bullish and bearish candle direction flags
- 15-minute bar return
- 20 bar and 50 bar moving averages
- 20 bar rolling volume average
- 20 bar rolling volatility
- Previous 20 bar rolling high and low levels
- Trading date, session open, session high, and session low
- Distance from current session high and low
- Average 20 bar candle range
- Range strength and volume strength
- Strong volume, strong range, and candle significance filters

Scoring includes:

- Bullish patterns add positive points
- Bearish patterns add negative points
- More recent patterns have larger weights
- Trend direction adds extra context to the final score
- Conflicting same-candle patterns are filtered by priority
- Confidence is reduced when the market is choppy or balanced

## Important Note

This project is an educational analytical model, not financial advice. It is designed to demonstrate rule-based intraday technical analysis concepts and should not be used as the sole basis for trading or investment decisions.
