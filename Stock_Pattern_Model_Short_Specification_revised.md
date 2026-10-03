Stock Pattern Model - Short Specification (Revised)

1. Project Overview / Introduction
Stock Pattern Model is a rule-based technical-analysis system for OHLCV stock market data across multiple timeframes, from intraday bars (as short as 15 minutes) up to daily and weekly bars. It loads market data for a chosen instrument, detects recognized chart and candlestick patterns, analyzes the underlying trend, and produces a structured, human-readable report describing what was found, when it was detected, and how strong the signal is. The system focuses purely on analysis, it does not place trades or manage a portfolio.

2. Problem Statement
Manually tracking large volumes of price and volume data and spotting meaningful chart patterns in real time is difficult and error-prone for an individual investor. Important formations (such as reversal or breakout patterns) can be easy to miss, and it is hard to judge, without a consistent method, how strong or reliable a given signal is at a given moment. The system automates this detection and scoring process so the user gets a clear, consistent read of the market instead of manual, ad-hoc chart-watching.

3. Project Goals
* Automatically identify recognized chart and candlestick patterns.
* Report the exact, exchange-aware time each pattern was detected.
* Analyze the overall and local market trend.
* Calculate a score and confidence level for the resulting signal.
* Track each detected pattern through its lifecycle (tentative, confirmed, invalidated, expired, or awaiting confirmation) rather than treating detection as a one-time event.
* Allow the user to enter a ticker symbol (or an Israeli security number) to analyze.
* Present the results clearly, in both text and JSON formats.

4. Target Users / Use Case
The system is aimed at individual investors, traders, and students of technical analysis who want a quick, structured read on a stock without manually scanning charts. A user enters a ticker (e.g. AAPL) or an Israeli security number, chooses a time frame (e.g., one day, one month, one year), and receives a report showing the current trend, detected patterns, a directional bias, and a confidence score for further use.

5. Main Functional Requirements
* Accept a stock ticker, company name, or Israeli security number as input.
* Retrieve OHLCV (Open/High/Low/Close/Volume) market data, either live or from local files (CSV/Parquet).
* Validate the data for quality issues (missing values, duplicate or out-of-order timestamps, invalid prices, etc.).
* Detect chart and candlestick patterns from a defined pattern library.
* Distinguish regular-session trading from premarket/afterhours activity, and let the user choose which segments to include.
* Analyze the market trend and overall directional bias.
* Score the detected evidence, resolve conflicts when bullish and bearish evidence coexist, and compute a confidence level.
* Present results in clear text or JSON, including a final summary assessment.
* Record an exact, exchange-aware timestamp for every detected pattern (e.g. "2026-09-03 07:00:00+0300 Asia/Jerusalem", not a bare time of day).

6. System Overview / Architecture
At a high level, the system works as a pipeline:


User → CLI → Data Provider → Data Validation & Processing →


Feature Engineering → Pattern Detection → Trend & Scoring Engine → Results


* CLI: entry point where the user provides a ticker/security number and analysis settings (time frame, display timezone, exchange timezone, session mode/segments, output format, caching options, and an optional "as-of" point-in-time cutoff for historical/leakage-free testing).
* Data Provider: fetches market data either live (via Yahoo Finance, with retry logic and optional local caching) or from local CSV/Parquet files, optionally resolving Israeli security numbers to tickers through a CSV mapping file.
* Data Validation & Processing: checks data quality and filters out incomplete candles.
* Feature Engineering: computes supporting indicators (e.g., moving averages, session highs/lows, volume baselines).
* Pattern Detection: a registry of independent detectors, each identifying one pattern type, with results tracked through a lifecycle (tentative, confirmed, invalidated, expired, directionally confirmed, or awaiting confirmation) across analysis runs.
* Trend & Scoring Engine: combines trend analysis and detected patterns into scores, a directional bias, and a confidence level, tempering the result when bullish and bearish evidence conflict.
* Results: a structured report including a final "recommend to buy" / "not recommended" / "neutral" style assessment with an explanation.

7. Main Patterns / Algorithms
The system currently detects the following patterns (without going into their internal formulas):


* Bullish Engulfing
* Bearish Engulfing
* Bullish Pin Bar
* Hammer
* Shooting Star
* Inside Bar / Inside Bar Failure
* Breakout / Breakdown
* Doji
* Morning Star / Evening Star
* Double Top / Double Bottom

8. Data Sources
* Live data: fetched through a Yahoo Finance–based provider, with retry logic, metadata capture, and optional local caching.
* Offline data: CSV or Parquet files supplied directly by the user, following a required OHLCV column structure.
* Interval / time frame: configurable, ranging from 15-minute intraday bars up to weekly bars, via preset time-frame options (one day, one week, one month, three/six months, one year, five years) or manual period/interval settings.
* Additional mapping data: an optional CSV mapping file used to resolve Israeli security numbers to their corresponding ticker symbols.
* Stored/output information: each analysis run can be saved as a structured JSON or text file containing the detected patterns, scores, timestamps, and explanations.

9. Expected Output
A typical result includes fields such as:


Trend: Uptrend


Pattern: Bullish Engulfing


Detected at: 2026-09-03 07:00:00+0300 Asia/Jerusalem


Rule Confidence: 78/100 (Confidence Level: HIGH)


Full output additionally includes the overall market bias, individual scoring components, a list of supporting/conflicting evidence, data-quality warnings, and a final structured assessment ("RECOMMEND TO BUY" / "NOT RECOMMENDED" / "NEUTRAL") with a short explanation and a disclaimer. Rule Confidence is reported as an uncalibrated 0-100 rule-strength score, not a statistical probability.

10. Limitations and Future Development
* The system does not guarantee trading success and does not provide financial advice. It is an educational, rule-based technical-analysis tool.
* Pattern detection and scoring are based entirely on historical/past price and volume data, not on predictions of future events.
* Confidence scores are rule-based strength indicators, not statistically calibrated probabilities.
* Signal-level historical evaluation and backtesting (forward returns, MFE/MAE, target-vs-stop outcomes, grouped performance summaries) is already implemented at the library level, but is not yet exposed as an interactive CLI command.
* The system currently does not include brokerage integration, live trade execution, portfolio management, paper trading, portfolio-level trading simulation, portfolio equity curves, or Sharpe-ratio calculations.
* Potential future directions might include: exposing the existing backtesting engine through the CLI, portfolio-level simulation, a graphical user interface (GUI), additional tested chart patterns, and potential use of machine learning to complement the rule-based approach.
