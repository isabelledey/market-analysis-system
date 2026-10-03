Stock Market Pattern Analysis System - Extended Specification (Revised)

# STOCK MARKET PATTERN ANALYSIS SYSTEM
## Extended Project Specification
### Technical Analysis, Pattern Recognition, Scoring, Backtesting and System Architecture

Academic Project Documentation
September 2026

---

## Executive Summary

This project develops a modular software system for automated technical analysis of financial instruments. The system ingests OHLCV market data, validates and normalizes the data, computes candle and market-context features, detects well-known chart and candlestick patterns, evaluates market trend across several horizons, manages the lifecycle of technical signals, and combines the resulting evidence into an explainable market assessment.

The purpose of the system is not to guarantee price prediction. Instead, it converts a manual and subjective chart-reading process into a deterministic, explainable and empirically testable analysis pipeline. A central design principle is the strict separation between pattern detection, rule-based evidence strength, and historical predictive performance. Accordingly, Rule Confidence describes the internal consistency and quality of the current evidence; it is not presented as a probability that a trade will succeed.

The specification is based on the current project architecture: market-data providers, instrument resolution, AnalysisContext, feature engineering, pattern registry, broad and local trend analysis, structural deduplication, signal lifecycle, scoring, CLI output and historical evaluation. The document also defines a rigorous backtesting methodology intended to minimize look-ahead bias and data snooping.

### Primary Objectives

* Automate recognition of technical patterns using explicit, reproducible rules.
* Report the exact time at which a pattern becomes detectable, not merely the candle on which its visual structure began.
* Separate raw candle geometry from market context and confirmation requirements.
* Combine pattern, trend and volume evidence without double counting correlated signals.
* Provide an explainable technical assessment through text and structured JSON output.
* Evaluate every signal historically using forward returns, direction accuracy, MFE, MAE, target/stop outcomes and expectancy.
* Provide a modular foundation for future visualization, alerts, paper trading and machine-learning extensions.

---

## Table of Contents

1. Project Purpose and the Problem It Solves
2. Review of Scientific Research and Related Work
3. Functional and Non-Functional Requirements
4. System Description and Architecture
5. Pattern-Detection Algorithms
6. Trend, Confidence and Signal-Score Computation
7. Data Sources and Data-Quality Handling
8. CLI and Future User Interface
9. Testing, Backtesting and Accuracy Metrics
10. Limitations, Risks and Future Development
11. Diagrams, Tables and Example Outputs
12. Conclusion
References

---

## 1. Project Purpose and the Problem It Solves

### 1.1 Background

Technical analysis attempts to interpret price, volume and market structure in order to describe current market conditions and identify scenarios that may have predictive value. A common approach is to inspect candlestick formations and chart patterns such as Bullish Engulfing, Bearish Engulfing, Hammer, Shooting Star, Morning Star, Inside Bar, Breakout, Double Top and Double Bottom.

Manual chart analysis is difficult to standardize. Two analysts may assign different names to the same structure, detect the same structure at different times, or disagree on whether the surrounding trend makes the pattern meaningful. In addition, human analysis is vulnerable to hindsight bias: once future candles are visible, a past formation may appear more obvious than it actually was at the time.

### 1.2 Core Problem

The central engineering and research problem is therefore: how can a subjective visual analysis process be converted into a deterministic, traceable and testable algorithm without using future information and without presenting rule strength as statistically proven prediction accuracy?

### 1.3 Problems Addressed by the System

1. Subjectivity in visual recognition of candlestick and chart patterns.
2. Inconsistent pattern definitions across analysts and data sources.
3. Ambiguity between pattern formation time, completion time and first valid detection time.
4. Look-ahead bias caused by using future candles to confirm a signal retroactively.
5. Failure to distinguish candle geometry from the market context required for a classical pattern.
6. Ignoring trading-session boundaries, time zones and incomplete candles.
7. Double counting several labels that describe the same underlying market event.
8. Confusing an internal rule score with an empirically calibrated probability of success.
9. Lack of systematic historical evaluation for measuring whether detected signals have value.

### 1.4 Project Scope

The current system is best described as a rule-based technical-analysis and decision-support platform. It analyzes a selected financial instrument and returns technical evidence, trend state, pattern lifecycle information, scores, confidence and explanations. It does not automatically execute trades and it does not claim to forecast future prices with certainty.

| Included in the current scope | Outside the current scope |
|---|---|
| Ticker / security-number resolution | Live broker order execution |
| OHLCV ingestion and validation | Portfolio optimization |
| Candlestick and structural pattern detection | Institutional-grade market-data feed |
| Trend and local-trend estimation | Calibrated probability of profit |
| Rule-based scoring and confidence | Fully trained machine-learning predictor |
| Historical signal evaluation | Production paper-trading engine with full portfolio accounting |
| CLI and JSON output | Automated financial advice |

---

## 2. Review of Scientific Research and Related Work

### 2.1 Research Context

Academic evidence on technical analysis is mixed. Some studies report that technical rules or price patterns contain incremental information about future return distributions, while other studies find that popular candlestick strategies fail to generate economically significant excess returns after appropriate controls. This mixed evidence strongly motivates the architecture of the present project: detection is separated from empirical validation, and no detected pattern is treated as a guaranteed prediction.

| Study | Topic | Main finding | Relevance to this project |
|---|---|---|---|
| Brock, Lakonishok & LeBaron (1992) | Moving averages and trading-range breaks | Reported evidence that simple technical rules contained information inconsistent with several common random-walk models. | Supports empirical testing of trend and breakout rules. |
| Caginalp & Laurent (1998) | Predictive power of price patterns | Found predictive value for selected price-pattern rules in the sample studied. | Motivates quantitative pattern definitions. |
| Sullivan, Timmermann & White (1999) | Data snooping in technical rules | Showed that repeated testing of many rules can create apparently strong results by chance. | Supports strict out-of-sample and walk-forward evaluation. |
| Lo, Mamaysky & Wang (2000) | Automated technical pattern recognition | Demonstrated computational pattern-recognition methods and found information in some technical formations. | Closely aligned with replacing subjective visual analysis with algorithms. |
| Marshall, Young & Rose (2006) | Candlestick strategies | Found no value for several candlestick strategies on the DJIA sample examined. | Demonstrates why pattern detection must not be equated with profitability. |
| Park & Irwin (2007) | Survey of technical-analysis profitability | Reviewed a large body of studies and reported mixed evidence, with substantial methodological concerns. | Supports careful methodology and explicit limitations. |
| Chen & Tsai (2020) | Candlestick image encoding and CNN classification | Showed that candlestick information can be encoded for deep-learning classification. | Provides a possible future ML direction. |
| Cagliero, Fior & Garza (2023) | Pattern recognition with machine learning | Combined candlestick filtering with ML models and backtesting. | Demonstrates a possible hybrid architecture. |
| Agah et al. (2026) | Neural trend prediction and candlestick recognition | Proposed a hybrid framework combining trend prediction with candlestick recognition. | Highly relevant future extension of the current rule-based baseline. |

### 2.2 Methodological Consequences for the Project

* Pattern Detection is not equivalent to Prediction.
* Rule Confidence is not equivalent to Probability of Success.
* Rules must be frozen before final out-of-sample evaluation.
* The system should report signal counts together with accuracy metrics.
* Performance must be examined across patterns, market regimes, time of day and confidence buckets.
* Walk-forward testing is preferable to random train/test shuffling for time-series data.

---

## 3. Functional and Non-Functional Requirements

### 3.1 Functional Requirements

| ID | Requirement | Description |
|---|---|---|
| FR-01 | Instrument input | The system shall accept a ticker symbol and, where a mapping exists, an Israeli security number, and shall resolve the input to a canonical tradable symbol. |
| FR-02 | Timeframe selection | The user shall be able to select a predefined timeframe or provide period and interval parameters explicitly. |
| FR-03 | Market-data loading | The system shall support online market data and offline CSV/Parquet data providers. |
| FR-04 | Data validation | The system shall validate timestamps, OHLC integrity, volume, missing values, duplicate rows and interval continuity before analysis. |
| FR-05 | Completed-candle filtering | The system shall exclude a candle that has not completed at the requested as-of time. |
| FR-06 | Feature engineering | The system shall compute candle geometry, moving averages, rolling highs/lows, volatility, session context and volume baselines. |
| FR-07 | Pattern detection | Registered detectors shall return structured PatternEvent objects with name, family, bias, status, timestamps, score and explanation. |
| FR-08 | Exact timestamps | The system shall distinguish pattern_start_at, pattern_end_at and detected_at. |
| FR-09 | Trend analysis | The system shall calculate broad and local trends across multiple horizons and shall keep latest-candle direction separate from trend. |
| FR-10 | Pattern lifecycle | The system shall support candidate/tentative/confirmed/failed/expired states and pattern-specific confirmation logic. |
| FR-11 | Scoring | The system shall calculate bullish, bearish, pattern, volume, trend, net-signal and rule-confidence values. |
| FR-12 | Deduplication | The system shall prevent multiple correlated labels from being counted as independent evidence. |
| FR-13 | Overall bias | The system shall classify the final bias as Bullish, Bearish or Neutral while considering evidence conflict and confirmation. |
| FR-14 | Historical evaluation | The system shall replay historical cutoffs using only information available at each cutoff and shall evaluate future outcomes separately. |
| FR-15 | Structured output | The system shall support human-readable text and machine-readable JSON output. |

### 3.2 Non-Functional Requirements

| ID | Quality attribute | Requirement |
|---|---|---|
| NFR-01 | Determinism | Identical data, configuration and as-of time should produce identical results. |
| NFR-02 | No look-ahead | No detector may use information unavailable at detected_at. |
| NFR-03 | Explainability | Every meaningful signal should include a human-readable explanation and invalidation condition where applicable. |
| NFR-04 | Modularity | New detectors should be addable without redesigning the entire system. |
| NFR-05 | Testability | Detectors, scoring components and data utilities should be independently testable. |
| NFR-06 | Robustness | External provider failures or data problems must not silently produce misleading results. |
| NFR-07 | Time-zone correctness | Exchange time and display time shall remain explicitly separated. |
| NFR-08 | Traceability | The data-quality report should expose cleaning actions, row counts, warnings and gaps. |
| NFR-09 | Maintainability | Data, feature, detection, scoring, evaluation and presentation layers should remain separated. |
| NFR-10 | Performance | Interactive analysis of several thousand candles should complete within a practical response time on a standard personal computer. |

---

## 4. System Description and Architecture

### 4.1 Architectural Overview

The project follows a modular pipeline architecture. Input and instrument resolution are separated from market-data acquisition; validation and feature engineering are performed before trend and pattern logic; detected events are normalized, deduplicated and processed through a lifecycle; scoring then produces a market state and final assessment; historical evaluation operates as a separate layer so that future data cannot leak into live detection logic.

*Figure 1. High-level system architecture.*

### 4.2 Main Modules

| Module | Responsibility |
|---|---|
| cli.py | User input, command-line arguments and top-level interaction. |
| resolver.py | Converts a ticker or security identifier into a resolved instrument. |
| context.py | Builds a consistent AnalysisContext including time zone, interval, session and as-of information. |
| market_data.py | Market-data providers, caching, loading and validation. |
| features.py | Candle geometry, moving averages, volatility, rolling levels and volume-related features. |
| pattern_detector.py | Pattern registry, detector execution and trend-related pattern logic. |
| analysis.py | Orchestrates the full analysis pipeline. |
| scoring.py | Computes pattern, trend, volume, net score, confidence, market state and overall bias. |
| evaluation.py | Performs historical evaluation and forward-outcome measurement. |
| formatters.py | Text and JSON output formatting. |
| domain.py | Domain objects and dataclasses used throughout the system. |
| session_utils.py | Regular / premarket / after-hours session utilities. |
| datetime_utils.py | Interval parsing, time-zone conversion and candle-completion utilities. |
| config.py | Central configuration values and thresholds. |

### 4.3 End-to-End Processing Pipeline

1. Accept ticker/security identifier and analysis parameters.
2. Resolve the instrument and identify exchange metadata.
3. Create an AnalysisContext with interval, period, session mode, exchange time zone and display time zone.
4. Load OHLCV data from the selected provider.
5. Validate and, when allowed, clean the dataset.
6. Remove the currently forming candle.
7. Compute engineered features.
8. Calculate broad and local trends.
9. Execute the registered pattern detectors.
10. Normalize events, resolve conflicts and prevent structural duplicates.
11. Update pattern lifecycle and confirmation state.
12. Compute pattern, volume, trend and net signal scores.
13. Compute Rule Confidence and Market State.
14. Generate text/JSON output or pass the historical event to the evaluation layer.

---

## 5. Pattern-Detection Algorithms

### 5.1 Common Candle Geometry

```
Body = |Close - Open|
Range = High - Low
UpperWick = High - max(Open, Close)
LowerWick = min(Open, Close) - Low
```

The feature layer also derives normalized body and wick ratios, bar return, close location within the range, moving averages, rolling highs/lows, volume strength, range strength and session-relative variables. These normalized features allow pattern thresholds to remain more comparable across instruments with different price levels.

### 5.2 Bullish and Bearish Engulfing

A Bullish Engulfing event requires two candles in the same trading session. The first candle is bearish and the second is bullish. The second body must engulf the first body, subject to the project tolerance rules. At least one candle should also be significant with respect to range or volume so that very small random candles do not receive the same importance as meaningful displacement.

* Second open <= previous close.
* Second close >= previous open.
* Second candle is bullish and previous candle is bearish.
* At least one of the two candles satisfies significance criteria.
* Default base score: 15.

Bearish Engulfing uses the mirrored conditions and the same default base score.

> **Note (multi-timeframe continuity):** "Same trading session" is enforced differently depending on granularity. For intraday bars, it means the same exchange trading day and session segment (regular/premarket/afterhours). For daily-or-larger bars, where a single candle already represents a full session by construction, continuity is instead validated by checking that consecutive bars are not separated by an abnormally large data gap (e.g. an extended trading halt), so that a legitimate multi-day pattern spanning a weekend or holiday is not rejected. This distinction was fixed in the current codebase after testing showed multi-bar patterns were previously undetectable on daily bars; see Section 9.1.

### 5.3 Hammer / Bullish Pin Bar

```
LowerWickRatio >= 0.55
BodyRatio <= 0.35
```

Geometry alone is not sufficient. A classical Hammer interpretation requires relevant downside context, while a long-lower-wick candle near support may enter a confirmation lifecycle as a lower-wick rejection even when the broader decline is not strong enough for a classical Hammer label. If neither downside context nor nearby support exists, the candle should not receive a strong bullish interpretation merely because it has a long lower wick.

Default base score: 10.

### 5.4 Shooting Star / Upper-Wick Rejection

```
UpperWickRatio >= 0.55
BodyRatio <= 0.35
```

A classical Shooting Star requires an appropriate preceding advance, meaningful displacement and positioning near resistance or a recent swing high. A similar candle without an uptrend may be classified as an upper-wick or resistance rejection rather than a full Shooting Star. This separation reduces false semantic labeling based only on candle shape.

### 5.5 Doji

```
BodyRatio <= 0.08
```

A Doji primarily represents indecision and therefore has a neutral directional bias by default. Its value comes mainly from the context in which it appears, such as after an extended move or near an important level.

### 5.6 Inside Bar and Inside-Bar Failure

```
High_current < High_mother
Low_current > Low_mother
```

The Inside Bar itself is treated as neutral consolidation. A directional signal may arise from a later failure or false break. For example, a bearish inside-bar failure may temporarily trade above the mother-bar high and then close back inside the mother range with bearish characteristics. Default base score for the failure pattern: 11.

### 5.7 Breakout and Breakdown

```
Close_t > RollingHigh_20
Close_(t-1) <= PreviousReferenceHigh
```

A breakout is recorded as a crossing event rather than as a persistent state. This avoids emitting the same breakout repeatedly on every candle that remains above the level. Volume confirmation is required by the detector configuration, and a cooldown prevents immediate duplicate detections. Default base score: 18; strong breakout: 26; default cooldown: 3 bars. Breakdown uses the mirrored logic with a rolling low.

### 5.8 Morning Star and Evening Star

A Morning Star consists of a bearish impulse, a small-bodied star candle and a bullish recovery candle that closes sufficiently far into the first candle, using an intraday-appropriate gap tolerance. An Evening Star uses the mirrored construction. Default base score: 16.

### 5.9 Double Top

The Double Top detector uses confirmed swing highs rather than simply comparing arbitrary local candles. Two peaks must be separated by an allowed number of bars and be sufficiently close in price. The default peak-similarity tolerance is approximately 0.6%. A meaningful valley between the peaks defines the neckline. The structure remains tentative until price closes below the neckline, adjusted by tolerance. A decisive move above the peaks before confirmation invalidates the structure.

```
|Peak1 - Peak2| / max(Peak1, Peak2) <= 0.006
```

Default base score: 20.

### 5.10 Double Bottom

The Double Bottom is the symmetric bullish structure. Two comparable swing lows and an intervening rally define the potential pattern. Confirmation occurs only after a close above the neckline, while a breakdown below the lows can invalidate the setup. Default base score: 20.

### 5.11 Pattern Timing

Every PatternEvent should distinguish at least three timestamps: pattern_start_at is the beginning of the visual structure; pattern_end_at is the final candle that belongs to the formation; detected_at is the earliest point at which the system had enough information to declare the event. For structures that rely on confirmed pivots or later confirmation, detected_at may be later than the visually obvious peak or trough. This distinction is essential for honest backtesting.

### 5.12 Pattern Lifecycle

*Figure 2. Simplified pattern lifecycle.*

The lifecycle separates visual formation from actionable confirmation. A rejection candle may first become Tentative and only later become Confirmed when price behaves consistently with the directional hypothesis. It may also become Failed or Expired. This prevents the system from assigning full weight to every visually plausible shape immediately at candle close.

---

## 6. Trend, Confidence and Signal-Score Computation

### 6.1 Trend Components

The trend engine does not rely on a single moving average. Each horizon combines several independent structural components, including regression slope, moving-average structure, directional persistence, swing structure and recent structural breaks.

```
TrendScore = SlopeScore + MAScore + PersistenceScore + SwingScore + BreakScore
TrendScore is clipped to the interval [-100, 100]
```

A typical classification threshold is Uptrend at +18 or above, Downtrend at -18 or below, and Neutral between the two thresholds.

### 6.2 Multi-Horizon Trend

When sufficient history is available, the project combines short-, medium- and long-horizon trend snapshots. The short horizon receives the largest weight because the system is intended to respond to recent market structure without discarding longer context.

```
CompositeTrend = 0.50 * Short + 0.35 * Medium + 0.15 * Long
```

### 6.3 Local Trend

Broad trend can remain bullish even when the latest portion of the session has begun to decline sharply. For this reason the project also computes Local Trend over a more recent window. It uses local structure, cumulative return, ATR-normalized displacement and the price position within the window range. The local trend therefore provides an explicit representation of short-term intraday change without overwriting the broader market structure.

```
CumulativeReturnATR = (Close - WindowOpen) / ATRScale
```

### 6.4 Latest Candle Direction

Latest Candle Direction is intentionally separate from trend. It combines body direction, body ratio, close location and wick asymmetry, and can be classified as Strong Bullish, Bullish, Neutral, Bearish or Strong Bearish. A single bullish candle must never be interpreted as an uptrend by itself.

```
CandleDirectionScore ≈ 0.5*BodyDirection + 0.3*ClosePosition + 0.2*WickAsymmetry
```

### 6.5 Pattern Score and Recency

```
PatternContribution = Direction * BaseScore * RecencyWeight * Multipliers
RecencyWeight = 0.85^(CandlesAgo)
```

The exact relevance horizon depends on the pattern family. Reversal signals may decay faster than structural signals. Strong signals may receive a multiplier of approximately 1.15, while volume-confirmed events may receive an additional volume contribution. The system should ensure that the same volume surge is not counted multiple times for several labels generated from one market event.

```
BullishScore = sum(positive pattern contributions)
BearishScore = sum(|negative pattern contributions|)
PatternScore = BullishScore - BearishScore
```

### 6.6 Trend and Net Signal Score

At the final scoring layer, trend contributes a compact directional term, for example +12 for Uptrend, -12 for Downtrend and 0 for Neutral. This score is deliberately smaller than the total possible contribution of multiple confirmed patterns so that the pattern layer cannot be completely overridden by a broad trend label.

```
NetSignalScore = PatternScore + VolumeScore + TrendSignalScore
```

### 6.7 Evidence Conflict and Overall Bias

```
ConflictRatio = min(BullishScore, BearishScore) / max(BullishScore, BearishScore)
```

When both bullish and bearish evidence are strong, the system may return Neutral even when the arithmetic net score is not exactly zero. A default conflict threshold around 0.65 can be used to identify such conditions. The current design also avoids issuing a directional Overall Bias based solely on trend when no meaningful directional pattern evidence is confirmed.

### 6.8 Rule Confidence

Rule Confidence is an internal measure of the quality, agreement and independence of the rule-based evidence. It can incorporate bonuses for independent confirmed patterns, signal recency, strength, volume confirmation and trend alignment, as well as penalties for conflicting evidence, data-quality warnings, duplicates, family concentration and signal age.

```
Confidence = Base + Agreement + Confirmation + Recency + Strength + Volume + TrendAlignment - Penalties
```

The result is bounded, for example between 5 and 100. It must not be described as a probability of profit or as forecast accuracy. Only historical calibration can justify a later mapping from confidence buckets to empirical success rates.

### 6.9 Market State

Market State provides a richer summary than a simple Buy/Sell label. Example states include Neutral, Trend Only, Conflicted, Breakout Attempt, Breakdown Attempt, Bullish Continuation, Bearish Continuation, Bullish Setup, Bearish Setup, Reversal Watch, Bullish Trend with Bearish Reversal Attempt and Bearish Trend with Bullish Reversal Attempt.

---

## 7. Data Sources and Data-Quality Handling

### 7.1 Online Market Data

The current online provider is based on yfinance. It is convenient for academic work and prototyping because it provides OHLCV data for a large range of securities and supports multiple intervals. However, it is not an institutional real-time feed, metadata may occasionally be incomplete, and intraday historical depth depends on the requested interval. These limitations must be acknowledged in both the specification and final evaluation.

### 7.2 Offline File Provider

The system also supports CSV and Parquet data. Offline files are important for reproducibility, unit tests, backtesting, experiments with alternate providers and analysis without an internet connection.

### 7.3 Required Schema

| Field | Purpose |
|---|---|
| Datetime | Timestamp identifying the candle. |
| Open | Opening price. |
| High | Highest price within the candle. |
| Low | Lowest price within the candle. |
| Close | Closing price. |
| Volume | Traded volume or provider-specific volume measure. |

### 7.4 Integrity Rules

```
High >= Open, High >= Close, Low <= Open, Low <= Close, High >= Low
```

The validation layer checks missing values, duplicate timestamps, non-sorted data, non-positive prices, negative volume, impossible OHLC relationships, unexpected interval gaps and time-zone/session consistency. Strict mode may reject invalid data immediately; non-strict mode may clean selected problems while documenting every action in a data-quality report.

### 7.5 Time Zones

Exchange Timezone and Display Timezone are separate. Detection should be computed using the exchange-local interpretation of the trading session, while presentation may convert timestamps to the user-facing time zone. This prevents errors in which a signal detected at 15:45 New York time is incorrectly presented as 15:45 in Jerusalem.

### 7.6 Trading Sessions

The system distinguishes premarket, regular and after-hours sessions. The default mode is regular trading. Multi-candle patterns should not automatically join candles from unrelated session segments because price behavior and liquidity differ substantially between them.

### 7.7 Completed Candles Only

An incomplete candle can change its high, low, close and volume before the interval ends. Therefore a 15-minute candle that begins at 10:00 is not eligible for final pattern analysis before 10:15. This rule is one of the most important safeguards against accidental look-ahead and unstable signals.

### 7.8 Time-of-Day Volume Baseline

Intraday volume follows a strong time-of-day pattern, usually with elevated activity near the open and close. Comparing a 09:45 candle to a simple all-day rolling average can therefore be misleading. The system attempts to compare volume with candles from similar times on previous sessions. When insufficient history exists, it can fall back to a rolling 20-bar baseline and report that fallback.

```
VolumeStrength = CurrentVolume / VolumeBaseline
```

---

## 8. CLI and Future User Interface

### 8.1 Current CLI

The current primary interface is command-line based. The user can run the project interactively or provide a ticker directly. The CLI is appropriate for development, testing, reproducibility and automated evaluation.

```
python3 main.py
Enter a ticker or Israeli security number: <input>
Choose a timeframe: <1-7>

python3 main.py AAPL --timeframe 1_DAY --no-interactive
python3 -m stock_pattern_model analyze AAPL --timeframe 1_DAY --no-interactive
```

> **Note:** whenever `--timeframe`, `--period` and `--interval` are all omitted, the CLI defaults to `interactive=True` and always prompts for a timeframe choice before producing a report (this is the behavior demonstrated in every example run in this document's companion sessions). A one-shot, prompt-free run requires an explicit `--timeframe` (or `--period`/`--interval`) together with `--no-interactive`.

### 8.2 Important CLI Parameters

```
--ticker
--interval
--period
--timeframe
--lookback-bars
--top
--all-patterns
--pattern-history
--history-limit
--display-timezone
--session-mode
--format text/json
--output
--mapping-file
--data-file
--exchange-timezone
--cache-dir
--cache-ttl
--no-cache
--strict-data
--as-of
--interactive / --no-interactive
--verbose
```

### 8.3 Proposed Web Interface

A future web dashboard should keep the analytical logic on the server side while presenting the results visually. The most valuable visualization would be an interactive candlestick chart with explicit markers for pattern start, pattern completion, detected_at, confirmation, invalidation, support/resistance and neckline levels.

| UI Area | Recommended content |
|---|---|
| Instrument controls | Ticker/security number, interval, period, session mode, as-of time. |
| Market summary | Latest price, broad trend, local trend, latest candle direction, market state, overall bias, confidence. |
| Pattern panel | Current active patterns with status, detection time, strength and invalidation. |
| Interactive chart | Candles, volume, trend overlays and pattern markers. |
| History panel | Session pattern history and lifecycle changes. |
| Evaluation panel | Historical performance by pattern, horizon, confidence bucket and market regime. |

---

## 9. Testing, Backtesting and Accuracy Metrics

### 9.1 Testing Strategy

The project contains a broad automated test suite. The repository structure includes tests for CLI behavior, market-data validation, pattern registry behavior, scoring, lifecycle logic, time zones, trend calculations, rejection-pattern context, structural deduplication, pattern clustering, historical evaluation, and detector-level multi-bar pattern detection. As of the current codebase snapshot, static inspection identified 281 passing test functions across 18 test files; this count represents test definitions and should not be interpreted as proof that every test passes in every environment without executing the full dependency stack.

This count also reflects a concrete, recent example of why the look-ahead/regression testing described in Section 9.2 matters in practice: a session-continuity check (`_same_pattern_session`) originally keyed pattern continuity on the exchange calendar date. Because every daily-or-larger bar sits on its own calendar date by construction, this silently prevented every multi-bar pattern (Morning Star, Evening Star, Engulfing, Inside Bar, Breakout/Breakdown) from ever being detected on daily, weekly, or longer timeframes — with no test exercising real OHLC data against a detector on a daily interval to catch it. The fix made the continuity check interval-aware (exact session-key matching for intraday bars; an abnormally-large-gap check for daily-or-larger bars, so a pattern can still legitimately span a weekend or holiday) and added a dedicated `tests/test_multi_bar_pattern_detection.py` covering all 14 registered detectors, including boundary/near-miss cases and the gap-tolerance logic itself.

### 9.2 Required Unit-Test Categories

| Test category | Purpose |
|---|---|
| Positive case | Verify a canonical example is detected. |
| Negative case | Verify a visually similar but invalid structure is rejected. |
| Boundary case | Test exact threshold values such as BodyRatio = 0.08. |
| Session boundary | Prevent multi-candle patterns from crossing unrelated sessions. |
| Incomplete candle | Verify the forming candle is excluded. |
| Look-ahead test | Verify that adding future candles cannot alter past detection time retroactively. |
| Lifecycle test | Verify confirmation, failure and expiration occur only when the relevant future event actually becomes available. |
| Deduplication test | Verify correlated labels do not multiply evidence artificially. |
| Interval-continuity test | Verify a given detector reaches its multi-bar branch identically on intraday and on daily-or-larger intervals, and that a normal calendar gap (e.g. a weekend) does not block detection while an abnormal one (e.g. an extended halt) correctly does. |

### 9.3 Backtesting Principle

Backtesting must be separated from live detection. At historical time t, the analysis layer may access only data with timestamps up to t. The future segment may be exposed only after the signal has been created and stored. This separation is necessary to produce a valid estimate of performance.

```
Detection uses Data <= t; evaluation may inspect Data > t only after the signal exists
```

### 9.4 Forward Return

```
R_h = P_h / P_0 - 1
```

For bullish signals, positive forward returns are favorable. For bearish signals, the direction is inverted when computing directional success so that a downward move can be treated as a positive outcome in signal space.

### 9.5 Direction Accuracy

```
DirectionAccuracy = CorrectDirectionalSignals / TotalEvaluatedSignals
```

Direction Accuracy should always be reported together with the number of evaluated signals. A value of 80% over five examples is much weaker evidence than 61% over several thousand examples.

### 9.6 MFE and MAE

```
Bullish MFE = (max(FutureHigh) - Entry) / Entry
Bullish MAE = (min(FutureLow) - Entry) / Entry
```

Maximum Favorable Excursion measures the best movement in the signal direction before the evaluation horizon ends. Maximum Adverse Excursion measures the worst movement against the signal. These metrics are useful for later stop-loss, target and risk-management research.

### 9.7 Target / Stop Evaluation

A simple evaluation configuration may use a +1% target and -0.5% stop for a bullish signal, with mirrored values for bearish signals. The evaluator should record Target First, Stop First, Neither, or Both on the Same Candle. When both levels occur inside one OHLC candle, the true order is not knowable without higher-resolution data and should be marked as ambiguous rather than guessed.

### 9.8 Win Rate, False Positive Rate and Expectancy

```
WinRate = PositiveSimulatedTrades / EvaluatedTrades
FalsePositiveRate = IncorrectDirectionalSignals / EvaluatedSignals
Expectancy = (1/N) * Σ TradeReturn_i
```

Expectancy is often more informative than raw accuracy. A system can be correct less than half of the time yet still have positive expectancy if average gains are sufficiently larger than average losses. Conversely, a high win rate can still be economically poor if losses are much larger than gains.

### 9.9 Performance Segmentation

Evaluation should not collapse every signal into a single accuracy figure. Results should be segmented by pattern type, market regime, trend alignment, time of day, interval, symbol characteristics and confidence bucket. Confidence buckets are especially important because they reveal whether higher Rule Confidence is actually associated with better empirical results.

| Segment | Example |
|---|---|
| Pattern | Breakout vs. Shooting Star vs. Bullish Engulfing |
| Market context | Trend-aligned continuation vs. counter-trend reversal |
| Time of day | Opening hour, mid-session, final hour |
| Confidence bucket | 0-24, 25-49, 50-74, 75-100 |
| Volatility regime | Low, normal, high volatility |
| Liquidity | High-volume large cap vs. thinner instrument |

### 9.10 Out-of-Sample and Walk-Forward Validation

Thresholds should be developed on one period and evaluated on a chronologically later period that was not used during rule design. For time series, walk-forward testing is preferable to random shuffling because random shuffling destroys temporal order and can leak future market regimes into the development sample.

*Figure 3. Walk-forward evaluation concept.*

### 9.11 Data Snooping and Frozen Rules

Repeatedly inspecting a chart where the system failed and then changing thresholds to fix that specific example can gradually overfit the system to a small set of memorable securities. Once rule development is complete, the project should freeze the configuration and perform a final evaluation on unseen symbols and unseen time periods. This addresses the data-snooping concerns highlighted in the academic literature.

### 9.12 Recommended Evaluation Dashboard

| Metric | Interpretation |
|---|---|
| Direction Accuracy | Fraction of signals that moved in the predicted direction. |
| Precision | Quality of selected directional signals under a chosen outcome definition. |
| False Positive Rate | Fraction of evaluated signals classified as incorrect. |
| Average Forward Return | Mean directional return after the signal. |
| Median Forward Return | Robust central tendency less sensitive to outliers. |
| MFE | Best favorable excursion after the signal. |
| MAE | Worst adverse excursion after the signal. |
| Target-First Rate | Fraction where target is reached before stop. |
| Stop-First Rate | Fraction where stop is reached before target. |
| Expectancy | Average simulated return per evaluated signal. |
| Signal Count | Sample size used to interpret every other metric. |

---

## 10. Limitations, Risks and Future Development

### 10.1 Rule-Based Thresholds

Thresholds such as a 0.55 wick ratio, a 0.35 body ratio, a 20-bar breakout window or a 0.6% Double Top tolerance are engineering choices rather than universal market laws. They may behave differently across equities, futures, cryptocurrencies, exchanges, volatility regimes and liquidity levels.

### 10.2 Market-Regime Dependence

A pattern that performs well in a strong bull market may perform poorly during high-volatility selloffs, low-liquidity periods or range-bound regimes. Historical results must therefore be segmented by regime rather than generalized without evidence.

### 10.3 Data-Provider Limitations

yfinance is suitable for prototyping and academic work but should not be treated as an institutional execution-grade source. Intraday depth, metadata, corporate-action handling and occasional provider anomalies can affect results.

### 10.4 Transaction Costs and Execution

Signal-level historical evaluation does not automatically imply real-world profitability. A trading simulation must account for commission, bid-ask spread, slippage, latency, order type, partial fills and position sizing before profitability claims can be made.

### 10.5 Corporate Actions

Splits, dividends and symbol changes can distort historical price series if not handled consistently. Adjusted and unadjusted data must not be mixed inadvertently.

### 10.6 Confidence Is Not Calibration

A Rule Confidence value of 90 does not mean that 90% of comparable signals historically succeeded. Only a large out-of-sample dataset can establish an empirical mapping from confidence values to observed success rates.

### 10.7 Pattern Correlation

Several labels can describe the same underlying event, such as Hammer and Bullish Pin Bar on the same candle. The current deduplication design reduces this problem, but residual correlation between related families should still be measured during evaluation.

### 10.8 Principal Risks

| Risk | Description | Mitigation |
|---|---|---|
| False positives | Random market noise can resemble known patterns. | Require context, confirmation, significance and empirical evaluation. |
| Overfitting | Thresholds may become specialized to examples used during development. | Freeze rules and use unseen periods/symbols. |
| Look-ahead bias | Future candles may influence historical detection. | Use as-of cutoffs and separate detection from evaluation. |
| Survivorship bias | Testing only currently listed successful stocks may inflate results. | Use broader historical universes where possible. |
| Selection bias | Manually choosing interesting examples distorts conclusions. | Evaluate a predefined symbol universe and time window. |
| Multiple testing | Trying many rules increases chance findings. | Limit degrees of freedom and use robust statistical validation. |
| Provider error | Bad timestamps or prices can create false patterns. | Strict validation, data-quality reporting and alternate-provider checks. |
| Interval-scoped logic gaps | A rule (e.g. a continuity or session check) validated only on one bar granularity may silently fail, partially or completely, on another. | Test every detector against both intraday and daily-or-larger fixtures explicitly, as in Section 9.2's interval-continuity test category. |

### 10.9 Future Development Roadmap

**Phase 1 - Extended Backtesting**
* Transaction costs and spread.
* Slippage assumptions.
* Equity curve.
* Maximum drawdown.
* Sharpe and Sortino ratios.
* Profit factor.

**Phase 2 - Multi-Timeframe Analysis**

Combine lower-timeframe entry signals with higher-timeframe context, for example a 15-minute Bullish Engulfing inside a one-hour uptrend near a daily support area.

**Phase 3 - Empirical Confidence Calibration**

After accumulating enough unseen historical signals, compare Rule Confidence buckets with observed outcome rates. If the relationship is stable and monotonic, a separate Empirical Probability layer could be developed without redefining the original Rule Confidence metric.

| Rule Confidence bucket | Illustrative historical success rate |
|---|---|
| 0-24 | 48% |
| 25-49 | 53% |
| 50-74 | 61% |
| 75-100 | 67% |

The percentages above are illustrative only; they must be estimated from real out-of-sample results before use.

**Phase 4 - Machine Learning**

The existing rule-based system can provide interpretable features for a later classifier. Potential inputs include trend score, pattern identifiers, pattern strength, volume strength, local trend, distance from support/resistance, time of day and volatility. The ML model should be evaluated against the rule-based baseline so that its added value is measurable rather than assumed.

**Phase 5 - Web Dashboard and Alerts**

A dashboard can visualize candles, pattern markers, lifecycle changes, watchlists and historical results. Alerts may notify the user when a confirmed event occurs, while leaving the final trading decision to the user.

**Phase 6 - Paper Trading**

A paper-trading simulation can open virtual positions after eligible signals, apply target/stop and position-sizing rules, maintain a transaction log, and calculate an equity curve. This step should precede any integration with a real broker.

---

## 11. Diagrams, Tables and Example Outputs

### 11.1 Example Data Flow

```
User enters: MU
   |
Resolve instrument
   |
Fetch OHLCV
   |
Validate timestamps / prices / volume
   |
Filter incomplete candle
   |
Feature engineering
   |
Broad + local trend
   |
Pattern registry
   |
Conflict / duplicate resolution
   |
Lifecycle
   |
Scoring + confidence
   |
Market state + overall bias
   |
Text / JSON output
```

### 11.2 Example Text Output

> **Note:** the block below is a *condensed* illustration. A real run of `python3 main.py AAPL --timeframe 1_DAY --no-interactive` prints roughly 25+ fields per individual pattern (Geometry Status, Context Status, Directional Confirmation, Follow-Through, full Pattern Candle OHLC, Invalidation Condition, lifecycle state, recency weight, score-eligibility, exclusion reasons, and more), not the single summary line shown here. This example keeps only the fields most relevant for illustrating the overall report shape.

```
Instrument: AAPL
Input Identifier: aapl
Resolved Symbol: AAPL
Name: Apple Inc.
Exchange: NMS
Currency: USD

Interval: 15m
Analysis Time: 2026-07-28 20:15:00+0300 Asia/Jerusalem
Exchange Timezone: America/New_York
Display Timezone: Asia/Jerusalem
Session Mode: regular

Latest Completed Candle Start: 2026-07-28 20:00:00+0300 Asia/Jerusalem
Latest Completed Candle End:   2026-07-28 20:15:00+0300 Asia/Jerusalem
Latest Close: 215.40

Broad Trend: Uptrend
Broad Trend Score: 37.42
Local Trend: Neutral
Local Trend Score: 7.31
Latest Candle Direction: Bullish

Market State: Bullish Continuation
Overall Bias: Bullish

Bullish Pattern Score: 20.84
Bearish Pattern Score: 4.31
Pattern Score: 16.53
Volume Score: 2.12
Trend Signal Score: 12.00
Net Signal Score: 30.65
Rule Confidence: 67.8

Current Relevant Patterns (summarized; see note above for full field set):
- 20-Bar Breakout | confirmed | detected 2026-07-28 13:45:00-0400 America/New_York | volume confirmed
- Bullish Engulfing | confirmed | detected 2026-07-28 14:15:00-0400 America/New_York

Conflicting Evidence:
- Older Shooting Star evidence has decayed and no longer dominates the current score.

Final Assessment:
The broad trend and confirmed bullish pattern evidence are aligned. No sufficiently strong bearish conflict currently neutralizes the signal.

Disclaimer: This output is a technical-analysis research result and is not financial advice.
All numerical values in this example are illustrative and do not represent a specific live AAPL analysis.
```

### 11.3 Example JSON Output

```json
{
 "symbol": "AAPL",
 "interval": "15m",
 "trend": "Uptrend",
 "local_trend": "Neutral",
 "latest_candle_direction": "Bullish",
 "market_state": "Bullish Continuation",
 "overall_bias": "Bullish",
 "pattern_score": 16.53,
 "volume_score": 2.12,
 "net_signal_score": 30.65,
 "rule_confidence": 67.8,
 "current_relevant_patterns": [],
 "session_pattern_history": [],
 "data_quality_report": {},
 "structured_explanation": {}
}
```

### 11.4 Primary Use Case

| Field | Description |
|---|---|
| Actor | End user / project evaluator. |
| Preconditions | Python environment and a valid data source are available; input identifier can be resolved. |
| Trigger | User requests analysis for a ticker or security number. |
| Main flow | Resolve instrument -> load/validate data -> filter incomplete candle -> engineer features -> trend analysis -> pattern detection -> deduplication -> lifecycle -> scoring -> explanation/output. |
| Postcondition | The user receives a technical snapshot based only on information available up to the analysis cutoff. |

### 11.5 Project Success Criteria

1. Patterns are detected deterministically and consistently.
2. Each event receives a correct detected_at timestamp.
3. No future candle influences a past detection decision.
4. Pattern geometry is separated from context and confirmation.
5. Incomplete candles never affect final analysis.
6. Duplicate or correlated patterns do not inflate scores artificially.
7. Broad Trend, Local Trend and Latest Candle Direction remain conceptually separate.
8. Data-quality warnings reduce trust appropriately and are visible to the user.
9. Historical evaluation uses future data only after the historical signal has been generated.
10. Performance can be segmented by pattern, context, time of day and confidence.
11. New detectors can be added without redesigning the whole application.
12. A detector's multi-bar continuity logic behaves correctly across every supported bar granularity (intraday through weekly), not only the granularity it happened to be tested against.
13. The final output explains why each important conclusion was reached.

---

## 12. Conclusion

The Stock Market Pattern Analysis System is a modular rule-based technical-analysis platform designed to transform subjective chart interpretation into a reproducible computational process. Its main contribution is not merely recognizing candlestick shapes; it combines validated market data, candle geometry, context-sensitive pattern recognition, multi-horizon trend analysis, local trend, signal lifecycle, conflict resolution, recency decay, volume confirmation and structured scoring.

The design intentionally separates three different concepts that are often confused in technical-analysis software: a pattern can be detected, the current rule evidence can be strong, and yet the historical probability of success may still be unknown. By keeping Rule Confidence separate from Historical Evaluation, the project preserves explainability while creating a path toward empirical calibration.

The most important next step is therefore broad, frozen-rule, out-of-sample and walk-forward evaluation across a large and diverse dataset. Only after this stage should the project make stronger claims about predictive value or introduce machine learning. This approach is consistent with the mixed academic evidence on technical analysis and with the need to control overfitting, data snooping and look-ahead bias.

---

## References

Brock, W., Lakonishok, J., & LeBaron, B. (1992). Simple Technical Trading Rules and the Stochastic Properties of Stock Returns. *The Journal of Finance*, 47(5), 1731-1764. https://doi.org/10.1111/j.1540-6261.1992.tb04681.x

Caginalp, G., & Laurent, H. (1998). The Predictive Power of Price Patterns. *Applied Mathematical Finance*, 5, 181-205. https://doi.org/10.1080/135048698334637

Sullivan, R., Timmermann, A., & White, H. (1999). Data-Snooping, Technical Trading Rule Performance, and the Bootstrap. *The Journal of Finance*, 54(5), 1647-1691. https://doi.org/10.1111/0022-1082.00163

Lo, A. W., Mamaysky, H., & Wang, J. (2000). Foundations of Technical Analysis: Computational Algorithms, Statistical Inference, and Empirical Implementation. *The Journal of Finance*, 55(4), 1705-1765. https://doi.org/10.1111/0022-1082.00265

White, H. (2000). A Reality Check for Data Snooping. *Econometrica*, 68(5), 1097-1126. https://doi.org/10.1111/1468-0262.00152

Marshall, B. R., Young, M. R., & Rose, L. C. (2006). Candlestick Technical Trading Strategies: Can They Create Value for Investors? *Journal of Banking & Finance*, 30(8), 2303-2323. https://doi.org/10.1016/j.jbankfin.2005.08.001

Park, C.-H., & Irwin, S. H. (2007). What Do We Know About the Profitability of Technical Analysis? *Journal of Economic Surveys*, 21(4), 786-826. https://doi.org/10.1111/j.1467-6419.2007.00519.x

Chen, J.-H., & Tsai, Y.-C. (2020). Encoding Candlesticks as Images for Pattern Classification Using Convolutional Neural Networks. *Financial Innovation*, 6, 26. https://doi.org/10.1186/s40854-020-00187-0

Cagliero, L., Fior, J., & Garza, P. (2023). Shortlisting Machine Learning-Based Stock Trading Recommendations Using Candlestick Pattern Recognition. *Expert Systems with Applications*, 216, 119493. https://doi.org/10.1016/j.eswa.2022.119493

Agah, S. A., Yazdian Varjani, A., Zamani Boroujeni, F., & Yazdani, S. (2026). A Hybrid Framework for Algorithmic Trading: Combining Lightweight Neural Network Trend Prediction with Candlestick Pattern Recognition. *Expert Systems with Applications*, 316, 131365.

---

*Extended Project Specification | September 2026*
