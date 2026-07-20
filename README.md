# Stock Pattern Model

## Project Overview

`stock-pattern-model` is an educational, rule-based technical-analysis application for intraday OHLCV data. It focuses on the analysis layer only: loading validated market data, engineering features, detecting chart and candlestick patterns, scoring recent evidence, and producing structured text or JSON output.

This project currently supports:

- Offline analysis from CSV and Parquet files
- Provider-based loading through `YFinanceProvider` and `FileDataProvider`
- Completed-candle filtering with injectable `as_of`
- Exact pattern timestamps in exchange time and display time
- Ticker analysis and Israeli security-number resolution through a CSV mapping file
- A registry-based pattern system with confirmed, tentative, failed, and expired events
- Structured scoring, market-state logic, and human-readable explanations

This project does **not** currently include:

- Brokerage integration
- Live order execution
- Portfolio management
- Historical trading simulation
- Backtesting
- Paper trading
- Stop-loss order simulation
- Take-profit order simulation
- Trade-level profit and loss calculations
- Portfolio equity curves
- Sharpe-ratio calculations
- Guaranteed market predictions
- Statistically calibrated success probabilities

Backtesting and trading simulation were not implemented in this project version.

Supported Python version: `Python 3.9+`

## Installation

```bash
cd stock-pattern-model
python3 -m pip install -r requirements.txt
```

## Package Usage

The package exposes analysis and provider APIs without doing any work at import time:

```python
from stock_pattern_model import analyze_dataframe, analyze_stock
```

Importing `main` or `stock_pattern_model` does not trigger analysis, network calls, or terminal output.

## CLI Usage

Package CLI:

```bash
python3 -m stock_pattern_model --help
python3 -m stock_pattern_model analyze --help
```

Root-level compatibility wrapper:

```bash
python3 main.py --help
python3 main.py
python3 main.py AAPL
python3 main.py TEVA.TA
python3 main.py analyze AAPL
```

`main.py` is only a thin compatibility wrapper around the packaged CLI. It does not implement a separate analysis flow.

## Analysis Usage

Interactive input:

```bash
python3 -m stock_pattern_model analyze
python3 main.py
```

Both commands prompt:

```text
Enter a ticker or Israeli security number:
```

Direct ticker analysis:

```bash
python3 -m stock_pattern_model analyze AAPL
python3 main.py AAPL
```

Exchange-qualified tickers are preserved:

```bash
python3 -m stock_pattern_model analyze TEVA.TA
python3 -m stock_pattern_model analyze BRK-B
```

Israeli security numbers require a real mapping file:

```bash
python3 -m stock_pattern_model analyze 1084128 --mapping-file data/tase_securities.csv
```

Example mapping schema:

```csv
security_number,yahoo_symbol,name,exchange,currency,timezone
1084128,TEVA.TA,Teva Pharmaceutical Industries,TASE,ILS,Asia/Jerusalem
```

`data/tase_securities.example.csv` is only an example schema and must not be treated as authoritative real-world data.

JSON output:

```bash
python3 -m stock_pattern_model analyze AAPL --format json
```

Text output:

```bash
python3 -m stock_pattern_model analyze AAPL --format text
```

Write output to a file:

```bash
python3 -m stock_pattern_model analyze AAPL --format json --output outputs/aapl_analysis.json
```

Show all patterns:

```bash
python3 -m stock_pattern_model analyze AAPL --all-patterns
```

Limit top patterns:

```bash
python3 -m stock_pattern_model analyze AAPL --top 5
```

Choose a display timezone:

```bash
python3 -m stock_pattern_model analyze AAPL --display-timezone Asia/Jerusalem
```

Provide an explicit `as_of`:

```bash
python3 -m stock_pattern_model analyze AAPL --as-of 2026-07-10T16:46:00-04:00
```

Empty interactive input is rejected with a nonzero exit code. The CLI never falls back to a hardcoded default symbol or batch list.

## Market-Data Layer

### Provider Interfaces

- `MarketDataProvider`: abstract interface used by the analysis layer
- `YFinanceProvider`: live-provider implementation with validation, retry, metadata capture, and optional Parquet cache
- `FileDataProvider`: offline provider for CSV and Parquet analysis

### File Inputs

Supported file formats:

- CSV
- Parquet

Required canonical columns:

- `Datetime`
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`

Common case and naming variations are normalized when unambiguous, such as `timestamp`, `open`, and `volume`.

Offline analysis:

```bash
python3 -m stock_pattern_model analyze TEST \
  --data-file tests/fixtures/sample_ohlcv.csv \
  --exchange-timezone America/New_York
```

### Cache Behavior

`YFinanceProvider` supports:

- Local Parquet cache
- Configurable cache directory
- Configurable cache expiration through `--cache-ttl`
- Cache bypass through `--no-cache`

Cached data is not committed to the repository and should stay outside versioned fixtures.

### Provider Errors

Provider failures preserve the original cause. Normal CLI usage prints concise user-facing errors. `--verbose` enables debug logging and more useful failure context.

### Data Validation

The validator checks:

- Missing columns
- Missing OHLCV values
- Duplicate timestamps
- Unsorted timestamps
- Negative prices
- Zero prices
- Negative volume
- Invalid OHLC relationships
- Irregular same-session interval gaps
- Timezone-naive timestamps without an explicit exchange timezone

Strict mode is the default. In strict mode, invalid OHLC rows are rejected instead of silently repaired.

Duplicate timestamps are reported.

Missing bars generate warnings.

Timezone-naive file data is never silently interpreted as UTC.

Daylight-saving transitions are handled during localization with explicit timezone rules.

Cleaning behavior, when enabled through non-strict validation, is explicit and recorded in `DataQualityReport.cleaning_actions`.

## Timestamp Semantics

- `bar_start_at`: the candle start timestamp
- `bar_end_at`: the candle close timestamp
- `pattern_start_at`: the first candle that belongs to the pattern
- `pattern_end_at`: the final candle that completes the pattern structure
- `detected_at`: the earliest time the pattern became knowable

A candle pattern is only detectable after its final candle closes.

Multi-candle patterns use:

- The first participating candle for `pattern_start_at`
- The final participating candle close for `pattern_end_at`
- The earliest knowable completion time for `detected_at`

Confirmed swing-pivot patterns such as Double Top and Double Bottom may have a `detected_at` later than the original peak or low because pivot confirmation requires future bars.

Incomplete candles are filtered out before features, patterns, and scoring are calculated.

`as_of` is injectable so tests and offline analysis can be deterministic.

Exchange timezone is preserved internally. Display conversion is applied only to user-facing fields.

## Pattern System

### Registry Architecture

Pattern detection is registry-based. Each detector:

- Has a unique `pattern_id`
- Declares a pattern family
- Declares minimum required history
- Returns zero or more `PatternEvent` objects
- Uses only information available by `detected_at`
- Does not mutate caller-owned data
- Does not score final analysis output
- Does not make network calls
- Is independently testable

### Implemented Patterns

Implemented and tested patterns:

- Bullish Engulfing
- Bearish Engulfing
- Bullish Pin Bar
- Shooting Star
- Inside Bar
- Inside Bar Failure
- Breakout
- Breakdown
- Doji
- Morning Star
- Evening Star
- Double Top
- Double Bottom

### Pattern Metadata

Each serialized pattern event includes structured metadata such as:

- `event_id`
- `setup_id`
- `evidence_group`
- `event_state`
- `pattern_id`
- `pattern_name`
- `pattern_family`
- `bias`
- `status`
- `pattern_start_at`
- `pattern_end_at`
- `detected_at`
- `relevant_prices`
- `relevant_indices`
- `detection_reason`
- `signal_strength`

### Event Status

- `confirmed`: fully confirmed and score-eligible by default
- `tentative`: visible in output, but not score-eligible by default
- `failed`: a setup invalidated before confirmation
- `expired`: a setup or older event that is no longer treated like fresh active evidence

### Key Detection Rules

- Breakout and breakdown use crossing-event semantics, not persistent repeated signals
- Strong breakout is represented as one stronger event, not both strong and regular events
- Inside Bar Failure uses explicit mother-bar confirmation
- Double Top requires confirmed swing highs, a meaningful valley, and neckline confirmation
- Double Bottom requires confirmed swing lows, an intervening rally, and neckline confirmation
- Pattern detection is no-look-ahead with respect to historical detection correctness

Configurable tolerances cover:

- Doji body tolerance
- Star gap tolerance
- Pivot confirmation strength
- Double-pattern price tolerance
- Minimum and maximum pattern separation

## Features And Volume

Continuous features:

- `Continuous_MA_20`
- `Continuous_MA_50`

Session-reset features:

- `Session_MA_20`
- `Session_High`
- `Session_Low`
- `Session_Open`

Continuous moving averages may span multiple sessions and are not described as session-only indicators.

Volume handling includes:

- Time-of-day volume baseline when enough prior same-time candles exist
- Rolling 20-bar fallback when time-of-day history is insufficient

The chosen volume baseline is surfaced in pattern explanations and result metadata.

## Scoring And Output

The analysis separates:

- `trend`
- `market_state`
- `overall_bias`

Meaning:

- `trend`: moving-average structure only
- `market_state`: the current technical context
- `overall_bias`: final directional leaning after deduplicated evidence is scored

Scoring fields:

- `trend_score`
- `pattern_score`
- `volume_score`
- `bullish_score`
- `bearish_score`
- `net_signal_score`
- `rule_confidence`

Evidence deduplication uses:

- `event_id`
- `setup_id`
- `evidence_group`

Only the strongest event in an overlapping evidence group gets full scoring weight. Related overlapping events may still appear in output historically, but as suppressed evidence.

Recency and expiration:

- Newer events receive more weight
- Old events decay and eventually stop influencing directional scores
- Expired patterns may still be displayed historically but should not be treated like fresh evidence

Conflict handling:

- Bullish and bearish evidence can coexist
- Conflicts reduce confidence
- Conflicts can neutralize final bias even when one side is slightly stronger

Data-quality warnings reduce confidence when relevant.

Tentative patterns do not affect default directional signals.

Neutral patterns do not contribute bullish or bearish directional scores.

Structured explanation output includes:

- `summary`
- `bullish_evidence`
- `bearish_evidence`
- `conflicts`
- `data_warnings`
- `reason_for_bias`
- `reason_for_confidence`

Important:

```text
Rule confidence is an uncalibrated rule-strength score.
It is not a statistical probability and is not a prediction accuracy percentage.
```

Also note:

- A bullish trend can coexist with a neutral overall bias
- High rule confidence does not guarantee future market movement
- Conflicting evidence can reduce the final bias
- Neutral output should not be interpreted as a strong recommendation

## Text And JSON Output

Text output includes:

- Instrument and resolved symbol
- Input identifier
- Security number, when relevant
- Name, when available
- Exchange and currency
- Interval
- Analysis time
- Exchange timezone and display timezone
- Latest completed candle start and end
- Latest close
- Trend, market state, overall bias
- Trend score, pattern score, volume score
- Bullish score, bearish score, net signal score
- Rule confidence
- Pattern details with exact detection times
- Warnings and data-quality summary
- Structured explanation

JSON output:

- Uses valid JSON
- Uses ISO-8601 timestamps with timezone offsets
- Includes serialized pattern metadata
- Includes data-quality information
- Includes scoring fields
- Includes explanations
- Avoids raw Python objects such as `Timestamp`, `Enum`, `NaN`, and tuples

## Exit Codes

CLI exit codes are consistent and nonzero for common failure classes:

- `2`: invalid input or invalid general configuration
- `3`: unknown Israeli security number
- `4`: missing mapping file
- `5`: invalid mapping file
- `6`: market-data provider failure
- `7`: data-validation failure
- `8`: output-file failure
- `9`: invalid timezone
- `10`: no completed bars
- `11`: missing data file
- `12`: unexpected internal failure

## Testing And Verification

Run the full test suite:

```bash
pytest
```

Run compilation checks:

```bash
python3 -m compileall stock_pattern_model
python3 -m compileall main.py
```

Offline fixture tests cover:

- Pattern detection
- Timestamp semantics
- Provider and cache behavior
- Instrument resolution
- CLI behavior
- Scoring and explanation consistency

## Known Limitations

- Live `yfinance` availability depends on internet access and external provider behavior
- Provider metadata may be incomplete or unavailable
- Israeli security-number resolution is only as complete as the mapping file you provide
- Data-quality validation catches many structural issues, but rule-based analysis still depends on input quality
- Pattern recognition is deterministic and educational, not exhaustive
- Rule confidence is not statistically calibrated

## Educational And Financial Disclaimer

This repository is an educational technical-analysis application.

It does not provide financial advice, guaranteed predictions, or statistically calibrated success probabilities. Use it to study data handling, timestamp correctness, rule-based pattern detection, and structured analysis output.

## Optional Future Work

Possible future improvements, not implemented in the current version:

- Broader instrument metadata sources
- Additional tested chart patterns
- Packaged project metadata such as a publishable `pyproject.toml`
- Historical simulation or backtesting in a separate, explicitly scoped module
