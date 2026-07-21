from __future__ import annotations

import json
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import stock_pattern_model.analysis as analysis_module
from stock_pattern_model.analysis import analyze_dataframe
from stock_pattern_model.features import add_features
from stock_pattern_model.pattern_detector import _relative_move
from stock_pattern_model.pattern_detector import _rolling_break_levels
from stock_pattern_model.pattern_detector import _trend_horizons
from stock_pattern_model.pattern_detector import _trend_label
from stock_pattern_model.pattern_detector import classify_intraday_trend


EXCHANGE_TZ = ZoneInfo("America/New_York")


def make_benchmark_df(length: int, start: str = "2026-07-10 09:30") -> pd.DataFrame:
    index = np.arange(length, dtype=float)
    datetimes = pd.date_range(start=start, periods=length, freq="15min", tz=EXCHANGE_TZ)
    close = 100.0 + (index * 0.015) + np.sin(index / 7.0) * 1.6 + np.sin(index / 17.0) * 0.9
    open_price = np.empty(length, dtype=float)
    open_price[0] = close[0] - 0.25
    open_price[1:] = close[:-1] + (np.sin(index[1:] / 5.0) * 0.18)
    high = np.maximum(open_price, close) + 0.42 + ((index % 5.0) * 0.03)
    low = np.minimum(open_price, close) - 0.38 - (((index + 2.0) % 5.0) * 0.025)
    volume = 900 + ((index.astype(int) % 17) * 55) + ((index.astype(int) % 41 == 0) * 1800)
    return pd.DataFrame(
        {
            "Datetime": datetimes,
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume.astype(int),
        }
    )


def analysis_as_of(df: pd.DataFrame) -> pd.Timestamp:
    return pd.Timestamp(df.iloc[-1]["Datetime"]) + pd.Timedelta(minutes=16)


def _confirmed_swings_reference(
    window: pd.DataFrame,
    *,
    pivot_left_bars: int,
    pivot_right_bars: int,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    highs: list[tuple[int, float]] = []
    lows: list[tuple[int, float]] = []
    if len(window) < pivot_left_bars + pivot_right_bars + 1:
        return highs, lows

    for idx in range(pivot_left_bars, len(window) - pivot_right_bars):
        high_value = float(window.iloc[idx]["High"])
        low_value = float(window.iloc[idx]["Low"])

        high_slice = window["High"].iloc[idx - pivot_left_bars : idx + pivot_right_bars + 1]
        low_slice = window["Low"].iloc[idx - pivot_left_bars : idx + pivot_right_bars + 1]

        if high_value == float(high_slice.max()) and int((high_slice == high_value).sum()) == 1:
            highs.append((idx, high_value))
        if low_value == float(low_slice.min()) and int((low_slice == low_value).sum()) == 1:
            lows.append((idx, low_value))

    return highs, lows


def _break_structure_score_reference(
    history: pd.DataFrame,
    *,
    breakout_lookback: int,
) -> tuple[float, list[str]]:
    evidence: list[str] = []
    if len(history) < max(breakout_lookback + 1, 3):
        return 0.0, evidence

    recent_history = history.reset_index(drop=True)
    rolling_high, rolling_low = _rolling_break_levels(recent_history, lookback=breakout_lookback)
    previous_reference_high = rolling_high.shift(1).fillna(rolling_high)
    previous_reference_low = rolling_low.shift(1).fillna(rolling_low)
    breakout_score = 0.0
    for idx in range(1, len(recent_history)):
        previous_row = recent_history.iloc[idx - 1]
        row = recent_history.iloc[idx]

        current_prev_high = rolling_high.iloc[idx]
        previous_prev_high = previous_reference_high.iloc[idx]
        current_prev_low = rolling_low.iloc[idx]
        previous_prev_low = previous_reference_low.iloc[idx]
        if str(recent_history.iloc[idx - 1]["Pattern_Session_Key"]) != str(recent_history.iloc[idx]["Pattern_Session_Key"]):
            continue

        if (
            pd.notna(current_prev_high)
            and pd.notna(previous_prev_high)
            and float(row["Close"]) > float(current_prev_high)
            and float(previous_row["Close"]) <= float(previous_prev_high)
        ):
            bars_ago = len(recent_history) - 1 - idx
            breakout_score = max(
                breakout_score,
                11.0
                - (bars_ago * 2.0)
                + (2.0 if bool(row.get("Strong_Volume", False)) else 0.0)
                + (1.5 if bool(row.get("Strong_Range", False)) else 0.0),
            )
        if (
            pd.notna(current_prev_low)
            and pd.notna(previous_prev_low)
            and float(row["Close"]) < float(current_prev_low)
            and float(previous_row["Close"]) >= float(previous_prev_low)
        ):
            bars_ago = len(recent_history) - 1 - idx
            bearish_break_score = max(
                4.0,
                11.0
                - (bars_ago * 2.0)
                + (2.0 if bool(row.get("Strong_Volume", False)) else 0.0)
                + (1.5 if bool(row.get("Strong_Range", False)) else 0.0),
            )
            breakout_score = min(
                breakout_score,
                -bearish_break_score,
            )

    if breakout_score >= 4.0:
        evidence.append(
            f"Price confirmed an upside break above the prior {breakout_lookback}-bar high."
        )
    elif breakout_score <= -4.0:
        evidence.append(
            f"Price confirmed a downside break below the prior {breakout_lookback}-bar low."
        )
    return breakout_score, evidence


def _trend_snapshot_reference(
    history: pd.DataFrame,
    *,
    horizon: int,
    pivot_left_bars: int,
    pivot_right_bars: int,
    breakout_lookback: int,
) -> tuple[float, str, list[str]]:
    window = history.tail(min(len(history), horizon)).reset_index(drop=True)
    if len(window) < 8:
        return 0.0, "Neutral", []

    closes = window["Close"].astype(float).to_numpy()
    highs = window["High"].astype(float).to_numpy()
    lows = window["Low"].astype(float).to_numpy()
    returns = window["Bar_Return"].fillna(0.0).astype(float).to_numpy()
    atr_scale = float(np.nanmean(highs - lows))
    atr_scale = atr_scale if atr_scale > 1e-9 else max(abs(float(closes[-1])) * 0.001, 1e-6)

    regression_x = np.arange(len(closes), dtype=float)
    slope = float(np.polyfit(regression_x, closes, 1)[0])
    slope_score = float(np.clip((slope / atr_scale) * 80.0, -24.0, 24.0))

    fast_period = min(20, max(5, len(window) // 3))
    slow_period = min(50, max(fast_period + 4, len(window)))
    fast_ma = float(np.mean(closes[-fast_period:]))
    slow_ma = float(np.mean(closes[-slow_period:]))
    price = float(closes[-1])

    ma_score = 0.0
    ma_separation = abs(fast_ma - slow_ma) / atr_scale
    if ma_separation >= 0.12:
        if fast_ma > slow_ma:
            ma_score += 10.0
        elif fast_ma < slow_ma:
            ma_score -= 10.0
    if price > fast_ma and price > slow_ma:
        ma_score += 5.0
    elif price < fast_ma and price < slow_ma:
        ma_score -= 5.0
    if len(closes) >= slow_period + 3:
        previous_fast = float(np.mean(closes[-fast_period - 3 : -3]))
        previous_slow = float(np.mean(closes[-slow_period - 3 : -3]))
        if fast_ma > previous_fast and slow_ma > previous_slow:
            ma_score += 5.0
        elif fast_ma < previous_fast and slow_ma < previous_slow:
            ma_score -= 5.0

    persistence_window = returns[-min(len(returns), max(8, horizon // 2)) :]
    bullish_persistence = float(np.mean(persistence_window > 0))
    bearish_persistence = float(np.mean(persistence_window < 0))
    persistence_score = (bullish_persistence - bearish_persistence) * 18.0

    swing_highs, swing_lows = _confirmed_swings_reference(
        window,
        pivot_left_bars=pivot_left_bars,
        pivot_right_bars=pivot_right_bars,
    )
    swing_tolerance = max((atr_scale / max(abs(price), 1.0)) * 0.35, 0.0015)
    swing_score = 0.0
    if len(swing_highs) >= 2:
        swing_score += 7.0 * _relative_move(swing_highs[-2][1], swing_highs[-1][1], swing_tolerance)
    if len(swing_lows) >= 2:
        swing_score += 7.0 * _relative_move(swing_lows[-2][1], swing_lows[-1][1], swing_tolerance)

    break_score, break_evidence = _break_structure_score_reference(
        history,
        breakout_lookback=breakout_lookback,
    )
    break_score *= 1.0 if horizon <= 60 else 0.5
    raw_score = slope_score + ma_score + persistence_score + swing_score + break_score
    score = float(np.clip(raw_score, -100.0, 100.0))

    evidence: list[str] = []
    if slope_score >= 6.0:
        evidence.append("Recent close regression slope was positive after range normalization.")
    elif slope_score <= -6.0:
        evidence.append("Recent close regression slope was negative after range normalization.")
    if ma_score >= 8.0:
        evidence.append("Price and moving-average alignment stayed bullish across the recent horizon.")
    elif ma_score <= -8.0:
        evidence.append("Price and moving-average alignment stayed bearish across the recent horizon.")
    if swing_score >= 7.0:
        evidence.append("Confirmed swing highs and lows were progressing upward.")
    elif swing_score <= -7.0:
        evidence.append("Confirmed swing highs and lows were progressing downward.")
    if persistence_score >= 4.0:
        evidence.append("Directional persistence favored bullish closes over the recent bars.")
    elif persistence_score <= -4.0:
        evidence.append("Directional persistence favored bearish closes over the recent bars.")
    evidence.extend(break_evidence)
    return round(score, 2), _trend_label(score), evidence


def classify_intraday_trend_reference(
    df: pd.DataFrame,
    *,
    lookback_bars: int = 12,
    pivot_left_bars: int = 2,
    pivot_right_bars: int = 2,
    breakout_lookback: int = 20,
) -> pd.DataFrame:
    pattern_df = df.copy()
    short_horizon, medium_horizon, long_horizon = _trend_horizons(lookback_bars)
    short_scores: list[float] = []
    medium_scores: list[float] = []
    long_scores: list[float] = []
    short_labels: list[str] = []
    medium_labels: list[str] = []
    long_labels: list[str] = []
    trend_scores: list[float] = []
    trend_labels: list[str] = []
    trend_evidence: list[list[str]] = []
    trend_horizons: list[str] = []
    trend_lookbacks: list[int] = []

    for index in range(len(pattern_df)):
        history = pattern_df.iloc[: index + 1].copy(deep=False)
        short_score, short_label, short_evidence = _trend_snapshot_reference(
            history,
            horizon=short_horizon,
            pivot_left_bars=pivot_left_bars,
            pivot_right_bars=pivot_right_bars,
            breakout_lookback=breakout_lookback,
        )
        medium_score, medium_label, medium_evidence = _trend_snapshot_reference(
            history,
            horizon=medium_horizon,
            pivot_left_bars=pivot_left_bars,
            pivot_right_bars=pivot_right_bars,
            breakout_lookback=breakout_lookback,
        )
        long_score, long_label, long_evidence = _trend_snapshot_reference(
            history,
            horizon=long_horizon,
            pivot_left_bars=pivot_left_bars,
            pivot_right_bars=pivot_right_bars,
            breakout_lookback=breakout_lookback,
        )

        weights: list[float] = []
        weighted_scores: list[float] = []
        for score, weight, horizon in (
            (short_score, 0.50, short_horizon),
            (medium_score, 0.35, medium_horizon),
            (long_score, 0.15, long_horizon),
        ):
            if len(history) >= min(8, horizon):
                weights.append(weight)
                weighted_scores.append(score * weight)

        composite_score = round(sum(weighted_scores) / sum(weights), 2) if weights else 0.0
        composite_label = _trend_label(composite_score)
        evidence = list(dict.fromkeys(short_evidence + medium_evidence))
        latest_row = history.iloc[-1]
        if composite_label == "Downtrend" and bool(latest_row.get("Is_Bullish", False)):
            evidence.append(
                "A recent bullish candle was treated as a counter-trend reaction, not a confirmed reversal."
            )
        elif composite_label == "Uptrend" and bool(latest_row.get("Is_Bearish", False)):
            evidence.append(
                "A recent bearish candle was treated as a counter-trend reaction, not a confirmed reversal."
            )
        if not evidence:
            evidence.append(
                "Slope, moving averages, swing structure, and recent breaks were too mixed to confirm a trend."
            )

        short_scores.append(short_score)
        medium_scores.append(medium_score)
        long_scores.append(long_score)
        short_labels.append(short_label)
        medium_labels.append(medium_label)
        long_labels.append(long_label)
        trend_scores.append(composite_score)
        trend_labels.append(composite_label)
        trend_evidence.append(evidence)
        trend_horizons.append("Short-to-medium term")
        trend_lookbacks.append(medium_horizon)

    pattern_df["Short_Term_Trend"] = short_labels
    pattern_df["Medium_Term_Trend"] = medium_labels
    pattern_df["Long_Term_Trend"] = long_labels
    pattern_df["Short_Term_Trend_Score"] = short_scores
    pattern_df["Medium_Term_Trend_Score"] = medium_scores
    pattern_df["Long_Term_Trend_Score"] = long_scores
    pattern_df["Trend"] = trend_labels
    pattern_df["Trend_Score"] = trend_scores
    pattern_df["Trend_Evidence"] = trend_evidence
    pattern_df["Trend_Horizon"] = trend_horizons
    pattern_df["Trend_Lookback_Bars"] = trend_lookbacks
    return pattern_df


def test_optimized_trend_classifier_matches_reference() -> None:
    df = make_benchmark_df(120)
    feature_df = add_features(df)

    optimized = classify_intraday_trend(feature_df, lookback_bars=12, pivot_left_bars=2, pivot_right_bars=2, breakout_lookback=20)
    reference = classify_intraday_trend_reference(
        feature_df,
        lookback_bars=12,
        pivot_left_bars=2,
        pivot_right_bars=2,
        breakout_lookback=20,
    )

    for column in (
        "Short_Term_Trend",
        "Medium_Term_Trend",
        "Long_Term_Trend",
        "Short_Term_Trend_Score",
        "Medium_Term_Trend_Score",
        "Long_Term_Trend_Score",
        "Trend",
        "Trend_Score",
        "Trend_Evidence",
        "Trend_Horizon",
        "Trend_Lookback_Bars",
    ):
        assert optimized[column].tolist() == reference[column].tolist()


def test_full_analysis_matches_reference_trend_pipeline(monkeypatch) -> None:
    df = make_benchmark_df(120)
    optimized = analyze_dataframe(df=df, symbol="MU", as_of=analysis_as_of(df))

    monkeypatch.setattr(analysis_module, "classify_intraday_trend", classify_intraday_trend_reference)
    reference = analyze_dataframe(df=df, symbol="MU", as_of=analysis_as_of(df))

    assert json.dumps(optimized, sort_keys=True, default=str) == json.dumps(reference, sort_keys=True, default=str)


def test_trend_classifier_preserves_no_lookahead() -> None:
    df = make_benchmark_df(140)
    feature_df = add_features(df)
    full = classify_intraday_trend(feature_df, lookback_bars=12, pivot_left_bars=2, pivot_right_bars=2, breakout_lookback=20)

    for cutoff in (32, 61, 97, 140):
        prefix = classify_intraday_trend(
            feature_df.iloc[:cutoff].copy(),
            lookback_bars=12,
            pivot_left_bars=2,
            pivot_right_bars=2,
            breakout_lookback=20,
        )
        assert prefix.iloc[-1]["Trend"] == full.iloc[cutoff - 1]["Trend"]
        assert float(prefix.iloc[-1]["Trend_Score"]) == float(full.iloc[cutoff - 1]["Trend_Score"])
        assert prefix.iloc[-1]["Trend_Evidence"] == full.iloc[cutoff - 1]["Trend_Evidence"]


def test_analysis_runtime_regression_is_fixed() -> None:
    analyze_dataframe(df=make_benchmark_df(80), symbol="WARM", as_of=analysis_as_of(make_benchmark_df(80)))
    df = make_benchmark_df(500)
    started_at = time.perf_counter()
    result = analyze_dataframe(df=df, symbol="MU", as_of=analysis_as_of(df), top_pattern_count=5)
    elapsed = time.perf_counter() - started_at

    assert result["symbol"] == "MU"
    assert elapsed < 5.0


def test_analysis_runtime_growth_is_near_linear() -> None:
    def timed_analysis(length: int) -> float:
        df = make_benchmark_df(length)
        started_at = time.perf_counter()
        analyze_dataframe(df=df, symbol="MU", as_of=analysis_as_of(df), top_pattern_count=5)
        return time.perf_counter() - started_at

    timed_analysis(120)
    runtime_500 = timed_analysis(500)
    runtime_1000 = timed_analysis(1000)

    assert runtime_1000 < runtime_500 * 3.0

