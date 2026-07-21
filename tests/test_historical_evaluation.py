from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from stock_pattern_model.config import HistoricalEvaluationConfig
from stock_pattern_model.evaluation import collect_historical_signals_from_dataframe
from stock_pattern_model.evaluation import evaluate_historical_dataframe


EXCHANGE_TZ = ZoneInfo("America/New_York")


def candle(
    open_price: float = 100.0,
    high_price: float = 100.6,
    low_price: float = 99.6,
    close_price: float = 100.1,
    volume: int = 1000,
) -> dict[str, float | int]:
    return {
        "Open": open_price,
        "High": high_price,
        "Low": low_price,
        "Close": close_price,
        "Volume": volume,
    }


def make_df(
    rows: list[dict[str, float | int]],
    *,
    start: str = "2026-07-10 09:30",
) -> pd.DataFrame:
    datetimes = pd.date_range(start=start, periods=len(rows), freq="15min", tz=EXCHANGE_TZ)
    return pd.DataFrame(
        [
            {
                "Datetime": timestamp,
                **row,
            }
            for timestamp, row in zip(datetimes, rows)
        ]
    )


def make_breakout_session(
    *,
    start: str,
    base_price: float = 100.0,
    winning_follow_through: bool = True,
    total_bars: int = 25,
) -> pd.DataFrame:
    rows = [
        candle(
            open_price=base_price,
            high_price=base_price + 0.6,
            low_price=base_price - 0.4,
            close_price=base_price + 0.1,
            volume=1000,
        )
        for _ in range(total_bars)
    ]
    breakout_index = 20
    rows[breakout_index] = candle(
        open_price=base_price + 0.1,
        high_price=base_price + 1.2,
        low_price=base_price,
        close_price=base_price + 0.9,
        volume=1200,
    )
    if winning_follow_through:
        if breakout_index + 1 < len(rows):
            rows[breakout_index + 1] = candle(
                open_price=base_price + 0.95,
                high_price=base_price + 1.7,
                low_price=base_price + 0.85,
                close_price=base_price + 1.4,
                volume=1180,
            )
        if breakout_index + 2 < len(rows):
            rows[breakout_index + 2] = candle(
                open_price=base_price + 1.4,
                high_price=base_price + 1.8,
                low_price=base_price + 1.2,
                close_price=base_price + 1.5,
                volume=1160,
            )
    else:
        if breakout_index + 1 < len(rows):
            rows[breakout_index + 1] = candle(
                open_price=base_price + 0.88,
                high_price=base_price + 1.0,
                low_price=base_price + 0.3,
                close_price=base_price + 0.45,
                volume=1180,
            )
        if breakout_index + 2 < len(rows):
            rows[breakout_index + 2] = candle(
                open_price=base_price + 0.48,
                high_price=base_price + 0.6,
                low_price=base_price + 0.2,
                close_price=base_price + 0.35,
                volume=1160,
            )
    return make_df(rows, start=start)


def analysis_as_of(df: pd.DataFrame) -> pd.Timestamp:
    last_bar = pd.Timestamp(df.iloc[-1]["Datetime"])
    return last_bar + pd.Timedelta(minutes=16)


def breakout_signal(result) -> object:
    return next(signal for signal in result.signals if signal.pattern_name == "20-Bar Breakout")


def breakout_outcome(result, horizon: int):
    signal = breakout_signal(result)
    return next(outcome for outcome in signal.outcomes if outcome.horizon_bars == horizon)


def test_collect_historical_signals_is_leakage_free() -> None:
    base_df = make_breakout_session(
        start="2026-07-10 09:30",
        base_price=100.0,
        winning_follow_through=True,
    )
    mutated_future = pd.concat(
        [
            base_df,
            make_df(
                [
                    candle(140.0, 145.0, 130.0, 144.0, 9000),
                    candle(144.0, 146.0, 120.0, 121.0, 9100),
                ],
                start="2026-07-10 15:45",
            ),
        ],
        ignore_index=True,
    )

    base_signals = collect_historical_signals_from_dataframe(
        base_df,
        symbol="LEAK",
        as_of=analysis_as_of(base_df),
    )
    mutated_signals = collect_historical_signals_from_dataframe(
        mutated_future,
        symbol="LEAK",
        as_of=analysis_as_of(mutated_future),
    )

    first_base = next(signal for signal in base_signals if signal.pattern_name == "20-Bar Breakout")
    first_mutated = next(signal for signal in mutated_signals if signal.detected_index == first_base.detected_index)

    assert first_base.pattern_name == first_mutated.pattern_name
    assert first_base.detected_at == first_mutated.detected_at
    assert first_base.rule_confidence == first_mutated.rule_confidence
    assert first_base.signal_confidence_bucket == first_mutated.signal_confidence_bucket


def test_forward_returns_direction_correctness_mfe_and_mae() -> None:
    df = make_breakout_session(
        start="2026-07-10 09:30",
        base_price=100.0,
        winning_follow_through=True,
    )
    result = evaluate_historical_dataframe(
        df,
        symbol="UP",
        as_of=analysis_as_of(df),
        evaluation_config=HistoricalEvaluationConfig(
            horizons_bars=(1, 2),
            target_return=0.004,
            stop_return=0.002,
        ),
    )

    outcome = breakout_outcome(result, 2)

    assert result.signal_count >= 1
    assert outcome.available is True
    assert outcome.direction_correct is True
    assert outcome.raw_forward_return > 0
    assert outcome.directional_forward_return > 0
    assert outcome.mfe_return >= outcome.directional_forward_return
    assert outcome.mae_return > -0.01
    assert outcome.first_touch == "target_first"


def test_target_stop_expectancy_win_rate_and_false_positive_rate() -> None:
    winning_df = make_breakout_session(
        start="2026-07-10 09:30",
        base_price=100.0,
        winning_follow_through=True,
    )
    losing_df = make_breakout_session(
        start="2026-07-13 09:30",
        base_price=101.0,
        winning_follow_through=False,
    )
    df = pd.concat([winning_df, losing_df], ignore_index=True)

    result = evaluate_historical_dataframe(
        df,
        symbol="MIX",
        as_of=analysis_as_of(df),
        evaluation_config=HistoricalEvaluationConfig(
            horizons_bars=(1,),
            target_return=0.004,
            stop_return=0.002,
        ),
    )

    breakout_summary = result.by_pattern["20-Bar Breakout"]["1"]

    assert breakout_summary.evaluated_signals == 2
    assert breakout_summary.precision == pytest.approx(0.5)
    assert breakout_summary.false_positive_rate == pytest.approx(0.5)
    assert breakout_summary.win_rate == pytest.approx(0.5)
    assert breakout_summary.expectancy == pytest.approx(0.001)
    assert breakout_summary.target_first_rate == pytest.approx(0.5)
    assert breakout_summary.stop_first_rate == pytest.approx(0.5)


def test_breakdowns_include_market_context_session_time_and_confidence_bucket() -> None:
    df = make_breakout_session(
        start="2026-07-10 09:30",
        base_price=100.0,
        winning_follow_through=True,
    )
    result = evaluate_historical_dataframe(
        df,
        symbol="CTX",
        as_of=analysis_as_of(df),
        evaluation_config=HistoricalEvaluationConfig(
            horizons_bars=(1,),
            target_return=0.004,
            stop_return=0.002,
            confidence_bucket_edges=(40.0, 60.0, 80.0),
        ),
    )

    signal = breakout_signal(result)
    market_context_key = f"{signal.trend} | {signal.market_state} | {signal.overall_bias}"
    session_time_key = f"{signal.session_segment} | {signal.session_time_exchange}"

    assert "20-Bar Breakout" in result.by_pattern
    assert market_context_key in result.by_market_context
    assert session_time_key in result.by_session_time
    assert signal.signal_confidence_bucket in result.by_confidence_bucket
    assert any("measured separately" in note.lower() for note in result.notes)


def test_insufficient_future_bars_are_retained_but_excluded_from_summary() -> None:
    df = make_breakout_session(
        start="2026-07-10 09:30",
        base_price=100.0,
        winning_follow_through=True,
        total_bars=22,
    )
    result = evaluate_historical_dataframe(
        df,
        symbol="SHORT",
        as_of=analysis_as_of(df),
        evaluation_config=HistoricalEvaluationConfig(
            horizons_bars=(3,),
            target_return=0.004,
            stop_return=0.002,
        ),
    )

    outcome = breakout_outcome(result, 3)
    summary = result.overall_by_horizon["3"]

    assert outcome.available is False
    assert outcome.first_touch == "insufficient_future_bars"
    assert summary.evaluated_signals == 0
    assert summary.expectancy is None


def test_historical_evaluation_result_is_json_serializable() -> None:
    df = make_breakout_session(
        start="2026-07-10 09:30",
        base_price=100.0,
        winning_follow_through=True,
    )
    result = evaluate_historical_dataframe(
        df,
        symbol="JSON",
        as_of=analysis_as_of(df),
        evaluation_config=HistoricalEvaluationConfig(
            horizons_bars=(1,),
            target_return=0.004,
            stop_return=0.002,
        ),
    )
    payload = result.to_dict()

    assert payload["symbol"] == "JSON"
    assert payload["signals"][0]["outcomes"][0]["horizon_bars"] == 1
    assert payload["overall_by_horizon"]["1"]["evaluated_signals"] >= 1
    assert "Signal Confidence buckets" in payload["notes"][1]
