from __future__ import annotations

import json
from zoneinfo import ZoneInfo

import pandas as pd

from stock_pattern_model.analysis import analyze_dataframe
from stock_pattern_model.config import ScoringConfig
from stock_pattern_model.domain import DataQualityReport
from stock_pattern_model.domain import PatternEvent
from stock_pattern_model.domain import PatternFamily
from stock_pattern_model.domain import PatternStatus
from stock_pattern_model.scoring import ScoringService


EXCHANGE_TZ = ZoneInfo("America/New_York")


def make_quality_report(warnings: list[str] | None = None) -> DataQualityReport:
    return DataQualityReport(
        row_count=30,
        completed_row_count=30,
        duplicate_count=0,
        missing_value_count=0,
        invalid_ohlc_count=0,
        irregular_gap_count=0,
        warnings=warnings or [],
        cleaning_actions=[],
    )


def make_event(
    *,
    pattern_id: str,
    pattern_name: str,
    pattern_family: PatternFamily,
    bias: str,
    status: PatternStatus = PatternStatus.CONFIRMED,
    detected_at: str = "2026-07-10 15:45",
    pattern_start_at: str = "2026-07-10 15:30",
    pattern_end_at: str = "2026-07-10 15:45",
    bar_start_at: str = "2026-07-10 15:30",
    bar_end_at: str = "2026-07-10 15:45",
    relevant_prices: dict[str, float] | None = None,
    relevant_indices: list[int] | None = None,
    base_score: float = 18.0,
    signal_strength: float = 1.5,
) -> PatternEvent:
    return PatternEvent(
        pattern_id=pattern_id,
        pattern_name=pattern_name,
        pattern_family=pattern_family,
        bias=bias,
        status=status,
        pattern_start_at=pd.Timestamp(pattern_start_at, tz=EXCHANGE_TZ),
        pattern_end_at=pd.Timestamp(pattern_end_at, tz=EXCHANGE_TZ),
        bar_start_at=pd.Timestamp(bar_start_at, tz=EXCHANGE_TZ),
        bar_end_at=pd.Timestamp(bar_end_at, tz=EXCHANGE_TZ),
        detected_at=pd.Timestamp(detected_at, tz=EXCHANGE_TZ),
        relevant_prices=relevant_prices or {},
        relevant_indices=relevant_indices or [0],
        detection_reason=f"{pattern_name} was detected.",
        signal_strength=signal_strength,
        base_score=base_score,
        exchange_timezone="America/New_York",
    )


def make_pattern_record(
    *,
    event: PatternEvent,
    candles_ago: int,
    priority: int = 2,
    volume_confirmed: bool = True,
    strong_signal: bool = False,
    score_eligible: bool = True,
) -> dict[str, object]:
    return {
        "event": event,
        "pattern_id": event.pattern_id,
        "pattern_name": event.pattern_name,
        "bias": event.bias,
        "status": event.status.value,
        "pattern_family": event.pattern_family.value,
        "priority": priority,
        "base_score": float(event.base_score),
        "weighted_score": float(event.base_score),
        "candles_ago": candles_ago,
        "detection_reason": event.detection_reason,
        "exchange_timezone": event.exchange_timezone,
        "volume_confirmed": volume_confirmed,
        "strong_signal": strong_signal,
        "signal_strength": float(event.signal_strength),
        "strength_label": event.strength_label,
        "volume_baseline_source": event.volume_baseline_source,
        "score_eligible": score_eligible,
    }


def make_base_df(length: int = 30, start: str = "2026-07-10 09:30") -> pd.DataFrame:
    datetimes = pd.date_range(start=start, periods=length, freq="15min", tz=EXCHANGE_TZ)
    rows = []
    for offset, timestamp in enumerate(datetimes):
        price = 100.0 + (offset * 0.02)
        rows.append(
            {
                "Datetime": timestamp,
                "Open": price,
                "High": price + 0.5,
                "Low": price - 0.5,
                "Close": price + 0.05,
                "Volume": 1000,
            }
        )
    return pd.DataFrame(rows)


def test_no_pattern_analysis_outputs_neutral_scores() -> None:
    df = make_base_df()
    result = analyze_dataframe(df=df, symbol="TEST", as_of=pd.Timestamp("2026-07-10 17:01", tz=EXCHANGE_TZ))

    assert result["pattern_score"] == 0
    assert result["volume_score"] == 0
    assert result["overall_bias"] == "Neutral"
    assert result["market_state"] in {"Trend Only", "Neutral"}
    assert set(result["structured_explanation"]) == {
        "summary",
        "trend_evidence",
        "bullish_evidence",
        "bearish_evidence",
        "conflicts",
        "data_warnings",
        "reason_for_bias",
        "reason_for_confidence",
    }


def test_trend_only_analysis_keeps_bias_neutral() -> None:
    df = make_base_df(length=60)
    df["Close"] = pd.Series([100 + (i * 0.25) for i in range(len(df))], dtype=float)
    df["Open"] = df["Close"] - 0.05
    df["High"] = df["Close"] + 0.15
    df["Low"] = df["Open"] - 0.15

    result = analyze_dataframe(df=df, symbol="TREND", as_of=pd.Timestamp("2026-07-10 23:31", tz=EXCHANGE_TZ))

    assert result["trend"] == "Uptrend"
    assert result["overall_bias"] == "Neutral"
    assert result["market_state"] == "Trend Only"
    assert result["rule_confidence"] <= 12.0


def test_strong_bullish_agreement_increases_confidence() -> None:
    service = ScoringService(ScoringConfig())
    patterns = [
        make_pattern_record(
            event=make_event(
                pattern_id="breakout",
                pattern_name="Strong 20-Bar Breakout",
                pattern_family=PatternFamily.BREAKOUT,
                bias="Bullish",
                base_score=26,
                detected_at="2026-07-10 15:45",
                relevant_prices={"breakout_level": 101.2},
            ),
            candles_ago=0,
            strong_signal=True,
        ),
        make_pattern_record(
            event=make_event(
                pattern_id="bullish_engulfing",
                pattern_name="Bullish Engulfing",
                pattern_family=PatternFamily.ENGULFING,
                bias="Bullish",
                base_score=15,
                detected_at="2026-07-10 15:30",
            ),
            candles_ago=1,
        ),
    ]

    result = service.evaluate(
        symbol="BULL",
        trend="Uptrend",
        patterns=patterns,
        quality_report=make_quality_report(),
        latest_close=102.5,
        latest_bar_start_display="2026-07-10 22:30 Asia/Jerusalem",
        latest_bar_end_display="2026-07-10 22:45 Asia/Jerusalem",
        interval="15m",
        latest_volume_baseline_source="time_of_day",
    )

    assert result["overall_bias"] == "Bullish"
    assert result["market_state"] in {"Breakout Attempt", "Bullish Continuation", "Bullish Setup"}
    assert result["rule_confidence"] >= 55.0


def test_conflicting_evidence_reduces_confidence_and_bias() -> None:
    service = ScoringService(ScoringConfig())
    patterns = [
        make_pattern_record(
            event=make_event(
                pattern_id="breakout",
                pattern_name="20-Bar Breakout",
                pattern_family=PatternFamily.BREAKOUT,
                bias="Bullish",
                base_score=18,
            ),
            candles_ago=1,
        ),
        make_pattern_record(
            event=make_event(
                pattern_id="bearish_engulfing",
                pattern_name="Bearish Engulfing",
                pattern_family=PatternFamily.ENGULFING,
                bias="Bearish",
                base_score=18,
                detected_at="2026-07-10 15:30",
            ),
            candles_ago=1,
        ),
    ]

    result = service.evaluate(
        symbol="MIXED",
        trend="Neutral",
        patterns=patterns,
        quality_report=make_quality_report(),
        latest_close=100.0,
        latest_bar_start_display="2026-07-10 22:30 Asia/Jerusalem",
        latest_bar_end_display="2026-07-10 22:45 Asia/Jerusalem",
        interval="15m",
        latest_volume_baseline_source="time_of_day",
    )

    assert result["overall_bias"] == "Neutral"
    assert result["market_state"] == "Conflicted"
    assert result["rule_confidence"] < 60.0


def test_old_events_expire_and_do_not_drive_state() -> None:
    service = ScoringService(ScoringConfig(state_expiration_bars=3, pattern_max_age_bars=12))
    patterns = [
        make_pattern_record(
            event=make_event(
                pattern_id="breakout",
                pattern_name="20-Bar Breakout",
                pattern_family=PatternFamily.BREAKOUT,
                bias="Bullish",
                base_score=18,
            ),
            candles_ago=6,
        ),
    ]

    result = service.evaluate(
        symbol="OLD",
        trend="Uptrend",
        patterns=patterns,
        quality_report=make_quality_report(),
        latest_close=101.0,
        latest_bar_start_display="2026-07-10 22:30 Asia/Jerusalem",
        latest_bar_end_display="2026-07-10 22:45 Asia/Jerusalem",
        interval="15m",
        latest_volume_baseline_source="time_of_day",
    )

    assert result["patterns"][0]["event_state"] == "expired"
    assert result["market_state"] == "Trend Only"


def test_duplicate_evidence_is_not_counted_twice() -> None:
    service = ScoringService(ScoringConfig())
    detected_at = "2026-07-10 15:45"
    patterns = [
        make_pattern_record(
            event=make_event(
                pattern_id="double_top",
                pattern_name="Double Top",
                pattern_family=PatternFamily.DOUBLE_TOP,
                bias="Bearish",
                base_score=20,
                detected_at=detected_at,
                relevant_prices={"confirmation_price": 97.8, "neckline": 97.8},
            ),
            candles_ago=0,
            strong_signal=True,
        ),
        make_pattern_record(
            event=make_event(
                pattern_id="breakdown",
                pattern_name="Strong 20-Bar Breakdown",
                pattern_family=PatternFamily.BREAKOUT,
                bias="Bearish",
                base_score=26,
                detected_at=detected_at,
                relevant_prices={"breakdown_level": 97.7},
            ),
            candles_ago=0,
            strong_signal=True,
        ),
    ]

    result = service.evaluate(
        symbol="DEDUPE",
        trend="Downtrend",
        patterns=patterns,
        quality_report=make_quality_report(),
        latest_close=97.4,
        latest_bar_start_display="2026-07-10 22:30 Asia/Jerusalem",
        latest_bar_end_display="2026-07-10 22:45 Asia/Jerusalem",
        interval="15m",
        latest_volume_baseline_source="time_of_day",
    )

    primary_patterns = [pattern for pattern in result["patterns"] if pattern["group_primary"]]
    suppressed_patterns = [pattern for pattern in result["patterns"] if pattern["group_suppressed"]]
    assert len(primary_patterns) == 1
    assert len(suppressed_patterns) == 1
    assert result["score"]["bearish_score"] < 50


def test_data_quality_warnings_reduce_confidence() -> None:
    service = ScoringService(ScoringConfig())
    patterns = [
        make_pattern_record(
            event=make_event(
                pattern_id="breakout",
                pattern_name="20-Bar Breakout",
                pattern_family=PatternFamily.BREAKOUT,
                bias="Bullish",
                base_score=18,
            ),
            candles_ago=0,
        ),
    ]
    clean_result = service.evaluate(
        symbol="WARN",
        trend="Uptrend",
        patterns=patterns,
        quality_report=make_quality_report(),
        latest_close=101.2,
        latest_bar_start_display="2026-07-10 22:30 Asia/Jerusalem",
        latest_bar_end_display="2026-07-10 22:45 Asia/Jerusalem",
        interval="15m",
        latest_volume_baseline_source="time_of_day",
    )
    warning_result = service.evaluate(
        symbol="WARN",
        trend="Uptrend",
        patterns=patterns,
        quality_report=make_quality_report(["Unexpected interval gaps found."]),
        latest_close=101.2,
        latest_bar_start_display="2026-07-10 22:30 Asia/Jerusalem",
        latest_bar_end_display="2026-07-10 22:45 Asia/Jerusalem",
        interval="15m",
        latest_volume_baseline_source="time_of_day",
    )

    assert warning_result["rule_confidence"] < clean_result["rule_confidence"]


def test_tentative_patterns_do_not_affect_default_signals() -> None:
    service = ScoringService(ScoringConfig())
    patterns = [
        make_pattern_record(
            event=make_event(
                pattern_id="double_bottom",
                pattern_name="Double Bottom",
                pattern_family=PatternFamily.DOUBLE_BOTTOM,
                bias="Bullish",
                status=PatternStatus.TENTATIVE,
                base_score=20,
            ),
            candles_ago=0,
            score_eligible=False,
        ),
    ]
    result = service.evaluate(
        symbol="TENT",
        trend="Neutral",
        patterns=patterns,
        quality_report=make_quality_report(),
        latest_close=100.1,
        latest_bar_start_display="2026-07-10 22:30 Asia/Jerusalem",
        latest_bar_end_display="2026-07-10 22:45 Asia/Jerusalem",
        interval="15m",
        latest_volume_baseline_source="time_of_day",
    )

    assert result["score"]["pattern_score"] == 0
    assert result["overall_bias"] == "Neutral"


def test_scores_bias_and_explanation_remain_consistent() -> None:
    service = ScoringService(ScoringConfig())
    patterns = [
        make_pattern_record(
            event=make_event(
                pattern_id="breakdown",
                pattern_name="Strong 20-Bar Breakdown",
                pattern_family=PatternFamily.BREAKOUT,
                bias="Bearish",
                base_score=26,
            ),
            candles_ago=0,
            strong_signal=True,
        ),
    ]
    result = service.evaluate(
        symbol="CONSIST",
        trend="Downtrend",
        patterns=patterns,
        quality_report=make_quality_report(),
        latest_close=96.8,
        latest_bar_start_display="2026-07-10 22:30 Asia/Jerusalem",
        latest_bar_end_display="2026-07-10 22:45 Asia/Jerusalem",
        interval="15m",
        latest_volume_baseline_source="time_of_day",
    )

    assert result["score"]["net_signal_score"] < 0
    assert result["overall_bias"] == "Bearish"
    assert "Bearish" in result["structured_explanation"]["reason_for_bias"] or "Bearish" in result["structured_explanation"]["summary"]
    assert "not statistically calibrated" not in result["structured_explanation"]["reason_for_confidence"]


def test_analysis_result_is_json_serializable() -> None:
    df = make_base_df()
    result = analyze_dataframe(df=df, symbol="JSON", as_of=pd.Timestamp("2026-07-10 17:01", tz=EXCHANGE_TZ))

    serialized = json.dumps(result, ensure_ascii=False, sort_keys=True)

    assert '"structured_explanation"' in serialized
    assert '"volume_score"' in serialized
