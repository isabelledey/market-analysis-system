from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

from stock_pattern_model.analysis import analyze_dataframe
from stock_pattern_model.config import PatternConfig
from stock_pattern_model.domain import PatternEvent
from stock_pattern_model.domain import PatternFamily
from stock_pattern_model.domain import PatternStatus
from stock_pattern_model.features import add_features
from stock_pattern_model.pattern_detector import BasePatternDetector
from stock_pattern_model.pattern_detector import BearishEngulfingDetector
from stock_pattern_model.pattern_detector import BreakdownDetector
from stock_pattern_model.pattern_detector import BreakoutDetector
from stock_pattern_model.pattern_detector import DEFAULT_PATTERN_REGISTRY
from stock_pattern_model.pattern_detector import DojiDetector
from stock_pattern_model.pattern_detector import DoubleBottomDetector
from stock_pattern_model.pattern_detector import DoubleTopDetector
from stock_pattern_model.pattern_detector import EveningStarDetector
from stock_pattern_model.pattern_detector import InsideBarDetector
from stock_pattern_model.pattern_detector import MorningStarDetector
from stock_pattern_model.pattern_detector import ShootingStarDetector


EXCHANGE_TZ = ZoneInfo("America/New_York")


def candle(
    open_price: float = 100.0,
    high_price: float = 100.8,
    low_price: float = 99.6,
    close_price: float = 100.2,
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


def make_base_df(length: int = 30) -> pd.DataFrame:
    return make_df([candle() for _ in range(length)])


def analysis_as_of(df: pd.DataFrame) -> pd.Timestamp:
    last_bar = pd.Timestamp(df.iloc[-1]["Datetime"])
    return last_bar + pd.Timedelta(minutes=16)


def test_doji_positive_and_near_miss() -> None:
    positive_df = make_df(
        [
            candle(),
            candle(open_price=100.0, high_price=101.0, low_price=99.0, close_price=100.02, volume=1500),
        ]
    )
    detector = DojiDetector()
    positive_events = detector.detect(add_features(positive_df), PatternConfig(), "15m")

    assert len(positive_events) == 1
    assert positive_events[0].pattern_name == "Doji"
    assert positive_events[0].detected_at == pd.Timestamp("2026-07-10 10:00", tz=EXCHANGE_TZ)

    near_miss_df = make_df(
        [
            candle(),
            candle(open_price=100.0, high_price=101.0, low_price=99.0, close_price=100.25, volume=1500),
        ]
    )
    near_miss_events = detector.detect(add_features(near_miss_df), PatternConfig(), "15m")

    assert near_miss_events == []


def test_morning_star_detection_timestamp_and_no_future_data_use() -> None:
    df = make_df(
        [
            candle(),
            candle(),
            candle(open_price=101.2, high_price=101.3, low_price=99.7, close_price=99.9, volume=2800),
            candle(open_price=99.95, high_price=100.05, low_price=99.6, close_price=99.9, volume=1400),
            candle(open_price=99.95, high_price=101.4, low_price=99.9, close_price=100.8, volume=2600),
        ]
    )
    detector = MorningStarDetector()
    events = detector.detect(add_features(df), PatternConfig(), "15m")

    assert len(events) == 1
    event = events[0]
    assert event.pattern_start_at == pd.Timestamp("2026-07-10 10:00", tz=EXCHANGE_TZ)
    assert event.pattern_end_at == pd.Timestamp("2026-07-10 10:45", tz=EXCHANGE_TZ)
    assert event.detected_at == pd.Timestamp("2026-07-10 10:45", tz=EXCHANGE_TZ)
    assert event.relevant_indices == [2, 3, 4]


def test_breakout_strong_event_remains_single_event() -> None:
    df = make_base_df(30)
    df.loc[20, ["Open", "High", "Low", "Close", "Volume"]] = [100.10, 101.20, 100.00, 100.90, 2600]
    df.loc[21, ["Open", "High", "Low", "Close", "Volume"]] = [100.80, 101.40, 100.70, 101.10, 1300]
    df.loc[22, ["Open", "High", "Low", "Close", "Volume"]] = [101.00, 101.50, 100.90, 101.20, 1200]

    events = BreakoutDetector().detect(add_features(df), PatternConfig(), "15m")

    assert len(events) == 1
    assert events[0].pattern_name == "Strong 20-Bar Breakout"
    assert events[0].strength_label == "strong"


def test_additional_pattern_detectors_emit_expected_events() -> None:
    bearish_engulfing_df = make_base_df(25)
    bearish_engulfing_df.loc[23, ["Open", "High", "Low", "Close", "Volume"]] = [99.8, 101.2, 99.7, 101.0, 2600]
    bearish_engulfing_df.loc[24, ["Open", "High", "Low", "Close", "Volume"]] = [101.1, 101.3, 99.4, 99.6, 2800]
    bearish_events = BearishEngulfingDetector().detect(add_features(bearish_engulfing_df), PatternConfig(), "15m")
    assert len(bearish_events) == 1
    assert bearish_events[0].pattern_name == "Bearish Engulfing"

    shooting_star_df = make_base_df(25)
    shooting_star_df.loc[24, ["Open", "High", "Low", "Close", "Volume"]] = [100.2, 101.5, 100.1, 100.3, 2600]
    shooting_star_events = ShootingStarDetector().detect(add_features(shooting_star_df), PatternConfig(), "15m")
    assert len(shooting_star_events) == 1
    assert shooting_star_events[0].pattern_name == "Shooting Star"

    inside_bar_df = make_base_df(25)
    inside_bar_df.loc[23, ["Open", "High", "Low", "Close", "Volume"]] = [100.0, 105.0, 95.0, 102.0, 2600]
    inside_bar_df.loc[24, ["Open", "High", "Low", "Close", "Volume"]] = [101.0, 104.0, 96.0, 101.5, 1800]
    inside_bar_events = InsideBarDetector().detect(add_features(inside_bar_df), PatternConfig(), "15m")
    assert len(inside_bar_events) == 1
    assert inside_bar_events[0].pattern_name == "Inside Bar"

    breakdown_df = make_base_df(30)
    breakdown_df.loc[20, ["Open", "High", "Low", "Close", "Volume"]] = [100.10, 100.20, 99.00, 99.10, 2600]
    breakdown_df.loc[21, ["Open", "High", "Low", "Close", "Volume"]] = [99.20, 99.30, 98.80, 98.90, 1200]
    breakdown_events = BreakdownDetector().detect(add_features(breakdown_df), PatternConfig(), "15m")
    assert len(breakdown_events) == 1
    assert breakdown_events[0].pattern_name == "Strong 20-Bar Breakdown"

    evening_star_df = make_df(
        [
            candle(),
            candle(),
            candle(open_price=99.8, high_price=101.3, low_price=99.7, close_price=101.1, volume=2800),
            candle(open_price=101.15, high_price=101.4, low_price=100.9, close_price=101.2, volume=1400),
            candle(open_price=101.1, high_price=101.2, low_price=99.5, close_price=100.0, volume=2600),
        ]
    )
    evening_star_events = EveningStarDetector().detect(add_features(evening_star_df), PatternConfig(), "15m")
    assert len(evening_star_events) == 1
    assert evening_star_events[0].pattern_name == "Evening Star"


def make_double_top_df(
    second_peak_high: float = 104.9,
    valley_low: float = 97.8,
    confirmation_close: float | None = 97.2,
) -> pd.DataFrame:
    post_confirmation_close = 97.5 if confirmation_close is not None else 98.6
    rows = [
        candle(99.5, 100.0, 99.0, 99.7),
        candle(100.0, 101.0, 99.4, 100.6),
        candle(100.6, 105.0, 100.1, 104.5, 2200),
        candle(104.1, 104.3, 101.8, 102.0),
        candle(102.0, 102.4, 99.8, 100.3),
        candle(100.1, 100.5, 98.4, 98.9),
        candle(98.9, 99.2, valley_low, 98.4, 2100),
        candle(98.5, 100.4, 98.2, 100.0),
        candle(100.0, 101.2, 99.6, 100.8),
        candle(101.0, 103.6, 100.4, 103.0),
        candle(103.1, second_peak_high, 102.5, 104.2, 2200),
        candle(104.0, 104.1, 101.8, 102.3),
        candle(102.2, 102.4, 100.0, 100.5),
        candle(100.3, 100.5, 96.8, confirmation_close if confirmation_close is not None else 98.7, 2400),
        candle(
            98.2,
            max(99.0, post_confirmation_close + 0.1),
            min(97.9, post_confirmation_close - 0.1),
            post_confirmation_close,
        ),
    ]
    return make_df(rows)


def make_double_bottom_df(
    second_bottom_low: float = 95.1,
    neckline_high: float = 102.3,
    confirmation_close: float | None = 102.8,
) -> pd.DataFrame:
    post_confirmation_close = 103.0 if confirmation_close is not None else 101.7
    rows = [
        candle(100.4, 101.0, 99.9, 100.7),
        candle(100.8, 101.2, 99.5, 100.0),
        candle(100.1, 100.3, 95.0, 95.6, 2300),
        candle(95.8, 97.6, 95.3, 97.1),
        candle(97.2, 99.3, 96.8, 99.0),
        candle(99.1, neckline_high, 98.9, 101.8, 2100),
        candle(101.6, 101.8, 99.5, 100.1),
        candle(100.0, 100.2, 98.0, 98.6),
        candle(98.5, 99.0, second_bottom_low, 95.8, 2200),
        candle(95.9, 97.2, 95.6, 96.8),
        candle(96.8, 98.5, 96.6, 97.8),
        candle(97.9, 103.0, 97.7, confirmation_close if confirmation_close is not None else 101.6, 2500),
        candle(102.9, 103.3, 102.0, post_confirmation_close),
    ]
    return make_df(rows)


def test_double_top_tentative_and_confirmed_timestamps() -> None:
    df = make_double_top_df()
    detector = DoubleTopDetector()
    events = detector.detect(add_features(df), PatternConfig(), "15m")

    tentative = next(event for event in events if event.status is PatternStatus.TENTATIVE)
    confirmed = next(event for event in events if event.status is PatternStatus.CONFIRMED)

    assert tentative.pattern_start_at == pd.Timestamp("2026-07-10 10:00", tz=EXCHANGE_TZ)
    assert tentative.pattern_end_at == pd.Timestamp("2026-07-10 12:15", tz=EXCHANGE_TZ)
    assert tentative.detected_at == pd.Timestamp("2026-07-10 12:45", tz=EXCHANGE_TZ)
    assert tentative.detected_at > tentative.bar_end_at
    assert confirmed.detected_at == pd.Timestamp("2026-07-10 13:00", tz=EXCHANGE_TZ)
    assert confirmed.relevant_prices["neckline"] == 97.8


def test_double_top_without_neckline_break_is_not_confirmed_and_does_not_score() -> None:
    df = make_double_top_df(confirmation_close=None)
    result = analyze_dataframe(df, symbol="TOP", as_of=analysis_as_of(df))

    statuses = [pattern["status"] for pattern in result["all_detected_patterns"] if pattern["pattern_name"] == "Double Top"]
    assert "tentative" in statuses
    assert "confirmed" not in statuses
    assert result["pattern_score"] == 0
    assert result["overall_bias"] == "Neutral"


def test_double_bottom_without_neckline_break_is_not_confirmed() -> None:
    df = make_double_bottom_df(confirmation_close=None)
    detector = DoubleBottomDetector()
    events = detector.detect(add_features(df), PatternConfig(), "15m")

    statuses = [event.status for event in events]
    assert PatternStatus.TENTATIVE in statuses
    assert PatternStatus.CONFIRMED not in statuses


def test_similar_peaks_without_meaningful_valley_are_rejected() -> None:
    df = make_df(
        [
            candle(99.5, 100.0, 99.0, 99.7),
            candle(100.0, 101.0, 99.4, 100.6),
            candle(100.6, 105.0, 100.1, 104.5, 2200),
            candle(104.1, 104.4, 103.7, 104.1),
            candle(104.0, 104.3, 103.8, 104.0),
            candle(104.0, 104.4, 103.9, 104.2),
            candle(104.2, 104.9, 103.8, 104.4, 2200),
            candle(104.1, 104.2, 103.9, 104.0),
            candle(104.0, 104.1, 103.8, 103.9),
        ]
    )
    events = DoubleTopDetector().detect(add_features(df), PatternConfig(), "15m")

    assert events == []


def test_configurable_tolerance_changes_double_top_detection() -> None:
    df = make_double_top_df(second_peak_high=103.8)
    feature_df = add_features(df)
    detector = DoubleTopDetector()

    strict_events = detector.detect(feature_df, PatternConfig(), "15m")
    tolerant_events = detector.detect(
        feature_df,
        PatternConfig(double_pattern_price_tolerance_ratio=0.02),
        "15m",
    )

    assert strict_events == []
    assert any(event.status is PatternStatus.TENTATIVE for event in tolerant_events)


def test_registry_can_add_detector_without_editing_main_analysis_service() -> None:
    class CustomDetector(BasePatternDetector):
        def __init__(self) -> None:
            super().__init__(
                pattern_id="custom_test_pattern",
                pattern_name="Custom Test Pattern",
                family=PatternFamily.EXTENSION,
                minimum_required_history=1,
                default_bias="Bullish",
                default_base_score=9,
                default_priority=1,
            )

        def detect(self, data: pd.DataFrame, config: PatternConfig, interval: str) -> list[PatternEvent]:
            return [
                self._build_event(
                    data,
                    interval,
                    0,
                    pattern_start_index=0,
                    relevant_indices=[0],
                    relevant_prices={"close": float(data.iloc[0]["Close"])},
                    detection_reason="Injected by a custom registry detector.",
                )
            ]

    registry = DEFAULT_PATTERN_REGISTRY.register(CustomDetector())
    df = make_base_df(5)
    result = analyze_dataframe(df, symbol="EXT", as_of=analysis_as_of(df), registry=registry)

    assert any(pattern["pattern_id"] == "custom_test_pattern" for pattern in result["all_detected_patterns"])


def test_tentative_patterns_rank_below_confirmed_patterns() -> None:
    df = make_double_top_df()
    result = analyze_dataframe(df, symbol="TOP", as_of=analysis_as_of(df), top_pattern_count=5)

    double_top_patterns = [pattern for pattern in result["top_patterns"] if pattern["pattern_name"] == "Double Top"]
    assert double_top_patterns[0]["status"] == "confirmed"
