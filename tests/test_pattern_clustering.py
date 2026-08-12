"""Tests for Stage 3: grouping adjacent correlated patterns into a rejection cluster
before the final score is calculated.

Two (or more) same-direction, same-analytical-family patterns that occur within a small
number of bars of each other and share a rejection/reversal price zone are correlated
evidence, not independent signals, and must not have their raw scores simply summed.
Instead the cluster contributes a single bounded amount: the strongest member's own
score plus a small, configurable, capped repetition bonus.

These tests exercise `ScoringService` directly with synthetic `PatternEvent`s (following
the conventions in tests/test_scoring.py) so the clustering behavior itself can be
verified deterministically, independent of the candlestick-detector/context logic
covered in the Stage 1/2 test files.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from stock_pattern_model.config import ScoringConfig
from stock_pattern_model.domain import DataQualityReport
from stock_pattern_model.domain import PatternEvent
from stock_pattern_model.domain import PatternFamily
from stock_pattern_model.domain import PatternStatus
from stock_pattern_model.scoring import ScoringService

EXCHANGE_TZ = ZoneInfo("America/New_York")


def make_quality_report() -> DataQualityReport:
    return DataQualityReport(
        row_count=30,
        completed_row_count=30,
        duplicate_count=0,
        missing_value_count=0,
        invalid_ohlc_count=0,
        irregular_gap_count=0,
        warnings=[],
        cleaning_actions=[],
    )


def make_event(
    *,
    pattern_id: str,
    pattern_name: str,
    pattern_family: PatternFamily,
    bias: str,
    relevant_prices: dict[str, float],
    detected_at: str,
    base_score: float = 15.0,
    status: PatternStatus = PatternStatus.CONFIRMED,
) -> PatternEvent:
    timestamp = pd.Timestamp(detected_at, tz=EXCHANGE_TZ)
    return PatternEvent(
        pattern_id=pattern_id,
        pattern_name=pattern_name,
        pattern_family=pattern_family,
        bias=bias,
        status=status,
        pattern_start_at=timestamp,
        pattern_end_at=timestamp,
        bar_start_at=timestamp,
        bar_end_at=timestamp,
        detected_at=timestamp,
        relevant_prices=relevant_prices,
        relevant_indices=[0],
        detection_reason=f"{pattern_name} was detected.",
        signal_strength=1.5,
        base_score=base_score,
        exchange_timezone="America/New_York",
    )


def make_pattern_record(*, event: PatternEvent, candles_ago: int, priority: int = 5) -> dict[str, object]:
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
        "volume_confirmed": False,
        "strong_signal": False,
        "signal_strength": float(event.signal_strength),
        "strength_label": event.strength_label,
        "volume_baseline_source": event.volume_baseline_source,
        "score_eligible": True,
    }


def evaluate(patterns: list[dict[str, object]], *, trend: str = "Neutral", config: ScoringConfig | None = None):
    service = ScoringService(config or ScoringConfig())
    return service.evaluate(
        symbol="CLUSTER",
        trend=trend,
        patterns=patterns,
        quality_report=make_quality_report(),
        latest_close=118.0,
        latest_bar_start_display="2026-07-10 22:30 Asia/Jerusalem",
        latest_bar_end_display="2026-07-10 22:45 Asia/Jerusalem",
        interval="15m",
        latest_volume_baseline_source="time_of_day",
    )


def test_adjacent_same_zone_rejections_are_bounded_not_summed() -> None:
    older = make_pattern_record(
        event=make_event(
            pattern_id="shooting_star",
            pattern_name="Shooting Star",
            pattern_family=PatternFamily.PIN_BAR,
            bias="Bearish",
            relevant_prices={"high": 118.2},
            detected_at="2026-07-10 15:15",
            base_score=15,
        ),
        candles_ago=2,
    )
    newer = make_pattern_record(
        event=make_event(
            pattern_id="shooting_star",
            pattern_name="Shooting Star",
            pattern_family=PatternFamily.PIN_BAR,
            bias="Bearish",
            relevant_prices={"high": 118.25},
            detected_at="2026-07-10 15:45",
            base_score=18,
        ),
        candles_ago=0,
    )

    result = evaluate([older, newer])

    raw_sum = 18.0 + 10.84  # each member's own recency-weighted raw score, for comparison
    representative = next(p for p in result["patterns"] if p["pattern_score_contribution"] != 0)
    member = next(p for p in result["patterns"] if p["pattern_score_contribution"] == 0)

    assert representative["cluster_id"] is not None
    assert representative["cluster_id"] == member["cluster_id"]
    assert representative["cluster_type"] == "Repeated Upper Rejection Zone"
    assert representative["cluster_size"] == 2
    assert member["cluster_suppressed"] is True
    assert member["raw_pattern_score_contribution"] != 0

    # Bounded contribution must be strictly less than the naive sum of both raw scores,
    # but still reflect that a second, corroborating rejection occurred (bigger in magnitude
    # than the strongest member alone).
    assert abs(result["score"]["bearish_score"]) < raw_sum
    assert abs(result["score"]["bearish_score"]) > 18.0
    assert result["score"]["bearish_score"] == abs(representative["pattern_score_contribution"])


def test_rejections_far_apart_in_time_are_not_clustered() -> None:
    older = make_pattern_record(
        event=make_event(
            pattern_id="shooting_star",
            pattern_name="Shooting Star",
            pattern_family=PatternFamily.PIN_BAR,
            bias="Bearish",
            relevant_prices={"high": 118.2},
            detected_at="2026-07-10 13:15",
            base_score=15,
        ),
        candles_ago=6,
    )
    newer = make_pattern_record(
        event=make_event(
            pattern_id="shooting_star",
            pattern_name="Shooting Star",
            pattern_family=PatternFamily.PIN_BAR,
            bias="Bearish",
            relevant_prices={"high": 118.25},
            detected_at="2026-07-10 15:45",
            base_score=18,
        ),
        candles_ago=0,
    )

    result = evaluate([older, newer])

    assert all(pattern["cluster_id"] is None for pattern in result["patterns"])
    assert all(not pattern["cluster_suppressed"] for pattern in result["patterns"])
    # Both contribute their full, independent score -- a straight sum, not a bounded cluster.
    assert result["score"]["bearish_score"] == round(18.0 + 5.66, 2) or result["score"]["bearish_score"] > 18.0


def test_rejections_at_materially_different_price_levels_are_not_clustered() -> None:
    lower = make_pattern_record(
        event=make_event(
            pattern_id="shooting_star",
            pattern_name="Shooting Star",
            pattern_family=PatternFamily.PIN_BAR,
            bias="Bearish",
            relevant_prices={"high": 100.0},
            detected_at="2026-07-10 15:15",
            base_score=15,
        ),
        candles_ago=2,
    )
    higher = make_pattern_record(
        event=make_event(
            pattern_id="shooting_star",
            pattern_name="Shooting Star",
            pattern_family=PatternFamily.PIN_BAR,
            bias="Bearish",
            relevant_prices={"high": 150.0},
            detected_at="2026-07-10 15:45",
            base_score=18,
        ),
        candles_ago=0,
    )

    result = evaluate([lower, higher])

    assert all(pattern["cluster_id"] is None for pattern in result["patterns"])
    assert result["score"]["bearish_score"] == round(18.0 + 10.84, 2)


def test_bullish_and_bearish_patterns_are_never_grouped_together() -> None:
    bearish = make_pattern_record(
        event=make_event(
            pattern_id="shooting_star",
            pattern_name="Shooting Star",
            pattern_family=PatternFamily.PIN_BAR,
            bias="Bearish",
            relevant_prices={"high": 118.2},
            detected_at="2026-07-10 15:15",
            base_score=15,
        ),
        candles_ago=2,
    )
    bullish = make_pattern_record(
        event=make_event(
            pattern_id="bullish_pin_bar",
            pattern_name="Bullish Pin Bar",
            pattern_family=PatternFamily.PIN_BAR,
            bias="Bullish",
            relevant_prices={"low": 118.0},
            detected_at="2026-07-10 15:45",
            base_score=18,
        ),
        candles_ago=0,
    )

    result = evaluate([bearish, bullish])

    assert all(pattern["cluster_id"] is None for pattern in result["patterns"])
    assert result["score"]["bullish_score"] == 18.0
    assert result["score"]["bearish_score"] == 10.84


def test_different_families_representing_same_event_are_clustered() -> None:
    """A Shooting Star (pin_bar family) and an Evening Star (star family) rejecting the
    same resistance zone one bar apart represent the same underlying market event and
    must be clustered even though they come from different detectors/families.
    """
    shooting_star = make_pattern_record(
        event=make_event(
            pattern_id="shooting_star",
            pattern_name="Shooting Star",
            pattern_family=PatternFamily.PIN_BAR,
            bias="Bearish",
            relevant_prices={"high": 118.2},
            detected_at="2026-07-10 15:15",
            base_score=15,
        ),
        candles_ago=2,
    )
    evening_star = make_pattern_record(
        event=make_event(
            pattern_id="evening_star",
            pattern_name="Evening Star",
            pattern_family=PatternFamily.STAR,
            bias="Bearish",
            relevant_prices={"star_high": 118.22},
            detected_at="2026-07-10 15:45",
            base_score=16,
        ),
        candles_ago=0,
    )

    result = evaluate([shooting_star, evening_star])

    clustered = [pattern for pattern in result["patterns"] if pattern["cluster_id"] is not None]
    assert len(clustered) == 2
    assert clustered[0]["cluster_id"] == clustered[1]["cluster_id"]
    assert clustered[0]["cluster_type"] == "Repeated Upper Rejection Zone"
    raw_sum = 16.0 + 10.84
    assert abs(result["score"]["bearish_score"]) < raw_sum


def test_cluster_repetition_bonus_and_cap_are_configurable() -> None:
    older = make_pattern_record(
        event=make_event(
            pattern_id="shooting_star",
            pattern_name="Shooting Star",
            pattern_family=PatternFamily.PIN_BAR,
            bias="Bearish",
            relevant_prices={"high": 118.2},
            detected_at="2026-07-10 15:15",
            base_score=15,
        ),
        candles_ago=2,
    )
    newer = make_pattern_record(
        event=make_event(
            pattern_id="shooting_star",
            pattern_name="Shooting Star",
            pattern_family=PatternFamily.PIN_BAR,
            bias="Bearish",
            relevant_prices={"high": 118.25},
            detected_at="2026-07-10 15:45",
            base_score=18,
        ),
        candles_ago=0,
    )

    default_result = evaluate([older, newer])
    tight_cap_result = evaluate(
        [older, newer],
        config=ScoringConfig(cluster_max_contribution_multiplier=1.0, cluster_repetition_bonus=5.0),
    )

    # With a 1.0x cap, any repetition bonus is fully capped away -- the bounded contribution
    # can never exceed the strongest member's own score, regardless of the repetition bonus.
    tight_representative = next(p for p in tight_cap_result["patterns"] if p["pattern_score_contribution"] != 0)
    assert "capped_at_max_contribution" in tight_representative["cluster_penalties_applied"]
    assert abs(tight_representative["pattern_score_contribution"]) == 18.0
    assert abs(tight_representative["pattern_score_contribution"]) < abs(
        next(p for p in default_result["patterns"] if p["pattern_score_contribution"] != 0)["pattern_score_contribution"]
    )


def test_scoring_config_validates_cluster_and_dampener_thresholds() -> None:
    from stock_pattern_model.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError):
        ScoringConfig(cluster_max_bar_distance=0).validate()
    with pytest.raises(ConfigurationError):
        ScoringConfig(cluster_price_zone_tolerance=-0.01).validate()
    with pytest.raises(ConfigurationError):
        ScoringConfig(cluster_repetition_bonus=-0.01).validate()
    with pytest.raises(ConfigurationError):
        ScoringConfig(cluster_max_contribution_multiplier=0.9).validate()
    with pytest.raises(ConfigurationError):
        ScoringConfig(unconfirmed_rejection_dampener=-0.01).validate()

    # Sensible non-default values remain valid.
    ScoringConfig(
        cluster_max_bar_distance=3,
        cluster_price_zone_tolerance=0.01,
        cluster_repetition_bonus=2.0,
        cluster_max_contribution_multiplier=1.5,
        unconfirmed_rejection_dampener=2.0,
    ).validate()
