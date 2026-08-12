"""Tests for the evidence-classification, analytical-family, and volume-attribution fixes.

These cover:
- Score-contributing evidence must never be zeroed out just because the overall bias is
  Neutral or because directional conflict exists; alignment/conflict are separate facts.
- `independent_family_key` (family, bias) must not merge opposite-direction patterns from the
  same geometric family, while still merging same-direction, same-family correlated patterns.
- Volume evidence shared by multiple pattern labels completing on the same candle must be
  attributed only once to the aggregate Volume Score and to `volume_confirmed_ratio`.
"""

from __future__ import annotations

from test_scoring import make_event, make_pattern_record, make_quality_report

from stock_pattern_model.config import ScoringConfig
from stock_pattern_model.domain import PatternFamily
from stock_pattern_model.scoring import (
    ScoringService,
    analytical_family,
    independent_family_key,
)


def _evaluate(patterns, trend="Neutral"):
    service = ScoringService(ScoringConfig())
    return service.evaluate(
        symbol="TEST",
        trend=trend,
        patterns=patterns,
        quality_report=make_quality_report(),
        latest_close=100.0,
        latest_bar_start_display="2026-07-10 22:30 Asia/Jerusalem",
        latest_bar_end_display="2026-07-10 22:45 Asia/Jerusalem",
        interval="15m",
        latest_volume_baseline_source="time_of_day",
    )


def _mixed_bias_patterns(*, same_candle_bearish: bool):
    """One bullish engulfing (always on its own, distinct candle) plus a bearish engulfing
    + evening star pair.

    When `same_candle_bearish` is True, the bearish pair shares one completion candle
    (reproducing the NVDA volume double-counting report, `candles_ago=0` for both); otherwise
    they are at different candles (`candles_ago` 1 vs 2), so no volume sharing should occur.
    The bullish pattern always sits at a third, distinct candle (`candles_ago=5`) so it never
    accidentally shares a volume-evidence id with either bearish arrangement.
    """
    bearish_time = "2026-07-10 15:45"
    star_start = "2026-07-10 15:15" if same_candle_bearish else "2026-07-10 14:15"
    star_time = bearish_time if same_candle_bearish else "2026-07-10 15:00"
    return [
        make_pattern_record(
            event=make_event(
                pattern_id="bullish_engulfing",
                pattern_name="Bullish Engulfing",
                pattern_family=PatternFamily.ENGULFING,
                bias="Bullish",
                base_score=50,
                detected_at="2026-07-10 14:00",
            ),
            candles_ago=5,
            volume_confirmed=True,
        ),
        make_pattern_record(
            event=make_event(
                pattern_id="bearish_engulfing",
                pattern_name="Bearish Engulfing",
                pattern_family=PatternFamily.ENGULFING,
                bias="Bearish",
                base_score=18,
                detected_at=bearish_time,
            ),
            candles_ago=0 if same_candle_bearish else 1,
            volume_confirmed=True,
        ),
        make_pattern_record(
            event=make_event(
                pattern_id="evening_star",
                pattern_name="Evening Star",
                pattern_family=PatternFamily.STAR,
                bias="Bearish",
                base_score=20,
                detected_at=star_time,
                pattern_start_at=star_start,
            ),
            candles_ago=0 if same_candle_bearish else 2,
            volume_confirmed=True,
        ),
    ]


# --- Issue 1: evidence-classification semantics -----------------------------------------


def test_neutral_bias_with_mixed_bullish_and_bearish_scored_evidence() -> None:
    """Required test 1/2/4: a Neutral overall bias with genuinely conflicting, nonzero
    scored evidence must report a nonzero score-contributing count, not zero it out.
    """
    result = _evaluate(_mixed_bias_patterns(same_candle_bearish=False))

    assert result["overall_bias"] == "Neutral"
    primary_patterns = [p for p in result["patterns"] if p.get("group_primary")]
    assert len(primary_patterns) == 3
    for pattern in primary_patterns:
        assert pattern["score_eligible"] is True
        assert abs(pattern["combined_score_contribution"]) > 0

    breakdown = result["structured_explanation"]["confidence_breakdown"]
    assert breakdown["conflict_present"] is True


def test_two_independent_analytical_families_are_not_over_merged() -> None:
    """Required test 6/7: a bullish and a bearish candlestick-reversal pattern at different
    timestamps are contradicting, not redundant, and must count as two independent families.
    """
    patterns = _mixed_bias_patterns(same_candle_bearish=False)
    bullish = patterns[0]
    bearish_engulfing = patterns[1]
    evening_star = patterns[2]

    # Same geometric bucket, opposite bias -> different independent-family keys.
    assert analytical_family(bullish) == analytical_family(bearish_engulfing) == "candlestick_reversal"
    assert independent_family_key(bullish) != independent_family_key(bearish_engulfing)

    # Same geometric bucket, same bias -> same independent-family key (still correlated).
    assert independent_family_key(bearish_engulfing) == independent_family_key(evening_star)

    result = _evaluate(patterns)
    breakdown = result["structured_explanation"]["confidence_breakdown"]
    assert breakdown["independent_analytical_families"] == 2


def test_two_genuinely_independent_families_structure_and_candlestick() -> None:
    """Required test 6: a structural pattern (breakdown) and a candlestick-reversal pattern
    of the same bias are still two independent families -- independence isn't only about bias.
    """
    patterns = [
        make_pattern_record(
            event=make_event(
                pattern_id="breakdown",
                pattern_name="Strong 20-Bar Breakdown",
                pattern_family=PatternFamily.BREAKOUT,
                bias="Bearish",
                base_score=26,
                detected_at="2026-07-10 15:45",
                relevant_prices={"breakdown_level": 98.0},
            ),
            candles_ago=0,
        ),
        make_pattern_record(
            event=make_event(
                pattern_id="bearish_engulfing",
                pattern_name="Bearish Engulfing",
                pattern_family=PatternFamily.ENGULFING,
                bias="Bearish",
                base_score=18,
                detected_at="2026-07-10 15:00",
            ),
            candles_ago=3,
        ),
    ]
    assert independent_family_key(patterns[0]) != independent_family_key(patterns[1])

    result = _evaluate(patterns, trend="Downtrend")
    breakdown = result["structured_explanation"]["confidence_breakdown"]
    assert breakdown["independent_analytical_families"] == 2


def test_two_pattern_families_mapped_to_same_dependency_group() -> None:
    """Required test 5: engulfing and star are different `pattern_family` values but the same
    analytical dependency group when same-biased, and this must be exposed for transparency.
    """
    patterns = _mixed_bias_patterns(same_candle_bearish=True)
    bearish_engulfing, evening_star = patterns[1], patterns[2]
    assert bearish_engulfing["pattern_family"] != evening_star["pattern_family"]
    assert independent_family_key(bearish_engulfing) == independent_family_key(evening_star)
    assert independent_family_key(bearish_engulfing) == ("candlestick_reversal", "Bearish")


# --- Issue 3: shared-candle volume attribution -------------------------------------------


def test_shared_candle_volume_is_deduplicated_in_aggregate_score() -> None:
    """Required test 8/9: two pattern labels completing on the same candle must not double the
    aggregate Volume Score.
    """
    result = _evaluate(_mixed_bias_patterns(same_candle_bearish=True))
    primary_by_name = {p["pattern_name"]: p for p in result["patterns"] if p.get("group_primary")}

    bearish_engulfing = primary_by_name["Bearish Engulfing"]
    evening_star = primary_by_name["Evening Star"]
    bullish_engulfing = primary_by_name["Bullish Engulfing"]

    # Same candle, same bias -> shared volume_evidence_id; only one keeps a nonzero contribution.
    assert bearish_engulfing["volume_evidence_id"] == evening_star["volume_evidence_id"]
    pair = [bearish_engulfing, evening_star]
    zeroed = next(p for p in pair if p["volume_score_contribution"] == 0.0)
    kept = next(p for p in pair if p["volume_score_contribution"] != 0.0)
    assert zeroed["raw_volume_score_contribution"] == kept["volume_score_contribution"]
    assert zeroed["volume_deduplication_reason"] is not None

    # Bullish engulfing is on a different candle -> untouched, independently attributed.
    assert bullish_engulfing["volume_score_contribution"] != 0.0
    assert bullish_engulfing["volume_deduplication_reason"] is None

    expected_volume_score = round(
        bullish_engulfing["volume_score_contribution"] + kept["volume_score_contribution"],
        2,
    )
    assert result["score"]["volume_score"] == expected_volume_score


def test_combined_event_contribution_reflects_deduplicated_volume() -> None:
    """Required test 10: Combined Event Contribution (canonical, display-facing) must use the
    deduplicated volume contribution, not the raw per-pattern one.
    """
    patterns = _mixed_bias_patterns(same_candle_bearish=True)
    result = _evaluate(patterns)
    primary_by_name = {p["pattern_name"]: p for p in result["patterns"] if p.get("group_primary")}
    zeroed = next(
        primary_by_name[name]
        for name in ("Bearish Engulfing", "Evening Star")
        if primary_by_name[name]["volume_score_contribution"] == 0.0
    )
    # The deduplicated pattern is still independently pattern-scored, so its combined
    # contribution should equal its pattern_score_contribution alone (volume portion is zero),
    # not pattern_score_contribution + the raw (pre-dedup) volume value.
    assert zeroed["combined_score_contribution"] == round(
        zeroed["pattern_score_contribution"] + zeroed["volume_score_contribution"], 2
    )
    assert zeroed["raw_volume_score_contribution"] != 0.0
    assert zeroed["combined_score_contribution"] != round(
        zeroed["pattern_score_contribution"] + zeroed["raw_volume_score_contribution"], 2
    )


def test_volume_confirmed_ratio_uses_independent_evidence_not_duplicated_patterns() -> None:
    """Required test 11: when the two same-candle patterns are volume-confirmed but a third,
    independent pattern (different candle) is not, the ratio must be based on unique volume
    evidence units (1 of 2), not raw pattern count (2 of 3).
    """
    patterns = _mixed_bias_patterns(same_candle_bearish=True)
    patterns[0]["volume_confirmed"] = False  # the independent bullish engulfing is NOT confirmed
    result = _evaluate(patterns)

    ratio = result["structured_explanation"]["confidence_breakdown"]["volume_confirmed_ratio"]
    # 2 independent volume-evidence units total (shared bearish candle + bullish candle);
    # only the shared bearish one is confirmed -> 1/2, not 2/3.
    assert ratio == 0.5


# --- Cluster / current-score exclusion still holds after these changes -------------------


def test_clustered_events_stay_excluded_from_current_score() -> None:
    """Required test 12: price-zone-clustered members must stay excluded from the current
    score (and thus cannot leak volume contribution back in) after the volume-dedup change.
    """
    patterns = [
        make_pattern_record(
            event=make_event(
                pattern_id="shooting_star",
                pattern_name="Shooting Star",
                pattern_family=PatternFamily.STAR,
                bias="Bearish",
                base_score=22,
                detected_at="2026-07-10 15:00",
                bar_start_at="2026-07-10 14:45",
                bar_end_at="2026-07-10 15:00",
                relevant_prices={"high": 105.0},
            ),
            candles_ago=2,
            volume_confirmed=True,
        ),
        make_pattern_record(
            event=make_event(
                pattern_id="shooting_star",
                pattern_name="Shooting Star",
                pattern_family=PatternFamily.STAR,
                bias="Bearish",
                base_score=22,
                detected_at="2026-07-10 15:45",
                bar_start_at="2026-07-10 15:30",
                bar_end_at="2026-07-10 15:45",
                relevant_prices={"high": 105.02},
            ),
            candles_ago=0,
            volume_confirmed=True,
        ),
    ]
    result = _evaluate(patterns, trend="Downtrend")
    cluster_suppressed = [p for p in result["patterns"] if p.get("cluster_suppressed")]
    assert len(cluster_suppressed) == 1
    member = cluster_suppressed[0]
    assert member["score_eligible"] is False
    assert member["combined_score_contribution"] == 0.0
