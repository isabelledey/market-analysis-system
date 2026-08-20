"""Dedicated scoring and explanation service for analysis output."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Any

import pandas as pd

from stock_pattern_model.config import ScoringConfig
from stock_pattern_model.datetime_utils import format_compact_display_datetime, interval_to_timedelta
from stock_pattern_model.domain import DataQualityReport, PatternScoreEligibility
from stock_pattern_model.session_utils import session_date_for_timestamp

EVENT_STATE_PRIORITY = {
    "new": 0,
    "retest_pending": 1,
    "retest_rejected": 2,
    "reclaimed": 3,
    "awaiting_confirmation": 4,
    "active": 5,
    "directionally_confirmed": 5,
    "retested": 6,
    "invalidated": 7,
    "failed_breakout": 8,
    "failed_breakdown": 9,
    "failed": 10,
    "expired": 11,
}

INELIGIBLE_STATES = {
    "awaiting_confirmation",
    "expired",
    "invalidated",
    "failed",
    "failed_breakout",
    "failed_breakdown",
    "reclaimed",
}

ACTIVE_SIGNAL_STATES = {
    "new",
    "active",
    "retested",
    "retest_pending",
    "retest_rejected",
    "directionally_confirmed",
}

# Event states in which a Stage-4 lower-wick rejection (Hammer / Bullish Pin Bar) that is
# context-validated but not yet directionally confirmed may act as a small, bounded bearish-score
# dampener rather than contributing nothing (see ScoringService._dampener_contribution).
PENDING_DAMPENER_STATES = {"new", "active", "retested"}

# Pattern families with a genuine, price-action-driven directional confirmation stage (as opposed
# to being fully validated by geometry + context alone, with nothing further to confirm).
_STAGED_CONFIRMATION_PATTERN_IDS = {"hammer", "bullish_pin_bar", "shooting_star"}
_STRUCTURAL_CONFIRMATION_PATTERN_IDS = {"double_top", "double_bottom"}
_BREAK_LEVEL_PATTERN_IDS = {"breakout", "breakdown"}


def _confirmation_fields(pattern: dict[str, Any]) -> dict[str, str]:
    """Stage 5: separate the single, ambiguous pattern status into four independent questions --
    "Status: confirmed" alone cannot tell a reader whether only the candle shape was validated,
    whether it occurred in an appropriate market context, whether subsequent price action actually
    confirmed the expected direction, or whether that move has continued since.

    - Geometry Status: the candle always satisfies its shape requirements once an event exists at
      all -- detectors never emit a PatternEvent for geometry that didn't match.
    - Context Status: only meaningful for pattern families with a distinct geometry-vs-context
      split (`event.context_quality`); "Not Applicable" for families validated holistically
      (engulfing, doji, star, inside bar, breakout/breakdown).
    - Directional Confirmation: "Not Required" unless the family has a genuine forward price-
      action confirmation stage (Hammer/Bullish Pin Bar/Shooting Star's Stage-4 dampener model, or
      Double Top/Bottom's neckline break). Merely failing to invalidate is "Pending", not
      "Confirmed" -- this is the exact ambiguity Stage 5 removes.
    - Follow-Through: what happened *after* directional confirmation (or, for breakout/breakdown,
      after the break itself, which is its own directional trigger).
    """
    pattern_id = str(pattern.get("pattern_id") or "")
    status = str(pattern.get("status") or "")
    event_state = str(pattern.get("event_state") or "")
    event = pattern.get("event")
    context_quality = str(getattr(event, "context_quality", pattern.get("context_quality", "unknown")))

    geometry_status = "Validated"

    # Only pattern families with a genuine, backward-looking context check (Stage 2/4's
    # quality_passed / near-support / near-resistance gate) have a meaningful Context Status.
    # Other detectors default their `context_quality` field to "geometry_only" too, but that is
    # just an unused static default for them, not an actual rejection decision -- labeling it
    # "Rejected" would misleadingly imply a context check was performed and failed.
    if pattern_id not in _STAGED_CONFIRMATION_PATTERN_IDS:
        context_status = "Not Applicable"
    elif context_quality == "validated":
        context_status = "Validated"
    elif context_quality == "geometry_only":
        context_status = "Rejected"
    else:
        context_status = "Not Applicable"

    if pattern_id in _STAGED_CONFIRMATION_PATTERN_IDS:
        if event_state == "directionally_confirmed":
            directional_confirmation = "Confirmed"
            follow_through = "Present"
        elif event_state == "invalidated":
            directional_confirmation = "Failed"
            follow_through = "Failed"
        elif pattern.get("dampener_eligible"):
            directional_confirmation = "Pending"
            follow_through = "Pending"
        else:
            directional_confirmation = "Not Required"
            follow_through = "Not Applicable"
    elif pattern_id in _STRUCTURAL_CONFIRMATION_PATTERN_IDS:
        if status == "confirmed":
            directional_confirmation = "Confirmed"
            follow_through = "Present"
        elif status == "failed":
            directional_confirmation = "Failed"
            follow_through = "Failed"
        elif event_state == "awaiting_confirmation":
            directional_confirmation = "Pending"
            follow_through = "Pending"
        else:
            directional_confirmation = "Not Required"
            follow_through = "Not Applicable"
    elif pattern_id in _BREAK_LEVEL_PATTERN_IDS:
        # The break itself is the directional trigger -- there is no separate confirmation stage
        # to await. Follow-through describes what happened afterward: a retest that held (Present),
        # a retest still unresolved or no retest yet (Pending), or a reclaim/failure (Failed).
        directional_confirmation = "Not Required"
        if event_state == "retest_rejected":
            follow_through = "Present"
        elif event_state in {"reclaimed", "failed_breakout", "failed_breakdown"}:
            follow_through = "Failed"
        elif event_state in {"new", "active", "retest_pending"}:
            follow_through = "Pending"
        else:
            follow_through = "Not Applicable"
    else:
        directional_confirmation = "Not Required"
        follow_through = "Not Applicable"

    return {
        "geometry_status": geometry_status,
        "context_status": context_status,
        "directional_confirmation": directional_confirmation,
        "follow_through": follow_through,
    }

PATTERN_SEMANTIC_SPECIFICITY = {
    "hammer": 40,
    "shooting_star": 35,
    "morning_star": 32,
    "evening_star": 32,
    "bullish_engulfing": 28,
    "bearish_engulfing": 28,
    "breakout": 26,
    "breakdown": 26,
    "double_top": 24,
    "double_bottom": 24,
    "bullish_pin_bar": 18,
    "inside_bar_failure": 16,
    "inside_bar": 12,
    "doji": 8,
}


def _event_index(pattern: dict[str, Any], field_name: str) -> int | None:
    value = pattern.get(field_name)
    if value is None:
        return None
    return int(value)


def _eligibility_anchor(pattern: dict[str, Any]) -> tuple[str, int | None]:
    event_state = str(pattern.get("event_state") or "")
    pattern_id = str(pattern.get("pattern_id") or "")

    if event_state == "retest_rejected":
        return "rejection", _event_index(pattern, "rejection_index")
    if event_state == "retest_pending":
        return "retest", _event_index(pattern, "retest_index")
    if event_state == "reclaimed":
        return "reclaimed", _event_index(pattern, "reclaimed_index")
    if event_state in {"failed", "failed_breakout", "failed_breakdown"}:
        return "failed", _event_index(pattern, "failed_index")
    if event_state == "invalidated":
        return "invalidated", _event_index(pattern, "invalidation_index")
    if event_state == "expired":
        return "expired", _event_index(pattern, "last_completed_candle_index")
    if pattern_id in {"double_top", "double_bottom"}:
        if pattern.get("status") == "tentative":
            return "setup_completion", _event_index(pattern, "setup_completion_index")
        confirmation_index = _event_index(pattern, "confirmation_index")
        if confirmation_index is not None:
            return "confirmation", confirmation_index
    return "detected", _event_index(pattern, "detected_index")


def pattern_max_age_bars(pattern: dict[str, Any], config: ScoringConfig) -> int:
    event_state = str(pattern.get("event_state") or "")
    if event_state == "awaiting_confirmation" or pattern.get("status") == "tentative":
        return config.tentative_pattern_max_age_bars

    pattern_id = str(pattern.get("pattern_id") or "")
    family = str(pattern.get("pattern_family") or "")
    if pattern_id in {"breakout", "breakdown"}:
        return config.breakout_pattern_max_age_bars
    if family in {"double_top", "double_bottom"}:
        return config.structural_pattern_max_age_bars
    if family in {"inside_bar", "inside_bar_failure"}:
        return config.consolidation_pattern_max_age_bars
    if family in {"pin_bar", "engulfing", "star", "doji"}:
        return config.reversal_pattern_max_age_bars
    return config.pattern_max_age_bars


def _anchor_timestamp(pattern: dict[str, Any], anchor_type: str) -> Any:
    """Best-available real timestamp for the eligibility anchor picked by _eligibility_anchor."""
    anchor_field_map = {
        "rejection": "rejection_at",
        "retest": "retest_at",
        "reclaimed": "reclaimed_at",
        "failed": "failed_at",
        "invalidated": "invalidated_at",
        "expired": "last_completed_candle_at",
    }
    field_name = anchor_field_map.get(anchor_type)
    if field_name is not None:
        return pattern.get(field_name)
    event = pattern.get("event")
    if anchor_type == "setup_completion":
        return getattr(event, "setup_completion_at", None)
    if anchor_type == "confirmation":
        return getattr(event, "confirmation_at", None)
    return getattr(event, "detected_at", None)


def evaluate_pattern_eligibility(
    pattern: dict[str, Any],
    config: ScoringConfig,
    interval: str,
) -> PatternScoreEligibility:
    anchor_type, anchor_index = _eligibility_anchor(pattern)
    detection_age = int(pattern.get("candles_ago", 0))
    detected_index = _event_index(pattern, "detected_index")
    last_completed_index = _event_index(pattern, "last_completed_candle_index")
    if anchor_index is None or detected_index is None or last_completed_index is None:
        age_bars = detection_age
    else:
        age_bars = max(0, last_completed_index - anchor_index)

    max_age_bars = pattern_max_age_bars(pattern, config)
    event_state = str(pattern.get("event_state") or "")
    status = str(pattern.get("status") or "")
    bias = str(pattern.get("bias") or "")
    pattern_id = str(pattern.get("pattern_id") or "")

    # Bar-index age alone doesn't know about overnight gaps: a pattern anchored in the
    # last few bars of a prior session can look just as "recent" as one from a few bars
    # into the current session. Require same trading-session date on top of the bar-age
    # cutoff so yesterday's late-session patterns can't leak into today's score.
    #
    # This only makes sense for intraday intervals, where a session spans many bars: for
    # daily/weekly intervals each bar *is* its own calendar-day "session", so this check
    # would disqualify every pattern except one from the single latest bar, regardless of
    # max_age_bars -- the bar-index age check below already correctly captures "how many
    # trading days/weeks ago" for those intervals, without needing a same-day gate on top.
    is_intraday = interval_to_timedelta(interval) < pd.Timedelta(days=1)
    if is_intraday:
        anchor_at = _anchor_timestamp(pattern, anchor_type)
        last_completed_at = pattern.get("last_completed_candle_at")
        if anchor_at is not None and last_completed_at is not None:
            exchange_timezone = getattr(pattern.get("event"), "exchange_timezone", None)
            if session_date_for_timestamp(anchor_at, exchange_timezone) != session_date_for_timestamp(
                last_completed_at, exchange_timezone
            ):
                return PatternScoreEligibility(
                    False, "outside current trading session", anchor_type, anchor_index, age_bars, max_age_bars
                )

    if pattern.get("dependency_suppressed"):
        return PatternScoreEligibility(False, "linked confirmation duplicate", anchor_type, anchor_index, age_bars, max_age_bars)
    if pattern.get("group_suppressed"):
        return PatternScoreEligibility(False, "overlap duplicate", anchor_type, anchor_index, age_bars, max_age_bars)
    if bias == "Neutral":
        return PatternScoreEligibility(False, "informational only", anchor_type, anchor_index, age_bars, max_age_bars)
    if event_state == "directionally_confirmed":
        if age_bars > max_age_bars:
            return PatternScoreEligibility(False, "outside scoring horizon", anchor_type, anchor_index, age_bars, max_age_bars)
        return PatternScoreEligibility(True, None, anchor_type, anchor_index, age_bars, max_age_bars)
    if event_state == "awaiting_confirmation":
        reason = "awaiting neckline confirmation" if pattern_id in {"double_top", "double_bottom"} else "awaiting directional confirmation"
        return PatternScoreEligibility(False, reason, anchor_type, anchor_index, age_bars, max_age_bars)
    if pattern.get("dampener_eligible") and event_state in PENDING_DAMPENER_STATES:
        if age_bars > max_age_bars:
            return PatternScoreEligibility(False, "outside scoring horizon", anchor_type, anchor_index, age_bars, max_age_bars)
        return PatternScoreEligibility(True, None, anchor_type, anchor_index, age_bars, max_age_bars)
    if event_state in INELIGIBLE_STATES:
        state_reason_map = {
            "expired": "expired",
            "invalidated": "invalidated",
            "failed": "failed pattern",
            "failed_breakout": "failed breakout",
            "failed_breakdown": "failed breakdown",
            "reclaimed": "level reclaimed",
        }
        return PatternScoreEligibility(False, state_reason_map[event_state], anchor_type, anchor_index, age_bars, max_age_bars)
    if status == "tentative":
        return PatternScoreEligibility(False, "unconfirmed structural pattern", anchor_type, anchor_index, age_bars, max_age_bars)
    if status == "failed":
        return PatternScoreEligibility(False, "failed pattern", anchor_type, anchor_index, age_bars, max_age_bars)
    if status == "expired":
        return PatternScoreEligibility(False, "expired", anchor_type, anchor_index, age_bars, max_age_bars)
    if age_bars > max_age_bars:
        return PatternScoreEligibility(False, "outside scoring horizon", anchor_type, anchor_index, age_bars, max_age_bars)
    return PatternScoreEligibility(True, None, anchor_type, anchor_index, age_bars, max_age_bars)


def build_event_id(pattern: dict[str, Any]) -> str:
    event = pattern["event"]
    return (
        f"{pattern['pattern_id']}:{pattern['status']}:"
        f"{event.detected_at.isoformat()}:{'-'.join(map(str, event.relevant_indices))}"
    )


def build_setup_id(pattern: dict[str, Any]) -> str:
    event = pattern["event"]
    return (
        f"{pattern['pattern_id']}:{event.pattern_start_at.isoformat()}:"
        f"{event.pattern_end_at.isoformat()}:{pattern['bias']}"
    )


def build_evidence_group(pattern: dict[str, Any]) -> str:
    event = pattern["event"]
    relevant_prices = event.relevant_prices
    if pattern["pattern_family"] in {"pin_bar", "doji", "star"}:
        return f"candlestick:{event.bar_start_at.isoformat()}"
    if pattern["pattern_id"] == "breakout":
        key_price = relevant_prices.get("breakout_level") or relevant_prices.get("confirmation_price") or 0.0
        return f"breakout:{event.detected_at.isoformat()}:{round(float(key_price), 2)}"
    if pattern["pattern_id"] == "breakdown":
        key_price = relevant_prices.get("breakdown_level") or relevant_prices.get("confirmation_price") or 0.0
        return f"breakdown:{event.detected_at.isoformat()}:{round(float(key_price), 2)}"
    if pattern["pattern_id"] in {"double_top", "double_bottom"}:
        setup_completion = event.setup_completion_at or event.pattern_end_at
        neckline = relevant_prices.get("neckline") or relevant_prices.get("confirmation_price") or 0.0
        return (
            f"structural:{pattern['pattern_id']}:{pattern['bias']}:"
            f"{setup_completion.isoformat()}:{round(float(neckline), 2)}"
        )
    if pattern["pattern_family"] == "engulfing":
        return (
            f"engulfing:{pattern['bias']}:{event.pattern_start_at.isoformat()}:"
            f"{event.pattern_end_at.isoformat()}"
        )
    if pattern["pattern_family"] in {"inside_bar", "inside_bar_failure"}:
        return f"inside_structure:{pattern['bias']}:{event.pattern_end_at.isoformat()}"
    key_price = (
        relevant_prices.get("confirmation_price")
        or relevant_prices.get("breakout_level")
        or relevant_prices.get("breakdown_level")
        or 0.0
    )
    return (
        f"{pattern['pattern_id']}:{pattern['bias']}:{event.detected_at.isoformat()}:"
        f"{round(float(key_price), 2)}"
    )


def _cluster_zone_price(pattern: dict[str, Any]) -> float | None:
    """A single representative price for the pattern's rejection/reversal zone.

    Used only to test whether two correlated events occurred in "the same price zone" for
    clustering purposes. Falls back across detector families since `relevant_prices` keys
    vary (pin bars expose high/low, stars expose star_close, structural patterns expose a
    neckline/break level, etc.).
    """
    # Only keys that represent a genuine tested price extreme (the candle's own rejection wick,
    # a structural neckline, or a break level) are used. "resistance_reference"/"support_reference"
    # (the level going *into* the candle, from Stage 2 context validation) are checked last -- they
    # describe context, not the point actually tested, and using them first would mismatch a
    # candle against its own retest (the retest's resistance_reference is the first candle's own
    # high, not the first candle's pre-breakout resistance_reference). Two-candle patterns like
    # engulfing and indecision patterns like doji have no single comparable "zone" price, so they
    # intentionally return None here and are simply never clustered -- a deliberate choice, not a gap.
    prices = pattern["event"].relevant_prices
    if pattern["bias"] == "Bearish":
        keys = ("high", "star_high", "neckline", "breakdown_level", "resistance_reference")
    elif pattern["bias"] == "Bullish":
        keys = ("low", "star_low", "neckline", "breakout_level", "support_reference")
    else:
        return None
    for key in keys:
        value = prices.get(key)
        if value is not None:
            return float(value)
    return None


def _cluster_bar_index(pattern: dict[str, Any]) -> int:
    event = pattern["event"]
    if event.detected_index is not None:
        return int(event.detected_index)
    return int(pattern.get("candles_ago", 0))


def cluster_type_label(family_bucket: str, bias: str) -> str:
    if family_bucket == "candlestick_reversal":
        if bias == "Bearish":
            return "Repeated Upper Rejection Zone"
        if bias == "Bullish":
            return "Repeated Lower Rejection Zone"
    return f"Repeated {bias} {family_bucket.replace('_', ' ').title()} Zone"


def analytical_family(pattern: dict[str, Any]) -> str:
    pattern_id = str(pattern.get("pattern_id") or "")
    family = str(pattern.get("pattern_family") or "")
    if pattern_id in {"breakout", "breakdown", "double_top", "double_bottom"}:
        return "structure"
    if family in {"pin_bar", "engulfing", "star", "doji"}:
        return "candlestick_reversal"
    if family in {"inside_bar", "inside_bar_failure"}:
        return "consolidation"
    return family or "other"


def independent_family_key(pattern: dict[str, Any]) -> tuple[str, str]:
    """Identify a pattern's independent analytical dependency group.

    `analytical_family` alone groups by geometric method (e.g. every candlestick-
    reversal shape collapses into one bucket) so repeated same-direction confirmations
    are treated as redundant rather than independent evidence. But bias must stay part
    of the key: a bullish and a bearish pattern from the same geometric family are
    contradicting, not redundant, and collapsing them together understates how many
    genuinely independent lines of evidence exist.
    """
    return (analytical_family(pattern), str(pattern.get("bias") or ""))


def resolved_pattern_sort_key(pattern: dict[str, Any]) -> tuple[object, ...]:
    pattern_name = str(pattern.get("pattern_name") or "")
    context_quality = str(
        pattern.get("context_quality")
        or getattr(pattern.get("event"), "context_quality", "unknown")
    )
    status = str(pattern.get("status") or "")
    bias = str(pattern.get("bias") or "")
    pattern_id = str(pattern.get("pattern_id") or "")
    generic_geometry_name = pattern_name.endswith("Rejection")
    return (
        0 if context_quality == "validated" else 1 if context_quality == "supported" else 2,
        0 if status == "confirmed" else 1 if status == "tentative" else 2 if status == "candidate" else 3,
        0 if bias in {"Bullish", "Bearish"} else 1,
        -PATTERN_SEMANTIC_SPECIFICITY.get(pattern_id, 0),
        0 if not generic_geometry_name else 1,
        int(pattern.get("priority", 99)),
        int(pattern.get("score_anchor_candles_ago", pattern.get("candles_ago", 0))),
        -abs(float(pattern.get("base_score", 0.0))),
        pattern_name,
    )


@dataclass(frozen=True)
class ScoringService:
    """Calculate signal scores, market state, and structured explanations."""

    config: ScoringConfig

    def evaluate(
        self,
        *,
        symbol: str,
        trend: str,
        trend_structure_score: float | None = None,
        trend_evidence: list[str] | None = None,
        trend_evidence_structured: list[dict[str, Any]] | None = None,
        trend_horizon: str | None = None,
        local_trend: str | None = None,
        local_trend_score: float | None = None,
        display_timezone: str = "Asia/Jerusalem",
        patterns: list[dict[str, Any]],
        quality_report: DataQualityReport,
        latest_close: float,
        latest_bar_start_display: str,
        latest_bar_end_display: str,
        interval: str,
        latest_volume_baseline_source: str,
    ) -> dict[str, Any]:
        enriched_patterns = self._enrich_patterns(patterns, interval)
        score_patterns = [pattern for pattern in enriched_patterns if pattern["score_eligible"]]
        score_groups = self._group_primary_patterns(score_patterns)
        primary_patterns = list(score_groups["primary_patterns"])
        suppressed_patterns = list(score_groups["suppressed_patterns"])

        primary_patterns, cluster_suppressed_patterns = self._apply_cluster_bounding(primary_patterns)
        suppressed_patterns.extend(cluster_suppressed_patterns)

        self._deduplicate_shared_candle_volume(primary_patterns)

        score = self._calculate_scores(trend, primary_patterns)
        market_state = self._classify_market_state(trend, primary_patterns, score)
        preliminary_bias = self._derive_overall_bias(primary_patterns, score)
        rule_confidence = self._calculate_rule_confidence(
            trend=trend,
            primary_patterns=primary_patterns,
            suppressed_patterns=suppressed_patterns,
            score=score,
            quality_report=quality_report,
            market_state=market_state,
        )
        overall_bias = (
            preliminary_bias
            if preliminary_bias == "Neutral" or rule_confidence >= self.config.minimum_bias_confidence
            else "Neutral"
        )
        structured_explanation = self._build_structured_explanation(
            symbol=symbol,
            trend=trend,
            trend_structure_score=trend_structure_score,
            trend_evidence=trend_evidence or [],
            trend_evidence_structured=trend_evidence_structured or [],
            trend_horizon=trend_horizon,
            local_trend=local_trend,
            local_trend_score=local_trend_score,
            market_state=market_state,
            overall_bias=overall_bias,
            primary_patterns=primary_patterns,
            suppressed_patterns=suppressed_patterns,
            quality_report=quality_report,
            latest_close=latest_close,
            latest_bar_start_display=latest_bar_start_display,
            latest_bar_end_display=latest_bar_end_display,
            interval=interval,
            latest_volume_baseline_source=latest_volume_baseline_source,
            display_timezone=display_timezone,
            score=score,
            rule_confidence=rule_confidence,
        )
        explanation = self._build_text_explanation(structured_explanation)

        ranked_patterns = sorted(
            enriched_patterns,
            key=lambda item: (
                self._status_rank(item["status"]),
                EVENT_STATE_PRIORITY.get(item["event_state"], 9),
                -abs(item["weighted_score"]),
                item["candles_ago"],
                item["priority"],
                item["pattern_name"],
            ),
        )
        return {
            "patterns": ranked_patterns,
            "score": score,
            "market_state": market_state,
            "overall_bias": overall_bias,
            "rule_confidence": rule_confidence,
            "structured_explanation": structured_explanation,
            "explanation": explanation,
        }

    def _enrich_patterns(self, patterns: list[dict[str, Any]], interval: str) -> list[dict[str, Any]]:
        enriched = [dict(pattern) for pattern in patterns]
        for pattern in enriched:
            event = pattern["event"]
            pattern["event_id"] = pattern.get("event_id") or build_event_id(pattern)
            pattern["setup_id"] = pattern.get("setup_id") or build_setup_id(pattern)
            pattern["evidence_group"] = pattern.get("evidence_group") or build_evidence_group(pattern)
            pattern["event_state"] = pattern.get("event_state") or self._base_event_state(pattern)
            pattern.update(_confirmation_fields(pattern))
            decision = evaluate_pattern_eligibility(pattern, self.config, interval)
            pattern["score_eligibility"] = decision.to_dict()
            pattern["score_anchor_type"] = decision.anchor_type
            pattern["score_anchor_index"] = decision.anchor_index
            pattern["score_anchor_candles_ago"] = decision.age_bars
            pattern["score_max_age_bars"] = decision.max_age_bars
            pattern["recency_weight"] = self._recency_weight(
                decision.age_bars,
                max_age_bars=decision.max_age_bars,
            )
            pattern["score_eligible"] = bool(pattern.get("score_eligible", True)) and decision.eligible
            pattern["weighted_score"] = round(abs(self._pattern_score_contribution(pattern)), 2)
            pattern["score_ineligibility_reason"] = decision.reason
            pattern["volume_score_contribution"] = 0.0
            pattern["pattern_score_contribution"] = 0.0
            pattern["combined_score_contribution"] = 0.0
            pattern["group_primary"] = False
            pattern["group_suppressed"] = False
            pattern["dependency_suppressed"] = False
            pattern["cluster_suppressed"] = False
            pattern["cluster_id"] = None
            pattern["cluster_type"] = None
            pattern["cluster_member_ids"] = []
            pattern["cluster_size"] = 1
            pattern["cluster_price_zone"] = None
            pattern["cluster_strongest_score"] = None
            pattern["cluster_repetition_bonus"] = None
            pattern["cluster_penalties_applied"] = []
            pattern["cluster_bounded_contribution"] = None
            pattern["raw_pattern_score_contribution"] = None
            pattern["event_detected_display"] = pattern.get("detected_at_display")
            pattern["event_timestamp"] = event.detected_at
            # Stable ID for the single candle whose volume was actually observed/tested, so
            # multiple pattern labels completing on that same candle (e.g. an engulfing pattern
            # and a star pattern that are not eligible for price-zone clustering) can be detected
            # as sharing one underlying volume fact rather than each independently "confirming" it.
            pattern["volume_evidence_id"] = f"candle:{_cluster_bar_index(pattern)}"
            pattern["raw_volume_score_contribution"] = None
            pattern["volume_deduplication_reason"] = None

        self._infer_structural_relationships(enriched)
        return enriched

    def _infer_structural_relationships(self, patterns: list[dict[str, Any]]) -> None:
        structural_to_trigger = {
            "double_top": "breakdown",
            "double_bottom": "breakout",
        }
        for pattern in patterns:
            trigger_id = structural_to_trigger.get(pattern["pattern_id"])
            if trigger_id is None or pattern.get("related_event_ids"):
                continue
            reference_level = self._relationship_level(pattern)
            if reference_level is None:
                continue
            for candidate in patterns:
                if candidate["pattern_id"] != trigger_id:
                    continue
                if candidate["event"].detected_at != pattern["event"].detected_at:
                    continue
                candidate_level = self._relationship_level(candidate)
                if candidate_level is None:
                    continue
                tolerance = max(abs(reference_level) * 0.01, 0.5)
                if abs(candidate_level - reference_level) > tolerance:
                    continue
                pattern["related_event_ids"] = [candidate["event_id"]]
                pattern["relationship_type"] = "confirmed_by"
                candidate["related_event_ids"] = [pattern["event_id"]]
                candidate["relationship_type"] = "confirms"
                candidate["confirms_pattern_id"] = pattern["event_id"]
                break

    def _relationship_level(self, pattern: dict[str, Any]) -> float | None:
        prices = pattern["event"].relevant_prices
        for key in ("neckline", "confirmation_price", "breakout_level", "breakdown_level"):
            value = prices.get(key)
            if value is not None:
                return float(value)
        return None

    def _group_primary_patterns(
        self,
        patterns: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        primary_patterns: list[dict[str, Any]] = []
        suppressed_patterns: list[dict[str, Any]] = []
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for pattern in patterns:
            groups[pattern["evidence_group"]].append(pattern)

        for group_patterns in groups.values():
            ranked_group = sorted(
                group_patterns,
                key=resolved_pattern_sort_key,
            )
            primary = ranked_group[0]
            primary["group_primary"] = True
            primary["pattern_score_contribution"] = round(self._pattern_score_contribution(primary), 2)
            primary["volume_score_contribution"] = round(self._volume_contribution(primary), 2)
            primary["combined_score_contribution"] = round(
                primary["pattern_score_contribution"] + primary["volume_score_contribution"],
                2,
            )
            primary_patterns.append(primary)

            for pattern in ranked_group[1:]:
                pattern["group_suppressed"] = True
                pattern["score_eligible"] = False
                pattern["score_ineligibility_reason"] = "overlap duplicate"
                pattern["score_eligibility"] = PatternScoreEligibility(
                    eligible=False,
                    reason="overlap duplicate",
                    anchor_type=str(pattern.get("score_anchor_type", "detected")),
                    anchor_index=pattern.get("score_anchor_index"),
                    age_bars=int(pattern.get("score_anchor_candles_ago", pattern["candles_ago"])),
                    max_age_bars=int(pattern.get("score_max_age_bars", self.config.pattern_max_age_bars)),
                ).to_dict()
                suppressed_patterns.append(pattern)

        primary_patterns, dependency_suppressed = self._apply_dependency_suppression(primary_patterns)
        suppressed_patterns.extend(dependency_suppressed)

        return {
            "primary_patterns": primary_patterns,
            "suppressed_patterns": suppressed_patterns,
        }

    def _apply_dependency_suppression(
        self,
        primary_patterns: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        by_event_id = {pattern["event_id"]: pattern for pattern in primary_patterns}
        retained: list[dict[str, Any]] = []
        suppressed: list[dict[str, Any]] = []
        visited: set[str] = set()

        for pattern in sorted(
            primary_patterns,
            key=lambda item: (
                -abs(self._raw_pattern_score(item)),
                item.get("score_anchor_candles_ago", item["candles_ago"]),
                item["pattern_name"],
            ),
        ):
            event_id = pattern["event_id"]
            if event_id in visited:
                continue
            cluster_ids = {event_id, *(pattern.get("related_event_ids") or [])}
            cluster = [by_event_id[item] for item in cluster_ids if item in by_event_id]
            if len(cluster) == 1:
                retained.append(pattern)
                visited.add(event_id)
                continue

            ranked_cluster = sorted(
                cluster,
                key=lambda item: (
                    item.get("relationship_type") != "confirmed_by",
                    -abs(self._raw_pattern_score(item)),
                    item.get("score_anchor_candles_ago", item["candles_ago"]),
                    item["pattern_name"],
                ),
            )
            winner = ranked_cluster[0]
            retained.append(winner)
            for item in ranked_cluster:
                visited.add(item["event_id"])
                if item["event_id"] == winner["event_id"]:
                    continue
                item["dependency_suppressed"] = True
                item["group_primary"] = False
                item["score_eligible"] = False
                item["score_ineligibility_reason"] = "linked confirmation duplicate"
                item["score_eligibility"] = PatternScoreEligibility(
                    eligible=False,
                    reason="linked confirmation duplicate",
                    anchor_type=str(item.get("score_anchor_type", "detected")),
                    anchor_index=item.get("score_anchor_index"),
                    age_bars=int(item.get("score_anchor_candles_ago", item["candles_ago"])),
                    max_age_bars=int(item.get("score_max_age_bars", self.config.pattern_max_age_bars)),
                ).to_dict()
                suppressed.append(item)

        retained_sorted = sorted(
            retained,
            key=lambda item: (
                item.get("score_anchor_candles_ago", item["candles_ago"]),
                item["priority"],
                item["pattern_name"],
            ),
        )
        return retained_sorted, suppressed

    def _cluster_correlated_patterns(
        self,
        primary_patterns: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        """Chain adjacent, correlated, same-direction events into rejection/reversal clusters.

        Two consecutive events (by bar index, within the same analytical family bucket and the
        same directional bias) join a cluster only if they are within `cluster_max_bar_distance`
        bars of each other, their rejection/reversal price zones overlap within
        `cluster_price_zone_tolerance`, and no confirmed opposite-bias event occurred between
        them (a "structural break" that would invalidate treating them as one continuous zone).
        Patterns without a usable zone price, or with a Neutral bias, are never clustered.
        """
        directional = [pattern for pattern in primary_patterns if pattern["bias"] in {"Bullish", "Bearish"}]
        # Patterns without a resolvable zone price (e.g. engulfing, doji) can never participate in
        # a cluster; excluding them up front keeps them from fragmenting a chain of patterns that
        # *do* share a zone (e.g. two shooting stars separated by one such pattern).
        clusterable = [pattern for pattern in directional if _cluster_zone_price(pattern) is not None]
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for pattern in clusterable:
            grouped[(analytical_family(pattern), pattern["bias"])].append(pattern)

        clusters: list[list[dict[str, Any]]] = []
        for group in grouped.values():
            ordered = sorted(group, key=_cluster_bar_index)
            current: list[dict[str, Any]] = []
            for pattern in ordered:
                if current and self._can_extend_cluster(current[-1], pattern, directional):
                    current.append(pattern)
                else:
                    if current:
                        clusters.append(current)
                    current = [pattern]
            if current:
                clusters.append(current)
        return clusters

    def _can_extend_cluster(
        self,
        previous: dict[str, Any],
        candidate: dict[str, Any],
        directional_patterns: list[dict[str, Any]],
    ) -> bool:
        previous_zone = _cluster_zone_price(previous)
        candidate_zone = _cluster_zone_price(candidate)
        if previous_zone is None or candidate_zone is None:
            return False

        bar_distance = abs(_cluster_bar_index(candidate) - _cluster_bar_index(previous))
        if bar_distance > self.config.cluster_max_bar_distance:
            return False

        reference = max(abs(previous_zone), abs(candidate_zone), 1e-6)
        if abs(previous_zone - candidate_zone) / reference > self.config.cluster_price_zone_tolerance:
            return False

        return not self._structural_break_between(previous, candidate, directional_patterns)

    def _structural_break_between(
        self,
        previous: dict[str, Any],
        candidate: dict[str, Any],
        directional_patterns: list[dict[str, Any]],
    ) -> bool:
        lower = min(_cluster_bar_index(previous), _cluster_bar_index(candidate))
        upper = max(_cluster_bar_index(previous), _cluster_bar_index(candidate))
        opposite_bias = "Bullish" if candidate["bias"] == "Bearish" else "Bearish"
        for other in directional_patterns:
            if other is previous or other is candidate:
                continue
            other_index = _cluster_bar_index(other)
            if lower < other_index < upper and other["bias"] == opposite_bias and other["status"] == "confirmed":
                return True
        return False

    def _apply_cluster_bounding(
        self,
        primary_patterns: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        clusters = self._cluster_correlated_patterns(primary_patterns)
        cluster_suppressed: list[dict[str, Any]] = []
        removed_event_ids: set[str] = set()

        for cluster in clusters:
            if len(cluster) < 2:
                continue

            representative = max(cluster, key=lambda item: abs(item["pattern_score_contribution"]))
            direction = 1.0 if representative["bias"] == "Bullish" else -1.0
            strongest_magnitude = abs(representative["pattern_score_contribution"])
            extra_members = len(cluster) - 1
            repetition_bonus = extra_members * self.config.cluster_repetition_bonus
            uncapped_magnitude = strongest_magnitude + repetition_bonus
            cap = strongest_magnitude * self.config.cluster_max_contribution_multiplier
            bounded_magnitude = min(uncapped_magnitude, cap)
            capped = uncapped_magnitude > cap
            bounded_contribution = round(direction * bounded_magnitude, 2)

            member_ids = [item["event_id"] for item in cluster]
            zone_prices = [zone for zone in (_cluster_zone_price(item) for item in cluster) if zone is not None]
            cluster_id = f"cluster:{'|'.join(sorted(member_ids))}"
            family_bucket = analytical_family(representative)
            cluster_metadata = {
                "cluster_id": cluster_id,
                "cluster_type": cluster_type_label(family_bucket, representative["bias"]),
                "cluster_member_ids": member_ids,
                "cluster_size": len(cluster),
                "cluster_price_zone": {
                    "low": round(min(zone_prices), 2) if zone_prices else None,
                    "high": round(max(zone_prices), 2) if zone_prices else None,
                },
                "cluster_strongest_score": round(direction * strongest_magnitude, 2),
                "cluster_repetition_bonus": round(direction * repetition_bonus, 2),
                "cluster_penalties_applied": ["capped_at_max_contribution"] if capped else [],
                "cluster_bounded_contribution": bounded_contribution,
            }

            representative["raw_pattern_score_contribution"] = representative["pattern_score_contribution"]
            representative["pattern_score_contribution"] = bounded_contribution
            representative.update(cluster_metadata)

            for member in cluster:
                if member is representative:
                    continue
                member["raw_pattern_score_contribution"] = member["pattern_score_contribution"]
                member["pattern_score_contribution"] = 0.0
                member["combined_score_contribution"] = 0.0
                member["cluster_suppressed"] = True
                member["group_primary"] = False
                member["score_eligible"] = False
                member["score_ineligibility_reason"] = "clustered correlated evidence"
                member["score_eligibility"] = PatternScoreEligibility(
                    eligible=False,
                    reason="clustered correlated evidence",
                    anchor_type=str(member.get("score_anchor_type", "detected")),
                    anchor_index=member.get("score_anchor_index"),
                    age_bars=int(member.get("score_anchor_candles_ago", member["candles_ago"])),
                    max_age_bars=int(member.get("score_max_age_bars", self.config.pattern_max_age_bars)),
                ).to_dict()
                member.update(cluster_metadata)
                removed_event_ids.add(member["event_id"])
                cluster_suppressed.append(member)

        remaining_primary = [
            pattern for pattern in primary_patterns if pattern["event_id"] not in removed_event_ids
        ]
        return remaining_primary, cluster_suppressed

    def _deduplicate_shared_candle_volume(self, primary_patterns: list[dict[str, Any]]) -> None:
        """Prevent the same candle's volume from being counted once per pattern label.

        Price-zone clustering (`_apply_cluster_bounding`) never sees two-candle patterns like
        engulfing, or indecision patterns like doji, because they have no comparable zone price
        (see `_cluster_zone_price`). That is correct for *pattern-score* redundancy, but it leaves
        a gap for *volume*: an engulfing pattern and a star pattern that complete on the exact same
        candle read the exact same `Volume_Strength` value and, with matching bias and recency,
        produce the exact identical volume contribution -- so summing both into the aggregate
        Volume Score treats one candle's volume as two independent confirmations. This groups
        primary patterns by (source candle, bias) and keeps the full volume contribution on only
        one representative per group, zeroing the rest while preserving their raw observation for
        diagnostics. Pattern-score contributions are untouched: each pattern label can still be
        independent evidence at the pattern-recognition level even though they share one candle's
        volume fact.
        """
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for pattern in primary_patterns:
            if pattern["bias"] not in {"Bullish", "Bearish"}:
                continue
            groups[(pattern["volume_evidence_id"], pattern["bias"])].append(pattern)

        for group in groups.values():
            if len(group) < 2:
                continue
            ranked = sorted(group, key=resolved_pattern_sort_key)
            representative = ranked[0]
            for pattern in ranked[1:]:
                pattern["raw_volume_score_contribution"] = pattern["volume_score_contribution"]
                if pattern["volume_score_contribution"] != 0.0:
                    pattern["volume_deduplication_reason"] = (
                        f"already represented by {representative['pattern_name']} on the same candle"
                    )
                pattern["volume_score_contribution"] = 0.0
                pattern["combined_score_contribution"] = round(
                    pattern["pattern_score_contribution"] + pattern["volume_score_contribution"],
                    2,
                )

    def _calculate_scores(
        self,
        trend: str,
        primary_patterns: list[dict[str, Any]],
    ) -> dict[str, float]:
        bullish_score = round(
            sum(
                max(pattern["pattern_score_contribution"], 0.0)
                for pattern in primary_patterns
            ),
            2,
        )
        bearish_score = round(
            sum(
                abs(min(pattern["pattern_score_contribution"], 0.0))
                for pattern in primary_patterns
            ),
            2,
        )
        pattern_score = round(bullish_score - bearish_score, 2)
        volume_score = round(
            sum(pattern["volume_score_contribution"] for pattern in primary_patterns),
            2,
        )
        trend_score = round(self._trend_score(trend), 2)
        net_signal_score = round(pattern_score + volume_score + trend_score, 2)
        return {
            "trend_score": trend_score,
            "pattern_score": pattern_score,
            "volume_score": volume_score,
            "bullish_pattern_score": bullish_score,
            "bearish_pattern_score": bearish_score,
            "bullish_score": bullish_score,
            "bearish_score": bearish_score,
            "net_signal_score": net_signal_score,
        }

    def _classify_market_state(
        self,
        trend: str,
        primary_patterns: list[dict[str, Any]],
        score: dict[str, float],
    ) -> str:
        active_patterns = [
            pattern
            for pattern in primary_patterns
            if pattern["event_state"] in ACTIVE_SIGNAL_STATES
        ]
        if not active_patterns:
            return "Trend Only" if trend != "Neutral" else "Neutral"

        latest_pattern = min(
            active_patterns,
            key=lambda item: item.get("score_anchor_candles_ago", item["candles_ago"]),
        )
        bullish_score = score["bullish_score"]
        bearish_score = score["bearish_score"]
        conflict_ratio = self._conflict_ratio(bullish_score, bearish_score)

        if bullish_score > 0 and bearish_score > 0 and conflict_ratio >= self.config.conflict_neutrality_ratio:
            return "Conflicted"
        if (
            latest_pattern["pattern_id"] == "breakout"
            and latest_pattern.get("score_anchor_candles_ago", latest_pattern["candles_ago"]) <= self.config.breakout_state_max_age_bars
        ):
            return "Breakout Attempt"
        if (
            latest_pattern["pattern_id"] == "breakdown"
            and latest_pattern.get("score_anchor_candles_ago", latest_pattern["candles_ago"]) <= self.config.breakout_state_max_age_bars
        ):
            if latest_pattern["event_state"] == "retest_pending":
                return "Bearish Continuation Under Retest"
            if trend == "Downtrend":
                return "Breakdown Attempt"
        if trend == "Downtrend" and any(
            pattern["pattern_id"] in {"bullish_pin_bar", "hammer", "double_bottom"} for pattern in active_patterns
        ):
            return "Bearish Trend with Bullish Reversal Attempt"
        if trend == "Uptrend" and any(
            pattern["pattern_id"] in {"shooting_star", "double_top"} for pattern in active_patterns
        ):
            return "Bullish Trend with Bearish Reversal Attempt"
        if trend == "Uptrend" and score["pattern_score"] > 0:
            return "Bullish Continuation"
        if trend == "Downtrend" and score["pattern_score"] < 0:
            return "Bearish Continuation"
        if (
            (trend == "Uptrend" and score["pattern_score"] < 0)
            or (trend == "Downtrend" and score["pattern_score"] > 0)
        ):
            return "Reversal Watch"
        if score["pattern_score"] > 0:
            return "Bullish Setup"
        if score["pattern_score"] < 0:
            return "Bearish Setup"
        return "Neutral"

    def _derive_overall_bias(
        self,
        primary_patterns: list[dict[str, Any]],
        score: dict[str, float],
    ) -> str:
        confirmed_patterns = [
            pattern
            for pattern in primary_patterns
            if pattern["status"] == "confirmed" and pattern["score_eligible"]
        ]
        directional_confirmed_patterns = [
            pattern
            for pattern in confirmed_patterns
            if pattern["bias"] in {"Bullish", "Bearish"}
        ]
        if not directional_confirmed_patterns:
            return "Neutral"
        if abs(score["net_signal_score"]) < self.config.bias_threshold:
            return "Neutral"
        if self._conflict_ratio(score["bullish_score"], score["bearish_score"]) >= self.config.conflict_neutrality_ratio:
            return "Neutral"
        if score["net_signal_score"] > 0:
            return "Bullish"
        if score["net_signal_score"] < 0:
            return "Bearish"
        return "Neutral"

    def _calculate_rule_confidence(
        self,
        *,
        trend: str,
        primary_patterns: list[dict[str, Any]],
        suppressed_patterns: list[dict[str, Any]],
        score: dict[str, float],
        quality_report: DataQualityReport,
        market_state: str,
    ) -> float:
        confirmed_patterns = [
            pattern
            for pattern in primary_patterns
            if pattern["status"] == "confirmed" and pattern["score_eligible"]
        ]
        if not confirmed_patterns:
            return 12.0 if trend != "Neutral" else 5.0

        recency_values = [pattern["recency_weight"] for pattern in confirmed_patterns]
        strength_values = [
            min(pattern["signal_strength"], 3.0) / 3.0
            for pattern in confirmed_patterns
        ]
        volume_ratio = self._independent_volume_confirmed_ratio(confirmed_patterns)
        canonical_groups = len({pattern["evidence_group"] for pattern in confirmed_patterns})
        independent_families = len({independent_family_key(pattern) for pattern in confirmed_patterns})
        agreement_bonus = min(canonical_groups, independent_families + 1) * 7.0
        confirmation_bonus = min(len(confirmed_patterns), 4) * 6.0
        recency_bonus = mean(recency_values) * 18.0
        strength_bonus = mean(strength_values) * 14.0
        volume_bonus = volume_ratio * 8.0
        trend_alignment_bonus = 0.0
        if (
            (trend == "Uptrend" and score["pattern_score"] > 0)
            or (trend == "Downtrend" and score["pattern_score"] < 0)
        ):
            trend_alignment_bonus = 8.0

        conflict_penalty = self._conflict_ratio(score["bullish_score"], score["bearish_score"]) * 20.0
        data_penalty = min(
            len(quality_report.warnings) * self.config.data_warning_confidence_penalty,
            25.0,
        )
        duplicate_penalty = len(suppressed_patterns) * self.config.duplicate_group_confidence_penalty
        family_penalty = max(0, len(confirmed_patterns) - independent_families) * 3.0
        concentration_penalty = max(0, canonical_groups - independent_families) * 4.0
        age_penalty = max(0.0, 10.0 * (1.0 - mean(recency_values)))
        trend_only_penalty = 12.0 if market_state == "Trend Only" else 0.0

        confidence = (
            18.0
            + agreement_bonus
            + confirmation_bonus
            + recency_bonus
            + strength_bonus
            + volume_bonus
            + trend_alignment_bonus
            - conflict_penalty
            - data_penalty
            - duplicate_penalty
            - family_penalty
            - concentration_penalty
            - age_penalty
            - trend_only_penalty
        )
        return round(max(5.0, min(100.0, confidence)), 1)

    def _build_structured_explanation(
        self,
        *,
        symbol: str,
        trend: str,
        trend_structure_score: float | None,
        trend_evidence: list[str],
        trend_evidence_structured: list[dict[str, Any]],
        trend_horizon: str | None,
        local_trend: str | None,
        local_trend_score: float | None,
        market_state: str,
        overall_bias: str,
        primary_patterns: list[dict[str, Any]],
        suppressed_patterns: list[dict[str, Any]],
        quality_report: DataQualityReport,
        latest_close: float,
        latest_bar_start_display: str,
        latest_bar_end_display: str,
        interval: str,
        latest_volume_baseline_source: str,
        display_timezone: str,
        score: dict[str, float],
        rule_confidence: float,
    ) -> dict[str, Any]:
        bullish_patterns = [
            pattern for pattern in primary_patterns
            if pattern["bias"] == "Bullish"
        ]
        bearish_patterns = [
            pattern for pattern in primary_patterns
            if pattern["bias"] == "Bearish"
        ]
        supporting_trend_evidence = [
            item["explanation"]
            for item in trend_evidence_structured
            if item.get("supports_composite_trend")
        ]
        conflicting_trend_evidence = [
            item["explanation"]
            for item in trend_evidence_structured
            if item.get("conflicts_with_composite_trend")
        ]
        neutral_trend_evidence = [
            item["explanation"]
            for item in trend_evidence_structured
            if not item.get("supports_composite_trend") and not item.get("conflicts_with_composite_trend")
        ]
        # Stage 8: derive cluster and dampener narrative directly from the structured cluster_*/
        # dampener_eligible fields already computed on each primary pattern -- never hard-coded.
        cluster_notes: list[str] = []
        seen_cluster_ids: set[str] = set()
        for pattern in primary_patterns:
            cluster_id = pattern.get("cluster_id")
            if not cluster_id or cluster_id in seen_cluster_ids or pattern.get("cluster_size", 1) < 2:
                continue
            seen_cluster_ids.add(cluster_id)
            zone = pattern.get("cluster_price_zone") or {}
            zone_text = (
                f" near {zone['low']:.2f}-{zone['high']:.2f}"
                if zone.get("low") is not None and zone.get("high") is not None
                else ""
            )
            cluster_notes.append(
                f"{pattern.get('cluster_type')}: {pattern.get('cluster_size')} correlated "
                f"{pattern['bias'].lower()} detections{zone_text} were treated as one cluster "
                f"contributing {pattern['pattern_score_contribution']:.2f} (not the sum of the "
                "individual raw detections)."
            )

        dampener_notes: list[str] = []
        for pattern in primary_patterns:
            if not pattern.get("dampener_eligible") or pattern.get("directional_confirmation") != "Pending":
                continue
            direction_word = "bullish" if pattern["bias"] == "Bullish" else "bearish"
            edge_word = "session low" if pattern["bias"] == "Bullish" else "session high/resistance"
            dampener_notes.append(
                f"{pattern['pattern_name']} near the {edge_word} shows a partial, unconfirmed "
                f"{direction_word} response; it has not yet received directional price or volume "
                f"confirmation, so its contribution is capped at {pattern['pattern_score_contribution']:.2f}."
            )

        trend_diverges = bool(
            local_trend
            and trend in {"Uptrend", "Downtrend"}
            and local_trend in {"Uptrend", "Downtrend"}
            and local_trend != trend
        )

        conflicts: list[str] = []
        if bullish_patterns and bearish_patterns:
            conflicts.append(
                "Bullish and bearish confirmed evidence were both present, so the net signal was tempered."
            )
        if trend_diverges:
            conflicts.append(
                f"The broader trend ({trend}) and the local session trend ({local_trend}) point in "
                "opposite directions, so the near-term read follows the local session rather than "
                "the broader trend."
            )
        conflicts.extend(cluster_notes)
        conflicts.extend(dampener_notes)
        if suppressed_patterns:
            overlap_count = sum(1 for pattern in suppressed_patterns if pattern.get("group_suppressed"))
            dependency_count = sum(1 for pattern in suppressed_patterns if pattern.get("dependency_suppressed"))
            if overlap_count:
                conflicts.append(
                    f"{overlap_count} overlapping candle label(s) were grouped into shared canonical candle events to avoid double counting."
                )
            if dependency_count:
                conflicts.append(
                    f"{dependency_count} linked structural setup(s) or confirmation trigger(s) were kept as separate events, and dependency-aware scoring prevented double counting."
                )

        if overall_bias == "Bullish":
            reason_for_bias = (
                "Bullish confirmed evidence outweighed bearish evidence after recency, volume, and trend context were applied."
            )
        elif overall_bias == "Bearish":
            reason_for_bias = (
                "Bearish confirmed evidence outweighed bullish evidence after recency, volume, and trend context were applied."
            )
        elif bullish_patterns or bearish_patterns:
            lean = ""
            net_signal_score = score.get("net_signal_score", 0.0)
            if net_signal_score < 0:
                lean = " with a slight short-term bearish lean"
            elif net_signal_score > 0:
                lean = " with a slight short-term bullish lean"
            reason_for_bias = (
                "Confirmed evidence existed, but the net signal stayed too balanced or too weak to "
                f"justify a directional bias{lean}."
            )
        elif trend != "Neutral":
            reason_for_bias = (
                "The trend remained directional, but no recent confirmed pattern added enough fresh evidence to move the bias away from neutral."
            )
        else:
            reason_for_bias = "No recent confirmed pattern created a directional edge."

        confidence_reasons: list[str] = []
        if primary_patterns:
            canonical_groups = len({pattern["evidence_group"] for pattern in primary_patterns})
            independent_families = len({independent_family_key(pattern) for pattern in primary_patterns})
            confidence_reasons.append(
                f"{canonical_groups} canonical scored group(s) resolved into {independent_families} independent analytical family/families."
            )
            if canonical_groups > independent_families:
                dependency_labels = sorted(
                    {
                        f"{pattern['pattern_name']} -> {analytical_family(pattern)}:{pattern['bias']}"
                        for pattern in primary_patterns
                    }
                )
                confidence_reasons.append(
                    "Dependency groups: " + "; ".join(dependency_labels) + "."
                )
        if quality_report.warnings:
            confidence_reasons.append("Data-quality warnings reduced confidence.")
        if bullish_patterns and bearish_patterns:
            confidence_reasons.append("Conflicting evidence reduced confidence.")
        if not primary_patterns and trend != "Neutral":
            confidence_reasons.append("Trend-only output keeps confidence low because it lacks fresh confirmed patterns.")
        if latest_volume_baseline_source == "rolling_20":
            confidence_reasons.append("Rolling volume baseline was used because time-of-day history was limited.")
        if dampener_notes:
            confidence_reasons.append(
                "Confidence stays moderate because the pending signal(s) above still lack volume "
                "or directional price-structure confirmation."
            )
        if not confidence_reasons:
            confidence_reasons.append("The score reflects rule strength only and is not statistically calibrated.")

        trend_clause = f"Broad trend: {trend}."
        if trend_structure_score is not None:
            trend_clause = f"Broad trend: {trend} (score {trend_structure_score:.2f})."
        if trend_horizon:
            trend_clause = f"{trend_clause} Trend horizon: {trend_horizon}."
        if local_trend:
            local_trend_clause = f" Local session trend: {local_trend}."
            if local_trend_score is not None:
                local_trend_clause = f" Local session trend: {local_trend} (score {local_trend_score:.2f})."
            trend_clause = f"{trend_clause}{local_trend_clause}"

        summary = (
            f"{symbol} last traded at {latest_close:.2f} on the completed {interval} candle from "
            f"{latest_bar_start_display} to {latest_bar_end_display}. {trend_clause} "
            f"Market state: {market_state}. Overall bias: {overall_bias}. "
            f"Net signal score: {score['net_signal_score']:.2f}. Rule confidence: {rule_confidence:.1f}/100."
        )
        confidence_breakdown = {
            "confirmed_score_eligible_groups": len(primary_patterns),
            "independent_analytical_families": len({independent_family_key(pattern) for pattern in primary_patterns}),
            "conflict_present": bool(bullish_patterns and bearish_patterns),
            "volume_confirmed_ratio": round(
                self._independent_volume_confirmed_ratio(primary_patterns),
                2,
            ),
            "quality_warning_count": len(quality_report.warnings),
            "uncalibrated": True,
        }
        return {
            "summary": summary,
            "trend_evidence": trend_evidence,
            "trend_evidence_structured": trend_evidence_structured,
            "supporting_trend_evidence": supporting_trend_evidence,
            "conflicting_trend_evidence": conflicting_trend_evidence,
            "neutral_trend_evidence": neutral_trend_evidence,
            "bullish_evidence": [
                self._format_evidence_line(pattern, display_timezone=display_timezone)
                for pattern in bullish_patterns[:3]
            ],
            "bearish_evidence": [
                self._format_evidence_line(pattern, display_timezone=display_timezone)
                for pattern in bearish_patterns[:3]
            ],
            "conflicts": conflicts,
            "cluster_notes": cluster_notes,
            "dampener_notes": dampener_notes,
            "trend_diverges": trend_diverges,
            "data_warnings": list(quality_report.warnings),
            "reason_for_bias": reason_for_bias,
            "reason_for_confidence": " ".join(confidence_reasons),
            "confidence_breakdown": confidence_breakdown,
        }

    def _build_text_explanation(self, structured_explanation: dict[str, Any]) -> str:
        parts = [structured_explanation["summary"]]
        if structured_explanation.get("trend_evidence"):
            parts.append("Trend evidence: " + "; ".join(structured_explanation["trend_evidence"]) + ".")
        if structured_explanation["bullish_evidence"]:
            parts.append("Bullish evidence: " + "; ".join(structured_explanation["bullish_evidence"]) + ".")
        if structured_explanation["bearish_evidence"]:
            parts.append("Bearish evidence: " + "; ".join(structured_explanation["bearish_evidence"]) + ".")
        if structured_explanation["conflicts"]:
            parts.append("Conflicts: " + "; ".join(structured_explanation["conflicts"]) + ".")
        if structured_explanation["data_warnings"]:
            parts.append("Data warnings: " + "; ".join(structured_explanation["data_warnings"]) + ".")
        parts.append("Bias rationale: " + structured_explanation["reason_for_bias"])
        parts.append(
            "Confidence rationale: "
            + structured_explanation["reason_for_confidence"]
            + " This is an uncalibrated rule-strength score, not a probability."
        )
        return " ".join(parts)

    def _format_evidence_line(self, pattern: dict[str, Any], *, display_timezone: str) -> str:
        detected_at = format_compact_display_datetime(pattern["event"].detected_at, display_timezone)
        # Deliberately never renders bare "[confirmed]" for a directional signal: geometry and
        # context validation are not the same claim as price action having confirmed the
        # direction (Stage 5), so directional confirmation is always spelled out explicitly.
        return (
            f"{pattern['pattern_name']} [geometry: {pattern['geometry_status']}, "
            f"context: {pattern['context_status']}, "
            f"directional confirmation: {pattern['directional_confirmation']}, "
            f"follow-through: {pattern['follow_through']}] "
            f"detected at {detected_at} with {pattern['detection_reason']}"
        )

    def _base_event_state(self, pattern: dict[str, Any]) -> str:
        # Any TENTATIVE pattern -- not just double top/bottom -- is inherently "awaiting
        # confirmation": the geometry/context is in place, but the forward price action that
        # would confirm or invalidate it hasn't happened yet (or hasn't been re-evaluated).
        if pattern["status"] == "tentative":
            return "awaiting_confirmation"
        status = pattern["status"]
        if status == "candidate":
            if pattern["candles_ago"] > self.config.state_expiration_bars:
                return "expired"
            return "new" if pattern["candles_ago"] <= 1 else "active"
        if status == "failed":
            return "failed"
        if status == "expired" or pattern["candles_ago"] > pattern_max_age_bars(pattern, self.config):
            return "expired"
        if pattern["candles_ago"] <= 1:
            return "new"
        return "active"

    def _raw_pattern_score(self, pattern: dict[str, Any]) -> float:
        direction = 1.0 if pattern["bias"] == "Bullish" else -1.0 if pattern["bias"] == "Bearish" else 0.0
        multiplier = 1.0
        if pattern["strong_signal"]:
            multiplier *= self.config.strong_signal_multiplier
        if pattern["status"] == "tentative":
            multiplier *= self.config.tentative_signal_multiplier
        return direction * pattern["base_score"] * pattern["recency_weight"] * multiplier

    def _is_pending_dampener(self, pattern: dict[str, Any]) -> bool:
        return bool(pattern.get("dampener_eligible")) and str(pattern.get("event_state")) in PENDING_DAMPENER_STATES

    def _dampener_contribution(self, pattern: dict[str, Any]) -> float:
        """Small, configurable, capped nudge for a context-validated but not-yet-directionally-
        confirmed lower-wick rejection (Stage 4). Deliberately bypasses `_raw_pattern_score` (which
        scales with `base_score`) so this can never approach a directionally confirmed signal's
        contribution -- it is bounded purely by `unconfirmed_rejection_dampener`.
        """
        direction = 1.0 if pattern["bias"] == "Bullish" else -1.0 if pattern["bias"] == "Bearish" else 0.0
        return direction * self.config.unconfirmed_rejection_dampener * pattern["recency_weight"]

    def _pattern_score_contribution(self, pattern: dict[str, Any]) -> float:
        if self._is_pending_dampener(pattern):
            return self._dampener_contribution(pattern)
        return self._raw_pattern_score(pattern)

    def _volume_contribution(self, pattern: dict[str, Any]) -> float:
        if not pattern["volume_confirmed"]:
            return 0.0
        if self._is_pending_dampener(pattern):
            # The dampener is the one, well-understood small effect for an unconfirmed rejection;
            # stacking the full volume bonus on top of it would make the "small, capped" dampener
            # unbounded in practice.
            return 0.0
        direction = 1.0 if pattern["bias"] == "Bullish" else -1.0 if pattern["bias"] == "Bearish" else 0.0
        return direction * self.config.volume_confirmation_bonus * pattern["recency_weight"]

    def _recency_weight(self, candles_ago: int, *, max_age_bars: int | None = None) -> float:
        effective_max_age = self.config.pattern_max_age_bars if max_age_bars is None else max_age_bars
        if candles_ago > effective_max_age:
            return 0.0
        return round(self.config.recency_decay ** candles_ago, 4)

    def _trend_score(self, trend: str) -> float:
        if trend == "Uptrend":
            return self.config.trend_score_weight
        if trend == "Downtrend":
            return -self.config.trend_score_weight
        return 0.0

    def _conflict_ratio(self, bullish_score: float, bearish_score: float) -> float:
        if bullish_score <= 0 or bearish_score <= 0:
            return 0.0
        return min(bullish_score, bearish_score) / max(bullish_score, bearish_score)

    def _independent_volume_confirmed_ratio(self, patterns: list[dict[str, Any]]) -> float:
        """Fraction of independent volume evidence (one unit per shared candle) that is confirmed.

        Counting `volume_confirmed` per pattern label would count one candle's volume once for
        every correlated pattern detected on it (see `_deduplicate_shared_candle_volume`). This
        counts unique `volume_evidence_id`s instead, so two pattern labels sharing one candle
        contribute one unit of evidence to both the numerator and denominator, not two.
        """
        if not patterns:
            return 0.0
        all_ids = {pattern["volume_evidence_id"] for pattern in patterns}
        confirmed_ids = {
            pattern["volume_evidence_id"] for pattern in patterns if pattern["volume_confirmed"]
        }
        return len(confirmed_ids) / len(all_ids) if all_ids else 0.0

    def _status_rank(self, status: str) -> int:
        return {
            "confirmed": 0,
            "candidate": 1,
            "tentative": 2,
            "failed": 3,
            "expired": 4,
        }.get(status, 9)
