"""Formatting helpers for CLI text and JSON output."""

from __future__ import annotations

import json
from typing import Any

_PATTERN_ENTRY_MOVE_LABELS = {
    "Uptrend": "Advance",
    "Downtrend": "Decline",
    "Sideways": "Sideways",
    "Ambiguous": "Ambiguous",
    "Not Applicable": None,
}


def _pattern_entry_move_display(pattern_entry_trend: str | None) -> str | None:
    if not pattern_entry_trend:
        return None
    return _PATTERN_ENTRY_MOVE_LABELS.get(pattern_entry_trend, pattern_entry_trend)


def _append_transition_lines(lines: list[str], pattern: dict[str, Any], *, indent: str) -> None:
    for label, key in (
        ("State Updated", "state_updated_at_display"),
        ("Retest Time", "retest_at_display"),
        ("Rejection Time", "rejection_at_display"),
        ("Reclaimed Time", "reclaimed_at_display"),
        ("Failed Time", "failed_at_display"),
        ("Invalidated Time", "invalidated_at_display"),
        ("Expired Time", "expired_at_display"),
    ):
        value = pattern.get(key)
        if value:
            lines.append(f"{indent}{label}: {value}")


def format_analysis_json(result: dict[str, Any]) -> str:
    """Render a result dictionary as pretty JSON."""
    return json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True)


def format_analysis_text(
    result: dict[str, Any],
    include_all_patterns: bool = False,
    pattern_history_mode: str = "session",
    history_limit: int | None = None,
) -> str:
    """Render an analysis result as human-readable text."""
    instrument = result.get("instrument", {})
    current_patterns = result.get("current_relevant_patterns") or []
    session_history = result.get("session_pattern_history") or []
    current_contributing = result.get("current_contributing_evidence") or []
    awaiting_confirmation = result.get("awaiting_confirmation_evidence") or []
    directionally_conflicting = result.get("directionally_conflicting_scored_evidence") or []
    current_neutral = result.get("current_neutral_evidence") or []
    historical_lifecycle = result.get("historical_lifecycle_events") or []
    recent_non_contributing = result.get("recent_non_contributing_tracked_events") or []
    historical_summary = result.get("historical_lifecycle_summary") or {}
    raw_pattern_key = "all_detected_patterns" if include_all_patterns else "top_patterns"
    raw_patterns = result.get(raw_pattern_key, [])
    if history_limit is not None:
        session_history = session_history[:history_limit]
    total_history = result.get("session_history_total", len(session_history))
    shown_history = len(session_history)
    history_count_suffix = "" if total_history == shown_history else f" of {total_history}"
    lines = [
        f"Instrument: {instrument.get('symbol', result.get('symbol', 'UNKNOWN'))}",
        f"Input Identifier: {instrument.get('input_identifier', result.get('symbol', 'UNKNOWN'))}",
        f"Resolved Symbol: {result.get('symbol', instrument.get('symbol', 'UNKNOWN'))}",
        f"Security Number: {instrument.get('security_number') or 'None'}",
        f"Name: {instrument.get('name') or 'Unknown'}",
        f"Exchange: {instrument.get('exchange') or 'Unknown'}",
        f"Exchange Calendar: {result.get('exchange_calendar') or 'Unknown'}",
        f"Currency: {instrument.get('currency') or 'Unknown'}",
        f"Interval: {result.get('interval', 'Unknown')}",
        f"Analysis Time: {result.get('analysis_time', result.get('as_of', 'Unknown'))}",
        f"Exchange Timezone: {result.get('exchange_timezone') or 'Unknown'}",
        f"Display Timezone: {result.get('display_timezone') or 'Unknown'}",
        f"Session Mode: {result.get('session_mode') or 'Unknown'}",
        f"Included Segments: {', '.join(result.get('included_segments', [])) or 'Unknown'}",
        f"Excluded Segments: {', '.join(result.get('excluded_segments', [])) or 'None'}",
        f"Latest Completed Candle Start: {result.get('latest_bar_start', 'Unknown')}",
        f"Latest Completed Candle End: {result.get('latest_bar_end', 'Unknown')}",
        f"Latest Close: {result.get('latest_close', 'Unknown')}",
        f"Data Quality: {result.get('data_quality_report', {}).get('completed_row_count', 'Unknown')} "
        f"completed rows / {result.get('data_quality_report', {}).get('row_count', 'Unknown')} total rows",
        f"Trend: {result.get('trend', 'Unknown')}",
        f"Broad Trend: {result.get('broad_trend', result.get('trend', 'Unknown'))}",
        f"Local Session Trend: {result.get('local_trend', 'Unknown')} "
        f"({result.get('local_trend_score', 'Unknown')}, "
        f"lookback {result.get('local_trend_lookback_bars', 'Unknown')} bars)",
        f"Trend Horizon: {result.get('trend_horizon', 'Unknown')}",
        f"Market State: {result.get('market_state', 'Unknown')}",
        f"Overall Bias: {result.get('overall_bias', 'Unknown')}",
        f"Bullish Pattern Score: {result.get('bullish_pattern_score', result.get('bullish_score', 'Unknown'))}",
        f"Bearish Pattern Score: {result.get('bearish_pattern_score', result.get('bearish_score', 'Unknown'))}",
        f"Rule Confidence: {result.get('rule_confidence', 'Unknown')}",
        f"Trend Score: {result.get('trend_score', 'Unknown')}",
        f"Trend Signal Contribution: {result.get('trend_signal_score', 'Unknown')}",
        f"Pattern Score: {result.get('pattern_score', 'Unknown')}",
        f"Volume Score: {result.get('volume_score', 'Unknown')}",
        f"Net Signal Score: {result.get('net_signal_score', 'Unknown')}",
        f"Short-Term Trend: {result.get('short_term_trend', 'Unknown')} ({result.get('short_term_trend_score', 'Unknown')})",
        f"Medium-Term Trend: {result.get('medium_term_trend', 'Unknown')} ({result.get('medium_term_trend_score', 'Unknown')})",
        f"Long-Term Trend: {result.get('long_term_trend', 'Unknown')} ({result.get('long_term_trend_score', 'Unknown')})",
        f"Current Score-Contributing Evidence Count: {len(current_contributing)}",
        f"Score-Contributing Bullish Evidence Count: {result.get('score_contributing_bullish_count', 0)}",
        f"Score-Contributing Bearish Evidence Count: {result.get('score_contributing_bearish_count', 0)}",
        (
            "Bias-Aligned Evidence Count: "
            f"{result.get('bias_aligned_evidence_count') if result.get('bias_aligned_evidence_count') is not None else 'Not Applicable'}"
        ),
        f"Directional Conflict Present: {'Yes' if result.get('directional_conflict_present') else 'No'}",
        f"Awaiting Confirmation Count: {len(awaiting_confirmation)}",
        f"Directionally Conflicting Scored Evidence Count: {len(directionally_conflicting)}",
        f"Current Neutral Evidence Count: {len(current_neutral)}",
        f"Recent Non-Contributing Tracked Event Count: {len(recent_non_contributing)}",
        f"Archived Lifecycle Event Count: {len(historical_lifecycle)}",
    ]

    def append_event_section(title: str, patterns: list[dict[str, Any]]) -> None:
        lines.append(f"{title} ({len(patterns)}):")
        if not patterns:
            lines.append("  None")
            return
        for pattern in patterns:
            labels = ", ".join(pattern.get("pattern_labels", [pattern.get("primary_pattern_name", "Unknown")]))
            lines.append(f"  Name: {pattern.get('primary_pattern_name', 'Unknown')}")
            lines.append(f"  Pattern Labels: {labels}")
            if pattern.get("raw_geometry_labels"):
                lines.append(f"  Raw Geometry Labels: {', '.join(pattern['raw_geometry_labels'])}")
            lines.append(f"  Family: {pattern.get('family', 'unknown')}")
            lines.append(f"  State: {pattern.get('state', 'unknown')}")
            lines.append(f"  Status: {pattern.get('status', 'unknown')}")
            # Status alone is ambiguous ("confirmed" could mean geometry, context, or direction
            # was confirmed). These four fields spell out exactly which claim is being made.
            lines.append(f"  Geometry Status: {pattern.get('geometry_status', 'Validated')}")
            lines.append(f"  Context Status: {pattern.get('context_status', 'Not Applicable')}")
            lines.append(f"  Directional Confirmation: {pattern.get('directional_confirmation', 'Not Required')}")
            lines.append(f"  Follow-Through: {pattern.get('follow_through', 'Not Applicable')}")
            lines.append(f"  Bias: {pattern.get('bias', 'Unknown')}")
            lines.append(f"  Geometry: {pattern.get('geometry_label', 'unknown')}")
            lines.append(f"  Context Quality: {pattern.get('context_quality', 'unknown')}")
            pre_pattern_move = _pattern_entry_move_display(pattern.get("pattern_entry_trend"))
            if pre_pattern_move:
                lines.append(f"  Immediate Pre-Pattern Move: {pre_pattern_move}")
            if pattern.get("dampener_eligible") or pattern.get("rejection_confirmation_state") not in (None, "not_applicable"):
                lines.append(f"  Rejection Confirmation State: {pattern.get('rejection_confirmation_state')}")
                # dampener_eligible is a static, detection-time flag (true once context-validated)
                # and stays true even after later directional confirmation -- only show the
                # dampener note while it is actually the mechanism driving the contribution
                # (directional_confirmation == "Pending"), not just because it was once eligible.
                if pattern.get("dampener_eligible") and pattern.get("directional_confirmation") == "Pending":
                    # The dampener nudges the *opposite* side of the pattern's own bias: a Bullish
                    # pattern (e.g. Hammer) dampens the Bearish score, while a Bearish pattern
                    # (e.g. Shooting Star) dampens the Bullish score. Must match this pattern's own
                    # bias, not be hardcoded -- see the equivalent, correctly bias-aware wording in
                    # ScoringService's dampener_notes.
                    dampened_side = "Bearish" if pattern.get("bias") == "Bullish" else "Bullish"
                    lines.append(f"  {dampened_side}-Score Dampener: Active (unconfirmed, bounded)")
                if pattern.get("confirmed_at"):
                    lines.append(f"  Directionally Confirmed At: {pattern['confirmed_at']}")
            if pattern.get("cluster_id"):
                lines.append(f"  Cluster: {pattern.get('cluster_type')} ({pattern.get('cluster_size')} members)")
                lines.append(f"  Cluster Strongest Score: {pattern.get('cluster_strongest_score')}")
                lines.append(f"  Cluster Repetition Bonus: {pattern.get('cluster_repetition_bonus')}")
                lines.append(f"  Cluster Bounded Contribution: {pattern.get('cluster_bounded_contribution')}")
                if pattern.get("cluster_penalties_applied"):
                    lines.append(f"  Cluster Penalties Applied: {', '.join(pattern['cluster_penalties_applied'])}")
            lines.append(f"  Pattern Start: {pattern.get('pattern_start_display', 'Unknown')}")
            lines.append(f"  Setup Completion: {pattern.get('setup_completion_display', pattern.get('pattern_completion_display', 'Unknown'))}")
            lines.append(f"  Pattern Completion: {pattern.get('pattern_completion_display', 'Unknown')}")
            lines.append(f"  Detected at: {pattern.get('detected_at_display', 'Unknown')}")
            if pattern.get("confirmation_at_display"):
                lines.append(f"  Confirmation Time: {pattern['confirmation_at_display']}")
            _append_transition_lines(lines, pattern, indent="  ")
            lines.append(f"  Display Timezone: {pattern.get('display_timezone', 'Unknown')}")
            lines.append(f"  Signal Strength: {pattern.get('signal_strength', 'Unknown')}")
            lines.append(f"  Pattern Score Contribution: {pattern.get('pattern_score_contribution', 0.0)}")
            lines.append(f"  Volume Score Contribution (Applied): {pattern.get('volume_score_contribution', 0.0)}")
            raw_volume = pattern.get("raw_volume_score_contribution", pattern.get("volume_score_contribution", 0.0))
            if raw_volume != pattern.get("volume_score_contribution", 0.0):
                lines.append(f"  Raw Volume Contribution: {raw_volume}")
            if pattern.get("volume_evidence_id"):
                lines.append(f"  Volume Evidence ID: {pattern['volume_evidence_id']}")
            if pattern.get("volume_deduplication_reason"):
                lines.append(f"  Volume Deduplication Reason: {pattern['volume_deduplication_reason']}")
            lines.append(f"  Combined Event Contribution: {pattern.get('combined_event_contribution', pattern.get('current_weighted_score', 0.0))}")
            lines.append(f"  Recency Weight: {pattern.get('recency_weight', 'Unknown')}")
            lines.append(f"  Score Eligible: {'Yes' if pattern.get('score_eligible') else 'No'}")
            lines.append(f"  Included in Current Score: {'Yes' if pattern.get('included_in_current_score') else 'No'}")
            if pattern.get("invalidation_condition"):
                lines.append(f"  Invalidation Condition: {pattern['invalidation_condition']}")
            if pattern.get("current_score_exclusion_reason"):
                lines.append(f"  Current Score Exclusion Reason: {pattern['current_score_exclusion_reason']}")
            if pattern.get("overlap_note"):
                lines.append(f"  Overlap Note: {pattern['overlap_note']}")
            if pattern.get("related_note"):
                lines.append(f"  Relationship Note: {pattern['related_note']}")

    append_event_section("Current Score-Contributing Evidence", current_contributing)
    lines.append(f"Directionally Conflicting Scored Evidence ({len(directionally_conflicting)}):")
    if directionally_conflicting:
        lines.append(
            "  (Already included above under Current Score-Contributing Evidence; flagged here "
            "only to show which of those events disagree on direction.)"
        )
        for pattern in directionally_conflicting:
            lines.append(
                f"  - {pattern.get('primary_pattern_name', 'Unknown')} "
                f"[{pattern.get('bias', 'Unknown')}] (event_id={pattern.get('event_id', 'Unknown')})"
            )
    else:
        lines.append("  None")
    append_event_section("Awaiting Confirmation", awaiting_confirmation)
    append_event_section("Current Neutral / Informational Evidence", current_neutral)
    append_event_section("Recent Non-Contributing Tracked Events", recent_non_contributing)

    lines.append(f"Deprecated Current Relevant Patterns ({len(current_patterns)}):")
    if current_patterns:
        lines.append("  This alias is deprecated and mirrors the current display collections above.")
    else:
        lines.append("  None")

    if pattern_history_mode in {"session", "all"}:
        relevant_session = result.get("relevant_session", {})
        lines.append(
            f"Historical Session Detections ({shown_history} shown{history_count_suffix}):"
        )
        lines.append(
            f"  Relevant Session: {relevant_session.get('exchange_date', 'Unknown')} "
            f"({relevant_session.get('session_start_display', 'Unknown')} to "
            f"{relevant_session.get('session_end_display', 'Unknown')})"
        )
        lines.append(f"  Session Mode: {relevant_session.get('session_mode', result.get('session_mode', 'Unknown'))}")
        lines.append(
            f"  Included Segments: {', '.join(relevant_session.get('included_segments', result.get('included_segments', []))) or 'Unknown'}"
        )
        if total_history != shown_history:
            lines.append(f"  Showing {shown_history} of {total_history} detected events")
        if session_history:
            for index, pattern in enumerate(session_history, start=1):
                lines.append(f"  {index}. {pattern.get('primary_pattern_name', 'Unknown')}")
                lines.append(f"     Pattern Labels: {', '.join(pattern.get('pattern_labels', []))}")
                lines.append(f"     Detected at: {pattern.get('detected_at_display', 'Unknown')}")
                lines.append(f"     State: {pattern.get('state', 'unknown')}")
                _append_transition_lines(lines, pattern, indent="     ")
                lines.append(
                    f"     Included in Current Score: {'Yes' if pattern.get('included_in_current_score') else 'No'}"
                )
                if pattern.get("invalidation_condition"):
                    lines.append(f"     Invalidation Condition: {pattern['invalidation_condition']}")
                if pattern.get("current_score_exclusion_reason"):
                    lines.append(
                        f"     Current Score Exclusion Reason: {pattern['current_score_exclusion_reason']}"
                    )
                if pattern.get("overlap_note"):
                    lines.append(f"     Overlap Note: {pattern['overlap_note']}")
                if pattern.get("related_note"):
                    lines.append(f"     Relationship Note: {pattern['related_note']}")
        else:
            lines.append("  None")

    lines.append("Archived Lifecycle Summary (events no longer current-relevant; excludes anything shown above as current):")
    if historical_summary:
        lines.append(f"  Total Archived Lifecycle Events: {historical_summary.get('count', len(historical_lifecycle))}")
        if historical_summary.get("by_state"):
            lines.append(
                "  By State: "
                + ", ".join(f"{state}={count}" for state, count in historical_summary["by_state"].items())
            )
        if historical_summary.get("by_family"):
            lines.append(
                "  By Family: "
                + ", ".join(f"{family}={count}" for family, count in historical_summary["by_family"].items())
            )
        archived_active_count = historical_summary.get("by_state", {}).get("active", 0)
        if archived_active_count:
            lines.append(
                f"  Note: {archived_active_count} archived event(s) still carry lifecycle state "
                "'active' -- their setup was never invalidated, expired, or failed, but they aged "
                "out of the current display/scoring window. 'Active' here describes the pattern's "
                "own lifecycle state, not current relevance."
            )
    else:
        lines.append("  None")

    if include_all_patterns:
        lines.append(f"All Historical Detected Pattern Labels ({len(raw_patterns)}):")
        if raw_patterns:
            for pattern in raw_patterns:
                lines.append(f"  Name: {pattern['pattern_name']}")
                lines.append(f"  Family: {pattern.get('pattern_family', 'unknown')}")
                lines.append(f"  Status: {pattern.get('status', 'confirmed')}")
                lines.append(f"  State: {pattern.get('event_state', 'unknown')}")
                lines.append(f"  Geometry Status: {pattern.get('geometry_status', 'Validated')}")
                lines.append(f"  Context Status: {pattern.get('context_status', 'Not Applicable')}")
                lines.append(f"  Directional Confirmation: {pattern.get('directional_confirmation', 'Not Required')}")
                lines.append(f"  Follow-Through: {pattern.get('follow_through', 'Not Applicable')}")
                pre_pattern_move = _pattern_entry_move_display(pattern.get("pattern_entry_trend"))
                if pre_pattern_move:
                    lines.append(f"  Immediate Pre-Pattern Move: {pre_pattern_move}")
                lines.append(f"  Bias: {pattern['bias']}")
                lines.append(
                    f"  Pattern Start: {pattern.get('pattern_start_display', pattern.get('bar_start_display', 'Unknown'))}"
                )
                lines.append(
                    f"  Setup Completion: {pattern.get('setup_completion_display', pattern.get('pattern_end_display', 'Unknown'))}"
                )
                lines.append(
                    f"  Pattern Completion: {pattern.get('pattern_end_display', pattern.get('bar_end_display', 'Unknown'))}"
                )
                lines.append(f"  Detected at: {pattern['detected_at_display']}")
                if pattern.get("confirmation_at_display"):
                    lines.append(f"  Confirmation Time: {pattern['confirmation_at_display']}")
                lines.append(f"  Included in Current Score: {'Yes' if pattern.get('included_in_current_score') else 'No'}")
                if pattern.get("invalidation_condition"):
                    lines.append(f"  Invalidation Condition: {pattern['invalidation_condition']}")
                if pattern.get("exclusion_reason"):
                    lines.append(f"  Exclusion Reason: {pattern['exclusion_reason']}")

    warnings = result.get("warnings") or []
    lines.append("Warnings:")
    if warnings:
        for warning in warnings:
            lines.append(f"  - {warning}")
    else:
        lines.append("  None")

    structured = result.get("structured_explanation") or {}
    lines.append("Explanation:")
    lines.append(f"  {structured.get('summary', result.get('explanation', ''))}")
    if structured.get("trend_evidence"):
        lines.append("Trend Evidence:")
        for item in structured["trend_evidence"]:
            lines.append(f"  - {item}")
    if structured.get("supporting_trend_evidence"):
        lines.append("Supporting Trend Evidence:")
        for item in structured["supporting_trend_evidence"]:
            lines.append(f"  - {item}")
    if structured.get("conflicting_trend_evidence"):
        lines.append("Conflicting Trend Evidence:")
        for item in structured["conflicting_trend_evidence"]:
            lines.append(f"  - {item}")
    if structured.get("neutral_trend_evidence"):
        lines.append("Neutral Trend Evidence:")
        for item in structured["neutral_trend_evidence"]:
            lines.append(f"  - {item}")
    if structured.get("bullish_evidence"):
        lines.append("Bullish Evidence:")
        for item in structured["bullish_evidence"]:
            lines.append(f"  - {item}")
    if structured.get("bearish_evidence"):
        lines.append("Bearish Evidence:")
        for item in structured["bearish_evidence"]:
            lines.append(f"  - {item}")
    if structured.get("current_pattern_evidence"):
        lines.append("Current Pattern Evidence:")
        for item in structured["current_pattern_evidence"]:
            lines.append(f"  - {item}")
    if structured.get("session_context"):
        lines.append("Session Context:")
        for item in structured["session_context"]:
            lines.append(f"  - {item}")
    if structured.get("lifecycle_note"):
        lines.append("Lifecycle Note:")
        lines.append(f"  {structured['lifecycle_note']}")
    if structured.get("cluster_notes"):
        lines.append("Rejection/Reversal Clusters:")
        for item in structured["cluster_notes"]:
            lines.append(f"  - {item}")
    if structured.get("dampener_notes"):
        lines.append("Unconfirmed Dampener Signals:")
        for item in structured["dampener_notes"]:
            lines.append(f"  - {item}")
    if structured.get("conflicts"):
        lines.append("Conflicts:")
        for item in structured["conflicts"]:
            lines.append(f"  - {item}")
    if structured.get("data_warnings"):
        lines.append("Data Warnings:")
        for item in structured["data_warnings"]:
            lines.append(f"  - {item}")
    lines.append("Bias Rationale:")
    lines.append(f"  {structured.get('reason_for_bias', '')}")
    lines.append("Confidence Rationale:")
    lines.append(
        f"  {structured.get('reason_for_confidence', result.get('explanation', ''))} "
        "Rule confidence is an uncalibrated rule-strength score, not a probability."
    )
    if structured.get("confidence_breakdown"):
        lines.append("Confidence Breakdown:")
        for key, value in structured["confidence_breakdown"].items():
            lines.append(f"  - {key}: {value}")
    return "\n".join(lines)
