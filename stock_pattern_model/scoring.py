"""Dedicated scoring and explanation service for analysis output."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Any

from stock_pattern_model.config import ScoringConfig
from stock_pattern_model.domain import DataQualityReport


EVENT_STATE_PRIORITY = {
    "new": 0,
    "retested": 1,
    "active": 2,
    "failed": 3,
    "expired": 4,
}


@dataclass(frozen=True)
class ScoringService:
    """Calculate signal scores, market state, and structured explanations."""

    config: ScoringConfig

    def evaluate(
        self,
        *,
        symbol: str,
        trend: str,
        patterns: list[dict[str, Any]],
        quality_report: DataQualityReport,
        latest_close: float,
        latest_bar_start_display: str,
        latest_bar_end_display: str,
        interval: str,
        latest_volume_baseline_source: str,
    ) -> dict[str, Any]:
        enriched_patterns = self._enrich_patterns(patterns)
        score_patterns = [pattern for pattern in enriched_patterns if pattern["score_eligible"]]
        score_groups = self._group_primary_patterns(score_patterns)
        primary_patterns = list(score_groups["primary_patterns"])
        suppressed_patterns = list(score_groups["suppressed_patterns"])

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

    def _enrich_patterns(self, patterns: list[dict[str, Any]]) -> list[dict[str, Any]]:
        enriched = [dict(pattern) for pattern in patterns]
        for pattern in enriched:
            event = pattern["event"]
            pattern["event_id"] = self._build_event_id(pattern)
            pattern["setup_id"] = self._build_setup_id(pattern)
            pattern["evidence_group"] = self._build_evidence_group(pattern)
            pattern["recency_weight"] = self._recency_weight(pattern["candles_ago"])
            pattern["event_state"] = self._base_event_state(pattern)
            pattern["score_eligible"] = bool(pattern["score_eligible"]) and pattern["recency_weight"] > 0
            pattern["volume_score_contribution"] = 0.0
            pattern["pattern_score_contribution"] = 0.0
            pattern["group_primary"] = False
            pattern["group_suppressed"] = False
            pattern["event_detected_display"] = pattern.get("detected_at_display")
            pattern["event_timestamp"] = event.detected_at

        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for pattern in enriched:
            if pattern["event_state"] not in {"failed", "expired"}:
                groups[pattern["evidence_group"]].append(pattern)

        for group_patterns in groups.values():
            group_patterns.sort(key=lambda item: (item["candles_ago"], -abs(item["weighted_score"])))
            if len(group_patterns) > 1 and group_patterns[0]["event_state"] in {"new", "active"}:
                group_patterns[0]["event_state"] = "retested"

        return enriched

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
                key=lambda item: (
                    -abs(self._raw_pattern_score(item)),
                    item["candles_ago"],
                    item["priority"],
                ),
            )
            primary = ranked_group[0]
            primary["group_primary"] = True
            primary["pattern_score_contribution"] = round(self._raw_pattern_score(primary), 2)
            primary["volume_score_contribution"] = round(self._volume_contribution(primary), 2)
            primary_patterns.append(primary)

            for pattern in ranked_group[1:]:
                pattern["group_suppressed"] = True
                suppressed_patterns.append(pattern)

        return {
            "primary_patterns": primary_patterns,
            "suppressed_patterns": suppressed_patterns,
        }

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
            if pattern["event_state"] in {"new", "active", "retested"}
        ]
        if not active_patterns:
            return "Trend Only" if trend != "Neutral" else "Neutral"

        latest_pattern = min(active_patterns, key=lambda item: item["candles_ago"])
        bullish_score = score["bullish_score"]
        bearish_score = score["bearish_score"]
        conflict_ratio = self._conflict_ratio(bullish_score, bearish_score)

        if bullish_score > 0 and bearish_score > 0 and conflict_ratio >= self.config.conflict_neutrality_ratio:
            return "Conflicted"
        if (
            latest_pattern["pattern_id"] == "breakout"
            and latest_pattern["candles_ago"] <= self.config.breakout_state_max_age_bars
        ):
            return "Breakout Attempt"
        if (
            latest_pattern["pattern_id"] == "breakdown"
            and latest_pattern["candles_ago"] <= self.config.breakout_state_max_age_bars
        ):
            return "Breakdown Attempt"
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
            if pattern["status"] == "confirmed"
        ]
        if not confirmed_patterns:
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
            if pattern["status"] == "confirmed"
        ]
        if not confirmed_patterns:
            return 12.0 if trend != "Neutral" else 5.0

        recency_values = [pattern["recency_weight"] for pattern in confirmed_patterns]
        strength_values = [
            min(pattern["signal_strength"], 3.0) / 3.0
            for pattern in confirmed_patterns
        ]
        volume_ratio = (
            sum(1 for pattern in confirmed_patterns if pattern["volume_confirmed"])
            / len(confirmed_patterns)
        )
        independent_groups = len({pattern["evidence_group"] for pattern in confirmed_patterns})
        independent_families = len({pattern["pattern_family"] for pattern in confirmed_patterns})
        agreement_bonus = independent_groups * 7.0
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
            - age_penalty
            - trend_only_penalty
        )
        return round(max(5.0, min(100.0, confidence)), 1)

    def _build_structured_explanation(
        self,
        *,
        symbol: str,
        trend: str,
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
        conflicts: list[str] = []
        if bullish_patterns and bearish_patterns:
            conflicts.append(
                "Bullish and bearish confirmed evidence were both present, so the net signal was tempered."
            )
        if suppressed_patterns:
            conflicts.append(
                f"{len(suppressed_patterns)} overlapping pattern event(s) were grouped to avoid double counting."
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
            reason_for_bias = (
                "Confirmed evidence existed, but the net signal stayed too balanced or too weak to justify a directional bias."
            )
        elif trend != "Neutral":
            reason_for_bias = (
                "The trend remained directional, but no recent confirmed pattern added enough fresh evidence to move the bias away from neutral."
            )
        else:
            reason_for_bias = "No recent confirmed pattern created a directional edge."

        confidence_reasons: list[str] = []
        if primary_patterns:
            confidence_reasons.append(
                f"{len(primary_patterns)} independent evidence group(s) were scored after deduplication."
            )
        if quality_report.warnings:
            confidence_reasons.append("Data-quality warnings reduced confidence.")
        if bullish_patterns and bearish_patterns:
            confidence_reasons.append("Conflicting evidence reduced confidence.")
        if not primary_patterns and trend != "Neutral":
            confidence_reasons.append("Trend-only output keeps confidence low because it lacks fresh confirmed patterns.")
        if latest_volume_baseline_source == "rolling_20":
            confidence_reasons.append("Rolling volume baseline was used because time-of-day history was limited.")
        if not confidence_reasons:
            confidence_reasons.append("The score reflects rule strength only and is not statistically calibrated.")

        summary = (
            f"{symbol} last traded at {latest_close:.2f} on the completed {interval} candle from "
            f"{latest_bar_start_display} to {latest_bar_end_display}. Trend: {trend}. "
            f"Market state: {market_state}. Overall bias: {overall_bias}. "
            f"Net signal score: {score['net_signal_score']:.2f}. Rule confidence: {rule_confidence:.1f}/100."
        )
        return {
            "summary": summary,
            "bullish_evidence": [self._format_evidence_line(pattern) for pattern in bullish_patterns[:3]],
            "bearish_evidence": [self._format_evidence_line(pattern) for pattern in bearish_patterns[:3]],
            "conflicts": conflicts,
            "data_warnings": list(quality_report.warnings),
            "reason_for_bias": reason_for_bias,
            "reason_for_confidence": " ".join(confidence_reasons),
        }

    def _build_text_explanation(self, structured_explanation: dict[str, Any]) -> str:
        parts = [structured_explanation["summary"]]
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

    def _format_evidence_line(self, pattern: dict[str, Any]) -> str:
        detected_at = pattern["event"].detected_at
        return (
            f"{pattern['pattern_name']} [{pattern['status']}, {pattern['event_state']}] "
            f"detected at {detected_at.strftime('%Y-%m-%d %H:%M %Z')} with {pattern['detection_reason']}"
        )

    def _build_event_id(self, pattern: dict[str, Any]) -> str:
        event = pattern["event"]
        return (
            f"{pattern['pattern_id']}:{pattern['status']}:"
            f"{event.detected_at.isoformat()}:{'-'.join(map(str, event.relevant_indices))}"
        )

    def _build_setup_id(self, pattern: dict[str, Any]) -> str:
        event = pattern["event"]
        return (
            f"{pattern['pattern_id']}:{event.pattern_start_at.isoformat()}:"
            f"{event.pattern_end_at.isoformat()}:{pattern['bias']}"
        )

    def _build_evidence_group(self, pattern: dict[str, Any]) -> str:
        event = pattern["event"]
        relevant_prices = event.relevant_prices
        if pattern["pattern_id"] in {"breakout", "double_bottom"}:
            return f"upside_break:{event.detected_at.isoformat()}"
        if pattern["pattern_id"] in {"breakdown", "double_top"}:
            return f"downside_break:{event.detected_at.isoformat()}"
        if pattern["pattern_family"] in {"engulfing", "pin_bar", "doji", "star"}:
            return f"candlestick:{pattern['bias']}:{event.bar_start_at.isoformat()}"
        if pattern["pattern_family"] in {"inside_bar", "inside_bar_failure"}:
            return f"inside_structure:{pattern['bias']}:{event.pattern_end_at.isoformat()}"
        key_price = (
            relevant_prices.get("confirmation_price")
            or relevant_prices.get("breakout_level")
            or relevant_prices.get("breakdown_level")
            or 0.0
        )
        return f"{pattern['pattern_id']}:{pattern['bias']}:{event.detected_at.isoformat()}:{round(float(key_price), 2)}"

    def _base_event_state(self, pattern: dict[str, Any]) -> str:
        status = pattern["status"]
        if status == "failed":
            return "failed"
        if status == "expired" or pattern["candles_ago"] > self.config.state_expiration_bars:
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

    def _volume_contribution(self, pattern: dict[str, Any]) -> float:
        if not pattern["volume_confirmed"]:
            return 0.0
        direction = 1.0 if pattern["bias"] == "Bullish" else -1.0 if pattern["bias"] == "Bearish" else 0.0
        return direction * self.config.volume_confirmation_bonus * pattern["recency_weight"]

    def _recency_weight(self, candles_ago: int) -> float:
        if candles_ago > self.config.pattern_max_age_bars:
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

    def _status_rank(self, status: str) -> int:
        return {
            "confirmed": 0,
            "tentative": 1,
            "failed": 2,
            "expired": 3,
        }.get(status, 9)
