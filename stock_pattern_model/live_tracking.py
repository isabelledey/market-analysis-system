"""Incremental re-evaluation of pattern candidates as new candles complete.

``analyze_dataframe``/``analyze_stock`` already re-derive every pattern's full lifecycle from
scratch on each call: they rescan the (now one-bar-longer) historical DataFrame end to end,
including the forward-looking retest/confirmation/invalidation/expiration scan in
``analysis._apply_pattern_lifecycle``. That means simply calling one of them again after a new
candle completes already re-evaluates every existing pattern candidate against the new candle --
there is no separate incremental engine to build. What is missing is a way to see *only what
changed* between two such calls, so a caller does not have to diff the entire result set itself
every time a new candle appears. ``PatternLifecycleTracker`` is that thin diffing layer; it adds
no parallel pattern-detection or lifecycle logic of its own.

Patterns are tracked across calls by a "lineage key" -- ``(pattern_id, pattern_start_at)`` --
rather than the richer ``event_id``/``setup_id`` already computed by scoring.py. ``event_id``
embeds ``relevant_indices``, which are positions in whatever DataFrame was analyzed; a caller
that keeps a fixed trailing lookback window (as every ``--timeframe`` preset does) sees those
positions shift as old bars drop off the front, which would make ``event_id`` churn even for a
pattern whose calendar timestamps never changed. ``setup_id`` embeds ``pattern_end_at`` and
``bias``, which are exactly the two fields a multi-candle candidate (see
``pattern_detector.MorningStarDetector``/``EveningStarDetector``) changes the moment it resolves
from a forming candidate into a confirmed or failed event. ``pattern_start_at`` is the one
identifier that stays fixed across a pattern's entire candidate -> confirmed/failed/expired
lifetime, which is what continuity across candle-by-candle re-evaluation actually requires.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from stock_pattern_model.analysis import analyze_stock

LineageKey = tuple[str, str]


def _lineage_key(pattern: dict[str, Any]) -> LineageKey:
    return (str(pattern["pattern_id"]), str(pattern["pattern_start_at"]))


@dataclass(frozen=True)
class PatternTransition:
    """One pattern's state change between two successive completed-candle evaluations."""

    lineage_key: LineageKey
    pattern_id: str
    pattern_name: str
    bias: str
    kind: str  # "new_candidate" | "state_changed" | "aged_out_of_window"
    previous_status: str | None
    previous_event_state: str | None
    status: str | None
    event_state: str | None
    pattern: dict[str, Any] | None


@dataclass
class _TrackedPattern:
    status: str
    event_state: str
    pattern: dict[str, Any]


@dataclass
class IngestResult:
    """Result of feeding one newly-available snapshot of candles to the tracker."""

    analysis: dict[str, Any]
    has_new_completed_candle: bool
    transitions: list[PatternTransition]


class PatternLifecycleTracker:
    """Re-evaluates pattern candidates every time a new completed candle becomes available.

    Each ``ingest`` call re-runs the full analysis pipeline (via ``analyze_stock`` by default, or
    any injected callable with the same keyword-argument contract, e.g. ``analyze_dataframe`` fed
    a growing DataFrame in tests) and diffs the resulting patterns against what was tracked last
    time, keyed by lineage so that:

    * a pattern appearing for the first time is reported as ``"new_candidate"`` -- this covers
      both a brand new single-candle detection and a multi-candle candidate (e.g. a Morning Star's
      first two candles) just starting to form;
    * a pattern whose ``status``/``event_state`` changed since the last check (a candidate
      confirming or failing, a confirmed pattern getting retested/reclaimed/invalidated, a
      tentative structural pattern breaking its neckline) is reported as ``"state_changed"``,
      exactly once;
    * a pattern that is unchanged is silently updated in the tracker's memory and not re-reported
      -- this is what keeps repeated evaluation from producing duplicate detections;
    * a pattern that no longer appears at all (typically because it fell outside a sliding
      lookback window) is reported once as ``"aged_out_of_window"`` and then forgotten.

    Calling ``ingest`` again before a new candle has actually completed is a safe no-op: no
    transitions are computed and the tracker's memory is left untouched.
    """

    def __init__(
        self,
        *,
        analyze: Callable[..., dict[str, Any]] = analyze_stock,
        **analyze_kwargs: Any,
    ) -> None:
        self._analyze = analyze
        self._analyze_kwargs = analyze_kwargs
        self._tracked: dict[LineageKey, _TrackedPattern] = {}
        self._last_completed_candle: pd.Timestamp | None = None

    @property
    def last_completed_candle(self) -> pd.Timestamp | None:
        return self._last_completed_candle

    def ingest(self, **overrides: Any) -> IngestResult:
        """Run one analysis pass and report what changed since the last completed candle.

        ``overrides`` are merged over the kwargs supplied at construction time (e.g. a per-call
        ``df=`` when wrapping ``analyze_dataframe``, or a per-call ``as_of=`` to advance a
        simulated clock in tests). The still-forming latest candle, if any, continues to be
        excluded by the wrapped analyzer's own completed-candle filtering -- this tracker never
        evaluates a candle that has not closed.
        """
        kwargs = {**self._analyze_kwargs, **overrides}
        result = self._analyze(**kwargs)
        latest_completed = pd.Timestamp(result["latest_datetime"])
        has_new_candle = (
            self._last_completed_candle is None or latest_completed > self._last_completed_candle
        )

        if not has_new_candle:
            return IngestResult(analysis=result, has_new_completed_candle=False, transitions=[])

        transitions = self._diff(result["all_detected_patterns"])
        self._last_completed_candle = latest_completed
        return IngestResult(analysis=result, has_new_completed_candle=True, transitions=transitions)

    def _diff(self, patterns: list[dict[str, Any]]) -> list[PatternTransition]:
        transitions: list[PatternTransition] = []
        seen_keys: set[LineageKey] = set()

        for pattern in patterns:
            key = _lineage_key(pattern)
            seen_keys.add(key)
            status = str(pattern["status"])
            event_state = str(pattern["event_state"])
            previous = self._tracked.get(key)

            if previous is None:
                transitions.append(
                    PatternTransition(
                        lineage_key=key,
                        pattern_id=str(pattern["pattern_id"]),
                        pattern_name=str(pattern["pattern_name"]),
                        bias=str(pattern["bias"]),
                        kind="new_candidate",
                        previous_status=None,
                        previous_event_state=None,
                        status=status,
                        event_state=event_state,
                        pattern=pattern,
                    )
                )
            elif previous.status != status or previous.event_state != event_state:
                transitions.append(
                    PatternTransition(
                        lineage_key=key,
                        pattern_id=str(pattern["pattern_id"]),
                        pattern_name=str(pattern["pattern_name"]),
                        bias=str(pattern["bias"]),
                        kind="state_changed",
                        previous_status=previous.status,
                        previous_event_state=previous.event_state,
                        status=status,
                        event_state=event_state,
                        pattern=pattern,
                    )
                )

            self._tracked[key] = _TrackedPattern(status=status, event_state=event_state, pattern=pattern)

        for key in list(self._tracked):
            if key in seen_keys:
                continue
            tracked = self._tracked.pop(key)
            transitions.append(
                PatternTransition(
                    lineage_key=key,
                    pattern_id=str(tracked.pattern["pattern_id"]),
                    pattern_name=str(tracked.pattern["pattern_name"]),
                    bias=str(tracked.pattern["bias"]),
                    kind="aged_out_of_window",
                    previous_status=tracked.status,
                    previous_event_state=tracked.event_state,
                    status=None,
                    event_state=None,
                    pattern=None,
                )
            )

        return transitions
