from __future__ import annotations

import math

from ..schema import ScrollEvent

_EMPTY_FEATURES = {
    "scroll_event_count": 0.0,
    "scroll_total_distance": 0.0,
    "scroll_direction_reversals": 0.0,
    "scroll_interval_mean": 0.0,
    "scroll_interval_std": 0.0,
    "scroll_delta_mean": 0.0,
    "scroll_delta_std": 0.0,
}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def extract_scroll_features(events: list[ScrollEvent]) -> dict[str, float]:
    """Compute scroll-behavior features.

    Humans scroll in short, variable-speed bursts with direction
    reversals (re-reading, overshoot correction); scripted page
    traversal tends toward uniform-interval, single-direction, often
    larger jumps.
    """
    if len(events) < 2:
        features = dict(_EMPTY_FEATURES)
        features["scroll_event_count"] = float(len(events))
        return features

    ordered = sorted(events, key=lambda e: e.t)
    deltas = [e.delta_y for e in ordered]
    intervals = [b.t - a.t for a, b in zip(ordered, ordered[1:])]
    reversals = sum(
        1
        for a, b in zip(deltas, deltas[1:])
        if (a > 0 and b < 0) or (a < 0 and b > 0)
    )

    return {
        "scroll_event_count": float(len(ordered)),
        "scroll_total_distance": sum(abs(d) for d in deltas),
        "scroll_direction_reversals": float(reversals),
        "scroll_interval_mean": _mean(intervals),
        "scroll_interval_std": _std(intervals),
        "scroll_delta_mean": _mean(deltas),
        "scroll_delta_std": _std(deltas),
    }
