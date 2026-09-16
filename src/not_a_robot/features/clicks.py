from __future__ import annotations

import math

from ..schema import ClickEvent

_EMPTY_FEATURES = {
    "click_count": 0.0,
    "click_interval_mean": 0.0,
    "click_interval_std": 0.0,
    "click_position_std": 0.0,
}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def extract_click_features(events: list[ClickEvent]) -> dict[str, float]:
    """Compute click/tap behavior features.

    Coordinate-based automation often clicks the exact same pixel
    repeatedly (a fixed button center baked into a script); a human
    clicking the same element multiple times still lands with small,
    non-zero position variance.
    """
    if not events:
        return dict(_EMPTY_FEATURES)

    ordered = sorted(events, key=lambda e: e.t)
    intervals = [b.t - a.t for a, b in zip(ordered, ordered[1:])]
    xs = [e.x for e in ordered]
    ys = [e.y for e in ordered]

    return {
        "click_count": float(len(ordered)),
        "click_interval_mean": _mean(intervals),
        "click_interval_std": _std(intervals),
        "click_position_std": (_std(xs) + _std(ys)) / 2.0,
    }
