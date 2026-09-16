from __future__ import annotations

import math
from typing import Optional

from ..schema import KeyEvent

_EPS = 1e-9


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def extract_timing_features(
    key_events: list[KeyEvent],
    page_load_t: float,
    submit_t: Optional[float],
) -> dict[str, float]:
    """Compute keystroke-dynamics and overall pacing features.

    Dwell time (key down -> up) and flight time (key up -> next key down)
    are classic keystroke-dynamics signals: scripted form fills tend to have
    near-uniform, unrealistically short intervals compared to a human typing.
    """
    ordered = sorted(key_events, key=lambda k: k.t_down)

    dwell_times = [k.t_up - k.t_down for k in ordered]
    flight_times = [b.t_down - a.t_up for a, b in zip(ordered, ordered[1:])]

    time_to_first_key = ordered[0].t_down - page_load_t if ordered else 0.0
    time_to_submit = submit_t - page_load_t if submit_t is not None else 0.0

    return {
        "key_count": float(len(ordered)),
        "key_dwell_mean": _mean(dwell_times),
        "key_dwell_std": _std(dwell_times),
        "key_flight_mean": _mean(flight_times),
        "key_flight_std": _std(flight_times),
        "time_to_first_key_ms": max(time_to_first_key, 0.0),
        "time_to_submit_ms": max(time_to_submit, 0.0),
    }
