from __future__ import annotations

import math

from ..schema import MouseEvent

_EPS = 1e-9

_EMPTY_FEATURES = {
    "mouse_num_points": 0.0,
    "mouse_path_length": 0.0,
    "mouse_path_efficiency": 0.0,
    "mouse_duration_ms": 0.0,
    "mouse_velocity_mean": 0.0,
    "mouse_velocity_std": 0.0,
    "mouse_velocity_max": 0.0,
    "mouse_accel_mean": 0.0,
    "mouse_accel_std": 0.0,
    "mouse_jerk_mean": 0.0,
    "mouse_turning_angle_mean": 0.0,
    "mouse_turning_angle_std": 0.0,
    "mouse_direction_reversals": 0.0,
    "mouse_pause_count": 0.0,
}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def _derivative(values: list[float], times: list[float]) -> list[float]:
    out = []
    for i in range(len(values) - 1):
        dt = max(times[i + 1] - times[i], _EPS)
        out.append((values[i + 1] - values[i]) / dt)
    return out


def _turning_angles(points: list[MouseEvent]) -> list[float]:
    angles = []
    for p0, p1, p2 in zip(points, points[1:], points[2:]):
        v1 = (p1.x - p0.x, p1.y - p0.y)
        v2 = (p2.x - p1.x, p2.y - p1.y)
        n1 = math.hypot(*v1)
        n2 = math.hypot(*v2)
        if n1 < _EPS or n2 < _EPS:
            continue
        cos_theta = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        angles.append(math.acos(cos_theta))
    return angles


def extract_mouse_features(points: list[MouseEvent]) -> dict[str, float]:
    """Compute behavioral features from a raw mouse-movement trace.

    Bot-generated traces (scripted moveTo/click calls) tend toward near-
    straight paths, near-constant velocity, and low jitter; human traces
    tend to have curvier paths, variable speed, and more direction changes.
    None of these signals are individually conclusive -- they are inputs
    to a classifier, not a verdict on their own.
    """
    if len(points) < 3:
        features = dict(_EMPTY_FEATURES)
        features["mouse_num_points"] = float(len(points))
        return features

    ordered = sorted(points, key=lambda p: p.t)

    path_length = sum(
        math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(ordered, ordered[1:])
    )
    straight_line = math.hypot(
        ordered[-1].x - ordered[0].x, ordered[-1].y - ordered[0].y
    )
    duration = max(ordered[-1].t - ordered[0].t, _EPS)

    mid_times = [(a.t + b.t) / 2 for a, b in zip(ordered, ordered[1:])]
    velocities = [
        math.hypot(b.x - a.x, b.y - a.y) / max(b.t - a.t, _EPS)
        for a, b in zip(ordered, ordered[1:])
    ]
    accelerations = _derivative(velocities, mid_times)
    accel_times = mid_times[:-1]
    jerks = _derivative(accelerations, accel_times)

    angles = _turning_angles(ordered)
    reversals = sum(1 for a in angles if a > math.pi / 2)

    intervals = [b.t - a.t for a, b in zip(ordered, ordered[1:])]
    pause_threshold = max(_mean(intervals) * 3, 100.0)
    pauses = sum(1 for dt in intervals if dt > pause_threshold)

    return {
        "mouse_num_points": float(len(ordered)),
        "mouse_path_length": path_length,
        "mouse_path_efficiency": (
            straight_line / path_length if path_length > _EPS else 1.0
        ),
        "mouse_duration_ms": duration,
        "mouse_velocity_mean": _mean(velocities),
        "mouse_velocity_std": _std(velocities),
        "mouse_velocity_max": max(velocities, default=0.0),
        "mouse_accel_mean": _mean(accelerations),
        "mouse_accel_std": _std(accelerations),
        "mouse_jerk_mean": _mean(jerks),
        "mouse_turning_angle_mean": _mean(angles),
        "mouse_turning_angle_std": _std(angles),
        "mouse_direction_reversals": float(reversals),
        "mouse_pause_count": float(pauses),
    }
