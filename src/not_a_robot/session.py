from __future__ import annotations

from .features.mouse import extract_mouse_features
from .features.timing import extract_timing_features
from .schema import InteractionSession

FEATURE_NAMES: tuple[str, ...] = (
    "mouse_num_points",
    "mouse_path_length",
    "mouse_path_efficiency",
    "mouse_duration_ms",
    "mouse_velocity_mean",
    "mouse_velocity_std",
    "mouse_velocity_max",
    "mouse_accel_mean",
    "mouse_accel_std",
    "mouse_jerk_mean",
    "mouse_turning_angle_mean",
    "mouse_turning_angle_std",
    "mouse_direction_reversals",
    "mouse_pause_count",
    "key_count",
    "key_dwell_mean",
    "key_dwell_std",
    "key_flight_mean",
    "key_flight_std",
    "time_to_first_key_ms",
    "time_to_submit_ms",
)


def extract_features(session: InteractionSession) -> dict[str, float]:
    """Extract the full named feature set for one session."""
    features: dict[str, float] = {}
    features.update(extract_mouse_features(session.mouse_events))
    features.update(
        extract_timing_features(
            session.key_events, session.page_load_t, session.submit_t
        )
    )
    return features


def to_vector(features: dict[str, float]) -> list[float]:
    """Order a feature dict into the fixed vector a model expects."""
    return [features.get(name, 0.0) for name in FEATURE_NAMES]
