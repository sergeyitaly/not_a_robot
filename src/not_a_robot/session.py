from __future__ import annotations

from .features.enrichment import ENRICHMENT_FEATURE_NAMES, extract_enrichment_features
from .features.mouse import extract_mouse_features
from .features.timing import extract_timing_features
from .schema import InteractionSession

_BASE_FEATURE_NAMES: tuple[str, ...] = (
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

FEATURE_NAMES: tuple[str, ...] = _BASE_FEATURE_NAMES + ENRICHMENT_FEATURE_NAMES


def extract_features(session: InteractionSession) -> dict[str, float]:
    """Extract the full named feature set for one session.

    Combines the raw mouse/timing measurements with derived enrichment
    ratios (coefficients of variation, per-second rates) computed from
    them, so callers always get the complete, model-ready feature set.
    """
    features: dict[str, float] = {}
    features.update(extract_mouse_features(session.mouse_events))
    features.update(
        extract_timing_features(
            session.key_events, session.page_load_t, session.submit_t
        )
    )
    features.update(extract_enrichment_features(features))
    return features


def to_vector(features: dict[str, float]) -> list[float]:
    """Order a feature dict into the fixed vector a model expects."""
    return [features.get(name, 0.0) for name in FEATURE_NAMES]
