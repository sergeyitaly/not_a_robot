from __future__ import annotations

from .features.clicks import extract_click_features
from .features.engagement import extract_engagement_features
from .features.enrichment import ENRICHMENT_FEATURE_NAMES, extract_enrichment_features
from .features.mouse import extract_mouse_features
from .features.scroll import extract_scroll_features
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
    "scroll_event_count",
    "scroll_total_distance",
    "scroll_direction_reversals",
    "scroll_interval_mean",
    "scroll_interval_std",
    "scroll_delta_mean",
    "scroll_delta_std",
    "click_count",
    "click_interval_mean",
    "click_interval_std",
    "click_position_std",
    "blur_count",
    "paste_count",
    "paste_total_chars",
)

FEATURE_NAMES: tuple[str, ...] = _BASE_FEATURE_NAMES + ENRICHMENT_FEATURE_NAMES


def extract_features(session: InteractionSession) -> dict[str, float]:
    """Extract the full named feature set for one session.

    Combines the raw mouse/timing/scroll/click/engagement measurements
    with derived enrichment ratios (coefficients of variation, per-second
    rates) computed from them, so callers always get the complete,
    model-ready feature set.
    """
    features: dict[str, float] = {}
    features.update(extract_mouse_features(session.mouse_events))
    features.update(
        extract_timing_features(
            session.key_events, session.page_load_t, session.submit_t
        )
    )
    features.update(extract_scroll_features(session.scroll_events))
    features.update(extract_click_features(session.click_events))
    features.update(
        extract_engagement_features(session.focus_events, session.paste_events)
    )
    features.update(extract_enrichment_features(features))
    return features


def to_vector(features: dict[str, float]) -> list[float]:
    """Order a feature dict into the fixed vector a model expects."""
    return [features.get(name, 0.0) for name in FEATURE_NAMES]
