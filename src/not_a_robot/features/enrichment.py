from __future__ import annotations

_EPS = 1e-9

ENRICHMENT_FEATURE_NAMES: tuple[str, ...] = (
    "mouse_velocity_cv",
    "mouse_points_per_sec",
    "mouse_pause_rate",
    "key_rate_per_sec",
    "key_dwell_cv",
    "activity_balance",
)


def extract_enrichment_features(base: dict[str, float]) -> dict[str, float]:
    """Derive normalized ratio/rate features from the base feature set.

    Raw counts and statistics scale with session length and typing speed;
    these ratios and coefficients of variation normalize for that and tend
    to separate scripted, uniform behavior from naturally variable human
    behavior better than any single raw statistic does.
    """
    duration_s = max(base.get("mouse_duration_ms", 0.0), 1.0) / 1000.0
    vel_mean = base.get("mouse_velocity_mean", 0.0)
    vel_std = base.get("mouse_velocity_std", 0.0)
    dwell_mean = base.get("key_dwell_mean", 0.0)
    dwell_std = base.get("key_dwell_std", 0.0)
    mouse_activity = base.get("mouse_num_points", 0.0)
    key_activity = base.get("key_count", 0.0)
    total_activity = mouse_activity + key_activity

    return {
        "mouse_velocity_cv": vel_std / vel_mean if vel_mean > _EPS else 0.0,
        "mouse_points_per_sec": mouse_activity / duration_s,
        "mouse_pause_rate": base.get("mouse_pause_count", 0.0) / duration_s,
        "key_rate_per_sec": key_activity / duration_s,
        "key_dwell_cv": dwell_std / dwell_mean if dwell_mean > _EPS else 0.0,
        "activity_balance": (
            mouse_activity / total_activity if total_activity > _EPS else 0.0
        ),
    }
