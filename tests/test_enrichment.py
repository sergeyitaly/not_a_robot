from not_a_robot.features.enrichment import extract_enrichment_features


def test_enrichment_handles_zero_duration_and_activity():
    features = extract_enrichment_features({})
    assert features["mouse_velocity_cv"] == 0.0
    assert features["activity_balance"] == 0.0
    assert features["mouse_points_per_sec"] == 0.0


def test_enrichment_computes_expected_ratios():
    base = {
        "mouse_duration_ms": 2000.0,
        "mouse_velocity_mean": 10.0,
        "mouse_velocity_std": 5.0,
        "mouse_num_points": 20.0,
        "mouse_pause_count": 2.0,
        "key_count": 10.0,
        "key_dwell_mean": 100.0,
        "key_dwell_std": 20.0,
    }
    features = extract_enrichment_features(base)
    assert features["mouse_velocity_cv"] == 0.5
    assert features["mouse_points_per_sec"] == 10.0
    assert features["mouse_pause_rate"] == 1.0
    assert features["key_rate_per_sec"] == 5.0
    assert features["key_dwell_cv"] == 0.2
    assert features["activity_balance"] == 20.0 / 30.0
