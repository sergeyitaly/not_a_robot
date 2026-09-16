from not_a_robot.features.timing import extract_timing_features
from not_a_robot.schema import KeyEvent


def test_no_keys_returns_zeroed_features():
    features = extract_timing_features([], page_load_t=0.0, submit_t=None)
    assert features["key_count"] == 0.0
    assert features["key_dwell_mean"] == 0.0
    assert features["time_to_submit_ms"] == 0.0


def test_dwell_and_flight_times_computed_correctly():
    keys = [
        KeyEvent(t_down=100.0, t_up=150.0),
        KeyEvent(t_down=200.0, t_up=260.0),
    ]
    features = extract_timing_features(keys, page_load_t=0.0, submit_t=500.0)
    assert features["key_count"] == 2.0
    assert features["key_dwell_mean"] == 55.0
    assert features["key_flight_mean"] == 50.0
    assert features["time_to_first_key_ms"] == 100.0
    assert features["time_to_submit_ms"] == 500.0


def test_unordered_events_are_sorted_by_down_time():
    keys = [
        KeyEvent(t_down=200.0, t_up=250.0),
        KeyEvent(t_down=100.0, t_up=140.0),
    ]
    features = extract_timing_features(keys, page_load_t=0.0, submit_t=None)
    assert features["time_to_first_key_ms"] == 100.0
