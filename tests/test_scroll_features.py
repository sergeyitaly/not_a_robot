from not_a_robot.features.scroll import extract_scroll_features
from not_a_robot.schema import ScrollEvent


def test_no_events_returns_zeroed_features():
    features = extract_scroll_features([])
    assert features["scroll_event_count"] == 0.0
    assert features["scroll_total_distance"] == 0.0


def test_single_direction_scroll_has_no_reversals():
    events = [ScrollEvent(0.0, 100.0), ScrollEvent(200.0, 120.0), ScrollEvent(400.0, 90.0)]
    features = extract_scroll_features(events)
    assert features["scroll_direction_reversals"] == 0.0
    assert features["scroll_total_distance"] == 310.0


def test_direction_change_counts_as_reversal():
    events = [ScrollEvent(0.0, 100.0), ScrollEvent(200.0, -80.0), ScrollEvent(400.0, 60.0)]
    features = extract_scroll_features(events)
    assert features["scroll_direction_reversals"] == 2.0
