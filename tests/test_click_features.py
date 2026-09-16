from not_a_robot.features.clicks import extract_click_features
from not_a_robot.schema import ClickEvent


def test_no_events_returns_zeroed_features():
    features = extract_click_features([])
    assert features["click_count"] == 0.0
    assert features["click_position_std"] == 0.0


def test_repeated_exact_coordinate_has_zero_position_std():
    events = [ClickEvent(410.0, 512.0, 0.0), ClickEvent(410.0, 512.0, 500.0)]
    features = extract_click_features(events)
    assert features["click_count"] == 2.0
    assert features["click_position_std"] == 0.0


def test_varying_coordinates_have_nonzero_position_std():
    events = [ClickEvent(400.0, 500.0, 0.0), ClickEvent(430.0, 520.0, 500.0)]
    features = extract_click_features(events)
    assert features["click_position_std"] > 0.0
