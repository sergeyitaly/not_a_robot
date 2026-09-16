from not_a_robot.features.engagement import extract_engagement_features
from not_a_robot.schema import FocusEvent, PasteEvent


def test_no_events_returns_zeroed_features():
    features = extract_engagement_features([], [])
    assert features["blur_count"] == 0.0
    assert features["paste_count"] == 0.0
    assert features["paste_total_chars"] == 0.0


def test_blur_and_paste_counted_correctly():
    focus_events = [FocusEvent(100.0, False), FocusEvent(2000.0, True)]
    paste_events = [PasteEvent(50.0, 12), PasteEvent(3000.0, 8)]
    features = extract_engagement_features(focus_events, paste_events)
    assert features["blur_count"] == 1.0
    assert features["paste_count"] == 2.0
    assert features["paste_total_chars"] == 20.0
