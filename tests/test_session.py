from not_a_robot.schema import InteractionSession, KeyEvent, MouseEvent
from not_a_robot.session import FEATURE_NAMES, extract_features, to_vector


def test_extract_features_returns_every_named_feature():
    session = InteractionSession(
        mouse_events=[MouseEvent(0, 0, 0), MouseEvent(10, 5, 20), MouseEvent(20, 0, 40)],
        key_events=[KeyEvent(100, 150)],
        page_load_t=0.0,
        submit_t=300.0,
    )
    features = extract_features(session)
    assert set(features.keys()) == set(FEATURE_NAMES)


def test_to_vector_preserves_feature_order():
    features = {name: float(i) for i, name in enumerate(FEATURE_NAMES)}
    vector = to_vector(features)
    assert vector == [float(i) for i in range(len(FEATURE_NAMES))]


def test_to_vector_defaults_missing_features_to_zero():
    assert to_vector({}) == [0.0] * len(FEATURE_NAMES)
