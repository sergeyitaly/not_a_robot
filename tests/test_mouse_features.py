from not_a_robot.features.mouse import extract_mouse_features
from not_a_robot.schema import MouseEvent


def test_empty_and_short_traces_return_zeroed_features():
    assert extract_mouse_features([])["mouse_num_points"] == 0.0

    one_point = [MouseEvent(0, 0, 0)]
    features = extract_mouse_features(one_point)
    assert features["mouse_num_points"] == 1.0
    assert features["mouse_path_length"] == 0.0


def test_straight_line_has_path_efficiency_near_one():
    points = [MouseEvent(x=float(i) * 10, y=0.0, t=float(i) * 5) for i in range(10)]
    features = extract_mouse_features(points)
    assert features["mouse_path_efficiency"] > 0.99
    assert features["mouse_turning_angle_mean"] == 0.0
    assert features["mouse_direction_reversals"] == 0.0


def test_jittery_zigzag_has_lower_path_efficiency_and_reversals():
    points = []
    x, t = 0.0, 0.0
    for i in range(20):
        y = 10.0 if i % 2 == 0 else -10.0
        points.append(MouseEvent(x, y, t))
        x += 5.0
        t += 10.0
    features = extract_mouse_features(points)
    assert features["mouse_path_efficiency"] < 0.9
    assert features["mouse_direction_reversals"] > 0


def test_velocity_reflects_speed():
    fast = [MouseEvent(0, 0, 0), MouseEvent(100, 0, 10), MouseEvent(200, 0, 20)]
    slow = [MouseEvent(0, 0, 0), MouseEvent(100, 0, 1000), MouseEvent(200, 0, 2000)]
    fast_features = extract_mouse_features(fast)
    slow_features = extract_mouse_features(slow)
    assert fast_features["mouse_velocity_mean"] > slow_features["mouse_velocity_mean"]
