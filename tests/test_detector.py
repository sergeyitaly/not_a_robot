import pytest

from not_a_robot import BotDetector, InteractionSession
from examples.synthetic_data import make_synthetic_dataset


def test_fit_requires_at_least_two_labeled_sessions():
    detector = BotDetector()
    with pytest.raises(ValueError):
        detector.fit([InteractionSession(label=True)])


def test_fit_requires_both_classes():
    detector = BotDetector()
    sessions = [InteractionSession(label=True), InteractionSession(label=True)]
    with pytest.raises(ValueError):
        detector.fit(sessions)


def test_score_before_fit_raises():
    detector = BotDetector()
    with pytest.raises(RuntimeError):
        detector.score(InteractionSession())


def test_fit_and_predict_on_synthetic_data_beats_chance():
    sessions = make_synthetic_dataset(n_per_class=60, seed=1)
    train, test = sessions[:80], sessions[80:]

    detector = BotDetector()
    detector.fit(train)

    correct = sum(
        detector.predict(session) == bool(session.label) for session in test
    )
    accuracy = correct / len(test)
    assert accuracy > 0.8


def test_save_and_load_round_trip(tmp_path):
    sessions = make_synthetic_dataset(n_per_class=20, seed=2)
    detector = BotDetector()
    detector.fit(sessions)

    path = tmp_path / "detector.joblib"
    detector.save(path)
    loaded = BotDetector.load(path)

    for session in sessions[:5]:
        assert loaded.score(session) == pytest.approx(detector.score(session))


def test_fit_raises_clear_error_when_below_calibration_cv_minimum():
    # Default model is CalibratedClassifierCV(cv=3); 2 human + 2 bot is
    # fewer than 3 of each, so this should fail fast with a specific
    # message rather than surfacing sklearn's generic CV error.
    sessions = [
        InteractionSession(label=True),
        InteractionSession(label=True),
        InteractionSession(label=False),
        InteractionSession(label=False),
    ]
    detector = BotDetector()
    with pytest.raises(ValueError, match="3-fold calibration"):
        detector.fit(sessions)


def test_save_writes_library_version(tmp_path):
    import joblib

    sessions = make_synthetic_dataset(n_per_class=20, seed=3)
    detector = BotDetector()
    detector.fit(sessions)

    path = tmp_path / "detector.joblib"
    detector.save(path)
    payload = joblib.load(path)
    assert "library_version" in payload


def test_load_warns_on_library_version_mismatch(tmp_path):
    import joblib

    sessions = make_synthetic_dataset(n_per_class=20, seed=4)
    detector = BotDetector()
    detector.fit(sessions)

    path = tmp_path / "detector.joblib"
    detector.save(path)
    payload = joblib.load(path)
    payload["library_version"] = "0.0.0-does-not-exist"
    joblib.dump(payload, path)

    with pytest.warns(UserWarning, match="library_version|not-a-robot"):
        BotDetector.load(path)


def test_save_leaves_no_temp_file_behind(tmp_path):
    sessions = make_synthetic_dataset(n_per_class=20, seed=5)
    detector = BotDetector()
    detector.fit(sessions)

    path = tmp_path / "detector.joblib"
    detector.save(path)

    assert path.exists()
    assert not (tmp_path / "detector.joblib.tmp").exists()
