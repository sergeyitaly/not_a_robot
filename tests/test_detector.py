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
