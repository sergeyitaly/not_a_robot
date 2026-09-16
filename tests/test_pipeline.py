import pytest

from examples.synthetic_data import make_synthetic_dataset
from not_a_robot.pipeline import run_training_pipeline
from not_a_robot.schema import InteractionSession


def test_pipeline_produces_sane_report_on_synthetic_data():
    sessions = make_synthetic_dataset(n_per_class=60, seed=3)
    detector, report = run_training_pipeline(sessions, seed=3, data_source="test")

    assert report.n_train + report.n_test == len(sessions)
    assert 0.0 <= report.test_accuracy <= 1.0
    assert report.test_accuracy > 0.8
    assert 0.0 <= report.bot_catch_rate <= 1.0
    assert report.bot_catch_rate == pytest.approx(1.0 - report.false_accept_rate)
    assert report.test_recall == pytest.approx(1.0 - report.false_reject_rate)
    assert len(report.top_features) > 0

    sample = sessions[0]
    score = detector.score(sample)
    assert 0.0 <= score <= 1.0


def test_pipeline_requires_minimum_labeled_sessions():
    with pytest.raises(ValueError):
        run_training_pipeline([InteractionSession(label=True)])


def test_pipeline_requires_both_classes():
    sessions = [InteractionSession(label=True) for _ in range(5)]
    with pytest.raises(ValueError):
        run_training_pipeline(sessions)


def test_summary_reports_practical_verification_metrics():
    sessions = make_synthetic_dataset(n_per_class=40, seed=4)
    _, report = run_training_pipeline(sessions, seed=4, data_source="test")
    text = report.summary()
    assert "Bot catch rate" in text
    assert "Human pass rate" in text
    assert "False accept rate" in text
    assert "False reject rate" in text
