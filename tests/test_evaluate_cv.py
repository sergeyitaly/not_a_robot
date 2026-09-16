import pytest

from examples.synthetic_data import make_synthetic_dataset
from not_a_robot.pipeline import evaluate_cv
from not_a_robot.schema import InteractionSession


def test_evaluate_cv_runs_expected_number_of_folds_and_reports_variance():
    sessions = make_synthetic_dataset(n_per_class=60, seed=5)
    report = evaluate_cv(sessions, n_splits=5, n_repeats=3, seed=5, data_source="test")

    assert report.n_splits == 5
    assert report.n_repeats == 3
    assert report.n_sessions == len(sessions)
    assert 0.0 <= report.accuracy_mean <= 1.0
    assert report.accuracy_std >= 0.0


def test_evaluate_cv_breaks_out_recall_by_synthetic_archetype():
    sessions = make_synthetic_dataset(n_per_class=100, seed=6)
    report = evaluate_cv(sessions, n_splits=5, n_repeats=3, seed=6, data_source="test")

    expected_groups = {"human", "naive", "evasive", "headless", "sophisticated"}
    assert expected_groups.issubset(report.group_recall_mean.keys())

    # naive bots (straight-line path, uniform keystrokes) should be caught
    # far more often than the sophisticated archetype, which is designed
    # to be much harder (mouse/keyboard/click cloned from the human
    # generator, only missing scroll/focus/paste activity).
    assert report.group_recall_mean["naive"] > report.group_recall_mean["sophisticated"]

    # group weights should roughly match the generator's target proportions
    assert report.group_weight["human"] == pytest.approx(0.5, abs=0.02)


def test_evaluate_cv_requires_minimum_sessions_for_fold_count():
    with pytest.raises(ValueError):
        evaluate_cv([InteractionSession(label=True)] * 4, n_splits=5)


def test_evaluate_cv_requires_both_classes():
    sessions = [InteractionSession(label=True) for _ in range(10)]
    with pytest.raises(ValueError):
        evaluate_cv(sessions, n_splits=5)


def test_summary_reports_per_group_table():
    sessions = make_synthetic_dataset(n_per_class=60, seed=7)
    report = evaluate_cv(sessions, n_splits=5, n_repeats=3, seed=7, data_source="test")
    text = report.summary()
    assert "Per-group recall" in text
    assert "sophisticated" in text
    assert "+/-" in text
