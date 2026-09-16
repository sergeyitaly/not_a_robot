import pytest

from examples.synthetic_data import make_synthetic_dataset
from not_a_robot.pipeline import (
    cost_optimal_threshold,
    evaluate_cv,
    summarize_across_seeds,
)
from not_a_robot.schema import InteractionSession


def test_evaluate_cv_runs_expected_number_of_folds_and_reports_variance():
    sessions = make_synthetic_dataset(n_per_class=60, seed=5)
    report = evaluate_cv(sessions, n_splits=5, n_repeats=3, seed=5, data_source="test")

    assert report.n_splits == 5
    assert report.n_repeats == 3
    assert report.n_sessions == len(sessions)
    assert 0.0 <= report.accuracy_mean <= 1.0
    assert report.accuracy_std >= 0.0


def test_evaluate_cv_breaks_out_pooled_recall_by_synthetic_archetype():
    sessions = make_synthetic_dataset(n_per_class=100, seed=6)
    report = evaluate_cv(sessions, n_splits=5, n_repeats=3, seed=6, data_source="test")

    expected_groups = {"human", "naive", "evasive", "headless", "sophisticated"}
    assert expected_groups.issubset(report.group_recall.keys())

    # naive bots (straight-line path, uniform keystrokes) should be caught
    # far more often than the sophisticated archetype, which is designed
    # to be much harder (mouse/keyboard/click cloned from the human
    # generator, only missing scroll/focus/paste activity).
    assert report.group_recall["naive"] > report.group_recall["sophisticated"]

    # group weights should roughly match the generator's target proportions
    assert report.group_weight["human"] == pytest.approx(0.5, abs=0.02)

    # pooled n should be n_repeats * (number of sessions in that group)
    n_sophisticated_sessions = sum(1 for s in sessions if s.group == "sophisticated")
    assert report.group_n["sophisticated"] == n_sophisticated_sessions * 3

    # Wilson CI must bracket the point estimate (within float tolerance)
    # and stay within [0, 1]
    eps = 1e-6
    for group, recall in report.group_recall.items():
        lo, hi = report.group_recall_ci[group]
        assert -eps <= lo <= 1.0 + eps
        assert -eps <= hi <= 1.0 + eps
        assert lo - eps <= recall <= hi + eps


def test_evaluate_cv_reports_per_fold_group_counts():
    sessions = make_synthetic_dataset(n_per_class=100, seed=6)
    report = evaluate_cv(sessions, n_splits=5, n_repeats=3, seed=6, data_source="test")

    for group in report.group_weight:
        assert report.group_n_per_fold_min[group] >= 0
        assert report.group_n_per_fold_mean[group] >= report.group_n_per_fold_min[group]


def test_evaluate_cv_computes_cost_optimal_threshold_and_group_breakdown():
    sessions = make_synthetic_dataset(n_per_class=100, seed=6)
    report = evaluate_cv(
        sessions, n_splits=5, n_repeats=3, seed=6, data_source="test", cost_fa=10.0, cost_fr=1.0
    )

    assert report.cost_threshold is not None
    assert 0.0 <= report.cost_threshold <= 1.0
    assert 0.0 <= report.cost_far <= 1.0
    assert 0.0 <= report.cost_frr <= 1.0
    assert set(report.cost_group_recall.keys()) == set(report.group_recall.keys())


def test_evaluate_cv_requires_minimum_sessions_for_fold_count():
    with pytest.raises(ValueError):
        evaluate_cv([InteractionSession(label=True)] * 4, n_splits=5)


def test_evaluate_cv_requires_both_classes():
    sessions = [InteractionSession(label=True) for _ in range(10)]
    with pytest.raises(ValueError):
        evaluate_cv(sessions, n_splits=5)


def test_summary_reports_per_group_table_and_cost_curve():
    sessions = make_synthetic_dataset(n_per_class=60, seed=7)
    report = evaluate_cv(sessions, n_splits=5, n_repeats=3, seed=7, data_source="test")
    text = report.summary()
    assert "Per-group recall" in text
    assert "sophisticated" in text
    assert "95% CI" in text
    assert "Cost-optimal threshold" in text


def test_cost_optimal_threshold_prefers_far_penalty_when_c_fa_is_large():
    import numpy as np

    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p_human = np.array([0.2, 0.4, 0.6, 0.8, 0.3, 0.5, 0.7, 0.9])

    # a very high false-accept cost should push the threshold up (stricter
    # on letting bots through) relative to a very high false-reject cost
    thr_strict, _, _, _ = cost_optimal_threshold(y_true, p_human, c_fa=100.0, c_fr=1.0)
    thr_lenient, _, _, _ = cost_optimal_threshold(y_true, p_human, c_fa=1.0, c_fr=100.0)
    assert thr_strict >= thr_lenient


def test_summarize_across_seeds_pools_group_counts():
    sessions_a = make_synthetic_dataset(n_per_class=60, seed=10)
    sessions_b = make_synthetic_dataset(n_per_class=60, seed=11)

    report_a = evaluate_cv(sessions_a, n_splits=5, n_repeats=2, seed=10, data_source="a")
    report_b = evaluate_cv(sessions_b, n_splits=5, n_repeats=2, seed=11, data_source="b")

    combined = summarize_across_seeds([(10, report_a), (11, report_b)], data_source="a+b")

    assert combined.seeds == [10, 11]
    assert len(combined.accuracy) == 2
    for group in report_a.group_n:
        expected_n = report_a.group_n.get(group, 0) + report_b.group_n.get(group, 0)
        assert combined.group_n[group] == expected_n

    text = combined.summary()
    assert "multi-seed evaluation" in text
    assert "Bot catch rate range across seeds" in text
