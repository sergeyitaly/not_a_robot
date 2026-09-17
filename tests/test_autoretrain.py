import pytest

from examples.synthetic_data import make_synthetic_dataset
from not_a_robot.autoretrain import AutoRetrainStore
from not_a_robot.schema import InteractionSession


def test_record_session_requires_a_trusted_label(tmp_path):
    store = AutoRetrainStore(tmp_path / "proj")
    with pytest.raises(ValueError):
        store.record_session(InteractionSession(label=None))


def test_label_and_group_composition_reflect_recorded_sessions(tmp_path):
    store = AutoRetrainStore(tmp_path / "proj")
    sessions = make_synthetic_dataset(n_per_class=60, seed=42)
    for session in sessions:
        store.record_session(session)

    label_comp = store.label_composition()
    assert label_comp["human"] == sum(1 for s in sessions if s.label is True)
    assert label_comp["bot"] == sum(1 for s in sessions if s.label is False)
    assert label_comp["human"] + label_comp["bot"] == len(sessions)

    group_comp = store.group_composition()
    assert sum(group_comp.values()) == len(sessions)
    assert group_comp["human"] == label_comp["human"]
    assert set(group_comp) == {"human", "naive", "evasive", "headless", "sophisticated"}


def test_record_session_appends_and_pending_count_tracks_it(tmp_path):
    store = AutoRetrainStore(tmp_path / "proj", min_new_sessions=5)
    assert store.pending_session_count() == 0

    for session in make_synthetic_dataset(n_per_class=2, seed=1):
        store.record_session(session)

    assert store.pending_session_count() == 4
    assert store.sessions_path.exists()


def test_maybe_retrain_waits_for_threshold_then_trains(tmp_path):
    store = AutoRetrainStore(tmp_path / "proj", min_new_sessions=40, cv_seeds=(0, 1))

    for session in make_synthetic_dataset(n_per_class=10, seed=2):
        store.record_session(session)
    assert store.pending_session_count() == 20

    # below threshold: no retrain yet
    assert store.maybe_retrain() is None
    assert not store.model_path.exists()

    for session in make_synthetic_dataset(n_per_class=15, seed=3):
        store.record_session(session)
    assert store.pending_session_count() == 50

    record = store.maybe_retrain()
    assert record is not None
    assert record["n_sessions"] == 50
    assert record["n_new_sessions"] == 50
    assert store.model_path.exists()
    assert 0.0 <= record["bot_catch_rate_range"][0] <= record["bot_catch_rate_range"][1] <= 1.0
    assert sum(record["group_composition"].values()) == 50
    assert record["group_composition"] == store.group_composition()

    # threshold resets: no new sessions since this retrain
    assert store.pending_session_count() == 0
    assert store.maybe_retrain() is None


def test_maybe_retrain_force_bypasses_threshold(tmp_path):
    store = AutoRetrainStore(tmp_path / "proj", min_new_sessions=1000, cv_seeds=(0,))
    for session in make_synthetic_dataset(n_per_class=10, seed=4):
        store.record_session(session)

    assert store.maybe_retrain() is None  # far below threshold
    record = store.maybe_retrain(force=True)
    assert record is not None
    assert store.model_path.exists()


def test_maybe_retrain_backs_up_previous_model(tmp_path):
    store = AutoRetrainStore(tmp_path / "proj", min_new_sessions=20, cv_seeds=(0,))
    for session in make_synthetic_dataset(n_per_class=10, seed=5):
        store.record_session(session)
    store.maybe_retrain()
    first_model_bytes = store.model_path.read_bytes()

    for session in make_synthetic_dataset(n_per_class=10, seed=6):
        store.record_session(session)
    store.maybe_retrain()

    backups = list(store.root.glob("model.joblib.*.bak"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == first_model_bytes


def test_load_detector_before_any_retrain_raises(tmp_path):
    store = AutoRetrainStore(tmp_path / "proj")
    with pytest.raises(RuntimeError):
        store.load_detector()


def test_score_uses_the_deployed_model(tmp_path):
    store = AutoRetrainStore(tmp_path / "proj", min_new_sessions=20, cv_seeds=(0,))
    sessions = make_synthetic_dataset(n_per_class=15, seed=7)
    for session in sessions:
        store.record_session(session)
    store.maybe_retrain()

    score = store.score(sessions[0])
    assert 0.0 <= score <= 1.0


def test_state_persists_across_new_store_instances_pointed_at_same_root(tmp_path):
    root = tmp_path / "proj"
    store_a = AutoRetrainStore(root, min_new_sessions=20, cv_seeds=(0,))
    for session in make_synthetic_dataset(n_per_class=15, seed=8):
        store_a.record_session(session)
    record = store_a.maybe_retrain()
    assert record is not None

    store_b = AutoRetrainStore(root, min_new_sessions=20, cv_seeds=(0,))
    assert store_b.pending_session_count() == 0
    detector = store_b.load_detector()
    assert detector is not None
