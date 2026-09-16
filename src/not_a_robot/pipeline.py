from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)

from .detector import BotDetector
from .schema import InteractionSession
from .session import FEATURE_NAMES as _ALL_FEATURE_NAMES
from .session import extract_features, to_vector


@dataclass
class TrainingReport:
    """Practical, verification-facing results from one pipeline run.

    ``test_recall`` doubles as the human pass rate (how often a real user
    is correctly let through) and ``bot_catch_rate`` is the mirror image
    for bots -- these two numbers, plus their error counterparts, are the
    ones that matter for deciding whether a detector is production-ready.
    """

    n_train: int
    n_test: int
    train_class_balance: dict[str, int]
    test_class_balance: dict[str, int]
    cv_accuracy_mean: float
    cv_accuracy_std: float
    test_accuracy: float
    test_precision: float
    test_recall: float
    test_f1: float
    test_roc_auc: Optional[float]
    bot_catch_rate: float
    false_accept_rate: float
    false_reject_rate: float
    confusion_matrix: list[list[int]]
    top_features: list[tuple[str, float]]
    data_source: str

    def summary(self) -> str:
        lines = [
            "=== not_a_robot training report ===",
            f"Data source: {self.data_source}",
            f"Training sessions: {self.n_train} "
            f"({self.train_class_balance['human']} human / "
            f"{self.train_class_balance['bot']} bot)",
            f"Held-out test sessions: {self.n_test} "
            f"({self.test_class_balance['human']} human / "
            f"{self.test_class_balance['bot']} bot)",
            "",
            "Cross-validated accuracy: "
            f"{self.cv_accuracy_mean:.1%} +/- {self.cv_accuracy_std:.1%}",
            "",
            "Held-out test results:",
            f"  Overall accuracy:   {self.test_accuracy:.1%}",
            f"  Human pass rate:    {self.test_recall:.1%}  "
            "(real users correctly verified as human)",
            f"  Bot catch rate:     {self.bot_catch_rate:.1%}  "
            "(bots correctly blocked)",
            f"  False accept rate:  {self.false_accept_rate:.1%}  "
            "(bots that slipped through as human)",
            f"  False reject rate:  {self.false_reject_rate:.1%}  "
            "(real users wrongly blocked)",
        ]
        if self.test_roc_auc is not None:
            lines.append(f"  ROC-AUC:            {self.test_roc_auc:.3f}")
        lines.append("")
        lines.append("Top features by importance:")
        for i, (name, importance) in enumerate(self.top_features, start=1):
            lines.append(f"  {i}. {name:<28} {importance:.3f}")
        return "\n".join(lines)


def _class_balance(sessions: list[InteractionSession]) -> dict[str, int]:
    return {
        "human": sum(1 for s in sessions if s.label is True),
        "bot": sum(1 for s in sessions if s.label is False),
    }


def run_training_pipeline(
    sessions: list[InteractionSession],
    test_size: float = 0.25,
    cv_folds: int = 5,
    seed: int = 0,
    top_n_features: int = 8,
    data_source: str = "unspecified",
    feature_names: Optional[tuple[str, ...]] = None,
) -> tuple[BotDetector, TrainingReport]:
    """Enrich, split, cross-validate, train, and evaluate a BotDetector.

    1. Extracts the full (base + enrichment) feature set for every labeled
       session.
    2. Splits off a stratified held-out test set that the model never
       trains on.
    3. Runs stratified k-fold cross-validation on the remaining training
       data for a robustness estimate that isn't just one lucky split.
    4. Fits the final detector on the full training set and evaluates it
       once on the held-out test set.

    ``feature_names`` restricts training/scoring to a subset of
    ``not_a_robot.session.FEATURE_NAMES`` (default: all of them). This
    exists for ablation studies -- e.g. comparing a detector trained
    without the scroll/click/engagement features against one with them,
    on the exact same data and train/test split, to isolate what a new
    feature group actually contributes rather than confounding it with a
    change in data.

    Returns the fitted BotDetector plus a TrainingReport whose ``summary()``
    gives a practical read on verification success: human pass rate, bot
    catch rate, and the false accept/reject rates that trade off against
    each other.
    """
    names = feature_names or _ALL_FEATURE_NAMES

    labeled = [s for s in sessions if s.label is not None]
    if len(labeled) < 4:
        raise ValueError("need at least 4 labeled sessions to run the pipeline")
    if len({s.label for s in labeled}) < 2:
        raise ValueError("need both human and bot examples to run the pipeline")

    x = np.array([to_vector(extract_features(s), names) for s in labeled])
    y = np.array([1 if s.label else 0 for s in labeled])

    x_train, x_test, y_train, y_test, train_sessions, test_sessions = train_test_split(
        x, y, labeled, test_size=test_size, stratify=y, random_state=seed
    )

    detector = BotDetector()

    min_class_count = int(min(np.bincount(y_train)))
    n_splits = min(cv_folds, min_class_count)
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        cv_scores = []
        for train_idx, val_idx in cv.split(x_train, y_train):
            fold_model = clone(detector.model)
            fold_model.fit(x_train[train_idx], y_train[train_idx])
            preds = fold_model.predict(x_train[val_idx])
            cv_scores.append(accuracy_score(y_train[val_idx], preds))
        cv_scores_arr = np.array(cv_scores)
    else:
        cv_scores_arr = np.array([float("nan")])

    detector.model.fit(x_train, y_train)
    detector._fitted = True

    y_pred = detector.model.predict(x_test)
    y_proba = detector.model.predict_proba(x_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    bot_catch_rate = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    false_accept_rate = fp / (tn + fp) if (tn + fp) > 0 else float("nan")
    false_reject_rate = fn / (fn + tp) if (fn + tp) > 0 else float("nan")

    try:
        roc_auc: Optional[float] = float(roc_auc_score(y_test, y_proba))
    except ValueError:
        roc_auc = None

    importances = getattr(detector.model, "feature_importances_", None)
    if importances is not None:
        ranked = sorted(zip(names, importances), key=lambda kv: -kv[1])
        top_features = [(name, float(v)) for name, v in ranked[:top_n_features]]
    else:
        top_features = []

    report = TrainingReport(
        n_train=len(train_sessions),
        n_test=len(test_sessions),
        train_class_balance=_class_balance(train_sessions),
        test_class_balance=_class_balance(test_sessions),
        cv_accuracy_mean=float(np.nanmean(cv_scores_arr)),
        cv_accuracy_std=float(np.nanstd(cv_scores_arr)),
        test_accuracy=float(accuracy_score(y_test, y_pred)),
        test_precision=float(precision_score(y_test, y_pred, zero_division=0)),
        test_recall=float(recall_score(y_test, y_pred, zero_division=0)),
        test_f1=float(f1_score(y_test, y_pred, zero_division=0)),
        test_roc_auc=roc_auc,
        bot_catch_rate=float(bot_catch_rate),
        false_accept_rate=float(false_accept_rate),
        false_reject_rate=float(false_reject_rate),
        confusion_matrix=cm,
        top_features=top_features,
        data_source=data_source,
    )
    return detector, report


@dataclass
class CVReport:
    """Distribution of detection metrics across repeated stratified CV.

    A single train/test split's metrics are a point estimate with real
    sampling variance, especially on a dataset this size -- one lucky (or
    unlucky) split is not a number you can defend. This reports the mean
    and standard deviation of each metric across ``n_splits * n_repeats``
    independent test folds, plus per-group recall (via
    ``InteractionSession.group``) so an aggregate bot-catch-rate number
    can't hide that all the errors are concentrated in one sub-population.
    """

    n_sessions: int
    n_splits: int
    n_repeats: int
    accuracy_mean: float
    accuracy_std: float
    human_pass_rate_mean: float
    human_pass_rate_std: float
    bot_catch_rate_mean: float
    bot_catch_rate_std: float
    false_accept_rate_mean: float
    false_accept_rate_std: float
    false_reject_rate_mean: float
    false_reject_rate_std: float
    roc_auc_mean: Optional[float]
    roc_auc_std: Optional[float]
    group_weight: dict[str, float]
    group_recall_mean: dict[str, float]
    group_recall_std: dict[str, float]
    feature_names: tuple[str, ...]
    data_source: str

    def summary(self) -> str:
        n_folds = self.n_splits * self.n_repeats
        lines = [
            "=== not_a_robot repeated-CV evaluation ===",
            f"Data source: {self.data_source}",
            f"Sessions: {self.n_sessions} "
            f"({self.n_splits}-fold x {self.n_repeats} repeats = {n_folds} test folds)",
            "",
            f"Accuracy:           {self.accuracy_mean:.1%} +/- {self.accuracy_std:.1%}",
            f"Human pass rate:    {self.human_pass_rate_mean:.1%} +/- "
            f"{self.human_pass_rate_std:.1%}",
            f"Bot catch rate:     {self.bot_catch_rate_mean:.1%} +/- "
            f"{self.bot_catch_rate_std:.1%}",
            f"False accept rate:  {self.false_accept_rate_mean:.1%} +/- "
            f"{self.false_accept_rate_std:.1%}",
            f"False reject rate:  {self.false_reject_rate_mean:.1%} +/- "
            f"{self.false_reject_rate_std:.1%}",
        ]
        if self.roc_auc_mean is not None:
            lines.append(
                f"ROC-AUC:            {self.roc_auc_mean:.3f} +/- {self.roc_auc_std:.3f}"
            )
        lines.append("")
        lines.append("Per-group recall (mean +/- std across the folds each session")
        lines.append("appeared in as a test example):")
        header = f"  {'group':<16}{'weight':>8}   recall"
        lines.append(header)
        ranked_groups = sorted(
            self.group_recall_mean, key=lambda g: -self.group_weight.get(g, 0.0)
        )
        for g in ranked_groups:
            w = self.group_weight.get(g, 0.0)
            mean_r = self.group_recall_mean[g]
            std_r = self.group_recall_std[g]
            lines.append(f"  {g:<16}{w:>7.1%}   {mean_r:.1%} +/- {std_r:.1%}")
        return "\n".join(lines)


@dataclass
class _FoldResult:
    accuracy: float
    bot_catch: float
    false_accept: float
    human_pass: float
    false_reject: float
    auc: Optional[float]


def _evaluate_fold(model, x_train, y_train, x_test, y_test) -> tuple[_FoldResult, np.ndarray]:
    """Fit one clone of ``model`` on a train fold and score it on the test
    fold. Returns the fold's metrics plus its raw predictions (the caller
    tracks per-session hit rates for the per-group recall breakdown)."""
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    try:
        proba = model.predict_proba(x_test)[:, 1]
        auc = float(roc_auc_score(y_test, proba))
    except ValueError:
        auc = None

    result = _FoldResult(
        accuracy=accuracy_score(y_test, preds),
        bot_catch=tn / (tn + fp) if (tn + fp) > 0 else float("nan"),
        false_accept=fp / (tn + fp) if (tn + fp) > 0 else float("nan"),
        human_pass=tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
        false_reject=fn / (tp + fn) if (tp + fn) > 0 else float("nan"),
        auc=auc,
    )
    return result, preds


def _aggregate_group_recall(
    groups: list[str], hit_counts: list[int], total_counts: list[int]
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Turn per-session hit/total counts into per-group weight and recall
    mean/std, using each session's own hit rate across the folds it
    appeared in as the unit of aggregation."""
    per_group_rates: dict[str, list[float]] = {}
    per_group_count: dict[str, int] = {}
    for i, g in enumerate(groups):
        per_group_count[g] = per_group_count.get(g, 0) + 1
        if total_counts[i] > 0:
            per_group_rates.setdefault(g, []).append(hit_counts[i] / total_counts[i])

    total = len(groups)
    weight = {g: c / total for g, c in per_group_count.items()}
    mean = {g: float(np.mean(vals)) for g, vals in per_group_rates.items()}
    std = {g: float(np.std(vals)) for g, vals in per_group_rates.items()}
    return weight, mean, std


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def evaluate_cv(
    sessions: list[InteractionSession],
    n_splits: int = 5,
    n_repeats: int = 10,
    seed: int = 0,
    feature_names: Optional[tuple[str, ...]] = None,
    data_source: str = "unspecified",
) -> CVReport:
    """Evaluate detection performance via repeated stratified k-fold CV.

    Unlike ``run_training_pipeline`` (which fits one deployable detector
    and evaluates it on one held-out split), this doesn't return a
    detector -- it exists purely to characterize expected performance and
    its variance. Runs ``n_splits x n_repeats`` independent stratified
    folds (a fresh model fit per fold) and reports mean +/- std for each
    metric, plus recall broken out by ``InteractionSession.group`` when
    set (e.g. a synthetic archetype label, or a known bot sub-type in
    real data), computed from each session's hit rate across every fold
    where it landed in the test split.
    """
    names = feature_names or _ALL_FEATURE_NAMES

    labeled = [s for s in sessions if s.label is not None]
    if len(labeled) < n_splits * 2:
        raise ValueError(
            f"need at least {n_splits * 2} labeled sessions for {n_splits}-fold CV"
        )
    if len({s.label for s in labeled}) < 2:
        raise ValueError("need both human and bot examples to run evaluation")

    x = np.array([to_vector(extract_features(s), names) for s in labeled])
    y = np.array([1 if s.label else 0 for s in labeled])
    groups = [
        s.group if s.group is not None else ("human" if s.label else "bot")
        for s in labeled
    ]

    base_model = BotDetector().model
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )

    fold_results: list[_FoldResult] = []
    hit_counts = [0] * len(labeled)
    total_counts = [0] * len(labeled)

    for train_idx, test_idx in rskf.split(x, y):
        result, preds = _evaluate_fold(
            clone(base_model), x[train_idx], y[train_idx], x[test_idx], y[test_idx]
        )
        fold_results.append(result)
        for local_i, global_i in enumerate(test_idx):
            total_counts[global_i] += 1
            if preds[local_i] == y[global_i]:
                hit_counts[global_i] += 1

    group_weight, group_recall_mean, group_recall_std = _aggregate_group_recall(
        groups, hit_counts, total_counts
    )

    acc_mean, acc_std = _mean_std([r.accuracy for r in fold_results])
    human_pass_mean, human_pass_std = _mean_std([r.human_pass for r in fold_results])
    bot_catch_mean, bot_catch_std = _mean_std([r.bot_catch for r in fold_results])
    far_mean, far_std = _mean_std([r.false_accept for r in fold_results])
    frr_mean, frr_std = _mean_std([r.false_reject for r in fold_results])
    aucs = [r.auc for r in fold_results if r.auc is not None]
    auc_mean, auc_std = _mean_std(aucs) if aucs else (None, None)

    return CVReport(
        n_sessions=len(labeled),
        n_splits=n_splits,
        n_repeats=n_repeats,
        accuracy_mean=acc_mean,
        accuracy_std=acc_std,
        human_pass_rate_mean=human_pass_mean,
        human_pass_rate_std=human_pass_std,
        bot_catch_rate_mean=bot_catch_mean,
        bot_catch_rate_std=bot_catch_std,
        false_accept_rate_mean=far_mean,
        false_accept_rate_std=far_std,
        false_reject_rate_mean=frr_mean,
        false_reject_rate_std=frr_std,
        roc_auc_mean=auc_mean,
        roc_auc_std=auc_std,
        group_weight=group_weight,
        group_recall_mean=group_recall_mean,
        group_recall_std=group_recall_std,
        feature_names=names,
        data_source=data_source,
    )
