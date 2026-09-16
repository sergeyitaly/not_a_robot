from __future__ import annotations

import math
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
    roc_curve,
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
    for bots. These are a **single train/test split's** metrics -- a
    point estimate with real sampling variance on a small dataset. Prefer
    ``evaluate_cv()`` (repeated CV, mean +/- std, per-group recall) for
    any number you intend to defend; this report is for producing the
    deployable detector, not for characterizing its accuracy.
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
            "=== not_a_robot training report (single split -- see evaluate_cv for a defensible number) ===",
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


def _get_feature_importances(model) -> Optional[np.ndarray]:
    """Extract feature importances, reaching through a CalibratedClassifierCV
    wrapper if needed (its ``feature_importances_`` isn't exposed at the top
    level -- it lives on each internally-fitted base estimator, one per
    calibration fold, so this averages across them)."""
    importances = getattr(model, "feature_importances_", None)
    if importances is not None:
        return importances

    calibrated = getattr(model, "calibrated_classifiers_", None)
    if not calibrated:
        return None

    per_fold = []
    for c in calibrated:
        estimator = getattr(c, "estimator", None) or getattr(c, "base_estimator", None)
        fold_importances = getattr(estimator, "feature_importances_", None)
        if fold_importances is not None:
            per_fold.append(fold_importances)
    return np.mean(per_fold, axis=0) if per_fold else None


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
    without the keystroke-timing features against one with them (a
    mouse-only capture surface), on the exact same data and train/test
    split, to isolate what a feature group actually contributes.

    Returns the fitted BotDetector plus a TrainingReport. See its
    docstring: this is a single-split point estimate, not the number to
    quote -- use ``evaluate_cv()`` for that.
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

    importances = _get_feature_importances(detector.model)
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


def _wilson_interval(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (default 95%).

    Unlike a normal-approximation +/- std, this stays well-behaved (and
    doesn't overshoot [0, 1]) at the small sample sizes a rare sub-group
    can have even after pooling across many CV folds.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def cost_optimal_threshold(
    y_true: np.ndarray, p_human: np.ndarray, c_fa: float = 10.0, c_fr: float = 1.0
) -> tuple[float, float, float, float]:
    """Find the P(human) threshold minimizing ``c_fa*FAR + c_fr*FRR``.

    ``c_fa``/``c_fr`` should reflect your deployment's actual cost
    asymmetry -- e.g. a false accept on a login or checkout form is
    usually far worse than a false reject (the default 10:1 is
    illustrative, not a recommendation; pass your own ratio). Returns
    ``(threshold, far, frr, cost)`` at the minimizing point.
    """
    fpr, tpr, thresholds = roc_curve(y_true, p_human, pos_label=1)
    far = fpr  # bots (y=0) scored >= threshold -> accepted as human
    frr = 1 - tpr  # humans (y=1) scored < threshold -> rejected
    cost = c_fa * far + c_fr * frr
    i = int(np.argmin(cost))
    threshold = float(np.clip(thresholds[i], 0.0, 1.0))
    return threshold, float(far[i]), float(frr[i]), float(cost[i])


def _pooled_group_stats(
    groups: list[str], hits: list[int], totals: list[int]
) -> tuple[dict[str, float], dict[str, float], dict[str, tuple[float, float]], dict[str, int], dict[str, int]]:
    """Pool per-session (hit, total) counts by group.

    Returns (weight, recall, 95% Wilson CI, pooled hit count, pooled
    total count) per group -- pooling raw counts, not averaging each
    session's own rate, so the resulting sample size is the sum across
    every fold instance (or a single instance per session, if called with
    totals of 1), not deflated by treating each session as one data point.
    """
    group_hits: dict[str, int] = {}
    group_totals: dict[str, int] = {}
    group_count: dict[str, int] = {}
    for g, h, t in zip(groups, hits, totals):
        group_count[g] = group_count.get(g, 0) + 1
        group_hits[g] = group_hits.get(g, 0) + h
        group_totals[g] = group_totals.get(g, 0) + t

    n_sessions = len(groups)
    weight = {g: c / n_sessions for g, c in group_count.items()}
    recall = {
        g: (group_hits[g] / group_totals[g] if group_totals[g] > 0 else float("nan"))
        for g in group_count
    }
    ci = {g: _wilson_interval(group_hits[g], group_totals[g]) for g in group_count}
    return weight, recall, ci, group_hits, group_totals


@dataclass
class CVReport:
    """Distribution of detection metrics across repeated stratified CV.

    A single train/test split's metrics are a point estimate with real
    sampling variance, especially on a dataset this size -- one lucky (or
    unlucky) split is not a number you can defend. This reports the mean
    and standard deviation of each metric across ``n_splits * n_repeats``
    independent test folds, plus recall pooled by ``InteractionSession.group``
    with a Wilson 95% confidence interval (not a +/- std of per-fold rates,
    which is close to meaningless for a rare group with only a handful of
    members per fold), and the operating point (threshold, FAR, FRR, and
    per-group recall at that threshold) that minimizes a stated false-accept
    vs. false-reject cost ratio instead of the classifier's default 0.5 cut.
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
    group_recall: dict[str, float]
    group_recall_ci: dict[str, tuple[float, float]]
    group_hits: dict[str, int]
    group_n: dict[str, int]
    group_n_per_fold_mean: dict[str, float]
    group_n_per_fold_min: dict[str, int]
    cost_fa: float
    cost_fr: float
    cost_threshold: Optional[float]
    cost_far: Optional[float]
    cost_frr: Optional[float]
    cost_group_recall: dict[str, float]
    cost_group_recall_ci: dict[str, tuple[float, float]]
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
        lines.append(
            "Per-group recall at the default 0.5 threshold (pooled hits/n "
            "across all folds, Wilson 95% CI -- not a mean of per-fold rates):"
        )
        lines.append(f"  {'group':<14}{'weight':>7}  {'n/fold':>12}  recall [95% CI]")
        ranked_groups = sorted(
            self.group_recall, key=lambda g: -self.group_weight.get(g, 0.0)
        )
        any_small_n = False
        for g in ranked_groups:
            w = self.group_weight.get(g, 0.0)
            n_mean = self.group_n_per_fold_mean.get(g, 0.0)
            n_min = self.group_n_per_fold_min.get(g, 0)
            r = self.group_recall[g]
            lo, hi = self.group_recall_ci[g]
            flag = " *" if n_min <= 2 else "  "
            any_small_n = any_small_n or n_min <= 2
            lines.append(
                f"  {g:<14}{w:>6.1%}  {n_mean:>5.1f} (min {n_min}){flag}  "
                f"{r:.1%} [{lo:.1%}-{hi:.1%}]  (n={self.group_n.get(g, 0)})"
            )
        if any_small_n:
            lines.append(
                "  * this group has <=2 members in its smallest fold -- the pooled"
            )
            lines.append(
                "    recall above (summed across all folds) is the defensible number,"
            )
            lines.append("    not any single fold's rate.")

        if self.cost_threshold is not None:
            lines.append("")
            lines.append(
                f"Cost-optimal threshold (c_fa={self.cost_fa:g}, c_fr={self.cost_fr:g}): "
                f"{self.cost_threshold:.3f}"
            )
            lines.append(
                f"  At that threshold: FAR {self.cost_far:.1%}, FRR {self.cost_frr:.1%}"
            )
            lines.append("  Per-group recall at the cost-optimal threshold:")
            for g in sorted(
                self.cost_group_recall, key=lambda g: -self.group_weight.get(g, 0.0)
            ):
                r = self.cost_group_recall[g]
                lo, hi = self.cost_group_recall_ci[g]
                lines.append(f"    {g:<14}{r:.1%} [{lo:.1%}-{hi:.1%}]")

        return "\n".join(lines)


@dataclass
class _FoldResult:
    accuracy: float
    bot_catch: float
    false_accept: float
    human_pass: float
    false_reject: float
    auc: Optional[float]


def _evaluate_fold(
    model, x_train, y_train, x_test, y_test
) -> tuple[_FoldResult, np.ndarray, Optional[np.ndarray]]:
    """Fit one clone of ``model`` on a train fold and score it on the test
    fold. Returns the fold's metrics, its raw predictions, and P(human)
    per test example when the model supports predict_proba (None
    otherwise) -- the caller tracks both for the per-group breakdowns."""
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    proba: Optional[np.ndarray]
    try:
        proba = model.predict_proba(x_test)[:, 1]
        auc = float(roc_auc_score(y_test, proba))
    except (ValueError, AttributeError):
        proba = None
        auc = None

    result = _FoldResult(
        accuracy=accuracy_score(y_test, preds),
        bot_catch=tn / (tn + fp) if (tn + fp) > 0 else float("nan"),
        false_accept=fp / (tn + fp) if (tn + fp) > 0 else float("nan"),
        human_pass=tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
        false_reject=fn / (tp + fn) if (tp + fn) > 0 else float("nan"),
        auc=auc,
    )
    return result, preds, proba


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
    cost_fa: float = 10.0,
    cost_fr: float = 1.0,
) -> CVReport:
    """Evaluate detection performance via repeated stratified k-fold CV.

    Unlike ``run_training_pipeline`` (which fits one deployable detector
    and evaluates it on one held-out split), this doesn't return a
    detector -- it exists purely to characterize expected performance and
    its variance. Runs ``n_splits x n_repeats`` independent stratified
    folds (a fresh model fit per fold) and reports:

    - mean +/- std for accuracy/human-pass/bot-catch/FAR/FRR/ROC-AUC
      across folds (this describes fold-partition variance on this one
      sample of sessions -- see ``pipeline.summarize_across_seeds`` for
      the more important cross-*sample* variance, by running this at
      several seeds and pooling);
    - recall pooled by ``InteractionSession.group`` (e.g. a synthetic
      archetype, or a known bot sub-type in real data) with a Wilson 95%
      CI and the mean/min number of that group's members per fold, since
      a rare group's per-fold recall is dominated by sampling noise, not
      model behavior, until pooled;
    - the threshold minimizing ``cost_fa*FAR + cost_fr*FRR`` (via
      out-of-fold P(human), averaged per session across every fold it
      appeared in as a test example), and per-group recall at that
      threshold instead of the classifier's default 0.5 cut.
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
    unique_groups = sorted(set(groups))

    base_model = BotDetector().model
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )

    fold_results: list[_FoldResult] = []
    hit_counts = [0] * len(labeled)
    total_counts = [0] * len(labeled)
    proba_sum = [0.0] * len(labeled)
    proba_complete = True
    group_fold_n: dict[str, list[int]] = {g: [] for g in unique_groups}

    for train_idx, test_idx in rskf.split(x, y):
        result, preds, proba = _evaluate_fold(
            clone(base_model), x[train_idx], y[train_idx], x[test_idx], y[test_idx]
        )
        fold_results.append(result)

        fold_group_tally: dict[str, int] = {}
        for local_i, global_i in enumerate(test_idx):
            total_counts[global_i] += 1
            if preds[local_i] == y[global_i]:
                hit_counts[global_i] += 1
            g = groups[global_i]
            fold_group_tally[g] = fold_group_tally.get(g, 0) + 1
            if proba is not None:
                proba_sum[global_i] += float(proba[local_i])
            else:
                proba_complete = False
        for g in unique_groups:
            group_fold_n[g].append(fold_group_tally.get(g, 0))

    group_weight, group_recall, group_recall_ci, group_hits, group_n = (
        _pooled_group_stats(groups, hit_counts, total_counts)
    )
    group_n_per_fold_mean = {g: float(np.mean(v)) for g, v in group_fold_n.items()}
    group_n_per_fold_min = {g: int(min(v)) for g, v in group_fold_n.items()}

    acc_mean, acc_std = _mean_std([r.accuracy for r in fold_results])
    human_pass_mean, human_pass_std = _mean_std([r.human_pass for r in fold_results])
    bot_catch_mean, bot_catch_std = _mean_std([r.bot_catch for r in fold_results])
    far_mean, far_std = _mean_std([r.false_accept for r in fold_results])
    frr_mean, frr_std = _mean_std([r.false_reject for r in fold_results])
    aucs = [r.auc for r in fold_results if r.auc is not None]
    auc_mean, auc_std = _mean_std(aucs) if aucs else (None, None)

    cost_threshold = cost_far = cost_frr = None
    cost_group_recall: dict[str, float] = {}
    cost_group_recall_ci: dict[str, tuple[float, float]] = {}
    if proba_complete and all(c > 0 for c in total_counts):
        avg_proba = np.array(
            [proba_sum[i] / total_counts[i] for i in range(len(labeled))]
        )
        cost_threshold, cost_far, cost_frr, _ = cost_optimal_threshold(
            y, avg_proba, c_fa=cost_fa, c_fr=cost_fr
        )
        decisions_correct = [
            int((avg_proba[i] >= cost_threshold) == bool(y[i]))
            for i in range(len(labeled))
        ]
        _, cost_group_recall, cost_group_recall_ci, _, _ = _pooled_group_stats(
            groups, decisions_correct, [1] * len(labeled)
        )

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
        group_recall=group_recall,
        group_recall_ci=group_recall_ci,
        group_hits=group_hits,
        group_n=group_n,
        group_n_per_fold_mean=group_n_per_fold_mean,
        group_n_per_fold_min=group_n_per_fold_min,
        cost_fa=cost_fa,
        cost_fr=cost_fr,
        cost_threshold=cost_threshold,
        cost_far=cost_far,
        cost_frr=cost_frr,
        cost_group_recall=cost_group_recall,
        cost_group_recall_ci=cost_group_recall_ci,
        feature_names=names,
        data_source=data_source,
    )


@dataclass
class MultiSeedReport:
    """Cross-seed summary: the honest operating characteristic.

    Repeated CV within one seed only captures fold-partition variance --
    all folds share the same underlying sample of sessions. Running
    ``evaluate_cv`` at several seeds and combining the results here
    exposes the variance that actually matters: how much the numbers
    move when the *sample* of sessions changes, not just how it's split.
    Group recall is pooled (raw hit/total counts summed) across every
    seed for the tightest defensible Wilson CI.
    """

    seeds: list[int]
    accuracy: list[float]
    human_pass_rate: list[float]
    bot_catch_rate: list[float]
    false_accept_rate: list[float]
    false_reject_rate: list[float]
    group_weight: dict[str, float]
    group_recall: dict[str, float]
    group_recall_ci: dict[str, tuple[float, float]]
    group_n: dict[str, int]
    data_source: str

    def summary(self) -> str:
        lines = [
            "=== not_a_robot multi-seed evaluation ===",
            f"Data source: {self.data_source}",
            f"Seeds: {self.seeds}",
            "",
            f"  {'seed':<6}{'accuracy':>10}{'human pass':>13}{'bot catch':>12}{'FAR':>8}{'FRR':>8}",
        ]
        for i, s in enumerate(self.seeds):
            lines.append(
                f"  {s:<6}{self.accuracy[i]:>9.1%} {self.human_pass_rate[i]:>12.1%} "
                f"{self.bot_catch_rate[i]:>11.1%} {self.false_accept_rate[i]:>7.1%} "
                f"{self.false_reject_rate[i]:>7.1%}"
            )
        lines.append("")
        lines.append(
            f"Bot catch rate range across seeds: {min(self.bot_catch_rate):.1%} - "
            f"{max(self.bot_catch_rate):.1%}  <- the honest operating characteristic"
        )
        lines.append("")
        lines.append("Combined per-group recall (pooled across all seeds, Wilson 95% CI):")
        lines.append(f"  {'group':<14}{'weight':>7}{'n':>8}   recall [95% CI]")
        ranked = sorted(self.group_recall, key=lambda g: -self.group_weight.get(g, 0.0))
        for g in ranked:
            w = self.group_weight.get(g, 0.0)
            n = self.group_n.get(g, 0)
            r = self.group_recall[g]
            lo, hi = self.group_recall_ci[g]
            lines.append(f"  {g:<14}{w:>6.1%}{n:>8}   {r:.1%} [{lo:.1%}-{hi:.1%}]")
        return "\n".join(lines)


def summarize_across_seeds(
    seed_reports: list[tuple[int, CVReport]], data_source: str = "multi-seed"
) -> MultiSeedReport:
    """Combine per-seed ``evaluate_cv`` results into a cross-seed summary.

    See ``MultiSeedReport`` for why this, not a single seed's repeated-CV
    std, is the number to quote as the pipeline's operating characteristic.
    """
    seeds = [s for s, _ in seed_reports]
    reports = [r for _, r in seed_reports]

    combined_hits: dict[str, int] = {}
    combined_n: dict[str, int] = {}
    weight_sum: dict[str, float] = {}
    for r in reports:
        for g, h in r.group_hits.items():
            combined_hits[g] = combined_hits.get(g, 0) + h
        for g, n in r.group_n.items():
            combined_n[g] = combined_n.get(g, 0) + n
        for g, w in r.group_weight.items():
            weight_sum[g] = weight_sum.get(g, 0.0) + w

    n_reports = len(reports)
    group_weight = {g: w / n_reports for g, w in weight_sum.items()}
    group_recall = {
        g: (combined_hits[g] / combined_n[g] if combined_n.get(g, 0) > 0 else float("nan"))
        for g in combined_n
    }
    group_recall_ci = {
        g: _wilson_interval(combined_hits.get(g, 0), combined_n[g]) for g in combined_n
    }

    return MultiSeedReport(
        seeds=seeds,
        accuracy=[r.accuracy_mean for r in reports],
        human_pass_rate=[r.human_pass_rate_mean for r in reports],
        bot_catch_rate=[r.bot_catch_rate_mean for r in reports],
        false_accept_rate=[r.false_accept_rate_mean for r in reports],
        false_reject_rate=[r.false_reject_rate_mean for r in reports],
        group_weight=group_weight,
        group_recall=group_recall,
        group_recall_ci=group_recall_ci,
        group_n=combined_n,
        data_source=data_source,
    )
