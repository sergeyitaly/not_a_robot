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
from sklearn.model_selection import StratifiedKFold, train_test_split

from .detector import BotDetector
from .schema import InteractionSession
from .session import FEATURE_NAMES, extract_features, to_vector


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

    Returns the fitted BotDetector plus a TrainingReport whose ``summary()``
    gives a practical read on verification success: human pass rate, bot
    catch rate, and the false accept/reject rates that trade off against
    each other.
    """
    labeled = [s for s in sessions if s.label is not None]
    if len(labeled) < 4:
        raise ValueError("need at least 4 labeled sessions to run the pipeline")
    if len({s.label for s in labeled}) < 2:
        raise ValueError("need both human and bot examples to run the pipeline")

    x = np.array([to_vector(extract_features(s)) for s in labeled])
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
        ranked = sorted(zip(FEATURE_NAMES, importances), key=lambda kv: -kv[1])
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
