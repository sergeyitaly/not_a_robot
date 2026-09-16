from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

import joblib
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

from .schema import InteractionSession
from .session import FEATURE_NAMES, extract_features, to_vector


class BotDetector:
    """Scores interaction sessions on how human-like their behavior is.

    Train it on sessions captured and labeled from your own application
    (label=True for a known-human session, label=False for a known-bot
    or scripted session), then call ``score``/``predict`` on new sessions.

    This is a behavioral-risk signal meant to sit alongside an actual
    challenge or auth flow, not to replace one -- treat its output as one
    input to a decision, not a verdict.

    The default model wraps a ``RandomForestClassifier`` in
    ``CalibratedClassifierCV``. A raw random forest's ``predict_proba`` is
    a vote fraction, not a real probability -- checked against a
    reliability diagram, most of its mass sits at the extremes with a
    sparse, badly-calibrated middle (a handful of samples per 0.1-wide
    bin, mean predicted probability not tracking the observed human
    fraction there). That's fine for the classifier's own 0.5 decision,
    but it makes any *other* threshold -- e.g. the cost-optimal one in
    ``pipeline.cost_optimal_threshold`` -- unreliable, since the sweep is
    hunting through that noisy, sparsely-populated region. Calibration
    fixes that; pass your own uncalibrated model via ``model=`` if you
    specifically want to skip it (e.g. to reproduce that failure mode).
    """

    def __init__(self, model: Optional[ClassifierMixin] = None):
        self.model = model or CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=200, max_depth=8, random_state=0),
            method="isotonic",
            cv=3,
        )
        self._fitted = False

    def fit(self, sessions: Iterable[InteractionSession]) -> "BotDetector":
        labeled = [s for s in sessions if s.label is not None]
        if len(labeled) < 2:
            raise ValueError("fit() requires at least two labeled sessions")
        if len({s.label for s in labeled}) < 2:
            raise ValueError("fit() requires both human and bot examples")

        x = np.array([to_vector(extract_features(s)) for s in labeled])
        y = np.array([1 if s.label else 0 for s in labeled])
        self.model.fit(x, y)
        self._fitted = True
        return self

    def score(self, session: InteractionSession) -> float:
        """Return P(human) in [0, 1] for a single session."""
        if not self._fitted:
            raise RuntimeError("BotDetector must be fit() before scoring")
        x = np.array([to_vector(extract_features(session))])
        return float(self.model.predict_proba(x)[0][1])

    def predict(self, session: InteractionSession, threshold: float = 0.5) -> bool:
        """Return True if the session looks human, at the given threshold."""
        return self.score(session) >= threshold

    def save(self, path: Union[str, Path]) -> None:
        joblib.dump({"model": self.model, "features": FEATURE_NAMES}, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BotDetector":
        payload = joblib.load(path)
        if tuple(payload["features"]) != FEATURE_NAMES:
            raise ValueError(
                "saved model's feature set does not match this version of "
                "not_a_robot -- retrain before loading"
            )
        instance = cls(model=payload["model"])
        instance._fitted = True
        return instance
