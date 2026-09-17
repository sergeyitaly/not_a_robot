from __future__ import annotations

import os
import warnings
from collections import Counter
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Iterable, Optional, Union

import joblib
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

from .schema import InteractionSession
from .session import FEATURE_NAMES, extract_features, to_vector


def _library_version() -> str:
    try:
        return importlib_metadata.version("not-a-robot")
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


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
        labels = [s.label for s in labeled]
        if len(set(labels)) < 2:
            raise ValueError("fit() requires both human and bot examples")

        # The default model is a CalibratedClassifierCV, which internally
        # runs stratified k-fold CV and raises an opaque sklearn error
        # ("Requesting k-fold cross-validation but provided less than k
        # examples for at least one class") if a class has fewer members
        # than that. Catch it here with a message that says what to do.
        if isinstance(self.model, CalibratedClassifierCV) and isinstance(
            self.model.cv, int
        ):
            min_count = min(Counter(labels).values())
            if min_count < self.model.cv:
                raise ValueError(
                    f"fit() requires at least {self.model.cv} sessions of "
                    f"each label for {self.model.cv}-fold calibration (got "
                    f"{min_count} of the minority label) -- collect more "
                    f"labeled data, or pass BotDetector(model=...) with a "
                    f"classifier that needs less"
                )

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
        """Save this detector, plus the feature set and library version it
        was trained against, so ``load()`` can detect incompatibility.

        Writes to a temporary file in the same directory and ``os.replace``s
        it into place, so a crash or kill mid-write leaves the previous
        file at ``path`` intact instead of a truncated, unloadable one.
        """
        path = Path(path)
        tmp_path = path.with_name(path.name + ".tmp")
        payload = {
            "model": self.model,
            "features": FEATURE_NAMES,
            "library_version": _library_version(),
        }
        joblib.dump(payload, tmp_path)
        os.replace(tmp_path, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BotDetector":
        """Load a previously saved detector.

        Uses ``joblib.load``, which unpickles arbitrary Python objects --
        the same trust model as ``pickle.load``. Only load files your own
        training pipeline produced (this library's own ``save()`` output,
        or an ``AutoRetrainStore``'s ``model.joblib``); loading a file from
        an untrusted source means arbitrary code execution, not a
        theoretical risk.
        """
        payload = joblib.load(path)
        if tuple(payload["features"]) != FEATURE_NAMES:
            raise ValueError(
                "saved model's feature set does not match this version of "
                "not_a_robot -- retrain before loading"
            )
        saved_version = payload.get("library_version", "unknown")
        current_version = _library_version()
        if saved_version != current_version:
            warnings.warn(
                f"model was saved with not-a-robot {saved_version}, "
                f"currently running {current_version} -- the feature set "
                f"matches, but consider retraining if scores look off",
                stacklevel=2,
            )
        instance = cls(model=payload["model"])
        instance._fitted = True
        return instance
