"""Per-project local auto-retrain store.

Each project that uses ``not_a_robot`` gets its own :class:`AutoRetrainStore`
rooted at its own directory -- no data or model is shared across projects.
The store accumulates trusted-labeled sessions and periodically retrains and
redeploys its own detector once enough new ones have arrived.

Scope, deliberately:

- **Local per project.** This is not a shared service across applications;
  point each project at its own root directory. If you run several
  applications and genuinely want one detector trained on all of them,
  call ``record_session`` against the same store from each, but that's a
  decision to make explicitly per project, not a default.
- **Trusted labels only, structurally.** ``record_session`` raises if the
  session has no label. There is no code path anywhere in this store that
  invents a label from the detector's own predictions or from unverified
  traffic -- the caller is always the one asserting ground truth (a
  CAPTCHA outcome, a verified signup, manual review, etc.), exactly as
  with the rest of this library. Auto-retrain only automates *when*
  training happens, never *what counts as truth*.
- **Not for the request hot path.** ``maybe_retrain()`` fits a
  detector and runs multi-seed CV -- tens of seconds, not milliseconds.
  Call ``record_session()`` from your live scoring path (a cheap file
  append); call ``maybe_retrain()`` from a separate periodic job (cron,
  a scheduled task, `python -m not_a_robot.autoretrain` on a timer).
- **Single-writer.** File appends aren't coordinated across processes.
  If multiple processes call ``record_session`` concurrently, put a real
  datastore or a lock in front of this, or accept the small risk of an
  interleaved write on your platform.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence, Union

from .detector import BotDetector
from .io import append_session_jsonl, load_sessions_jsonl
from .pipeline import evaluate_cv, run_training_pipeline, summarize_across_seeds
from .schema import InteractionSession


class AutoRetrainStore:
    """Local, per-project session log + model + retrain history.

    Layout under ``root``:

    - ``sessions.jsonl`` -- every session passed to ``record_session``
    - ``model.joblib`` -- the currently deployed detector
    - ``model.joblib.<timestamp>.bak`` -- the model this replaced, kept
      for rollback (never deleted automatically)
    - ``state.json`` -- retrain bookkeeping and history
    """

    def __init__(
        self,
        root: Union[str, Path],
        min_new_sessions: int = 50,
        cv_seeds: Sequence[int] = (0, 1, 2),
        feature_names: Optional[tuple[str, ...]] = None,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.sessions_path = self.root / "sessions.jsonl"
        self.model_path = self.root / "model.joblib"
        self.state_path = self.root / "state.json"
        self.min_new_sessions = min_new_sessions
        self.cv_seeds = tuple(cv_seeds)
        self.feature_names = feature_names
        self._detector_cache: Optional[BotDetector] = None

    def record_session(self, session: InteractionSession) -> None:
        """Append one trusted-labeled session to this project's log.

        Raises ``ValueError`` if ``session.label`` is ``None`` -- this
        store never accepts a session without an asserted ground truth.
        """
        if session.label is None:
            raise ValueError(
                "AutoRetrainStore.record_session() requires a trusted "
                "label (session.label must be True or False). Score "
                "unlabeled sessions with score()/load_detector() instead "
                "of recording them here."
            )
        append_session_jsonl(session, self.sessions_path)

    def _load_all_sessions(self) -> list[InteractionSession]:
        if not self.sessions_path.exists():
            return []
        return load_sessions_jsonl(self.sessions_path)

    def _load_state(self) -> dict:
        if not self.state_path.exists():
            return {"n_sessions_at_last_retrain": 0, "history": []}
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _save_state(self, state: dict) -> None:
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def pending_session_count(self) -> int:
        """How many labeled sessions have arrived since the last retrain."""
        state = self._load_state()
        return len(self._load_all_sessions()) - state.get("n_sessions_at_last_retrain", 0)

    def maybe_retrain(self, force: bool = False) -> Optional[dict]:
        """Retrain and redeploy if enough new sessions have accumulated.

        Fits a fresh :class:`BotDetector` on every labeled session
        recorded so far, runs multi-seed repeated CV
        (``self.cv_seeds``) for an honest report, backs up the
        previous model (if any), and writes the new one. Returns a
        summary dict of the retrain, or ``None`` if the threshold
        hasn't been reached (or there's not enough data at all) and
        ``force`` is False.
        """
        sessions = self._load_all_sessions()
        state = self._load_state()
        n_last = state.get("n_sessions_at_last_retrain", 0)

        if len(sessions) < 4:
            return None
        if not force and len(sessions) - n_last < self.min_new_sessions:
            return None

        data_source = f"{self.sessions_path} ({len(sessions)} labeled sessions)"

        detector, _ = run_training_pipeline(
            sessions, data_source=data_source, feature_names=self.feature_names
        )

        seed_reports = [
            (
                s,
                evaluate_cv(
                    sessions,
                    seed=s,
                    data_source=data_source,
                    feature_names=self.feature_names,
                ),
            )
            for s in self.cv_seeds
        ]
        combined = summarize_across_seeds(seed_reports, data_source=data_source)

        if self.model_path.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup_path = self.model_path.parent / f"{self.model_path.name}.{stamp}.bak"
            shutil.copy2(self.model_path, backup_path)

        detector.save(self.model_path)
        self._detector_cache = detector

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_sessions": len(sessions),
            "n_new_sessions": len(sessions) - n_last,
            "seeds": list(self.cv_seeds),
            "accuracy_range": [min(combined.accuracy), max(combined.accuracy)],
            "human_pass_rate_range": [
                min(combined.human_pass_rate),
                max(combined.human_pass_rate),
            ],
            "bot_catch_rate_range": [
                min(combined.bot_catch_rate),
                max(combined.bot_catch_rate),
            ],
        }
        state["n_sessions_at_last_retrain"] = len(sessions)
        state.setdefault("history", []).append(record)
        self._save_state(state)

        return record

    def load_detector(self) -> BotDetector:
        """Load (and cache) the currently deployed model."""
        if self._detector_cache is not None:
            return self._detector_cache
        if not self.model_path.exists():
            raise RuntimeError(
                f"no trained model yet at {self.model_path} -- call "
                "record_session() enough times and maybe_retrain(), or "
                "maybe_retrain(force=True) to train immediately"
            )
        self._detector_cache = BotDetector.load(self.model_path)
        return self._detector_cache

    def score(self, session: InteractionSession) -> float:
        """Score a session with the currently deployed model."""
        return self.load_detector().score(session)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check/run an AutoRetrainStore's retrain threshold. "
        "Meant to be invoked periodically (cron, a scheduled task), not "
        "from a live request path."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--min-new-sessions", type=int, default=50)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    store = AutoRetrainStore(args.root, min_new_sessions=args.min_new_sessions, cv_seeds=seeds)

    pending = store.pending_session_count()
    record = store.maybe_retrain(force=args.force)

    if record is None:
        print(
            f"No retrain: {pending} new labeled session(s) since last retrain "
            f"(threshold {args.min_new_sessions})."
        )
        return 0

    print(f"Retrained on {record['n_sessions']} sessions ({record['n_new_sessions']} new).")
    print(f"Accuracy range:       {record['accuracy_range'][0]:.1%} - {record['accuracy_range'][1]:.1%}")
    print(f"Human pass rate range: {record['human_pass_rate_range'][0]:.1%} - {record['human_pass_rate_range'][1]:.1%}")
    print(f"Bot catch rate range:  {record['bot_catch_rate_range'][0]:.1%} - {record['bot_catch_rate_range'][1]:.1%}")
    print(f"Model saved to {store.model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
