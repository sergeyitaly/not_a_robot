"""End-to-end example: run the training pipeline and print its report.

Run from the repo root with:

    python -m examples.quickstart

Replace ``make_synthetic_dataset()`` with real, labeled sessions captured
from your own application (see ``not_a_robot.io``) before trusting these
numbers, or run ``python -m not_a_robot.train --data sessions.jsonl``
directly on a real session log.
"""

from __future__ import annotations

from examples.synthetic_data import make_synthetic_dataset
from not_a_robot.pipeline import run_training_pipeline


def main() -> None:
    sessions = make_synthetic_dataset(n_per_class=150)

    detector, report = run_training_pipeline(
        sessions,
        data_source="synthetic demo data (examples/synthetic_data.py) -- NOT real traffic",
    )

    print(report.summary())

    detector.save("bot_detector.joblib")
    print("\nSaved trained model to bot_detector.joblib")


if __name__ == "__main__":
    main()
