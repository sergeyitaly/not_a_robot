"""End-to-end example: train a BotDetector and score a session.

Run from the repo root with:

    python -m examples.quickstart

Replace ``make_synthetic_dataset()`` with real, labeled sessions captured
from your own application before using this for anything real.
"""

from __future__ import annotations

from examples.synthetic_data import make_synthetic_dataset
from not_a_robot import BotDetector


def main() -> None:
    sessions = make_synthetic_dataset(n_per_class=100)
    split = int(len(sessions) * 0.75)
    train, test = sessions[:split], sessions[split:]

    detector = BotDetector()
    detector.fit(train)

    correct = sum(
        detector.predict(session) == bool(session.label) for session in test
    )
    print(f"Accuracy on held-out synthetic data: {correct / len(test):.2%}")

    detector.save("bot_detector.joblib")
    print("Saved model to bot_detector.joblib")


if __name__ == "__main__":
    main()
