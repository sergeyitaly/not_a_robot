"""Command-line entry point for the training/enrichment pipeline.

Usage:
    python -m not_a_robot.train --data sessions.jsonl --model-out model.joblib
    python -m not_a_robot.train --synthetic --n-per-class 150

``--synthetic`` runs the pipeline against the bundled synthetic demo dataset
(examples/synthetic_data.py) so you can see the full pipeline and report
end-to-end before you have real, labeled traffic captured from your own
site. It must be run from the repository root. Everywhere else, point
``--data`` at a JSONL file of sessions written by ``not_a_robot.io``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .io import load_sessions_jsonl
from .pipeline import run_training_pipeline
from .schema import InteractionSession


def _load_synthetic(n_per_class: int, seed: int) -> list[InteractionSession]:
    try:
        from examples.synthetic_data import make_synthetic_dataset
    except ImportError as exc:
        raise SystemExit(
            "--synthetic requires running from the repository root "
            "(so the examples package is importable)"
        ) from exc
    return make_synthetic_dataset(n_per_class=n_per_class, seed=seed)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="JSONL file of labeled sessions")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="use the bundled synthetic demo dataset instead of --data",
    )
    parser.add_argument("--n-per-class", type=int, default=150)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-out", type=Path, default=Path("bot_detector.joblib"))
    parser.add_argument("--report-out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.synthetic:
        sessions = _load_synthetic(args.n_per_class, args.seed)
        data_source = (
            f"synthetic demo data (n_per_class={args.n_per_class}, "
            f"seed={args.seed}) -- NOT real traffic"
        )
    elif args.data:
        sessions = load_sessions_jsonl(args.data)
        data_source = str(args.data)
    else:
        parser.error("pass --data <file.jsonl> or --synthetic")
        return 2

    detector, report = run_training_pipeline(
        sessions,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        seed=args.seed,
        data_source=data_source,
    )

    print(report.summary())

    detector.save(args.model_out)
    print(f"\nSaved trained model to {args.model_out}")

    if args.report_out:
        with open(args.report_out, "w", encoding="utf-8") as f:
            json.dump(report.__dict__, f, indent=2)
        print(f"Saved report to {args.report_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
