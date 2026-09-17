from .autoretrain import AutoRetrainStore
from .detector import BotDetector
from .io import (
    load_sessions_jsonl,
    save_sessions_jsonl,
    session_from_dict,
    session_to_dict,
)
from .pipeline import (
    CVReport,
    MultiSeedReport,
    TrainingReport,
    cost_optimal_threshold,
    evaluate_cv,
    run_training_pipeline,
    summarize_across_seeds,
)
from .schema import (
    ClickEvent,
    FocusEvent,
    InteractionSession,
    KeyEvent,
    MouseEvent,
    PasteEvent,
    ScrollEvent,
)
from .session import FEATURE_NAMES, channel_coverage, extract_features

__all__ = [
    "BotDetector",
    "InteractionSession",
    "KeyEvent",
    "MouseEvent",
    "ScrollEvent",
    "ClickEvent",
    "FocusEvent",
    "PasteEvent",
    "FEATURE_NAMES",
    "extract_features",
    "channel_coverage",
    "load_sessions_jsonl",
    "save_sessions_jsonl",
    "session_from_dict",
    "session_to_dict",
    "run_training_pipeline",
    "TrainingReport",
    "evaluate_cv",
    "CVReport",
    "cost_optimal_threshold",
    "summarize_across_seeds",
    "MultiSeedReport",
    "AutoRetrainStore",
]

__version__ = "0.1.7"
