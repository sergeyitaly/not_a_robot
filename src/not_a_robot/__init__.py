from .detector import BotDetector
from .io import load_sessions_jsonl, save_sessions_jsonl
from .pipeline import CVReport, TrainingReport, evaluate_cv, run_training_pipeline
from .schema import (
    ClickEvent,
    FocusEvent,
    InteractionSession,
    KeyEvent,
    MouseEvent,
    PasteEvent,
    ScrollEvent,
)
from .session import FEATURE_NAMES, extract_features

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
    "load_sessions_jsonl",
    "save_sessions_jsonl",
    "run_training_pipeline",
    "TrainingReport",
    "evaluate_cv",
    "CVReport",
]

__version__ = "0.1.0"
