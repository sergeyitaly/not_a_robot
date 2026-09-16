from .detector import BotDetector
from .io import load_sessions_jsonl, save_sessions_jsonl
from .pipeline import TrainingReport, run_training_pipeline
from .schema import InteractionSession, KeyEvent, MouseEvent
from .session import FEATURE_NAMES, extract_features

__all__ = [
    "BotDetector",
    "InteractionSession",
    "KeyEvent",
    "MouseEvent",
    "FEATURE_NAMES",
    "extract_features",
    "load_sessions_jsonl",
    "save_sessions_jsonl",
    "run_training_pipeline",
    "TrainingReport",
]

__version__ = "0.1.0"
