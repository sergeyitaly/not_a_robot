from .detector import BotDetector
from .schema import InteractionSession, KeyEvent, MouseEvent
from .session import FEATURE_NAMES, extract_features

__all__ = [
    "BotDetector",
    "InteractionSession",
    "KeyEvent",
    "MouseEvent",
    "FEATURE_NAMES",
    "extract_features",
]

__version__ = "0.1.0"
