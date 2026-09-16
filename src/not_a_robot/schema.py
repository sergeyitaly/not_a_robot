from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class MouseEvent:
    """A single mousemove sample."""

    x: float
    y: float
    t: float  # milliseconds since page load


@dataclass(frozen=True)
class KeyEvent:
    """A single keydown/keyup pair (dwell interval)."""

    t_down: float
    t_up: float


@dataclass
class InteractionSession:
    """One user session captured on your own page for training/scoring.

    ``label`` is only required for sessions used to train a BotDetector:
    True for a known-human session, False for a known-bot session, None
    for a session you only want to score.
    """

    mouse_events: list[MouseEvent] = field(default_factory=list)
    key_events: list[KeyEvent] = field(default_factory=list)
    page_load_t: float = 0.0
    submit_t: Optional[float] = None
    label: Optional[bool] = None
