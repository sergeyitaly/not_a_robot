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


@dataclass(frozen=True)
class ScrollEvent:
    """A single scroll sample."""

    t: float
    delta_y: float  # positive = scrolled down, negative = scrolled up


@dataclass(frozen=True)
class ClickEvent:
    """A single click/tap sample."""

    x: float
    y: float
    t: float


@dataclass(frozen=True)
class FocusEvent:
    """A window/tab focus transition."""

    t: float
    focused: bool  # True = focus gained, False = focus lost (blur)


@dataclass(frozen=True)
class PasteEvent:
    """A paste-into-field event."""

    t: float
    length: int  # number of characters pasted


@dataclass
class InteractionSession:
    """One user session captured on your own page for training/scoring.

    ``label`` is only required for sessions used to train a BotDetector:
    True for a known-human session, False for a known-bot session, None
    for a session you only want to score.

    ``group`` is an optional finer-grained tag beyond the human/bot label
    -- e.g. a known bot sub-type ("credential-stuffing", "scraper"), a
    traffic source, or (in the synthetic demo data) the archetype that
    generated the session. It plays no role in training; it exists so
    evaluation can report recall broken out by sub-population instead of
    only an aggregate that can hide which group is driving the errors.
    """

    mouse_events: list[MouseEvent] = field(default_factory=list)
    key_events: list[KeyEvent] = field(default_factory=list)
    scroll_events: list[ScrollEvent] = field(default_factory=list)
    click_events: list[ClickEvent] = field(default_factory=list)
    focus_events: list[FocusEvent] = field(default_factory=list)
    paste_events: list[PasteEvent] = field(default_factory=list)
    page_load_t: float = 0.0
    submit_t: Optional[float] = None
    label: Optional[bool] = None
    group: Optional[str] = None
