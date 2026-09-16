from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from .schema import (
    ClickEvent,
    FocusEvent,
    InteractionSession,
    KeyEvent,
    MouseEvent,
    PasteEvent,
    ScrollEvent,
)


def session_to_dict(session: InteractionSession) -> dict:
    return {
        "mouse_events": [[e.x, e.y, e.t] for e in session.mouse_events],
        "key_events": [[e.t_down, e.t_up] for e in session.key_events],
        "scroll_events": [[e.t, e.delta_y] for e in session.scroll_events],
        "click_events": [[e.x, e.y, e.t] for e in session.click_events],
        "focus_events": [[e.t, e.focused] for e in session.focus_events],
        "paste_events": [[e.t, e.length] for e in session.paste_events],
        "page_load_t": session.page_load_t,
        "submit_t": session.submit_t,
        "label": session.label,
    }


def session_from_dict(data: dict) -> InteractionSession:
    return InteractionSession(
        mouse_events=[
            MouseEvent(x, y, t) for x, y, t in data.get("mouse_events", [])
        ],
        key_events=[
            KeyEvent(t_down, t_up) for t_down, t_up in data.get("key_events", [])
        ],
        scroll_events=[
            ScrollEvent(t, delta_y) for t, delta_y in data.get("scroll_events", [])
        ],
        click_events=[
            ClickEvent(x, y, t) for x, y, t in data.get("click_events", [])
        ],
        focus_events=[
            FocusEvent(t, focused) for t, focused in data.get("focus_events", [])
        ],
        paste_events=[
            PasteEvent(t, length) for t, length in data.get("paste_events", [])
        ],
        page_load_t=data.get("page_load_t", 0.0),
        submit_t=data.get("submit_t"),
        label=data.get("label"),
    )


def load_sessions_jsonl(path: Union[str, Path]) -> list[InteractionSession]:
    """Load sessions from a JSON-lines file: one session object per line.

    This is the expected format for real, captured traffic: append one
    line per finished session from your own logging pipeline, with
    ``label`` set once you know the outcome (True=human, False=bot).
    """
    sessions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sessions.append(session_from_dict(json.loads(line)))
    return sessions


def save_sessions_jsonl(
    sessions: list[InteractionSession], path: Union[str, Path]
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for session in sessions:
            f.write(json.dumps(session_to_dict(session)))
            f.write("\n")
