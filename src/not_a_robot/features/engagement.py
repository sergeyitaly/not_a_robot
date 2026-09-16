from __future__ import annotations

from ..schema import FocusEvent, PasteEvent


def extract_engagement_features(
    focus_events: list[FocusEvent], paste_events: list[PasteEvent]
) -> dict[str, float]:
    """Compute tab-focus and paste behavior features.

    Frequent blur/focus cycling is consistent with a human multitasking
    across tabs; scripted fillers typically never blur. A paste event
    covering most/all of a field's characters suggests a value was
    injected rather than typed -- a signal to combine with, not replace,
    the keystroke-dynamics features, since a human can legitimately
    paste too (e.g. from a password manager).
    """
    blur_count = sum(1 for e in focus_events if not e.focused)
    paste_total_chars = sum(e.length for e in paste_events)

    return {
        "blur_count": float(blur_count),
        "paste_count": float(len(paste_events)),
        "paste_total_chars": float(paste_total_chars),
    }
