import math

import pytest

from not_a_robot.schema import (
    ClickEvent,
    FocusEvent,
    InteractionSession,
    KeyEvent,
    MouseEvent,
    PasteEvent,
    ScrollEvent,
)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_mouse_event_rejects_non_finite_values(bad):
    with pytest.raises(ValueError):
        MouseEvent(bad, 0, 0)
    with pytest.raises(ValueError):
        MouseEvent(0, bad, 0)
    with pytest.raises(ValueError):
        MouseEvent(0, 0, bad)


def test_key_event_rejects_non_finite_values():
    with pytest.raises(ValueError):
        KeyEvent(math.nan, 10)


def test_key_event_rejects_t_up_before_t_down():
    with pytest.raises(ValueError):
        KeyEvent(100, 50)


def test_key_event_allows_zero_dwell():
    KeyEvent(100, 100)


def test_scroll_event_rejects_non_finite_values():
    with pytest.raises(ValueError):
        ScrollEvent(math.inf, 10)


def test_click_event_rejects_non_finite_values():
    with pytest.raises(ValueError):
        ClickEvent(0, math.nan, 0)


def test_focus_event_rejects_non_finite_values():
    with pytest.raises(ValueError):
        FocusEvent(math.nan, True)


def test_paste_event_rejects_non_finite_time():
    with pytest.raises(ValueError):
        PasteEvent(math.nan, 5)


def test_paste_event_rejects_negative_length():
    with pytest.raises(ValueError):
        PasteEvent(0, -1)


def test_interaction_session_rejects_non_finite_page_load_t():
    with pytest.raises(ValueError):
        InteractionSession(page_load_t=math.nan)


def test_interaction_session_rejects_non_finite_submit_t():
    with pytest.raises(ValueError):
        InteractionSession(submit_t=math.inf)


def test_interaction_session_allows_none_submit_t():
    InteractionSession(submit_t=None)
