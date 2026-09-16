from not_a_robot.io import load_sessions_jsonl, save_sessions_jsonl
from not_a_robot.schema import (
    ClickEvent,
    FocusEvent,
    InteractionSession,
    KeyEvent,
    MouseEvent,
    PasteEvent,
    ScrollEvent,
)


def test_round_trip_through_jsonl(tmp_path):
    sessions = [
        InteractionSession(
            mouse_events=[MouseEvent(0, 0, 0), MouseEvent(5, 5, 20)],
            key_events=[KeyEvent(100, 150)],
            scroll_events=[ScrollEvent(40.0, 120.0)],
            click_events=[ClickEvent(410.0, 512.0, 250.0)],
            focus_events=[FocusEvent(60.0, False), FocusEvent(500.0, True)],
            paste_events=[PasteEvent(90.0, 12)],
            page_load_t=0.0,
            submit_t=300.0,
            label=True,
        ),
        InteractionSession(
            mouse_events=[MouseEvent(0, 0, 0)],
            key_events=[],
            page_load_t=0.0,
            submit_t=None,
            label=False,
        ),
    ]
    path = tmp_path / "sessions.jsonl"
    save_sessions_jsonl(sessions, path)
    loaded = load_sessions_jsonl(path)

    assert len(loaded) == 2
    assert loaded[0].label is True
    assert loaded[0].mouse_events[1].x == 5
    assert loaded[0].scroll_events[0].delta_y == 120.0
    assert loaded[0].click_events[0].x == 410.0
    assert loaded[0].focus_events[0].focused is False
    assert loaded[0].paste_events[0].length == 12
    assert loaded[1].submit_t is None
    assert loaded[1].label is False
    assert loaded[1].scroll_events == []
