"""Synthetic session generator for demos and tests only.

Produces one human archetype (a noisy random walk with occasional
hesitation pauses, variable keystroke dwell/flight times, occasional
scrolling/tab-switching/pasting) and four bot archetypes of increasing
sophistication, weighted toward the unsophisticated end since that's the
realistic mix:

- naive (45%): straight-line mouse path, perfectly uniform keystrokes,
  a single click at a fixed coordinate, no scroll/focus/paste activity
- evasive (35%): jitter added to path, timing, and click position, but
  drawn from a tighter/more uniform distribution than real humans
  produce; scripted scroll with near-uniform intervals
- headless (10%): little to no mouse/key/scroll/click activity,
  near-instant submit
- sophisticated (10%): reuses the human archetype's mouse/keyboard/click
  distributions (the signals that are well-documented and cheap for an
  attacker to fake, e.g. GAN-generated mouse trajectories -- see
  description.txt), but never scrolls, never blurs/refocuses, and never
  pastes -- the secondary channels that are more effort to convincingly
  automate. Mouse/keyboard features alone can't separate it from a human;
  the scroll/click/engagement features can, imperfectly (some real humans
  also don't scroll, blur, or paste in a given session).

This gives the training pipeline a non-trivially-separable dataset with
a genuine, non-zero error floor to validate against, instead of
archetypes that are perfectly separable by construction. None of this
models any real website's actual traffic; it exists only to exercise
this library's own pipeline mechanics before you have real, labeled
sessions from your own application.
"""

from __future__ import annotations

import random

from not_a_robot.schema import (
    ClickEvent,
    FocusEvent,
    InteractionSession,
    KeyEvent,
    MouseEvent,
    PasteEvent,
    ScrollEvent,
)


def _clip_positive(value: float, minimum: float) -> float:
    return max(value, minimum)


def _human_like_session(rng: random.Random) -> InteractionSession:
    x, y, t = rng.uniform(0, 50), rng.uniform(0, 50), 0.0
    points = [MouseEvent(x, y, t)]
    for _ in range(rng.randint(25, 70)):
        x += rng.gauss(0, 12)
        y += rng.gauss(0, 12)
        dt = _clip_positive(rng.gauss(35, 20), 5.0)
        if rng.random() < 0.08:  # occasional hesitation
            dt += rng.uniform(150, 500)
        t += dt
        points.append(MouseEvent(x, y, t))

    scroll_events = []
    st = rng.uniform(200, 1200)
    for _ in range(rng.randint(0, 6)):
        direction = rng.choices([1, -1], weights=[3, 1])[0]
        scroll_events.append(ScrollEvent(st, direction * rng.uniform(40, 220)))
        st += rng.uniform(150, 900)

    keys = []
    kt = t + rng.uniform(200, 900)
    for _ in range(rng.randint(6, 20)):
        down = kt
        dwell = _clip_positive(rng.gauss(110, 40), 30.0)
        up = down + dwell
        keys.append(KeyEvent(down, up))
        flight = _clip_positive(rng.gauss(130, 70), 10.0)
        if rng.random() < 0.05:  # occasional thinking pause
            flight += rng.uniform(200, 600)
        kt = up + flight

    click_events = [
        ClickEvent(410 + rng.gauss(0, 4), 512 + rng.gauss(0, 4), kt + rng.uniform(40, 180))
    ]

    focus_events = []
    if rng.random() < 0.25:  # tab-switch away and back
        blur_t = rng.uniform(300, kt)
        focus_events.append(FocusEvent(blur_t, False))
        focus_events.append(FocusEvent(blur_t + rng.uniform(800, 4000), True))

    paste_events = []
    if rng.random() < 0.15:  # e.g. pasted from a password manager
        paste_events.append(PasteEvent(kt - rng.uniform(50, 150), rng.randint(6, 20)))

    return InteractionSession(
        mouse_events=points,
        key_events=keys,
        scroll_events=scroll_events,
        click_events=click_events,
        focus_events=focus_events,
        paste_events=paste_events,
        page_load_t=0.0,
        submit_t=kt + rng.uniform(300, 900),
        label=True,
        group="human",
    )


def _bot_naive_session(rng: random.Random) -> InteractionSession:
    """Unsophisticated bot: straight-line path, perfectly uniform keys,
    a single click at a hardcoded coordinate, no scroll/focus/paste."""
    x0, y0 = rng.uniform(0, 50), rng.uniform(0, 50)
    x1, y1 = x0 + rng.uniform(100, 300), y0 + rng.uniform(100, 300)
    n = rng.randint(5, 10)
    points = [
        MouseEvent(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n, i * 5.0)
        for i in range(n + 1)
    ]

    keys = []
    kt = points[-1].t + 20.0
    for _ in range(rng.randint(5, 15)):
        down = kt
        up = down + 15.0
        keys.append(KeyEvent(down, up))
        kt = up + 15.0

    click_events = [ClickEvent(410.0, 512.0, kt + 15.0)]

    return InteractionSession(
        mouse_events=points,
        key_events=keys,
        click_events=click_events,
        page_load_t=0.0,
        submit_t=kt + 10.0,
        label=False,
        group="naive",
    )


def _bot_evasive_session(rng: random.Random) -> InteractionSession:
    """Bot with jitter added, but from a much tighter distribution than
    real humans -- gives the coefficient-of-variation features something
    non-trivial to separate on instead of an obviously straight line."""
    x0, y0 = rng.uniform(0, 50), rng.uniform(0, 50)
    x1, y1 = x0 + rng.uniform(100, 300), y0 + rng.uniform(100, 300)
    n = rng.randint(15, 30)
    points = []
    for i in range(n + 1):
        frac = i / n
        x = x0 + (x1 - x0) * frac + rng.gauss(0, 3)
        y = y0 + (y1 - y0) * frac + rng.gauss(0, 3)
        t = i * _clip_positive(rng.gauss(12, 2), 4.0)
        points.append(MouseEvent(x, y, t))

    scroll_events = []
    st = rng.uniform(50, 100)
    for _ in range(rng.randint(0, 3)):
        scroll_events.append(ScrollEvent(st, rng.gauss(150, 5)))
        st += rng.gauss(100, 5)

    keys = []
    kt = points[-1].t + rng.uniform(80, 200)
    for _ in range(rng.randint(6, 20)):
        down = kt
        dwell = _clip_positive(rng.gauss(60, 8), 20.0)
        up = down + dwell
        keys.append(KeyEvent(down, up))
        flight = _clip_positive(rng.gauss(45, 8), 10.0)
        kt = up + flight

    click_events = [
        ClickEvent(410 + rng.gauss(0, 0.5), 512 + rng.gauss(0, 0.5), kt + 10.0)
    ]

    return InteractionSession(
        mouse_events=points,
        key_events=keys,
        scroll_events=scroll_events,
        click_events=click_events,
        page_load_t=0.0,
        submit_t=kt + rng.uniform(30, 100),
        label=False,
        group="evasive",
    )


def _bot_headless_session(rng: random.Random) -> InteractionSession:
    """Headless/scripted bot: fields set programmatically, near-instant
    submit, little to no mouse/scroll/click activity."""
    points = []
    if rng.random() < 0.5:
        points = [MouseEvent(rng.uniform(0, 800), rng.uniform(0, 600), 0.0)]

    keys = []
    if rng.random() < 0.3:
        kt = 0.0
        for _ in range(rng.randint(3, 8)):
            keys.append(KeyEvent(kt, kt + 1.0))
            kt += 1.0

    return InteractionSession(
        mouse_events=points,
        key_events=keys,
        page_load_t=0.0,
        submit_t=rng.uniform(5, 80),
        label=False,
        group="headless",
    )


def _bot_sophisticated_session(rng: random.Random) -> InteractionSession:
    """An advanced bot that closely mimics human mouse/keyboard/click
    behavior -- the well-documented, cheap-to-fake signals (see
    description.txt on GAN-generated mouse trajectories) -- but skips the
    secondary channels that are more effort to convincingly automate:
    scroll physics, multi-tab focus/blur cycling, and paste-vs-type
    behavior. This is deliberately hard to catch on mouse/keyboard alone,
    but not undetectable outright: a classifier with the scroll/click/
    engagement features has real (if imperfect, since some real humans
    also don't scroll or paste in a given session) signal to work with.
    """
    human_reference = _human_like_session(rng)
    click_events = [
        ClickEvent(
            human_reference.click_events[0].x + rng.gauss(0, 2),
            human_reference.click_events[0].y + rng.gauss(0, 2),
            human_reference.click_events[0].t,
        )
    ]
    return InteractionSession(
        mouse_events=human_reference.mouse_events,
        key_events=human_reference.key_events,
        click_events=click_events,
        # no scroll_events/focus_events/paste_events: the channels this
        # archetype doesn't bother faking
        page_load_t=human_reference.page_load_t,
        submit_t=human_reference.submit_t,
        label=False,
        group="sophisticated",
    )


_BOT_GENERATORS = (
    _bot_naive_session,
    _bot_evasive_session,
    _bot_headless_session,
    _bot_sophisticated_session,
)
_BOT_WEIGHTS = (0.45, 0.35, 0.10, 0.10)


def make_synthetic_dataset(
    n_per_class: int = 50, seed: int = 0
) -> list[InteractionSession]:
    """Build a synthetic, non-trivially-separable demo dataset.

    ``n_per_class`` humans, and ``n_per_class`` bots drawn from the four
    weighted archetypes above. See the module docstring for what this is
    (and isn't) meant to demonstrate.
    """
    rng = random.Random(seed)
    sessions = [_human_like_session(rng) for _ in range(n_per_class)]

    for _ in range(n_per_class):
        generator = rng.choices(_BOT_GENERATORS, weights=_BOT_WEIGHTS, k=1)[0]
        sessions.append(generator(rng))

    rng.shuffle(sessions)
    return sessions
