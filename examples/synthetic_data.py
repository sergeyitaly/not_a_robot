"""Synthetic session generator for demos and tests only.

This produces two crude archetypes -- a jittery random walk (label=True)
and a near-straight, constant-speed path (label=False) -- purely so the
rest of the library has example data to train and test the feature
pipeline against. It does not model real bot behavior or real human
behavior; before relying on a BotDetector for anything, retrain it on
labeled sessions captured from your own application.
"""

from __future__ import annotations

import random

from not_a_robot.schema import InteractionSession, KeyEvent, MouseEvent


def _human_like_session(rng: random.Random) -> InteractionSession:
    x, y, t = rng.uniform(0, 50), rng.uniform(0, 50), 0.0
    points = [MouseEvent(x, y, t)]
    for _ in range(rng.randint(20, 60)):
        x += rng.uniform(-15, 15)
        y += rng.uniform(-15, 15)
        t += rng.uniform(10, 60)
        points.append(MouseEvent(x, y, t))

    keys = []
    kt = t + rng.uniform(200, 800)
    for _ in range(rng.randint(5, 15)):
        down = kt
        up = down + rng.uniform(60, 180)
        keys.append(KeyEvent(down, up))
        kt = up + rng.uniform(50, 250)

    return InteractionSession(
        mouse_events=points,
        key_events=keys,
        page_load_t=0.0,
        submit_t=kt + rng.uniform(200, 600),
        label=True,
    )


def _bot_like_session(rng: random.Random) -> InteractionSession:
    x0, y0 = rng.uniform(0, 50), rng.uniform(0, 50)
    x1, y1 = x0 + rng.uniform(100, 300), y0 + rng.uniform(100, 300)
    n = rng.randint(5, 10)
    points = [
        MouseEvent(
            x0 + (x1 - x0) * i / n,
            y0 + (y1 - y0) * i / n,
            i * 5.0,
        )
        for i in range(n + 1)
    ]

    keys = []
    kt = points[-1].t + 20.0
    for _ in range(rng.randint(5, 15)):
        down = kt
        up = down + 15.0
        keys.append(KeyEvent(down, up))
        kt = up + 15.0

    return InteractionSession(
        mouse_events=points,
        key_events=keys,
        page_load_t=0.0,
        submit_t=kt + 10.0,
        label=False,
    )


def make_synthetic_dataset(
    n_per_class: int = 50, seed: int = 0
) -> list[InteractionSession]:
    rng = random.Random(seed)
    sessions = [_human_like_session(rng) for _ in range(n_per_class)]
    sessions += [_bot_like_session(rng) for _ in range(n_per_class)]
    rng.shuffle(sessions)
    return sessions
