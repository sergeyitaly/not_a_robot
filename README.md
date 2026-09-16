# not-a-robot

A Python library for building the **detector side** of a "prove you're not a
robot" check: it extracts behavioral-telemetry features (mouse-movement
dynamics, keystroke timing, overall pacing) from an interaction session and
trains an ML classifier that scores how human-like the session looks.

This is meant to run on infrastructure you control, as one signal alongside
your own verification flow — not to defeat verification on someone else's
site. See [Scope](#scope) below.

## Install

```bash
pip install -e ".[dev]"
```

## Quickstart

```python
from not_a_robot import BotDetector, InteractionSession, MouseEvent, KeyEvent

# Sessions you've captured and labeled from your own application.
# label=True for a known-human session, label=False for a known-bot session.
sessions = [
    InteractionSession(
        mouse_events=[MouseEvent(x=10, y=12, t=0), MouseEvent(x=14, y=20, t=35), ...],
        key_events=[KeyEvent(t_down=500, t_up=560), ...],
        page_load_t=0.0,
        submit_t=4200.0,
        label=True,
    ),
    # ... more labeled sessions ...
]

detector = BotDetector()
detector.fit(sessions)
detector.save("bot_detector.joblib")

# Later, score a new session:
detector = BotDetector.load("bot_detector.joblib")
p_human = detector.score(new_session)          # float in [0, 1]
is_human = detector.predict(new_session)        # bool at the default 0.5 threshold
```

Run the end-to-end example (uses synthetic data, see below) from the repo
root:

```bash
python -m examples.quickstart
```

## What it extracts

- **Mouse dynamics** (`not_a_robot.features.mouse`): path length vs.
  straight-line distance ("path efficiency"), velocity/acceleration/jerk
  statistics, turning-angle statistics, direction reversals, pause count.
- **Timing / keystroke dynamics** (`not_a_robot.features.timing`): dwell
  time (key down -> up), flight time (key up -> next key down), time to
  first interaction, time to submit.

All features are combined into one fixed-order vector
(`not_a_robot.session.FEATURE_NAMES`) that feeds a scikit-learn classifier
(`RandomForestClassifier` by default — pass your own via `BotDetector(model=...)`).

## Capturing real training data

The library only defines the schema and the feature math; you own the
client-side capture. On the page you're protecting, record `mousemove`
coordinates + timestamps and `keydown`/`keyup` timestamps into
`MouseEvent`/`KeyEvent` objects, tag each finished session with a label
(from a secondary signal you trust — e.g. a CAPTCHA outcome, an email
verification, or manual review), and pass the collected `InteractionSession`
objects to `BotDetector.fit()`.

`examples/synthetic_data.py` generates crude synthetic sessions (a jittery
random walk vs. a near-straight constant-speed path) purely so the rest of
the pipeline has example data to run against before you have real, labeled
traffic. It is not a model of real bot or human behavior — replace it with
your own data before relying on this for anything.

## Scope

This library builds a defensive behavioral classifier for a system you run
and control. It intentionally does **not** include: CAPTCHA-solving (OCR,
image-grid classifiers), browser automation for clicking through third-party
challenges, integrations with CAPTCHA-solving services, or synthetic
mouse-trajectory generation meant to fool someone else's bot detection.
Those are a different (and, outside authorized testing of your own systems,
frequently abusive) category of tool.

## Development

```bash
pip install -e ".[dev]"
pytest
```
