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
- **Enrichment ratios** (`not_a_robot.features.enrichment`): coefficients
  of variation and per-second rates derived from the two feature groups
  above (e.g. `mouse_velocity_cv`, `key_rate_per_sec`), which normalize
  for session length/typing speed and tend to separate scripted, uniform
  behavior from naturally variable human behavior better than any single
  raw statistic.

All features are combined into one fixed-order vector
(`not_a_robot.session.FEATURE_NAMES`) that feeds a scikit-learn classifier
(`RandomForestClassifier` by default — pass your own via `BotDetector(model=...)`).

## Training pipeline and success-rate validation

`not_a_robot.pipeline.run_training_pipeline()` is the full training and
enrichment pipeline: it extracts the enriched feature set, splits off a
stratified held-out test set, runs stratified k-fold cross-validation on
the remaining training data, fits the final `BotDetector`, and evaluates it
once on the untouched test set. It returns the fitted detector plus a
`TrainingReport` framed around the numbers that actually matter for a
"not a robot" check:

- **Human pass rate** — how often a real user is correctly verified as
  human (test recall on the human class).
- **Bot catch rate** — how often a bot session is correctly blocked.
- **False accept rate** — bots that slipped through as human (the security
  cost).
- **False reject rate** — real users wrongly blocked (the UX cost).
- Overall accuracy, precision, F1, ROC-AUC, the full confusion matrix, and
  the top features by importance.

```python
from not_a_robot import run_training_pipeline

detector, report = run_training_pipeline(sessions, data_source="prod-2026-09")
print(report.summary())
detector.save("bot_detector.joblib")
```

Or from the command line, against a real captured session log:

```bash
python -m not_a_robot.train --data sessions.jsonl --model-out bot_detector.joblib --report-out report.json
```

`--synthetic` runs the same pipeline against the bundled demo dataset (see
below) so you can see a real, computed report before you have real traffic:

```bash
python -m not_a_robot.train --synthetic --n-per-class 150
```

That produced, on one run against the synthetic demo data (150 sessions per
class, 75/25 train/test split, 5-fold CV):

```
Cross-validated accuracy: 100.0% +/- 0.0%

Held-out test results:
  Overall accuracy:   100.0%
  Human pass rate:    100.0%  (real users correctly verified as human)
  Bot catch rate:     100.0%  (bots correctly blocked)
  False accept rate:  0.0%    (bots that slipped through as human)
  False reject rate:  0.0%    (real users wrongly blocked)
  ROC-AUC:            1.000
```

That 100% is expected and not meaningful on its own: the synthetic
generator's "bot" archetype (a near-straight, constant-speed path) and
"human" archetype (a jittery random walk with variable timing) are
trivially separable by design, so this only proves the pipeline's
mechanics (splitting, CV, fitting, metrics, reporting) work end to end.
The report format is real; the input data for this particular run is not.
Run `python -m not_a_robot.train --data <your sessions.jsonl>` on real,
labeled traffic from your own site to get numbers you can actually trust.

## Capturing real training data

The library only defines the schema and the feature math; you own the
client-side capture. On the page you're protecting, record `mousemove`
coordinates + timestamps and `keydown`/`keyup` timestamps into
`MouseEvent`/`KeyEvent` objects, tag each finished session with a label
(from a secondary signal you trust — e.g. a CAPTCHA outcome, an email
verification, or manual review), and either pass the collected
`InteractionSession` objects straight to `run_training_pipeline()`, or
persist them with `not_a_robot.io.save_sessions_jsonl()` (one JSON object
per line) so `python -m not_a_robot.train --data sessions.jsonl` can pick
them up later.

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
