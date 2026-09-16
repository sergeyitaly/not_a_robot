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
- **Scroll behavior** (`not_a_robot.features.scroll`): total distance,
  direction reversals, interval and delta statistics.
- **Click/tap behavior** (`not_a_robot.features.clicks`): click count,
  interval statistics, position variance (scripted clicks tend to land on
  the exact same pixel repeatedly).
- **Tab-focus and paste behavior** (`not_a_robot.features.engagement`):
  blur/refocus count, paste count and total pasted characters.
- **Enrichment ratios** (`not_a_robot.features.enrichment`): coefficients
  of variation and per-second rates derived from the feature groups above
  (e.g. `mouse_velocity_cv`, `key_rate_per_sec`, `scroll_rate_per_sec`,
  `typed_vs_pasted_ratio`), which normalize for session length/typing
  speed and tend to separate scripted, uniform behavior from naturally
  variable human behavior better than any single raw statistic.

Every field on `InteractionSession` (`mouse_events`, `key_events`,
`scroll_events`, `click_events`, `focus_events`, `paste_events`) is
optional and defaults to empty — you don't have to capture all of them to
use the library, but the more of them you wire up client-side, the more
signal the detector has to work with.

All features are combined into one fixed-order vector
(`not_a_robot.session.FEATURE_NAMES`) that feeds a scikit-learn classifier
(`RandomForestClassifier` by default — pass your own via `BotDetector(model=...)`).

## Training pipeline and success-rate validation

There are two evaluation paths, and they answer different questions.

**`run_training_pipeline()`** fits the detector you'd actually deploy: it
extracts the enriched feature set, splits off one stratified held-out test
set, fits a `BotDetector` on the rest, and evaluates it once on that split.
Useful for producing a model + a quick report, but its metrics are a
**single point estimate** — on a dataset in the hundreds of sessions, one
particular 75/25 split can look meaningfully better or worse than another
just from sampling luck, not from anything about the model.

**`evaluate_cv()`** answers "how much should I trust that number": it runs
repeated stratified k-fold cross-validation (`n_splits x n_repeats`
independent folds, a fresh model per fold, default 5x10 = 50), and reports
the **mean and standard deviation** of every metric across folds, plus
**recall broken out by `InteractionSession.group`** — a sub-population tag
(e.g. a known bot type in real data, or the synthetic archetype below) —
so an aggregate "bot catch rate" can't hide that all the errors are
concentrated in one group. This is the number to actually trust; the CLI
runs it by default alongside the single-split report.

```python
from not_a_robot import run_training_pipeline, evaluate_cv

detector, report = run_training_pipeline(sessions, data_source="prod-2026-09")
detector.save("bot_detector.joblib")

cv_report = evaluate_cv(sessions, data_source="prod-2026-09")
print(cv_report.summary())
```

From the command line, against a real captured session log:

```bash
python -m not_a_robot.train --data sessions.jsonl --model-out bot_detector.joblib --report-out report.json
```

`--synthetic` runs the same pipeline against the bundled demo dataset (see
below) so you can see a real, computed report before you have real traffic:

```bash
python -m not_a_robot.train --synthetic --n-per-class 200
```

That produced, on seed 0 (400 sessions, 5-fold x 10-repeat CV, full
feature set):

```
=== not_a_robot repeated-CV evaluation ===
Sessions: 400 (5-fold x 10 repeats = 50 test folds)

Accuracy:           92.8% +/- 2.2%
Human pass rate:    96.2% +/- 3.6%
Bot catch rate:     89.4% +/- 4.6%
False accept rate:  10.7% +/- 4.6%
False reject rate:  3.9% +/- 3.6%
ROC-AUC:            0.990 +/- 0.006

Per-group recall (mean +/- std across the folds each session
appeared in as a test example):
  group             weight   recall
  human             50.0%   96.2% +/- 14.2%
  naive             19.8%   100.0% +/- 0.0%
  evasive           18.5%   100.0% +/- 0.0%
  sophisticated      6.2%   14.8% +/- 23.7%
  headless           5.5%   100.0% +/- 0.0%
```

Seeds 1 and 2 tell the same story: naive/evasive/headless bots caught at
100.0% every time, `sophisticated` caught at 0.7% (seed 1) to 52.1% (seed
2) with very high per-fold variance (that variance is itself the honest
finding — a rare, weak signal isn't something to build a security decision
on). Aggregate accuracy across the three seeds: 92.8-96.3%. **This is the
correct way to read this benchmark: not "95% accurate," but "consistently
catches unsophisticated and evasive bots, and catches a real but unreliable
fraction of bots that mimic mouse/keyboard behavior closely."** A
single-split `TrainingReport` on the same data can show bot catch rate
anywhere from 78% to 100% depending on which sessions happened to land in
the test split — that swing is sampling noise on a ~400-session synthetic
set, not the model changing.

The synthetic generator (`examples/synthetic_data.py`) draws bots from
four weighted archetypes: naive (straight-line path, uniform keystrokes,
fixed click coordinate, 45%), evasive (jittered but still tighter than
human, scripted scroll, 35%), headless (near-instant submit, little/no
activity, 10%), and sophisticated (10%) — which reuses the human
archetype's mouse/keyboard/click distributions (the signals that are
well-documented and cheap for an attacker to fake — see `description.txt`
on GAN-generated mouse trajectories) but never scrolls, blurs, or pastes,
since those channels are more effort to convincingly automate. That's why
`sophisticated` is hard but not literally 0% catchable: the scroll/click/
engagement features give real, if noisy, signal against it, while
mouse/keyboard features alone cannot separate it from a human at all.

The report format and numbers above are real, computed output from this
repo. The input data is not: it's synthetic, generated locally, with no
interaction with any real website. Run
`python -m not_a_robot.train --data <your sessions.jsonl>` on real,
labeled traffic from your own site to get numbers you can actually trust
for a production decision.

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

`examples/synthetic_data.py` generates crude synthetic sessions (one
human archetype and four weighted bot archetypes, see above) purely so
the rest of the pipeline has example data to run against before you have
real, labeled traffic. It is not a model of real bot or human behavior —
replace it with your own data before relying on this for anything.

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
