# not-a-robot

A Python library for building the **detector side** of a "prove you're not a
robot" check: it extracts behavioral-telemetry features (mouse-movement
dynamics, keystroke timing, overall pacing) from an interaction session and
trains an ML classifier that scores how human-like the session looks.

**Scope:** builds the detector side of a "prove you're not a robot" check,
for infrastructure you run yourself. Does not include CAPTCHA-solving,
browser automation for third-party challenges, or trajectory generation
meant to fool someone else's detection. Full statement under
[Scope](#scope).

**Try it:** <https://not-a-robot-demo.onrender.com/> (free-tier hosting —
may take a moment to cold-start after inactivity), or run
[dry-run/](dry-run/) locally with Docker or plain Python. One button,
one real result: generates a synthetic batch across every archetype the
library ships, scores and trains on it, reports the actual human-pass /
bot-catch numbers the run just produced (not a mock), and runs the
environment checks below against a **real headless Chromium instance
launched via Selenium**, not a hardcoded example. See
[dry-run/README.md](dry-run/README.md) for what that found.

**Status:** alpha. Validated only on synthetic data so far, plus a small
real-automation capture (see [Capturing real training data](#capturing-real-training-data));
the pipeline ships here, real-traffic numbers are yours.

## Install

```bash
pip install not-a-robot
```

(For an editable install from a checkout, see [Development](#development).)

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

- **Mouse dynamics** (`not_a_robot.features.mouse`): path efficiency,
  velocity/acceleration/jerk, turning-angle statistics, direction
  reversals, pause count.
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
  of variation and per-second rates derived from the groups above (e.g.
  `mouse_velocity_cv`, `key_rate_per_sec`, `typed_vs_pasted_ratio`),
  which tend to separate scripted, uniform behavior from naturally
  variable human behavior better than any single raw statistic.

Every field on `InteractionSession` is optional and defaults to empty —
the more of them you wire up client-side, the more signal the detector
has. An empty channel is ambiguous, though (no mouse activity vs. the
tracking script never firing) — `not_a_robot.channel_coverage(session)`
reports which channels captured anything at all, for debugging/audit,
not as model input.

All features combine into one fixed-order vector
(`not_a_robot.session.FEATURE_NAMES`) feeding a scikit-learn classifier
(`RandomForestClassifier` by default, wrapped in `CalibratedClassifierCV`
— pass your own via `BotDetector(model=...)`).

## Training pipeline and success-rate validation

Three evaluation paths: `run_training_pipeline()` (one train/test split
— a single point estimate), `evaluate_cv()` (repeated stratified k-fold
CV on one sample, with per-group Wilson-CI recall and a cost-optimal
threshold), and `summarize_across_seeds()` (CLI: `--seeds 0,1,2,3`, the
one to actually quote — it's the only path that captures how much the
numbers move when the *sample itself* changes, not just the fold
partition).

```bash
python -m not_a_robot.train --data sessions.jsonl --seeds 0,1,2,3       # real traffic
python -m not_a_robot.train --synthetic --n-per-class 200 --seeds 0,1,2,3  # bundled demo data
```

Headline numbers from the bundled synthetic dataset (1,600 sessions, 4
seeds x 5-fold x 10-repeat CV, `BotDetector`'s calibrated default
model): **bot catch rate stable at 92–94%** across resamples.
`naive`/`evasive`/`headless` archetypes caught near 100%.
`sophisticated` (reuses the human archetype's mouse/keyboard behavior
by design, so those channels carry zero signal against it) is caught at
34.3% pooled — but **10.7–54.3% per seed**, a ~40-point spread that's
the honest finding: which seed the model trains on visibly changes how
much of this class gets caught, so don't quote either the 54.3% high or
the 34.3% pooled number alone as "real-world performance against
mimicry bots."

**Full methodology — calibration's effect (and what it didn't fix), the
`--drop-keys` ablation, the synthetic generator's design, and the
real-Selenium/real-Playwright capture validation — is in
[docs/validation.md](docs/validation.md).** The synthetic data is
useful for exercising the pipeline before you have real traffic; it is
not a model of real bot or human behavior.

## Deterministic automation checks (separate from the behavioral model)

`not_a_robot.environment` checks browser-observable automation artifacts
— `navigator.webdriver`, Selenium/ChromeDriver's `cdc_*` globals,
Playwright/Puppeteer markers, software-rendered WebGL (consistent with
headless without GPU passthrough):

```python
from not_a_robot.environment import EnvironmentSignals, score_environment

env = score_environment(EnvironmentSignals(
    webdriver_flag=True,
    cdc_properties_present=False,
    webgl_renderer="Google SwiftShader",
))
env.is_automated  # True
env.reasons        # ["navigator.webdriver is true", "WebGL is software-rendered ..."]
```

Not a fourth behavioral feature group, and not imported from the
top-level package — it returns a boolean plus which signal fired, not a
probability, so there's no calibration/CV story here. Combine both at
your application layer:

```python
if score_environment(signals).is_automated:
    block()      # near-certain; skip the behavioral score
else:
    decide(detector.score(session))  # falls back to the statistical layer
```

**What this does and doesn't buy you:** every signal here is exactly
what stealth plugins and anti-detect browsers patch by default. A
positive result is cheap, strong evidence of unsophisticated automation.
A negative result means "no artifact observed," not "this is a human" —
a stealth-patched bot passes on purpose, which is the gap
`BotDetector`'s behavioral scoring exists for. Verified against a real
browser (not just asserted) — see
[docs/validation.md](docs/validation.md#verifying-the-environment-layer-against-a-real-browser).

**Deliberately excludes TLS/JA3-JA4 fingerprinting** — that happens at
the TCP/TLS handshake, before application code sees the request, and
needs a reverse proxy/WAF, not a Python library.

### A third layer: `not_a_robot.request_fingerprint`

Same pattern, one level up the stack: HTTP header/User-Agent
plausibility. Flags known non-browser User-Agents (`python-requests`,
`curl`, `Scrapy`, `HeadlessChrome`, ...), and — only when the UA claims
Chromium — missing headers real Chromium sends automatically
(`Sec-Fetch-*`, `Sec-CH-UA`, `Accept-Language`/`Accept-Encoding`):

```python
from not_a_robot.request_fingerprint import signals_from_headers, score_request

report = score_request(signals_from_headers(request.headers))  # Flask/Werkzeug or any dict
report.is_suspicious  # True for a known scraper UA, or missing Chromium headers
```

Deliberately does not check header order or TLS fingerprints (a WSGI
app behind a proxy/CDN commonly sees headers reordered before your code
sees them — a check that silently misbehaves per-infrastructure is
worse than no check).

**Also considered and declined: a bespoke rate limiter.** Rate limiting
is inherently distributed/stateful (multiple workers/pods) — a naive
in-process counter is silently wrong past one worker, which is nearly
every real deployment. Mature dedicated tools (Flask-Limiter, nginx,
your CDN/WAF) already solve this; duplicating it badly here would be
worse than not having it.

## Auto-retrain per project

`AutoRetrainStore` automates *when* a project's detector retrains, not
*what counts as ground truth*. Each project gets its own store; nothing
trains on a session without an explicit label:

```python
from not_a_robot import AutoRetrainStore

store = AutoRetrainStore("path/to/project/.not_a_robot", min_new_sessions=50)

store.record_session(session)   # live path -- raises if session.label is None
p_human = store.score(new_session)

# Separate periodic job (cron/scheduled task), NOT the request path --
# fitting + multi-seed CV takes tens of seconds, not ms:
record = store.maybe_retrain()  # None if under min_new_sessions since last retrain
```

Or as a scheduled command: `python -m not_a_robot.autoretrain --root path/to/project/.not_a_robot --min-new-sessions 50`.

The store trusts your labels — a honeypot that fires on humans teaches
the detector humans are bots; the model backup
(`model.joblib.<timestamp>.bak`) is the only rollback. Each retrain
fits on every session recorded so far, runs the same multi-seed CV as
above (so the record's ranges are defensible, not a single split),
backs up the model it replaces, and records `group_composition()` at
that retrain alongside the metrics — so a human-pass/bot-catch shift
between retrains is explainable (a skewed archetype mix), not
mysterious. Not built here, deliberately: any mechanism that would
label sessions from the detector's own predictions or unverified live
traffic — see [Scope](#scope).

## Client-side capture: `not-a-robot.js`

The library only defines the schema and feature math; you own serving
the client-side capture. `js/not-a-robot.js` is a reference
implementation matching the `InteractionSession` schema exactly — plain
JS, no dependencies, no build step. **Not part of the PyPI package**
(a JS file isn't Python wheel content); copy it into your own static
assets.

```html
<script src="/not-a-robot.js"></script>
<script>
  var collector = new NotARobot.Collector({ endpoint: "/telemetry" });
  collector.attachToForm("#signup-form");
</script>
```

Fire-and-forget by default (`fetch(..., { keepalive: true })` alongside
the real submit) — a telemetry failure should never block a real user.
`examples/integrations/flask_app.py` and `fastapi_app.py` show the
server side end to end (serving the script, receiving `/telemetry`,
scoring with `session_from_dict()` + `BotDetector.score()`):

```bash
pip install flask  # or: fastapi uvicorn
PYTHONPATH=. python examples/integrations/flask_app.py
```

Verified against a real browser: a real headless Chromium session
(`ActionChains` mouse movement, `send_keys()` typing, real form submit)
produced a captured session the server correctly parsed and scored.

## Capturing real training data

Tag each finished session with a label from a secondary signal you
trust (a CAPTCHA outcome, email verification, manual review), then pass
`InteractionSession` objects to `run_training_pipeline()` or persist
with `not_a_robot.io.save_sessions_jsonl()` for
`python -m not_a_robot.train --data sessions.jsonl` later. Tag `group`
when you want recall broken out by population (`"known_bot_honeypot"`,
`"known_bot_asn"`, etc.) — without it you only get the aggregate bot
catch rate, not which bots are slipping through.

`examples/synthetic_data.py`'s bundled archetypes exist only so the
pipeline has example data before you have real traffic — replace it
before relying on this for anything. What real validation exists so
far: `BotDetector` trained purely on synthetic data caught **20/20**
real captured Selenium sessions and **20/20** real captured Playwright
sessions (two different automation frameworks' default idioms, against
a plain local test page) — comfortably, not just barely. Full capture
methodology, the bug it found (a synthetic archetype's keystroke timing
was 50x too slow), and what this result does and doesn't generalize to,
are in [docs/validation.md](docs/validation.md#validating-against-real-automation-not-just-synthetic-bots).

## Realistic value by scenario

| Scenario | Value |
|---|---|
| Small site, comment spam, occasional scraping | High — pre-filter, reduce CAPTCHA frequency |
| Login/checkout on a mid-size site | Medium — worth adding, but IP reputation + rate limits + device fingerprinting do more |
| High-value target (banking, ticketing, account creation at scale) | Low on its own — needs to be one of 5–10 signals, most of which this package doesn't cover |
| Research, teaching, detector template | High — the methodology is the product |
| Replacing a commercial bot-management vendor | Not viable |

**The honest one-liner:** useful the way a smoke detector is useful —
catches the common cases cheaply, doesn't replace a fire-suppression
system. A detector whose limits are documented (this one prints
`sophisticated` passing, states the per-seed spread, says what
calibration didn't fix) is one you can build a layered defense around.

## Coverage by bot class, if deployed as a pre-auth signal

| Bot class | Environment layer | Behavioral layer | Overall |
|---|---|---|---|
| Unpatched Selenium/Puppeteer | Caught | Caught | Caught |
| Headless Chrome (no GPU) | Caught | Caught | Caught |
| Stealth-patched, container (no GPU) | Caught (WebGL) | Sometimes caught | Usually caught |
| Stealth-patched, GPU passthrough | Passes | ~34% caught | Often passes |
| Anti-detect browser + human-like automation | Passes | Weak signal | Passes |

The bottom two rows aren't a gap this package can close with more
checks: the environment layer only sees what JavaScript can observe
(once every property is either patched or genuinely real, there's
nothing left to detect), and the behavioral layer is a per-session
statistical classifier that a GAN-trajectory generator is specifically
trained to defeat. Closing those rows needs signals outside a
per-session, client-observable library's reach entirely — cross-session
fleet correlation, IP/ASN reputation, or your own accumulated
production data.

**What to do about it: don't gate on them, challenge on them.** Treat
`BotDetector.score()` as a step-up trigger — allow above a
high-confidence threshold, block below a low-confidence one, route the
ambiguous middle (where rows 4-5 land) to an actual challenge, using
`pipeline.cost_optimal_threshold()` to pick the boundaries for your cost
ratio. This package "reduces how often you need that challenge," not
"replaces it."

## Scope

This library builds defensive detection for a system you run and
control: a behavioral classifier (`BotDetector`), a deterministic
automation-artifact check (`not_a_robot.environment`), and a
deterministic HTTP header/User-Agent check
(`not_a_robot.request_fingerprint`). It intentionally does **not**
include: CAPTCHA-solving, browser automation for clicking through
third-party challenges, integrations with CAPTCHA-solving services,
synthetic mouse-trajectory generation meant to fool someone else's bot
detection, TLS/JA3-JA4 fingerprinting (needs a reverse proxy/WAF, not a
Python library), or rate limiting (a distributed/stateful problem with
mature dedicated tools already — Flask-Limiter, nginx, your CDN/WAF).
Those are a different (and, outside authorized testing of your own
systems, frequently abusive) category of tool.

## Development

```bash
pip install -e ".[dev]"
pytest
```
