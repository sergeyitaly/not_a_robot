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

This README can assert the pipeline works; [dry-run/](dry-run/) shows it:
a live local demo (Docker or plain Python), one button, a real result --
generates a synthetic batch across every archetype the library ships,
scores and trains on it, and reports the actual accuracy / human-pass /
bot-catch numbers the run just produced, not a mock.

**Status:** 0.1.1, alpha. Validated only on synthetic data so far; the
pipeline ships here, real-traffic numbers are yours. See
[Training pipeline and success-rate validation](#training-pipeline-and-success-rate-validation).

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

There are three evaluation paths, answering three different questions.

**`run_training_pipeline()`** fits the detector you'd actually deploy: one
stratified train/test split, fit on train, evaluated once on test. Useful
for producing a model + a quick report, but its metrics are a **single
point estimate** — on a dataset in the hundreds of sessions, one 75/25
split can look meaningfully better or worse than another from sampling
luck alone, before the model is even a variable.

**`evaluate_cv()`** runs repeated stratified k-fold CV (`n_splits x
n_repeats` independent folds, default 5x10=50) on *one* sample of
sessions, and reports mean +/- std per metric, **recall pooled by
`InteractionSession.group`** with a Wilson 95% confidence interval (not a
mean/std of per-fold rates — a rare group can have 0-2 members in a given
fold, where std is close to meaningless; pooling raw hit/total counts
across all folds is the number that's actually defensible), and the
**cost-optimal decision threshold** for a stated false-accept-vs-reject
cost ratio, with per-group recall at that threshold instead of just the
classifier's default 0.5 cut.

**`summarize_across_seeds()`** (CLI: `--seeds 0,1,2,3`) is the one to
actually quote. Repeated CV within one seed only captures fold-partition
variance — every fold in that run shares the same 400 sessions. Running
`evaluate_cv` at several seeds and pooling exposes the variance that
matters: how much the numbers move when the *sample itself* changes.

```python
from not_a_robot import run_training_pipeline, evaluate_cv, summarize_across_seeds

detector, report = run_training_pipeline(sessions, data_source="prod-2026-09")
detector.save("bot_detector.joblib")

cv_report = evaluate_cv(sessions, data_source="prod-2026-09")
print(cv_report.summary())
```

From the command line, against a real captured session log:

```bash
python -m not_a_robot.train --data sessions.jsonl --model-out bot_detector.joblib --report-out report.json
python -m not_a_robot.train --data sessions.jsonl --seeds 0,1,2,3   # the defensible report
```

`--synthetic` runs the same pipeline against the bundled demo dataset (see
below) so you can see a real, computed report before you have real traffic:

```bash
python -m not_a_robot.train --synthetic --n-per-class 200 --seeds 0,1,2,3
```

That produced (**1,600 sessions total**: 400/seed x 4 seeds, 5-fold x
10-repeat CV per seed, full feature set, `c_fa=10 : c_fr=1` for the
cost-optimal threshold, `BotDetector`'s calibrated default model — see
below):

```
  seed    accuracy   human pass   bot catch     FAR     FRR
  0         92.5%        92.9%       92.1%    7.9%    7.1%
  1         96.0%        98.2%       93.7%    6.3%    1.8%
  2         94.7%        95.8%       93.6%    6.4%    4.2%
  3         94.4%        96.5%       92.3%    7.7%    3.5%

Bot catch rate range across seeds: 92.1% - 93.7%  <- the honest operating characteristic

Combined per-group recall (pooled across all seeds, Wilson 95% CI):
  group          weight       n   recall [95% CI]
  human          50.0%    8000   95.9% [95.4%-96.3%]
  naive          21.9%    3500   100.0% [99.9%-100.0%]
  evasive        17.2%    2760   100.0% [99.9%-100.0%]
  headless        5.5%     880   100.0% [99.6%-100.0%]
  sophisticated   5.4%     860   34.3% [31.2%-37.5%]
```

**Read it as:** bot catch rate is stable at 92-94% across resamples, not a
single point estimate. `naive`/`evasive`/`headless` are caught at ~100%
with a tight interval (n in the thousands, pooled). `sophisticated` is
caught at 34.3% [31.2-37.5%] pooled — but **per-seed it ranges 10.7% to
54.3%**, a ~40-point spread the pooled interval doesn't show on its own.
That per-seed spread, not the pooled point estimate, is the honest
finding about this group: the only signal separating it from humans is
the scroll/click/engagement channels, and it's weak enough that which
seed the model happens to train on visibly changes how much of it gets
caught. **Do not treat any single seed's `sophisticated` recall as an
estimate of real-world performance against mimicry bots** — not the
54.3% from seed 2, and not the pooled 34.3% either, without also carrying
that per-seed range.

**Calibration, and what it did and didn't fix.** `BotDetector`'s default
model wraps its `RandomForestClassifier` in `CalibratedClassifierCV`
(isotonic) — a raw random forest's `predict_proba` is a vote fraction,
not a real probability, and a reliability check on the raw model showed
the predicted-vs-observed relationship breaking down badly in a sparse
mid-range (a handful of test sessions per 0.1-wide probability bin, not
tracking the observed human fraction there) while a real, if partial,
overlap between `sophisticated` bots and humans sits in exactly that
region. Calibrating moved where the default 0.5 threshold sits on the
ROC curve, which raised default-threshold `sophisticated` recall from
23.7% (pooled, pre-calibration) to 34.3% (post) and nudged overall bot
catch rate up a couple points. **It did not change the ROC curve itself,
and it did not change the cost-optimal operating point** — the cost-curve
behavior at `c_fa=10:c_fr=1` was unaffected: the
cost-optimal threshold is still 0.85-0.89 across seeds, with FAR pushed
to ~0% at the cost of a 13-16% false reject rate on real humans, both
before and after calibration. That similarity is itself informative: it
means that behavior was never primarily a calibration artifact — it's
what a 10:1 cost ratio actually does when `sophisticated` bots and a
minority of real humans (the ones who also don't scroll, blur, or paste
in a given session) genuinely overlap in score. **Whether trading a
~1-in-7 real-user rejection rate for catching most `sophisticated` bots
is worth it depends entirely on your own false-accept-vs-reject cost,
which is why `cost_fa`/`cost_fr` are parameters, not constants** — the
10:1 default here is illustrative, not a recommendation; pass
`--cost-fa`/`--cost-fr` with your actual deployment's asymmetry (a login
form and a comment form do not have the same one), and don't ship the
cost-optimal threshold without deciding you actually want that trade.
Rules of thumb to start from, not to ship blindly: a login or payment
form, start around `--cost-fa 100 --cost-fr 1`; a comment or search form,
`--cost-fa 10 --cost-fr 1` is closer.

**`--drop-keys` ablation** (excludes keystroke-timing features, simulating
a mouse-only capture surface): removing them barely moved anything — bot
catch rate range 91.6-94.2% (vs. 92.1-93.7% with keys), combined
`sophisticated` recall 34.2% [31.1-37.4%] (vs. 34.3% with keys),
statistically indistinguishable. This holds both before and after
calibration, and contradicts what the single-split top-feature-importance
list suggested earlier (keystroke features ranked highest) — that ranking
reflected `naive`/`evasive` separability, not what actually separates
`sophisticated`. The reason is in the generator: `sophisticated` reuses
the human archetype's keystroke timing *and* mouse trajectory exactly, so
neither channel ever carried separating signal against it — only the
scroll/click/engagement features it doesn't fake do. Keystroke timing
helps separate `naive`/`evasive` (which fake it badly), but mouse
geometry alone already separates those too, so dropping keys is
redundant there, not costly. **The lesson isn't "keystroke timing matters
most" — it's "the channels a specific bot doesn't bother faking are what
catch it," a property of the bot, not of any one feature group.** Run
this against your own real data before assuming it transfers; a real
mouse-only capture surface (e.g. a slider puzzle with no text field) will
likely have worse `naive`/`evasive` separability than this synthetic set,
since here they still fail on mouse geometry too.

The synthetic generator (`examples/synthetic_data.py`) draws bots from
four weighted archetypes: naive (straight-line path, uniform keystrokes,
fixed click coordinate, 45%), evasive (jittered but still tighter than
human, scripted scroll, 35%), headless (near-instant submit, little/no
activity, 10%), and sophisticated (10%) — which reuses the human
archetype's mouse and keyboard distributions *exactly*, so those two
channels carry zero separable signal against it by construction (see
`description.txt` on GAN-generated mouse trajectories and keystroke
mimicry for why an attacker would specifically invest there). The
non-zero recall it shows comes entirely from the scroll/click/engagement
channels it does not mimic, plus (at the cost-optimal threshold) trading
human pass rate for `sophisticated`-bot recall. That is the pipeline
correctly recovering the partial signal the generator leaves available —
not a demonstration of general robustness against every kind of mimicry.

The report format and numbers above are real, computed output from this
repo. The input data is not: it's synthetic, generated locally, with no
interaction with any real website. Run
`python -m not_a_robot.train --data <your sessions.jsonl> --seeds 0,1,2,3`
on real, labeled traffic from your own site to get numbers you can
actually trust for a production decision.

## Auto-retrain per project

`AutoRetrainStore` automates *when* a project's detector gets retrained,
not *what counts as ground truth*. Each project gets its own store rooted
at its own directory -- no data or model is shared across projects, and
there's no code path that trains on anything but a session you've
explicitly labeled:

```python
from not_a_robot import AutoRetrainStore

store = AutoRetrainStore("path/to/project/.not_a_robot", min_new_sessions=50)

# From your live scoring path (cheap -- just a file append):
store.record_session(session)  # raises if session.label is None
p_human = store.score(new_session)
```

The store trusts your labels. A honeypot that fires on humans teaches
the detector that humans are bots; the model backup
(`model.joblib.<timestamp>.bak`) is the only rollback. Label quality is
upstream of this library -- the `label is not None` guard stops an
*unlabeled* session from being trained on, not a *wrongly* labeled one.

```python
# From a separate periodic job (cron, a scheduled task) -- NOT the
# request path: fitting + multi-seed CV takes tens of seconds, not ms.
record = store.maybe_retrain()  # None if under min_new_sessions since last retrain
```

Or as a scheduled command:

```bash
python -m not_a_robot.autoretrain --root path/to/project/.not_a_robot --min-new-sessions 50
```

Real output from a run (30 sessions recorded, below the 50 threshold, then
20 more crossing it):

```
pending after 30 sessions: 30
maybe_retrain() result: None
pending after 50 sessions: 50
{
  "timestamp": "2026-09-16T20:40:15.396101+00:00",
  "n_sessions": 50,
  "n_new_sessions": 50,
  "seeds": [0, 1, 2],
  "accuracy_range": [0.942, 0.946],
  "human_pass_rate_range": [0.964, 0.972],
  "bot_catch_rate_range": [0.92, 0.92]
}
model file exists: True
```

Each retrain fits on every session recorded so far, runs the same
multi-seed `evaluate_cv` used above (so the record's ranges are the
defensible cross-seed numbers, not a single split), backs up the model it
replaces (`model.joblib.<timestamp>.bak`, never deleted automatically --
rollback is a file copy), and appends the summary to `state.json`. Not
built here, deliberately: any mechanism that would label sessions from
the detector's own predictions or from unverified live traffic. That's
the difference between "automates when you retrain" (this) and "trains
itself on whatever it sees" (a real risk of training-data poisoning, and
out of scope for this library — see [Scope](#scope)).

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

Tag `group` when you have a population label you want recall broken out
by: `"human"`, `"known_bot_honeypot"` / `"known_bot_asn"` /
`"known_bot_review"` (one per label provenance), `"unknown"` for sessions
you score but haven't labeled. `evaluate_cv()` pools recall per group
with a Wilson CI. Without a group tag, you get the aggregate bot catch
rate and none of the per-group breakdown — which is the part that tells
you which bots are slipping through.

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
