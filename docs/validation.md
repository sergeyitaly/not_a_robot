# Validation methodology and findings

The full detail behind the headline numbers in the main
[README](../README.md#training-pipeline-and-success-rate-validation).
Read this when you want to know *why* a number is what it is, not just
what it is.

## Three evaluation paths, three different questions

**`run_training_pipeline()`** fits the detector you'd actually deploy:
one stratified train/test split, fit on train, evaluated once on test.
Useful for producing a model + a quick report, but its metrics are a
**single point estimate** — on a dataset in the hundreds of sessions,
one 75/25 split can look meaningfully better or worse than another from
sampling luck alone, before the model is even a variable.

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

```bash
python -m not_a_robot.train --synthetic --n-per-class 200 --seeds 0,1,2,3
```

That produced (**1,600 sessions total**: 400/seed x 4 seeds, 5-fold x
10-repeat CV per seed, full feature set, `c_fa=10 : c_fr=1` for the
cost-optimal threshold, `BotDetector`'s calibrated default model):

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

## Calibration: what it did and didn't fix

`BotDetector`'s default model wraps its `RandomForestClassifier` in
`CalibratedClassifierCV` (isotonic) — a raw random forest's
`predict_proba` is a vote fraction, not a real probability, and a
reliability check on the raw model showed the predicted-vs-observed
relationship breaking down badly in a sparse mid-range (a handful of
test sessions per 0.1-wide probability bin, not tracking the observed
human fraction there) while a real, if partial, overlap between
`sophisticated` bots and humans sits in exactly that region. Calibrating
moved where the default 0.5 threshold sits on the ROC curve, which
raised default-threshold `sophisticated` recall from 23.7% (pooled,
pre-calibration) to 34.3% (post) and nudged overall bot catch rate up a
couple points. **It did not change the ROC curve itself, and it did not
change the cost-optimal operating point** — the cost-curve behavior at
`c_fa=10:c_fr=1` was unaffected: the cost-optimal threshold is still
0.85-0.89 across seeds, with FAR pushed to ~0% at the cost of a 13-16%
false reject rate on real humans, both before and after calibration.
That similarity is itself informative: it means that behavior was never
primarily a calibration artifact — it's what a 10:1 cost ratio actually
does when `sophisticated` bots and a minority of real humans (the ones
who also don't scroll, blur, or paste in a given session) genuinely
overlap in score.

**Whether trading a ~1-in-7 real-user rejection rate for catching most
`sophisticated` bots is worth it depends entirely on your own
false-accept-vs-reject cost, which is why `cost_fa`/`cost_fr` are
parameters, not constants** — the 10:1 default is illustrative, not a
recommendation; pass `--cost-fa`/`--cost-fr` with your actual
deployment's asymmetry (a login form and a comment form do not have the
same one). Rules of thumb to start from, not to ship blindly: a login or
payment form, start around `--cost-fa 100 --cost-fr 1`; a comment or
search form, `--cost-fa 10 --cost-fr 1` is closer.

## The `--drop-keys` ablation

Excluding keystroke-timing features (simulating a mouse-only capture
surface) barely moved anything: bot catch rate range 91.6-94.2% (vs.
92.1-93.7% with keys), combined `sophisticated` recall 34.2%
[31.1-37.4%] (vs. 34.3% with keys) — statistically indistinguishable,
both before and after calibration. This contradicts what a single-split
top-feature-importance list suggests (keystroke features ranked
highest) — that ranking reflected `naive`/`evasive` separability, not
what actually separates `sophisticated`. The reason is in the generator:
`sophisticated` reuses the human archetype's keystroke timing *and*
mouse trajectory exactly, so neither channel ever carried separating
signal against it — only the scroll/click/engagement features it
doesn't fake do. Keystroke timing helps separate `naive`/`evasive`
(which fake it badly), but mouse geometry alone already separates those
too, so dropping keys is redundant there, not costly.

**The lesson isn't "keystroke timing matters most" — it's "the channels
a specific bot doesn't bother faking are what catch it," a property of
the bot, not of any one feature group.** Run this against your own real
data before assuming it transfers; a real mouse-only capture surface
(e.g. a slider puzzle with no text field) will likely have worse
`naive`/`evasive` separability than this synthetic set, since here they
still fail on mouse geometry too.

## The synthetic generator's design

`examples/synthetic_data.py` draws bots from four weighted archetypes:
naive (straight-line path, uniform keystrokes, fixed click coordinate,
45%), evasive (jittered but still tighter than human, scripted scroll,
35%), headless (near-instant submit, little/no activity, 10%), and
sophisticated (10%) — which reuses the human archetype's mouse and
keyboard distributions *exactly*, so those two channels carry zero
separable signal against it by construction (see `description.txt` on
GAN-generated mouse trajectories and keystroke mimicry for why an
attacker would specifically invest there). The non-zero recall it shows
comes entirely from the scroll/click/engagement channels it does not
mimic, plus (at the cost-optimal threshold) trading human pass rate for
`sophisticated`-bot recall. That is the pipeline correctly recovering
the partial signal the generator leaves available — not a demonstration
of general robustness against every kind of mimicry.

The report format and numbers above are real, computed output from this
repo. The input data is not: it's synthetic, generated locally, with no
interaction with any real website. Run
`python -m not_a_robot.train --data <your sessions.jsonl> --seeds 0,1,2,3`
on real, labeled traffic from your own site to get numbers you can
actually trust for a production decision.

## Validating against real automation, not just synthetic bots

`dry-run/capture_real_automation.py` drives real Selenium sessions
(`ActionChains` mouse movement, `send_keys()` typing, a direct
`scrollTop` assignment for scrolling) against a local test page
(`dry-run/static/capture.html`) and records whatever the browser's own
event listeners actually captured — real automation telemetry, not an
assumption about what "a scripted bot" looks like. A sample of 20
captured sessions ships at `examples/data/real_selenium_sample.jsonl`.

Comparing that real data against the synthetic archetypes' feature
distributions found a genuine bug: `_bot_naive_session` hardcoded a
15ms keystroke dwell/flight time, but real `send_keys()` fires
keydown/keyup back-to-back in the same JS tick — actual dwell was
~0.3ms, flight ~0.05ms, roughly 50x faster than the archetype assumed.
That's now fixed to match the evidence.

Before and after that fix, a `BotDetector` trained purely on
`examples/synthetic_data.py` correctly classified **20/20** of the real
captured Selenium sessions as bot: `score()` (P(human)) clusters at
0.21–0.24 per session for a fixed training seed, and averages
0.11–0.37 across 5 independent training seeds — comfortably under the
0.5 threshold, but nowhere near saturated at 0.0. That's a genuinely
useful result, but a narrow one: it is not the 10,000-session
real-human benchmark this project still doesn't have.

Two artifacts in the capture worth knowing about if you look at the raw
data: the ~240ms (±52ms) mouse-move interval comes from
`capture_real_automation.py` issuing one `ActionChains.perform()` call
per move — a real WebDriver round-trip per command, not a client-side
polling collector (`capture.html` uses `addEventListener("mousemove")`)
— so it's a signal specific to that scripting pattern, not automation
in general. And 6 of the 20 sessions carry a spurious duplicate
`focus=true` pair 7–28ms after page load with no matching blur, which
looks like headless Chromium's own window-init behavior rather than
anything about user tab-switching; it doesn't affect the label, but
don't read "has focus events" as a human signal in this dataset.

**Does the Selenium result generalize to a different framework?**
`dry-run/capture_playwright.py` drives the same `capture.html` page
with Playwright instead — `page.mouse.move()` (single-jump calls,
matching `ActionChains.move_by_offset()`'s granularity so the
comparison isolates framework from movement smoothness) and
`page.keyboard.type()` (like `send_keys()`, one real keydown/keyup pair
per character via CDP, not synthesized in bulk). A sample ships at
`examples/data/real_playwright_sample.jsonl`. Same detector, same 5
training seeds: **20/20** Playwright sessions also caught, and more
confidently than the Selenium sample (mean P(human) 0.05–0.20 across
seeds, vs. 0.11–0.37 for Selenium). The reason shows up in the raw
features: Playwright's CDP-based dispatch has far lower per-command
latency than Selenium's WebDriver HTTP round-trips, so its captured
sessions are faster overall (`mouse_duration_ms` 357ms vs. 1524ms,
`time_to_submit_ms` 732ms vs. 1787ms) while keystroke dwell stays far
below human range either way (3.8ms mean for Playwright, 0.3ms for
Selenium — both reflect "no human types keys back-to-back with
near-zero, near-uniform dwell," just by different margins). That's
evidence the result isn't a Selenium-specific quirk of one WebDriver
implementation's timing — it's the same underlying signal (scripted
typing doesn't hold keys down the way hands do) surviving a change of
framework. It is still only two frameworks' *default* idioms, both
against a plain, unthrottled local form; it says nothing about
Puppeteer, CDP-driven mouse paths tuned to look human, or either
framework deliberately slowed down to mimic human timing.

## Verifying the environment layer against a real browser

[dry-run/](../dry-run/) launches an actual headless Chromium via
Selenium and runs its real captured signals through
`score_environment()`. Finding from building that: a naive one-line
stealth patch (only overriding `navigator.webdriver`) does **not** evade
detection — the `cdc_*` properties still give it away, since that patch
never touches them. A more thorough patch (also stripping `cdc_*` from
`window` via CDP before page load) genuinely defeats both of those
checks, but the WebGL check still caught it in that container, because
the container has no real GPU — not because the patch was incomplete. A
stealth-patched instance with real GPU passthrough would pass this
layer entirely. See [dry-run/README.md](../dry-run/README.md) for the
full walkthrough.
