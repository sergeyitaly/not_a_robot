# not_a_robot -- live dry-run demo

A local, self-contained web demo of the `not_a_robot` library: a page
that captures your real mouse/keyboard/scroll/click/focus/paste
behavior, scores it against a live `not_a_robot.AutoRetrainStore`, and
lets you watch the model retrain and its metrics change as you feed it
more labeled sessions.

## Run it

**Docker (from the repository root):**

```bash
docker build -f dry-run/Dockerfile -t not-a-robot-demo .
docker run --rm -p 8000:8000 not-a-robot-demo
```

Then open <http://localhost:8000>.

**Without Docker (from the repository root):**

```bash
pip install -r dry-run/requirements.txt
NOT_A_ROBOT_STORE=./dry-run-data PYTHONPATH=. python dry-run/app.py
```

`dry-run/requirements.txt` installs `not-a-robot` from PyPI (pinned to
the version this demo was built against) plus Flask -- the same as
`pip install not-a-robot` for anyone else. `PYTHONPATH=.` is still
needed so `examples/synthetic_data.py` is importable: it's demo-only
(baseline seeding + the "simulate" buttons), not part of the published
package, so it's only available from this checkout.

## What you're looking at

1. **Interaction capture area** -- move your mouse, type in the field,
   scroll the box, click around. Everything is captured client-side in
   `dry-run/static/app.js` into the same `InteractionSession` shape the
   library's own `not_a_robot.io` module reads and writes.
2. **Run Test** -- sends your just-captured session to the currently
   deployed model and shows P(human) as a score bar.
3. **Record & self-enrich** -- you declare a label (see the caveat
   below) and the session is appended to the store. Once
   `min_new_sessions` (5, for this demo -- see `dry-run/app.py`) new
   labeled sessions have accumulated, the store automatically retrains:
   fits a fresh `BotDetector`, runs multi-seed repeated CV, and
   redeploys. The status panel and retrain history table update to show
   the new accuracy/human-pass/bot-catch ranges. "Force retrain now"
   skips the threshold for an immediate demo.
4. **Simulate bot/human archetypes** -- instantly score a synthetic
   session from each archetype in `examples/synthetic_data.py` (naive,
   evasive, headless, sophisticated, human), without having to actually
   act like a bot yourself.
5. **Store status & retrain history** -- polls `/api/status` every 5s.

## The label caveat, made explicit

The radio buttons in step 3 ("I'm acting like a human" / "I'm simulating
bot-like behavior") are **you self-declaring the label**, purely so this
demo has something to train on. That is a stand-in for a real trusted
signal (a CAPTCHA outcome, a verified signup, manual review) -- it is
**not** how a production deployment should get its labels. See
`AutoRetrainStore`'s docstring in the library for why that distinction is
structural, not just a suggestion: `record_session()` will always accept
whatever label you assert here, because at the API level there's no way
to distinguish a demo's self-declared label from a real one. Don't wire
a production deployment up to accept labels from the same page a user is
trying to get verified on.

## Scope

This demo scores sessions captured on this page, in your own browser,
against a model you're running and can inspect locally. It has no code
path that reaches out to, or attempts to influence, any third-party
site's verification system -- consistent with the rest of the project
(see the main [README's Scope section](../README.md#scope)).
