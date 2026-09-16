# not_a_robot -- live dry-run demo

A one-button local demo of the `not_a_robot` library, answering "does
this actually work": generates 20 synthetic sessions across every
archetype the library ships (naive/evasive/headless/sophisticated bots,
and humans), each with its real, known label from the generator itself,
scores and records each against a live `not_a_robot.AutoRetrainStore`,
retrains, and reports the actual accuracy/human-pass/bot-catch numbers.

## Run it

**Docker (from the repository root):**

```bash
docker build -f dry-run/Dockerfile -t not-a-robot-demo .
docker run --rm -p 8000:8000 not-a-robot-demo
```

Then open <http://localhost:8000> and click **Run Tests**.

**Without Docker (from the repository root):**

```bash
pip install -r dry-run/requirements.txt
NOT_A_ROBOT_STORE=./dry-run-data PYTHONPATH=. python dry-run/app.py
```

`dry-run/requirements.txt` installs `not-a-robot` from PyPI (pinned to
the version this demo was built against) plus Flask -- the same as
`pip install not-a-robot` for anyone else. `PYTHONPATH=.` is still
needed so `examples/synthetic_data.py` is importable: it's demo-only
(the archetype generators the button uses), not part of the published
package, so it's only available from this checkout.

## What happens when you click it

`dry-run/app.py` is a thin Flask wrapper around the library -- the
button drives real calls, not a mock:

1. Generates 20 sessions (4 each of human/naive/evasive/headless/
   sophisticated) via `examples/synthetic_data.py`'s archetype
   generators. Each one's label is the generator's own ground truth, not
   self-declared -- `label=True` for the human archetype, `False` for
   every bot archetype.
2. Scores each with the currently deployed model (`store.score()`), then
   records it (`store.record_session()`).
3. Once the batch is in, forces a retrain (`store.maybe_retrain(force=True)`):
   fits a fresh `BotDetector` on every session recorded so far and
   redeploys it.
4. Reports the retrain's real numbers: total sessions trained on, and
   the accuracy / human-pass-rate / bot-catch-rate ranges from
   `not_a_robot`'s own multi-seed `evaluate_cv` (see the main
   [README's Training pipeline section](../README.md#training-pipeline-and-success-rate-validation)
   for what those ranges mean and why they're reported as ranges, not
   single numbers), plus the 20 individual test scores from this run.

## Beyond the button

The UI only exposes the one button, but the backend still has the full
API if you want to poke at it directly:

```bash
curl http://localhost:8000/api/status
curl "http://localhost:8000/api/simulate/sophisticated?record=true"
curl -X POST http://localhost:8000/api/retrain -H "Content-Type: application/json" -d '{"force": true}'
```

Or skip the demo server and use the library directly -- see the main
[README](../README.md) and `AutoRetrainStore`'s docstring in
`src/not_a_robot/autoretrain.py`.

## Scope

This demo scores and trains on synthetic sessions generated locally. It
has no code path that reaches out to, or attempts to influence, any
third-party site's verification system -- consistent with the rest of
the project (see the main [README's Scope section](../README.md#scope)).
