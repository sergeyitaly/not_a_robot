# not_a_robot -- live dry-run demo

A one-button local demo of the `not_a_robot` library, answering "does
this actually work": generates 20 synthetic sessions across every
archetype the library ships (naive/evasive/headless/sophisticated bots,
and humans), each with its real, known label from the generator itself,
scores and records each against a live `not_a_robot.AutoRetrainStore`,
retrains, reports the actual accuracy/human-pass/bot-catch numbers, and
separately runs the deterministic `not_a_robot.environment` checks
against a handful of fixed scenarios so both detection layers are
exercised, not just the behavioral one.

## Run it

**Docker (from the repository root):**

```bash
docker build -f dry-run/Dockerfile -t not-a-robot-demo .
docker run --rm -p 8000:8000 not-a-robot-demo
```

Then open <http://localhost:8000> and click **Run Tests**.

**Or with Docker Compose** (works from any directory; paths in the
compose file are relative to itself, not your shell):

```bash
docker compose -f dry-run/docker-compose.yml up --build
```

The only difference from the plain `docker build`/`docker run` above:
Compose adds a named volume for `/app/dry-run-data`, so the
`AutoRetrainStore`'s session log/model survive a container restart
instead of reseeding from scratch every time. To get back to the clean
baseline the `--rm` flow above gives you, tear the volume down too:
`docker compose -f dry-run/docker-compose.yml down -v`.

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

**Deploying it publicly:** `render.yaml` at the repo root is a Render
Blueprint -- on Render, "New +" -> "Blueprint", connect this repo, it's
picked up automatically. It builds from `dry-run/Dockerfile` as-is; no
separate config needed. Two things to know before sharing the link:

- The free plan's filesystem is ephemeral -- the session log/model
  reset on every redeploy or cold start after 15 minutes idle. That's
  fine here (it's synthetic data, and it caps unbounded growth from
  public traffic); add a paid persistent disk at
  `NOT_A_ROBOT_STORE`'s path if you want state to survive.
- `/api/run_tests` and `/api/run_real_browser_checks` both launch real
  headless Chromium and/or a multi-seed CV retrain per call -- expensive
  enough that a public link needs *some* abuse guard. Both routes carry
  a per-IP cooldown (`app.py`'s `cooldown()` decorator, a plain
  in-process dict; 20s locally, `NOT_A_ROBOT_COOLDOWN_SECONDS` overrides
  it -- `render.yaml` sets 90s). That's deliberately not the thing the
  main README declines to build into the library: this is one Flask
  process behind one demo link, not a distributed deployment, so an
  in-process counter is the right tool here, not the same mistake at a
  different scale.
- **Render's free tier (0.1 vCPU) is genuinely weak for this app's
  default CV depth.** Measured taking multiple minutes per "Run Tests"
  click at the local defaults (2 seeds x 3 splits x 2 repeats, x3 more
  for isotonic calibration's own internal folds per fit) -- it does
  complete and save state correctly, but that's a bad experience for a
  real visitor. `NOT_A_ROBOT_CV_SEEDS`/`NOT_A_ROBOT_CV_N_SPLITS`/
  `NOT_A_ROBOT_CV_N_REPEATS`/`NOT_A_ROBOT_MIN_NEW_SESSIONS` let a
  deployment run a lighter depth; `render.yaml` sets `CV_SEEDS=0`,
  `CV_N_SPLITS=2`, `CV_N_REPEATS=1` (roughly a 3-4x reduction in RF fits
  vs. the local default) specifically for this reason. Still the same
  multi-seed methodology, just fewer seeds/folds -- an explicit,
  documented trade of statistical breadth for response time on a public
  link, not a silent shortcut.

## What happens when you click it

`dry-run/app.py` is a thin Flask wrapper around the library -- the
button drives real calls, not a mock:

1. Generates 20 sessions via `examples.synthetic_data.make_synthetic_dataset()`
   -- the same function the CLI's `--synthetic` mode and the README's own
   benchmark use, so the archetype mix always matches the library's real
   design weights (45/35/10/10 naive/evasive/headless/sophisticated,
   50/50 human/bot). An earlier version of this demo picked a fixed
   per-archetype count by hand, which silently drifted from those
   weights; delegating to the library's own generator makes that class of
   bug structurally impossible. Each session's label is the generator's
   own ground truth, not self-declared -- `label=True` for the human
   archetype, `False` for every bot archetype.
2. Scores each with the currently deployed model (`store.score()`), then
   records it (`store.record_session()`).
3. Once the batch is in, forces a retrain (`store.maybe_retrain(force=True)`):
   fits a fresh `BotDetector` on every session recorded so far and
   redeploys it.
4. Reports the retrain's real numbers: total sessions trained on, the
   accuracy / human-pass-rate / bot-catch-rate ranges from
   `not_a_robot`'s own multi-seed `evaluate_cv` (see the main
   [README's Training pipeline section](../README.md#training-pipeline-and-success-rate-validation)
   for what those ranges mean and why they're reported as ranges, not
   single numbers), the store's full cumulative label/archetype
   composition (`AutoRetrainStore.label_composition()` /
   `.group_composition()` -- printed so a future composition bug is
   visible immediately instead of requiring someone to export and
   inspect the session log by hand), and the 20 individual test scores
   from this run.
5. Separately, runs `not_a_robot.environment.score_environment()` against
   five fixed scenarios (clean browser, unpatched Selenium, headless
   without GPU passthrough, a Playwright/Puppeteer marker left in place,
   and a stealth-patched Selenium session) and reports each one's
   automated/not-automated verdict and reasons. This is a **separate
   table, not merged into the behavioral results** -- see the main
   README's [Deterministic automation checks section](../README.md#deterministic-automation-checks-separate-from-the-behavioral-model)
   for why. The stealth-patched scenario assumes real GPU passthrough,
   so it's deliberately identical, signal-for-signal, to the
   clean-browser one: that's the honest limit of this layer, not a demo
   bug.
6. **In parallel** with the above (both fired from the same click via
   `Promise.allSettled`, and actually processed concurrently server-side
   -- see `app.run(..., threaded=True)` -- not just dispatched that way
   by the browser), launches two **real** headless Chromium instances in
   this container via Selenium: one unpatched, one with a real CDP
   stealth patch (overrides `navigator.webdriver` *and* strips the
   injected `cdc_*` properties from `window`), reads each one's actual
   captured signals, and scores them with the same `score_environment()`
   call. This isn't a hardcoded example -- it's the practical answer to
   "can this package actually detect a real scraper." Verified while
   building this: the stealth patch genuinely defeats both the
   `navigator.webdriver` and `cdc_*` checks (both correctly read `false`
   afterward), but this container has no real GPU, so WebGL falls back
   to software rendering (`ANGLE (Google, Vulkan ... SwiftShader ...)`)
   and still gets caught by that one signal alone. **Both live instances
   run in this container**, so this demo can only ever show the
   no-GPU case -- it structurally cannot show what a stealth-patched
   instance with real GPU passthrough would look like. That instance
   would show a real renderer string and pass the environment layer
   entirely (`is_automated: false`), which is exactly the class
   `not_a_robot`'s behavioral scoring exists to catch, not something
   environment-layer checks can close on their own.

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
