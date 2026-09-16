"""Local dry-run demo: a live frontend showing not_a_robot in action.

Runs an AutoRetrainStore pre-seeded with the bundled synthetic dataset
(so it has a working model immediately), serves a page that captures
real mouse/keyboard/scroll/click/focus/paste events from whoever is
using it, scores sessions with the current model, and lets you record
new labeled sessions to watch the model retrain and its metrics update.

This is a local, self-contained demo. It captures events from this page
only, in your own browser, and never interacts with any third-party
site. See dry-run/README.md before treating anything here as more than
a demo: the "label" you pick in the UI is self-declared for
demonstration purposes, not a trusted ground-truth signal (see
AutoRetrainStore's own docstring on why that distinction matters).
"""

from __future__ import annotations

import os
import random
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from not_a_robot import AutoRetrainStore
from not_a_robot.environment import EnvironmentSignals, score_environment
from not_a_robot.io import session_from_dict, session_to_dict

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.chrome.service import Service as ChromeService

    _SELENIUM_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on optional dry-run dependency
    _SELENIUM_AVAILABLE = False

# Reaches into the demo generator's private archetype functions on
# purpose -- this is the same repo, not a public library boundary.
from examples.synthetic_data import (
    _bot_evasive_session,
    _bot_headless_session,
    _bot_naive_session,
    _bot_sophisticated_session,
    _human_like_session,
    make_synthetic_dataset,
)

STORE_ROOT = Path(os.environ.get("NOT_A_ROBOT_STORE", "/app/dry-run-data"))
STATIC_DIR = Path(__file__).parent / "static"

app = Flask(__name__, static_folder=None)

store = AutoRetrainStore(
    STORE_ROOT,
    min_new_sessions=5,
    cv_seeds=(0, 1),
    cv_n_splits=3,
    cv_n_repeats=2,
)

_ARCHETYPES = {
    "human": _human_like_session,
    "naive": _bot_naive_session,
    "evasive": _bot_evasive_session,
    "headless": _bot_headless_session,
    "sophisticated": _bot_sophisticated_session,
}

_REAL_GPU_RENDERER = "NVIDIA GeForce RTX 3080/PCIe/SSE2"

# Fixed, illustrative scenarios for the separate deterministic layer --
# not randomly generated, since the point is to show specific, named
# situations, not a statistical sample. The last one is deliberately
# identical (from this module's point of view) to the clean-browser
# scenario: that's not a bug in the demo, it's the honest limit
# score_environment() documents -- a stealth-patched bot passes every
# check here on purpose.
_ENVIRONMENT_SCENARIOS = [
    (
        "Clean browser",
        EnvironmentSignals(
            webdriver_flag=False,
            cdc_properties_present=False,
            automation_globals_present=False,
            webgl_renderer=_REAL_GPU_RENDERER,
        ),
    ),
    (
        "Unpatched Selenium",
        EnvironmentSignals(webdriver_flag=True, cdc_properties_present=True),
    ),
    (
        "Headless Chrome, no GPU passthrough",
        EnvironmentSignals(webdriver_flag=True, webgl_renderer="Google SwiftShader"),
    ),
    (
        "Playwright/Puppeteer, automation global left in place",
        EnvironmentSignals(automation_globals_present=True),
    ),
    (
        "Stealth-patched Selenium",
        EnvironmentSignals(
            webdriver_flag=False,
            cdc_properties_present=False,
            webgl_renderer=_REAL_GPU_RENDERER,
        ),
    ),
]


# Selenium's execute_script() wraps the given source as a function body,
# so a top-level `return` is valid -- no extra (function(){...})() needed.
_WEBGL_PROBE_JS = """
const canvas = document.createElement('canvas');
const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
if (!gl) { return [null, null]; }
const dbg = gl.getExtension('WEBGL_debug_renderer_info');
if (!dbg) { return [gl.getParameter(gl.RENDERER), gl.getParameter(gl.VENDOR)]; }
return [gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL), gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL)];
"""

_CDC_PROBE_JS = "return Object.keys(window).some((k) => k.indexOf('cdc_') === 0);"

# The standard stealth technique real plugins use: patch
# navigator.webdriver via CDP *before* any page script runs, not after
# with a plain post-load execute_script -- some checks (and real
# detection code) read the property during page load, so patching too
# late doesn't count as a fair stealth test.
_STEALTH_PATCH_JS = """
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
for (const key of Object.getOwnPropertyNames(window)) {
    if (key.indexOf('cdc_') === 0 || key.indexOf('$cdc_') === 0) {
        try { delete window[key]; } catch (e) { /* ignore */ }
    }
}
"""


def _launch_headless_chrome(stealth: bool):
    options = ChromeOptions()
    options.binary_location = os.environ.get("CHROME_BIN", "/usr/bin/chromium")
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    if stealth:
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)

    service = ChromeService(
        executable_path=os.environ.get("CHROMEDRIVER_BIN", "/usr/bin/chromedriver")
    )
    driver = webdriver.Chrome(service=service, options=options)
    if stealth:
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument", {"source": _STEALTH_PATCH_JS}
        )
    return driver


def _capture_real_browser_signals(stealth: bool) -> dict:
    """Launch an actual headless Chromium via Selenium (not a hardcoded
    example), capture its real navigator.webdriver / cdc_* / WebGL
    signals, and run them through the real score_environment(). This is
    the practical counterpart to _ENVIRONMENT_SCENARIOS: proof against a
    live automation process running in this same container, not just an
    illustrative dict."""
    driver = _launch_headless_chrome(stealth)
    try:
        driver.get("about:blank")
        webdriver_flag = driver.execute_script("return navigator.webdriver === true;")
        cdc_present = driver.execute_script(_CDC_PROBE_JS)
        renderer, vendor = driver.execute_script(_WEBGL_PROBE_JS)
        signals = EnvironmentSignals(
            webdriver_flag=bool(webdriver_flag),
            cdc_properties_present=bool(cdc_present),
            webgl_renderer=renderer,
            webgl_vendor=vendor,
        )
        report = score_environment(signals)
        return {
            "signals": {
                "webdriver_flag": signals.webdriver_flag,
                "cdc_properties_present": signals.cdc_properties_present,
                "webgl_renderer": signals.webgl_renderer,
                "webgl_vendor": signals.webgl_vendor,
            },
            "is_automated": report.is_automated,
            "reasons": report.reasons,
        }
    finally:
        driver.quit()


def _seed_baseline() -> None:
    """Pre-seed with the bundled synthetic dataset so the demo has a
    working model from the first request, instead of erroring on too
    little data for the first few clicks."""
    if store.total_session_count() > 0:
        return
    for session in make_synthetic_dataset(n_per_class=40, seed=0):
        store.record_session(session)
    store.maybe_retrain(force=True)


_seed_baseline()


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)


@app.route("/api/status")
def status():
    return jsonify(
        {
            "trained": store.model_path.exists(),
            "total_sessions": store.total_session_count(),
            "pending_sessions": store.pending_session_count(),
            "min_new_sessions": store.min_new_sessions,
            "history": store.history(),
        }
    )


@app.route("/api/score", methods=["POST"])
def score():
    data = request.get_json(force=True)
    session = session_from_dict(data)
    try:
        p_human = store.score(session)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409
    return jsonify({"score": p_human})


@app.route("/api/record", methods=["POST"])
def record():
    data = request.get_json(force=True)
    if data.get("label") is None:
        return jsonify({"error": "label (true/false) is required"}), 400
    session = session_from_dict(data)
    store.record_session(session)
    return jsonify({"pending": store.pending_session_count()})


@app.route("/api/retrain", methods=["POST"])
def retrain():
    body = request.get_json(silent=True) or {}
    force = bool(body.get("force", False))
    result = store.maybe_retrain(force=force)
    if result is None:
        return jsonify({"retrained": False, "pending": store.pending_session_count()})
    return jsonify({"retrained": True, **result})


@app.route("/api/simulate/<archetype>")
def simulate(archetype: str):
    """Generate one synthetic session from the given archetype and score
    it. With ?record=true, also records it -- using the *generator's own*
    true label (human archetype -> True, every bot archetype -> False),
    not a self-declared one, so this is real ground truth, unlike the
    label radio buttons in the manual "Record & self-enrich" section.

    Kept for direct API exploration (see dry-run/README.md); the Run
    Tests button uses /api/run_tests instead, which draws its batch from
    make_synthetic_dataset() rather than fixed per-archetype counts, so
    it can't drift from the library's own archetype weights the way a
    hand-picked count easily can (see that endpoint's docstring)."""
    generator = _ARCHETYPES.get(archetype)
    if generator is None:
        return jsonify({"error": f"unknown archetype {archetype!r}"}), 404
    session = generator(random.Random())
    try:
        p_human = store.score(session)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409

    recorded = False
    if request.args.get("record") == "true":
        store.record_session(session)
        recorded = True

    return jsonify(
        {
            "score": p_human,
            "archetype": archetype,
            "recorded": recorded,
            "pending": store.pending_session_count(),
            "session": session_to_dict(session),
        }
    )


@app.route("/api/run_tests", methods=["POST"])
def run_tests():
    """One atomic call behind the Run Tests button: generate a batch via
    make_synthetic_dataset() -- the same function the CLI's --synthetic
    mode and the README's benchmark use, so the archetype mix always
    matches the library's actual design weights (45/35/10/10) instead of
    a hand-picked count that can silently drift from them (an earlier
    version of this demo used a fixed 3:3:2:2 split that overweighted
    headless/sophisticated 2x relative to design -- this endpoint can't
    have that class of bug, because it delegates archetype selection
    entirely to the library's own weighted draw).

    Scores each session against the model as it stands *before* this
    call (useful evidence of what the previous model does with fresh
    data), records it, then forces a retrain and returns the retrain
    result plus the store's full cumulative composition -- so a pass/
    catch-rate change always comes with the numbers needed to tell
    "composition drifted" from "the model genuinely learned something"
    apart.
    """
    batch = make_synthetic_dataset(n_per_class=10, seed=random.randrange(2**31))

    results = []
    for session in batch:
        try:
            p_human = store.score(session)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409
        store.record_session(session)
        results.append(
            {
                "archetype": session.group,
                "label": "human" if session.label else "bot",
                "score": p_human,
            }
        )

    retrain_result = store.maybe_retrain(force=True)

    environment_results = [
        {
            "scenario": name,
            "is_automated": (report := score_environment(signals)).is_automated,
            "reasons": report.reasons,
            "checked": report.checked,
        }
        for name, signals in _ENVIRONMENT_SCENARIOS
    ]

    return jsonify(
        {
            "results": results,
            "retrain": retrain_result,
            "label_composition": store.label_composition(),
            "group_composition": store.group_composition(),
            "environment_results": environment_results,
        }
    )


@app.route("/api/run_real_browser_checks", methods=["POST"])
def run_real_browser_checks():
    """Launches two real headless Chromium instances via Selenium -- one
    unpatched, one with the standard CDP stealth patch -- and reports
    their actual captured signals through score_environment(). Meant to
    be called concurrently with /api/run_tests (see static/app.js), not
    sequentially after it: this endpoint's latency is dominated by
    browser startup, which is independent of the behavioral batch's ML
    training, so running them in parallel is a real wall-clock win, not
    just a UI trick -- app.run(threaded=True) below is what makes that
    actually concurrent server-side, not just dispatched concurrently by
    the browser.
    """
    if not _SELENIUM_AVAILABLE:
        return jsonify({"error": "selenium is not installed in this image"}), 503

    results = []
    for label, stealth in (
        ("Real Selenium, unpatched (launched just now)", False),
        ("Real Selenium, stealth-patched (launched just now)", True),
    ):
        try:
            outcome = _capture_real_browser_signals(stealth)
            results.append({"scenario": label, **outcome})
        except Exception as exc:  # noqa: BLE001 - report any launch failure to the UI
            results.append({"scenario": label, "error": str(exc)})

    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        debug=False,
        threaded=True,
    )
