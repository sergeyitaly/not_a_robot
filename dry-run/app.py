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
from not_a_robot.io import session_from_dict, session_to_dict

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
    generator = _ARCHETYPES.get(archetype)
    if generator is None:
        return jsonify({"error": f"unknown archetype {archetype!r}"}), 404
    session = generator(random.Random())
    try:
        p_human = store.score(session)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409
    return jsonify(
        {"score": p_human, "archetype": archetype, "session": session_to_dict(session)}
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=False)
