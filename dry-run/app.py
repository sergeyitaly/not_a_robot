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

    return jsonify(
        {
            "results": results,
            "retrain": retrain_result,
            "label_composition": store.label_composition(),
            "group_composition": store.group_composition(),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=False)
