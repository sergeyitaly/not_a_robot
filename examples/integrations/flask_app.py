"""Minimal Flask integration for not-a-robot.js.

Shows the whole wire end to end: a form page serves the collector
script, the collector POSTs a captured InteractionSession to /telemetry
on submit, and this app scores it with a BotDetector.

The detector here is trained on the bundled synthetic dataset purely so
this example runs standalone with a real, non-placeholder score -- see
the main README's "Training pipeline and success-rate validation" for
why that number isn't meaningful for a real deployment. Load your own
trained model (BotDetector.load(...)) in real use.

Run from the repo root (PYTHONPATH=. so `examples.synthetic_data` --
demo-only, not part of the installed package -- resolves):
    pip install flask
    pip install -e .
    PYTHONPATH=. python examples/integrations/flask_app.py
Then open http://127.0.0.1:5000/
"""

from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from not_a_robot import BotDetector, session_from_dict
from examples.synthetic_data import make_synthetic_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).parent / "static"
JS_DIR = REPO_ROOT / "js"

app = Flask(__name__, static_folder=None)

detector = BotDetector()
detector.fit(make_synthetic_dataset(n_per_class=200, seed=0))

_last_score: dict = {}


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/not-a-robot.js")
def collector_js():
    return send_from_directory(JS_DIR, "not-a-robot.js")


@app.route("/telemetry", methods=["POST"])
def telemetry():
    payload = request.get_json(force=True)
    session = session_from_dict(payload)
    p_human = detector.score(session)
    _last_score.clear()
    _last_score.update(
        {
            "p_human": round(p_human, 4),
            "predicted_human": detector.predict(session),
            "mouse_events_captured": len(session.mouse_events),
            "key_events_captured": len(session.key_events),
        }
    )
    return jsonify(_last_score)


@app.route("/last-score")
def last_score():
    return jsonify(_last_score or {"note": "no telemetry received yet"})


@app.route("/submitted")
def submitted():
    return "form submitted (this route exists only so the demo form has somewhere to go)"


if __name__ == "__main__":
    app.run(port=5000, debug=True)
