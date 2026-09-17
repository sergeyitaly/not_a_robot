"""Minimal FastAPI integration for not-a-robot.js.

Same wiring as flask_app.py, in an ASGI framework: the collector script
is served as a static file, a POST to /telemetry carries the captured
InteractionSession, and this app scores it with a BotDetector. See
flask_app.py's docstring for the caveat on the synthetic-trained
detector used here.

Run from the repo root (PYTHONPATH=. so `examples.synthetic_data` --
demo-only, not part of the installed package -- resolves):
    pip install fastapi uvicorn
    pip install -e .
    PYTHONPATH=. uvicorn examples.integrations.fastapi_app:app --reload
Then open http://127.0.0.1:8000/
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from not_a_robot import BotDetector, session_from_dict
from examples.synthetic_data import make_synthetic_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).parent / "static"
JS_DIR = REPO_ROOT / "js"

app = FastAPI()

detector = BotDetector()
detector.fit(make_synthetic_dataset(n_per_class=200, seed=0))

_last_score: dict = {}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/not-a-robot.js")
def collector_js():
    return FileResponse(JS_DIR / "not-a-robot.js", media_type="application/javascript")


@app.post("/telemetry")
async def telemetry(request: Request):
    payload = await request.json()
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
    return JSONResponse(_last_score)


@app.get("/last-score")
def last_score():
    return JSONResponse(_last_score or {"note": "no telemetry received yet"})


@app.get("/submitted")
def submitted():
    return "form submitted (this route exists only so the demo form has somewhere to go)"
