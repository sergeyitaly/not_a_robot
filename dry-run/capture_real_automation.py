"""Capture REAL automation-tool telemetry against a local test page.

Launches real Selenium sessions that perform scripted mouse movement
(ActionChains), typing (send_keys), and scrolling (a direct scrollTop
assignment via JS -- the most common automation pattern, which never
fires a wheel event at all), navigates to this demo's own
/static/capture.html (served locally, not any third-party site), and
reads back whatever real browser events actually fired. This replaces
an assumption about what "a scripted bot" looks like with evidence: what
Selenium's own APIs actually produce, captured by the same kind of
client-side listeners the rest of this project uses.

Run inside the dry-run container (it needs Selenium + Chromium, which
are only installed there) while the demo server is running:

    docker exec not-a-robot-demo python dry-run/capture_real_automation.py \
        --count 20 --out /app/dry-run-data/real_selenium.jsonl

Every captured session is tagged label=False, group="real_selenium" --
these are, definitionally, automated sessions. There is no honest way
for this script to also produce a "human" counterpart; see the project
README's discussion of why 10,000 real human sessions can't be
manufactured this way.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path
from typing import Optional

from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By

from not_a_robot.io import save_sessions_jsonl
from not_a_robot.schema import (
    ClickEvent,
    FocusEvent,
    InteractionSession,
    KeyEvent,
    MouseEvent,
    PasteEvent,
    ScrollEvent,
)


def _launch_chrome():
    options = ChromeOptions()
    options.binary_location = os.environ.get("CHROME_BIN", "/usr/bin/chromium")
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=800,600")
    service = ChromeService(
        executable_path=os.environ.get("CHROMEDRIVER_BIN", "/usr/bin/chromedriver")
    )
    return webdriver.Chrome(service=service, options=options)


def _capture_one_scripted_session(
    base_url: str, rng: random.Random
) -> Optional[InteractionSession]:
    """Drive one real Selenium session through the capture test page
    and read back whatever it actually produced."""
    driver = _launch_chrome()
    try:
        driver.get(f"{base_url}/static/capture.html")
        time.sleep(0.2)

        text_field = driver.find_element(By.ID, "text-field")
        submit_btn = driver.find_element(By.ID, "submit-btn")

        # Real Selenium ActionChains mouse movement: a small number of
        # discrete moves, which is what the common ActionChains pattern
        # actually produces (each move_by_offset dispatches one event),
        # not a continuous stream the way real hardware does.
        actions = ActionChains(driver)
        actions.move_to_element(text_field)
        actions.perform()
        for _ in range(rng.randint(3, 6)):
            actions = ActionChains(driver)
            actions.move_by_offset(rng.randint(-40, 40), rng.randint(-20, 20))
            actions.perform()

        # Real scripted typing via send_keys.
        text_field.click()
        text_field.send_keys("the quick brown fox")

        # Real scripted scroll via a direct scrollTop assignment -- the
        # most common automation pattern, distinct from a real wheel
        # event.
        driver.execute_script("document.getElementById('scroll-zone').scrollTop = 80;")
        time.sleep(0.05)
        driver.execute_script("document.getElementById('scroll-zone').scrollTop = 20;")

        submit_btn.click()
        time.sleep(0.1)

        raw = driver.execute_script("return window.__session;")
    except Exception as exc:  # noqa: BLE001 - report and skip, don't abort the whole batch
        print(f"  session failed: {exc}")
        return None
    finally:
        driver.quit()

    return InteractionSession(
        mouse_events=[MouseEvent(x, y, t) for x, y, t in raw["mouse_events"]],
        key_events=[KeyEvent(t_down, t_up) for t_down, t_up in raw["key_events"]],
        scroll_events=[ScrollEvent(t, d) for t, d in raw["scroll_events"]],
        click_events=[ClickEvent(x, y, t) for x, y, t in raw["click_events"]],
        focus_events=[FocusEvent(t, f) for t, f in raw["focus_events"]],
        paste_events=[PasteEvent(t, n) for t, n in raw["paste_events"]],
        page_load_t=raw["page_load_t"],
        submit_t=raw["submit_t"],
        label=False,
        group="real_selenium",
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--out", type=Path, default=Path("real_selenium_sessions.jsonl"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    sessions = []
    for i in range(args.count):
        print(f"capturing session {i + 1}/{args.count}...")
        session = _capture_one_scripted_session(args.base_url, rng)
        if session is not None:
            sessions.append(session)

    save_sessions_jsonl(sessions, args.out)
    print(f"Saved {len(sessions)} real Selenium-captured sessions to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
