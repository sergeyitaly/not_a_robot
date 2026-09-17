"""Capture REAL Playwright automation telemetry against a local test page.

The Selenium counterpart (capture_real_automation.py) found the trained
BotDetector correctly caught 20/20 real Selenium ActionChains/send_keys
sessions -- a genuine result, but a narrow one: it only tests one
framework's dispatch pattern. This script exists to check whether that
result generalizes, or was specific to Selenium's per-command WebDriver
round-trip timing, by driving the same /static/capture.html page with a
different framework's own idioms: page.mouse.move() (single-jump calls,
matching Selenium's move_by_offset().perform() granularity so the
comparison isolates framework from movement smoothness) and
page.keyboard.type() (which, like send_keys(), dispatches a real
keydown/keyup pair per character via CDP -- not synthesized in bulk).

Run inside the dry-run container (it needs Playwright + Chromium; reuses
the same apt-installed /usr/bin/chromium the Selenium capture uses via
executable_path, so no separate Playwright browser download is needed)
while the demo server is running:

    pip install playwright   # once; no `playwright install` needed --
                              # see executable_path below
    docker exec not-a-robot-demo python dry-run/capture_playwright.py \
        --count 20 --out /app/dry-run-data/real_playwright.jsonl

Every captured session is tagged label=False, group="real_playwright" --
these are, definitionally, automated sessions.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path
from typing import Optional

from playwright.sync_api import sync_playwright

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


def _capture_one_scripted_session(
    playwright, base_url: str, rng: random.Random
) -> Optional[InteractionSession]:
    """Drive one real Playwright session through the capture test page
    and read back whatever it actually produced."""
    browser = playwright.chromium.launch(
        executable_path=os.environ.get("CHROME_BIN", "/usr/bin/chromium"),
        headless=True,
        args=["--no-sandbox", "--disable-dev-shm-usage"],
    )
    try:
        page = browser.new_page(viewport={"width": 800, "height": 600})
        page.goto(f"{base_url}/static/capture.html")
        time.sleep(0.2)

        field_box = page.locator("#text-field").bounding_box()
        target_x = field_box["x"] + field_box["width"] / 2
        target_y = field_box["y"] + field_box["height"] / 2

        # Single-jump moves (default steps=1), matching the granularity
        # of the Selenium capture's move_by_offset().perform() calls --
        # isolates "different framework" from "different smoothness".
        page.mouse.move(target_x, target_y)
        for _ in range(rng.randint(3, 6)):
            target_x += rng.randint(-40, 40)
            target_y += rng.randint(-20, 20)
            page.mouse.move(target_x, target_y)

        # Real scripted typing via keyboard.type() -- like send_keys(),
        # dispatches one real keydown/keyup pair per character via CDP.
        page.locator("#text-field").click()
        page.keyboard.type("the quick brown fox")

        # Same scrollTop-assignment pattern as the Selenium capture, for
        # a direct scroll_events comparison.
        page.evaluate("document.getElementById('scroll-zone').scrollTop = 80")
        time.sleep(0.05)
        page.evaluate("document.getElementById('scroll-zone').scrollTop = 20")

        page.locator("#submit-btn").click()
        time.sleep(0.1)

        raw = page.evaluate("() => window.__session")
    except Exception as exc:  # noqa: BLE001 - report and skip, don't abort the whole batch
        print(f"  session failed: {exc}")
        return None
    finally:
        browser.close()

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
        group="real_playwright",
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--out", type=Path, default=Path("real_playwright_sessions.jsonl"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    sessions = []
    with sync_playwright() as playwright:
        for i in range(args.count):
            print(f"capturing session {i + 1}/{args.count}...")
            session = _capture_one_scripted_session(playwright, args.base_url, rng)
            if session is not None:
                sessions.append(session)

    save_sessions_jsonl(sessions, args.out)
    print(f"Saved {len(sessions)} real Playwright-captured sessions to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
