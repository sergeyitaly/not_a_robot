"""Browser-observable automation-framework signals.

A separate, deterministic layer from :class:`~not_a_robot.detector.BotDetector`'s
behavioral classifier. Scope, deliberately narrow:

- **Client-observable only.** Every signal here is something JavaScript
  running in the browser can read and report to you: ``navigator.webdriver``,
  Selenium/ChromeDriver's injected ``cdc_*`` globals, Playwright/Puppeteer
  markers on ``window``, and the WebGL renderer/vendor strings. This
  module makes no network requests, runs no JavaScript, and does **not**
  fingerprint TLS/JA3-JA4 handshakes -- that happens before your
  application code ever sees the request, and requires a reverse proxy,
  WAF, or load balancer doing the fingerprinting, not a Python library.
  If you need that layer, it doesn't belong here; build or buy it
  separately and combine its output with this module's the same way you'd
  combine this module's output with ``BotDetector``'s.
- **Deterministic, not statistical.** :func:`score_environment` returns a
  boolean plus which specific signal(s) fired, not a probability -- these
  are near-certain markers when present (an unmodified Selenium session
  really does set ``navigator.webdriver``), so there's no calibration or
  cross-validation story here the way there is for ``BotDetector``.
- **Cheap and easily defeated.** Every signal here is exactly what
  stealth plugins (e.g. puppeteer-extra-plugin-stealth) and anti-detect
  browsers patch by default. A positive result is strong evidence of
  unsophisticated automation; a negative result means "no automation
  artifact was observed", not "this is a human" -- a stealth-patched bot
  passes every check here on purpose. That's exactly the case
  ``BotDetector``'s behavioral scoring exists for: catching what these
  checks can't.
- **Kept out of the behavioral feature vector and evaluation.**
  :class:`EnvironmentSignals` is not part of ``InteractionSession``,
  ``FEATURE_NAMES``, or ``evaluate_cv``'s per-group Wilson CI report --
  these are binary, deterministic markers, and mixing them into the
  CV/cost-curve framework built for a statistical classifier would
  misrepresent both. Combine the two scores at your application layer::

      env = score_environment(signals)
      behavioral = detector.score(session)
      if env.is_automated:
          block()  # near-certain, don't bother with the behavioral score
      else:
          decide(behavioral)  # env check passed (or wasn't run); fall back
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

_SOFTWARE_RENDERER_MARKERS = ("swiftshader", "llvmpipe", "mesa", "software rasterizer")


def _is_software_renderer(value: Optional[str]) -> bool:
    if not value:
        return False
    lowered = value.lower()
    return any(marker in lowered for marker in _SOFTWARE_RENDERER_MARKERS)


@dataclass(frozen=True)
class EnvironmentSignals:
    """Browser-observable signals, captured client-side and passed in as
    plain data. Every field is optional; leave a signal ``None`` if you
    didn't check for it, rather than guessing ``False`` -- ``None`` means
    "unknown", not "absent", and :func:`score_environment` only reports
    on the signals you actually provide.
    """

    webdriver_flag: Optional[bool] = None  # navigator.webdriver === true
    cdc_properties_present: Optional[bool] = None  # Selenium/ChromeDriver's cdc_* globals
    automation_globals_present: Optional[bool] = None  # e.g. window.__playwright__
    webgl_renderer: Optional[str] = None  # raw WebGL renderer string
    webgl_vendor: Optional[str] = None  # raw WebGL vendor string


@dataclass
class EnvironmentReport:
    """Result of :func:`score_environment`: which signals were checked,
    and which (if any) indicated automation."""

    is_automated: bool
    reasons: list[str] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.checked:
            return "No environment signals were provided."
        if self.is_automated:
            return "Automation detected: " + "; ".join(self.reasons)
        return f"No automation markers found ({len(self.checked)} signal(s) checked)."


def score_environment(signals: EnvironmentSignals) -> EnvironmentReport:
    """Deterministic check for browser-observable automation-framework
    artifacts. See the module docstring for what this is (and structurally
    cannot be): a cheap pre-filter for unmodified automation, not a
    substitute for behavioral scoring against stealth-patched bots.
    """
    reasons: list[str] = []
    checked: list[str] = []

    if signals.webdriver_flag is not None:
        checked.append("webdriver_flag")
        if signals.webdriver_flag:
            reasons.append("navigator.webdriver is true")

    if signals.cdc_properties_present is not None:
        checked.append("cdc_properties_present")
        if signals.cdc_properties_present:
            reasons.append("Selenium/ChromeDriver cdc_* properties present")

    if signals.automation_globals_present is not None:
        checked.append("automation_globals_present")
        if signals.automation_globals_present:
            reasons.append("a known automation-framework global was found on window")

    if signals.webgl_renderer is not None or signals.webgl_vendor is not None:
        checked.append("webgl_renderer")
        if _is_software_renderer(signals.webgl_renderer) or _is_software_renderer(
            signals.webgl_vendor
        ):
            reasons.append(
                f"WebGL is software-rendered (renderer={signals.webgl_renderer!r}, "
                f"vendor={signals.webgl_vendor!r}), consistent with headless "
                "without GPU passthrough"
            )

    return EnvironmentReport(is_automated=bool(reasons), reasons=reasons, checked=checked)
