"""HTTP request header / User-Agent plausibility checks.

Another separate, deterministic layer alongside
:mod:`not_a_robot.environment` -- the same rules apply: kept out of the
behavioral feature vector and ``evaluate_cv``'s statistical framework,
deterministic rather than probabilistic, and trivially defeated by
anyone who bothers to fully impersonate a real browser's header set. See
``environment``'s module docstring for the shared reasoning in full;
this one doesn't repeat it.

**Deliberately does not check header order.** Header order looks
consistent for a given HTTP library, but a WSGI app behind a reverse
proxy, load balancer, or CDN (nginx, Cloudflare, an ALB) commonly sees
headers normalized or reordered before they ever reach application
code, and Werkzeug's own header parsing doesn't guarantee wire order is
preserved either. A check that silently misbehaves depending on your
infrastructure is worse than no check. If you need real header-order or
TLS/JA3-JA4 fingerprinting, that has to happen at the proxy/WAF layer
where the actual wire-level data is still visible -- the same boundary
``not_a_robot.environment`` draws for TLS fingerprinting, and for the
same reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

_NON_BROWSER_UA_MARKERS = (
    "python-requests",
    "python-urllib",
    "curl/",
    "wget/",
    "scrapy/",
    "go-http-client",
    "okhttp",
    "postmanruntime",
    "axios/",
    "node-fetch",
    "libwww-perl",
    "java/",
    "apache-httpclient",
    "httpie",
    "headlesschrome",
)

_CHROMIUM_UA_MARKERS = ("chrome/", "chromium/", "edg/", "edge/")


def _lower(value: Optional[str]) -> str:
    return value.lower() if value else ""


def _is_known_non_browser_ua(user_agent: Optional[str]) -> bool:
    ua = _lower(user_agent)
    return any(marker in ua for marker in _NON_BROWSER_UA_MARKERS)


def _claims_chromium(user_agent: Optional[str]) -> bool:
    ua = _lower(user_agent)
    return any(marker in ua for marker in _CHROMIUM_UA_MARKERS)


@dataclass(frozen=True)
class RequestSignals:
    """Header-derived signals for one HTTP request. Every field is
    optional; leave a signal ``None`` if you didn't check for it --
    ``None`` means "unknown", not "absent". Build this from a raw
    headers mapping with :func:`signals_from_headers` instead of
    populating it by hand, unless you have a specific reason to.
    """

    user_agent: Optional[str] = None
    accept_language_present: Optional[bool] = None
    accept_encoding_present: Optional[bool] = None
    sec_fetch_present: Optional[bool] = None
    sec_ch_ua_present: Optional[bool] = None


def signals_from_headers(headers: Mapping[str, str]) -> RequestSignals:
    """Build :class:`RequestSignals` from a headers mapping (e.g.
    Flask/Werkzeug's ``request.headers``, or a plain dict) -- lookups
    are case-insensitive regardless of the mapping's own casing."""
    lower_keys = {k.lower(): v for k, v in headers.items()}

    def _present(name: str) -> bool:
        return bool(lower_keys.get(name))

    return RequestSignals(
        user_agent=lower_keys.get("user-agent"),
        accept_language_present=_present("accept-language"),
        accept_encoding_present=_present("accept-encoding"),
        sec_fetch_present=any(k.startswith("sec-fetch-") for k in lower_keys),
        sec_ch_ua_present=_present("sec-ch-ua"),
    )


@dataclass
class RequestReport:
    """Result of :func:`score_request`: which signals were checked, and
    which (if any) indicated a non-browser or implausible client."""

    is_suspicious: bool
    reasons: list[str] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.checked:
            return "No request signals were provided."
        if self.is_suspicious:
            return "Suspicious request: " + "; ".join(self.reasons)
        return f"No automation markers found ({len(self.checked)} signal(s) checked)."


def score_request(signals: RequestSignals) -> RequestReport:
    """Deterministic plausibility check for one HTTP request's headers.
    See the module docstring for what this is (and structurally cannot
    be): a cheap pre-filter for unmodified HTTP clients and scripted
    requests, not a substitute for behavioral scoring against a client
    that fully impersonates a real browser's header set.
    """
    reasons: list[str] = []
    checked: list[str] = []

    if signals.user_agent is not None:
        checked.append("user_agent")
        if _is_known_non_browser_ua(signals.user_agent):
            reasons.append(
                f"User-Agent identifies a known non-browser HTTP client: {signals.user_agent!r}"
            )

    claims_chromium = _claims_chromium(signals.user_agent)

    if signals.accept_language_present is not None:
        checked.append("accept_language_present")
        if not signals.accept_language_present:
            reasons.append(
                "Accept-Language header is missing (real browsers almost always send one)"
            )

    if signals.accept_encoding_present is not None:
        checked.append("accept_encoding_present")
        if not signals.accept_encoding_present:
            reasons.append(
                "Accept-Encoding header is missing (real browsers almost always send one)"
            )

    if claims_chromium and signals.sec_fetch_present is not None:
        checked.append("sec_fetch_present")
        if not signals.sec_fetch_present:
            reasons.append(
                "User-Agent claims a Chromium-based browser but no Sec-Fetch-* header is present"
            )

    if claims_chromium and signals.sec_ch_ua_present is not None:
        checked.append("sec_ch_ua_present")
        if not signals.sec_ch_ua_present:
            reasons.append(
                "User-Agent claims a Chromium-based browser but Sec-CH-UA is missing"
            )

    return RequestReport(is_suspicious=bool(reasons), reasons=reasons, checked=checked)
