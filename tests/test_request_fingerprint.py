from not_a_robot.request_fingerprint import (
    RequestSignals,
    score_request,
    signals_from_headers,
)

_REAL_CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def test_no_signals_provided_is_not_flagged():
    report = score_request(RequestSignals())
    assert report.is_suspicious is False
    assert report.checked == []
    assert "No request signals" in report.summary()


def test_known_non_browser_user_agent_is_flagged():
    report = score_request(RequestSignals(user_agent="python-requests/2.31.0"))
    assert report.is_suspicious is True
    assert any("python-requests" in r for r in report.reasons)


def test_curl_user_agent_is_flagged():
    report = score_request(RequestSignals(user_agent="curl/8.4.0"))
    assert report.is_suspicious is True


def test_headless_chrome_user_agent_is_flagged():
    report = score_request(
        RequestSignals(user_agent="Mozilla/5.0 HeadlessChrome/120.0.0.0 Safari/537.36")
    )
    assert report.is_suspicious is True
    assert any("headlesschrome" in r.lower() for r in report.reasons)


def test_real_browser_user_agent_alone_is_not_flagged():
    report = score_request(RequestSignals(user_agent=_REAL_CHROME_UA))
    assert report.is_suspicious is False
    assert report.checked == ["user_agent"]


def test_missing_accept_language_is_flagged():
    report = score_request(
        RequestSignals(user_agent=_REAL_CHROME_UA, accept_language_present=False)
    )
    assert report.is_suspicious is True
    assert any("Accept-Language" in r for r in report.reasons)


def test_missing_sec_fetch_flagged_only_when_chromium_claimed():
    # Firefox doesn't send Sec-CH-UA/Sec-Fetch the same way and isn't a
    # Chromium UA -- the check should not fire for it.
    firefox_ua = "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"
    report = score_request(
        RequestSignals(user_agent=firefox_ua, sec_fetch_present=False, sec_ch_ua_present=False)
    )
    assert report.is_suspicious is False
    assert "sec_fetch_present" not in report.checked
    assert "sec_ch_ua_present" not in report.checked


def test_missing_sec_fetch_flagged_when_chromium_claimed():
    report = score_request(
        RequestSignals(user_agent=_REAL_CHROME_UA, sec_fetch_present=False)
    )
    assert report.is_suspicious is True
    assert any("Sec-Fetch" in r for r in report.reasons)


def test_signals_from_headers_builds_correct_signals():
    headers = {
        "User-Agent": _REAL_CHROME_UA,
        "Accept-Language": "en-US,en;q=0.9",
        "Sec-Fetch-Site": "none",
        "Sec-CH-UA": '"Chromium";v="120"',
    }
    signals = signals_from_headers(headers)
    assert signals.user_agent == _REAL_CHROME_UA
    assert signals.accept_language_present is True
    assert signals.accept_encoding_present is False
    assert signals.sec_fetch_present is True
    assert signals.sec_ch_ua_present is True

    report = score_request(signals)
    assert report.is_suspicious is True  # missing Accept-Encoding
    assert any("Accept-Encoding" in r for r in report.reasons)


def test_signals_from_headers_is_case_insensitive():
    headers = {"user-agent": "curl/8.0", "ACCEPT-LANGUAGE": "en"}
    signals = signals_from_headers(headers)
    assert signals.user_agent == "curl/8.0"
    assert signals.accept_language_present is True
