from not_a_robot.environment import EnvironmentSignals, score_environment


def test_no_signals_provided_is_not_flagged_and_checks_nothing():
    report = score_environment(EnvironmentSignals())
    assert report.is_automated is False
    assert report.checked == []
    assert report.reasons == []
    assert "No environment signals" in report.summary()


def test_webdriver_flag_true_is_detected():
    report = score_environment(EnvironmentSignals(webdriver_flag=True))
    assert report.is_automated is True
    assert "webdriver_flag" in report.checked
    assert any("webdriver" in r for r in report.reasons)


def test_webdriver_flag_false_is_checked_but_not_flagged():
    report = score_environment(EnvironmentSignals(webdriver_flag=False))
    assert report.is_automated is False
    assert report.checked == ["webdriver_flag"]
    assert report.reasons == []


def test_cdc_properties_present_is_detected():
    report = score_environment(EnvironmentSignals(cdc_properties_present=True))
    assert report.is_automated is True
    assert any("cdc" in r.lower() for r in report.reasons)


def test_automation_globals_present_is_detected():
    report = score_environment(EnvironmentSignals(automation_globals_present=True))
    assert report.is_automated is True
    assert any("global" in r.lower() for r in report.reasons)


def test_software_webgl_renderer_is_detected():
    report = score_environment(EnvironmentSignals(webgl_renderer="Google SwiftShader"))
    assert report.is_automated is True
    assert any("software" in r.lower() for r in report.reasons)


def test_real_gpu_webgl_renderer_is_not_flagged():
    report = score_environment(
        EnvironmentSignals(webgl_renderer="NVIDIA GeForce RTX 3080/PCIe/SSE2")
    )
    assert report.is_automated is False
    assert report.checked == ["webgl_renderer"]


def test_multiple_signals_all_report_as_reasons():
    report = score_environment(
        EnvironmentSignals(webdriver_flag=True, cdc_properties_present=True)
    )
    assert report.is_automated is True
    assert len(report.reasons) == 2
    assert len(report.checked) == 2


def test_summary_mentions_check_count_when_clean():
    report = score_environment(
        EnvironmentSignals(webdriver_flag=False, cdc_properties_present=False)
    )
    assert "2 signal" in report.summary()
