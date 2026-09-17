(function () {
  "use strict";

  function pct(range) {
    return `${(range[0] * 100).toFixed(1)}-${(range[1] * 100).toFixed(1)}%`;
  }

  async function postJSON(url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    return res.json();
  }

  function addResultRow(archetype, score) {
    const tbody = document.querySelector("#results-table tbody");
    const row = document.createElement("tr");
    row.innerHTML = `<td>${archetype}</td><td>${(score * 100).toFixed(1)}%</td>`;
    tbody.appendChild(row);
  }

  function addEnvironmentRow(scenario, isAutomated, reasons) {
    const tbody = document.querySelector("#environment-table tbody");
    const row = document.createElement("tr");
    const verdict = isAutomated ? "yes" : "no";
    const reasonText = reasons.length ? reasons.join("; ") : "--";
    row.innerHTML = `<td>${scenario}</td><td>${verdict}</td><td>${reasonText}</td>`;
    tbody.appendChild(row);
  }

  function addRealBrowserRow(scenario, isAutomated, reasons, signals, error) {
    const tbody = document.querySelector("#real-browser-table tbody");
    const row = document.createElement("tr");
    if (error) {
      row.innerHTML = `<td>${scenario}</td><td colspan="3">Failed: ${error}</td>`;
      tbody.appendChild(row);
      return;
    }
    const verdict = isAutomated ? "yes" : "no";
    const reasonText = reasons.length ? reasons.join("; ") : "--";
    const signalText =
      `webdriver=${signals.webdriver_flag}, cdc_*=${signals.cdc_properties_present}, ` +
      `webgl=${signals.webgl_renderer || "(none)"}`;
    row.innerHTML =
      `<td>${scenario}</td><td>${verdict}</td><td>${reasonText}</td><td>${signalText}</td>`;
    tbody.appendChild(row);
  }

  function addHistoryRow(record) {
    const tbody = document.querySelector("#history-table-runs tbody");
    const row = document.createElement("tr");
    const when = new Date(record.timestamp).toLocaleTimeString();
    row.innerHTML =
      `<td>${when}</td><td>${record.n_sessions}</td>` +
      `<td>${pct(record.accuracy_range)}</td>` +
      `<td>${pct(record.human_pass_rate_range)}</td>` +
      `<td>${pct(record.bot_catch_rate_range)}</td>`;
    tbody.appendChild(row);
  }

  function addRequestFingerprintRow(scenario, userAgent, isSuspicious, reasons) {
    const tbody = document.querySelector("#your-request-table tbody");
    const row = document.createElement("tr");
    const verdict = isSuspicious ? "yes" : "no";
    const reasonText = reasons.length ? reasons.join("; ") : "--";
    row.innerHTML =
      `<td>${scenario}</td><td>${userAgent || "(none sent)"}</td>` +
      `<td>${verdict}</td><td>${reasonText}</td>`;
    tbody.appendChild(row);
  }

  function formatComposition(labelComp, groupComp) {
    const groupPart = groupComp
      ? ` (naive ${groupComp.naive || 0}, evasive ${groupComp.evasive || 0}, ` +
        `headless ${groupComp.headless || 0}, sophisticated ${groupComp.sophisticated || 0})`
      : "";
    return `${labelComp.human} human / ${labelComp.bot} bot${groupPart}`;
  }

  // One sentence, computed from this run's actual results -- not
  // written in advance. label is "human" or "bot" per session; score
  // is P(human), so a bot is "blocked" at score < 0.5 and a human
  // "passed" at score >= 0.5.
  //
  // Two lines, both explicitly labeled, not one ambiguous number: this
  // run's 20-session count and the cumulative multi-seed CV range
  // measure different things (one sample vs. the whole trained-on
  // history) and can disagree by design -- see the "Header:
  // post-retrain..." note above the table. Labeling both is structural,
  // not a footnote a reader has to go find.
  function updateVerdict(results, cumulative) {
    const humans = results.filter((r) => r.label === "human");
    const bots = results.filter((r) => r.label === "bot");
    const humansPassed = humans.filter((r) => r.score >= 0.5).length;
    const botsBlocked = bots.filter((r) => r.score < 0.5).length;
    document.getElementById("m-verdict").textContent =
      `This run: ${botsBlocked}/${bots.length} bots blocked, ` +
      `${humansPassed}/${humans.length} humans passed.`;

    const cumulativeEl = document.getElementById("m-verdict-cumulative");
    if (cumulative) {
      cumulativeEl.textContent =
        `Cumulative (${cumulative.n_sessions}): bot catch ${pct(cumulative.bot_catch_rate_range)}, ` +
        `human pass ${pct(cumulative.human_pass_rate_range)}.`;
    } else {
      cumulativeEl.textContent = "";
    }
  }

  // Only appears when this run actually drew a sophisticated row --
  // it's drawn at its 10% design weight, so most runs have 0-2.
  function updateSophisticatedNote(results) {
    const details = document.getElementById("sophisticated-details");
    const sophisticated = results.filter((r) => r.archetype === "sophisticated");
    if (sophisticated.length === 0) {
      details.hidden = true;
      return;
    }
    const passed = sophisticated.filter((r) => r.score >= 0.5);
    const scores = sophisticated.map((r) => `${(r.score * 100).toFixed(1)}%`).join(", ");
    const summary =
      passed.length > 0
        ? `Note: ${passed.length}/${sophisticated.length} sophisticated bot(s) passed (${scores}) -- the documented ceiling, not a bug.`
        : `Note: ${sophisticated.length} sophisticated bot(s) this run, all caught (${scores}).`;
    document.getElementById("sophisticated-summary").textContent = summary;
    details.hidden = false;
  }

  function updateEnvironmentSummary(environmentResults) {
    const flagged = environmentResults.filter((r) => r.is_automated).length;
    const stealthRow = environmentResults.find((r) => /stealth/i.test(r.scenario));
    const stealthNote =
      stealthRow && !stealthRow.is_automated ? " (1 stealth-patched passes -- expected)" : "";
    document.getElementById("environment-summary").textContent =
      `Environment checks: ${flagged}/${environmentResults.length} scenarios flagged as automated${stealthNote}.`;
  }

  function updateRequestSummary(requestResults) {
    const flagged = requestResults.filter((r) => r.is_suspicious).length;
    const yours = requestResults.find((r) => r.scenario === "Your actual request");
    const yoursNote = yours ? (yours.is_suspicious ? "your UA flagged" : "your UA clean") : "";
    document.getElementById("request-summary").textContent =
      `Request fingerprint: ${yoursNote}; ${flagged}/${requestResults.length} scenarios flagged overall.`;
  }

  function updateBrowserSummary(results) {
    const parts = results.map((r) => {
      const short = /stealth/i.test(r.scenario) ? "stealth-patched" : "unpatched";
      if (r.error) return `${short} failed`;
      return `${short} ${r.is_automated ? "caught" : "passed"}`;
    });
    document.getElementById("browser-summary").textContent =
      `Live Selenium check: ${parts.join(", ")}.`;
  }

  async function runBehavioralBatch() {
    const progressEl = document.getElementById("autopilot-progress");
    document.querySelector("#results-table tbody").innerHTML = "";
    document.querySelector("#environment-table tbody").innerHTML = "";
    document.querySelector("#your-request-table tbody").innerHTML = "";
    document.querySelector("#history-table-runs tbody").innerHTML = "";
    progressEl.textContent = "Running 20 tests and retraining... this can take a little while.";

    let data;
    try {
      data = await postJSON("/api/run_tests", {});
    } catch (e) {
      progressEl.textContent = `Failed: ${e}`;
      return;
    }
    if (data.error) {
      progressEl.textContent = data.error;
      return;
    }

    data.results.forEach((r) => addResultRow(r.archetype, r.score));
    updateSophisticatedNote(data.results);

    (data.environment_results || []).forEach((r) =>
      addEnvironmentRow(r.scenario, r.is_automated, r.reasons)
    );
    if (data.environment_results) updateEnvironmentSummary(data.environment_results);

    (data.request_fingerprint_results || []).forEach((r) =>
      addRequestFingerprintRow(r.scenario, r.user_agent, r.is_suspicious, r.reasons)
    );
    if (data.request_fingerprint_results) updateRequestSummary(data.request_fingerprint_results);

    (data.recent_history || []).forEach((r) => addHistoryRow(r));

    const retrainData = data.retrain;
    if (retrainData) {
      document.getElementById("m-sessions").textContent = retrainData.n_sessions;
      document.getElementById("m-human").textContent = pct(retrainData.human_pass_rate_range);
      document.getElementById("m-bot").textContent = pct(retrainData.bot_catch_rate_range);
      progressEl.textContent =
        `Done. Retrained on ${retrainData.n_sessions} sessions (${retrainData.n_new_sessions} new).`;
    } else {
      progressEl.textContent = "Done scoring the batch. Not enough new sessions to retrain yet.";
    }

    // Cumulative range for the verdict block: this run's own retrain if
    // it happened, else the most recent past one (still real, just not
    // from this click) -- never the ambiguous "no number at all".
    const cumulative = retrainData || (data.recent_history && data.recent_history[0]) || null;
    updateVerdict(data.results, cumulative);

    document.getElementById("m-composition").textContent =
      `Composition: ${formatComposition(data.label_composition, data.group_composition)} ` +
      "(design: 50/50 human/bot, bots 45/35/10/10 naive/evasive/headless/sophisticated)";
  }

  async function runRealBrowserChecks() {
    const statusEl = document.getElementById("real-browser-status");
    document.querySelector("#real-browser-table tbody").innerHTML = "";
    statusEl.textContent = "Launching two real headless Chromium instances...";

    let data;
    try {
      data = await postJSON("/api/run_real_browser_checks", {});
    } catch (e) {
      statusEl.textContent = `Failed: ${e}`;
      return;
    }
    if (data.error) {
      statusEl.textContent = data.error;
      return;
    }

    data.results.forEach((r) =>
      addRealBrowserRow(r.scenario, r.is_automated, r.reasons, r.signals, r.error)
    );
    updateBrowserSummary(data.results);
    statusEl.textContent = "Done.";
  }

  document.getElementById("btn-run-tests").addEventListener("click", async () => {
    const btn = document.getElementById("btn-run-tests");
    const resultsEl = document.getElementById("results");
    btn.disabled = true;
    resultsEl.hidden = false;

    // Two independent checks, fired concurrently (not one awaited after
    // the other): the behavioral batch and the real-browser check don't
    // depend on each other, and the server (app.run(threaded=True))
    // actually processes them in parallel, not just dispatches them
    // that way from the browser.
    await Promise.allSettled([runBehavioralBatch(), runRealBrowserChecks()]);

    btn.disabled = false;
  });
})();
