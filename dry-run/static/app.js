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

  function addResultRow(archetype, label, score) {
    const tbody = document.querySelector("#results-table tbody");
    const row = document.createElement("tr");
    row.innerHTML =
      `<td>${archetype}</td><td>${label}</td><td>${(score * 100).toFixed(1)}%</td>`;
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

  function formatComposition(labelComp, groupComp) {
    const groupPart = groupComp
      ? ` (naive ${groupComp.naive || 0}, evasive ${groupComp.evasive || 0}, ` +
        `headless ${groupComp.headless || 0}, sophisticated ${groupComp.sophisticated || 0})`
      : "";
    return `${labelComp.human} human / ${labelComp.bot} bot${groupPart}`;
  }

  document.getElementById("btn-run-tests").addEventListener("click", async () => {
    const btn = document.getElementById("btn-run-tests");
    const progressEl = document.getElementById("autopilot-progress");
    const resultsEl = document.getElementById("results");
    btn.disabled = true;
    resultsEl.hidden = true;
    document.querySelector("#results-table tbody").innerHTML = "";
    document.querySelector("#environment-table tbody").innerHTML = "";
    progressEl.textContent = "Running 20 tests and retraining... this can take a little while.";

    let data;
    try {
      data = await postJSON("/api/run_tests", {});
    } catch (e) {
      progressEl.textContent = `Failed: ${e}`;
      btn.disabled = false;
      return;
    }
    if (data.error) {
      progressEl.textContent = data.error;
      btn.disabled = false;
      return;
    }

    data.results.forEach((r) => addResultRow(r.archetype, r.label, r.score));
    (data.environment_results || []).forEach((r) =>
      addEnvironmentRow(r.scenario, r.is_automated, r.reasons)
    );

    const retrainData = data.retrain;
    if (retrainData) {
      document.getElementById("m-sessions").textContent = retrainData.n_sessions;
      document.getElementById("m-accuracy").textContent = pct(retrainData.accuracy_range);
      document.getElementById("m-human").textContent = pct(retrainData.human_pass_rate_range);
      document.getElementById("m-bot").textContent = pct(retrainData.bot_catch_rate_range);
      progressEl.textContent =
        `Done. Retrained on ${retrainData.n_sessions} sessions (${retrainData.n_new_sessions} new).`;
    } else {
      progressEl.textContent = "Done scoring the batch. Not enough new sessions to retrain yet.";
    }

    document.getElementById("m-composition").textContent =
      `Composition: ${formatComposition(data.label_composition, data.group_composition)} ` +
      "(design: 50/50 human/bot, bots 45/35/10/10 naive/evasive/headless/sophisticated)";

    resultsEl.hidden = false;
    btn.disabled = false;
  });
})();
