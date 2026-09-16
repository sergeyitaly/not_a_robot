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

  function shuffled(arr) {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }

  const BATCH = [
    ...Array(4).fill("human"),
    ...Array(4).fill("naive"),
    ...Array(4).fill("evasive"),
    ...Array(4).fill("headless"),
    ...Array(4).fill("sophisticated"),
  ];

  function addResultRow(archetype, score) {
    const tbody = document.querySelector("#results-table tbody");
    const row = document.createElement("tr");
    const trueLabel = archetype === "human" ? "human" : "bot";
    row.innerHTML =
      `<td>${archetype}</td><td>${trueLabel}</td><td>${(score * 100).toFixed(1)}%</td>`;
    tbody.appendChild(row);
  }

  document.getElementById("btn-run-tests").addEventListener("click", async () => {
    const btn = document.getElementById("btn-run-tests");
    const progressEl = document.getElementById("autopilot-progress");
    const resultsEl = document.getElementById("results");
    btn.disabled = true;
    resultsEl.hidden = true;
    document.querySelector("#results-table tbody").innerHTML = "";

    const batch = shuffled(BATCH);
    for (let i = 0; i < batch.length; i++) {
      const archetype = batch[i];
      progressEl.textContent = `Running test ${i + 1} / ${batch.length}: ${archetype}...`;
      try {
        const res = await fetch(`/api/simulate/${archetype}?record=true`);
        const data = await res.json();
        if (!data.error) {
          addResultRow(archetype, data.score);
        }
      } catch (e) {
        progressEl.textContent = `Test ${i + 1} failed: ${e}`;
      }
      await new Promise((resolve) => setTimeout(resolve, 100));
    }

    progressEl.textContent = "Batch done. Retraining...";
    const retrainData = await postJSON("/api/retrain", { force: true });

    if (retrainData.retrained) {
      document.getElementById("m-sessions").textContent = retrainData.n_sessions;
      document.getElementById("m-accuracy").textContent = pct(retrainData.accuracy_range);
      document.getElementById("m-human").textContent = pct(retrainData.human_pass_rate_range);
      document.getElementById("m-bot").textContent = pct(retrainData.bot_catch_rate_range);
      progressEl.textContent =
        `Done. Retrained on ${retrainData.n_sessions} sessions (${retrainData.n_new_sessions} new).`;
    } else {
      progressEl.textContent = "Done scoring the batch. Not enough new sessions to retrain yet.";
    }

    resultsEl.hidden = false;
    btn.disabled = false;
  });
})();
