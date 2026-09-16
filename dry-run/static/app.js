(function () {
  "use strict";

  const pageLoadT = performance.now();
  const now = () => performance.now() - pageLoadT;

  let mouseEvents = [];
  let keyEvents = [];
  let scrollEvents = [];
  let clickEvents = [];
  let focusEvents = [];
  let pasteEvents = [];
  let lastMouseT = -1000;
  const keyDownTimes = {};

  function clearBuffers() {
    mouseEvents = [];
    keyEvents = [];
    scrollEvents = [];
    clickEvents = [];
    focusEvents = [];
    pasteEvents = [];
    Object.keys(keyDownTimes).forEach((k) => delete keyDownTimes[k]);
  }

  document.addEventListener("mousemove", (e) => {
    const t = now();
    if (t - lastMouseT < 15) return;
    lastMouseT = t;
    mouseEvents.push([e.clientX, e.clientY, t]);
  });

  document.addEventListener("click", (e) => {
    clickEvents.push([e.clientX, e.clientY, now()]);
  });

  window.addEventListener("blur", () => focusEvents.push([now(), false]));
  window.addEventListener("focus", () => focusEvents.push([now(), true]));

  const textField = document.getElementById("text-field");
  textField.addEventListener("keydown", (e) => {
    if (keyDownTimes[e.code] === undefined) {
      keyDownTimes[e.code] = now();
    }
  });
  textField.addEventListener("keyup", (e) => {
    const down = keyDownTimes[e.code];
    if (down !== undefined) {
      keyEvents.push([down, now()]);
      delete keyDownTimes[e.code];
    }
  });
  textField.addEventListener("paste", (e) => {
    const clip = e.clipboardData || window.clipboardData;
    const text = clip ? clip.getData("text") : "";
    pasteEvents.push([now(), text.length]);
  });

  const scrollZone = document.getElementById("scroll-zone");
  scrollZone.addEventListener(
    "wheel",
    (e) => scrollEvents.push([now(), e.deltaY]),
    { passive: true }
  );

  function buildSessionPayload(label) {
    return {
      mouse_events: mouseEvents,
      key_events: keyEvents,
      scroll_events: scrollEvents,
      click_events: clickEvents,
      focus_events: focusEvents,
      paste_events: pasteEvents,
      page_load_t: 0,
      submit_t: now(),
      label: label,
    };
  }

  function setScore(score, label) {
    const fill = document.getElementById("score-fill");
    const text = document.getElementById("score-label");
    const display = document.getElementById("score-display");
    if (score === null || score === undefined) {
      fill.style.width = "0%";
      text.textContent = label || "No test run yet";
      display.dataset.state = "empty";
      return;
    }
    fill.style.width = Math.round(score * 100) + "%";
    text.textContent = `${label || "Score"}: ${(score * 100).toFixed(1)}% human-like`;
    display.dataset.state = score >= 0.5 ? "human" : "bot";
  }

  function addLogRow(source, label, score) {
    const tbody = document.querySelector("#log-table tbody");
    const row = document.createElement("tr");
    const time = new Date().toLocaleTimeString();
    row.innerHTML =
      `<td>${time}</td><td>${source}</td><td>${label}</td>` +
      `<td>${(score * 100).toFixed(1)}%</td>`;
    tbody.prepend(row);
  }

  function pct(range) {
    return `${(range[0] * 100).toFixed(0)}-${(range[1] * 100).toFixed(0)}%`;
  }

  async function postJSON(url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    return res.json();
  }

  async function refreshStatus() {
    const res = await fetch("/api/status");
    const data = await res.json();

    document.getElementById("status-summary").textContent =
      `Trained: ${data.trained ? "yes" : "no"} | ` +
      `Total sessions: ${data.total_sessions} | ` +
      `Pending until next auto-retrain: ${data.pending_sessions} / ${data.min_new_sessions}`;

    const tbody = document.querySelector("#history-table tbody");
    tbody.innerHTML = "";
    data.history
      .slice()
      .reverse()
      .forEach((h) => {
        const row = document.createElement("tr");
        const time = new Date(h.timestamp).toLocaleTimeString();
        row.innerHTML =
          `<td>${time}</td><td>${h.n_sessions}</td><td>${h.n_new_sessions}</td>` +
          `<td>${pct(h.accuracy_range)}</td><td>${pct(h.human_pass_rate_range)}</td>` +
          `<td>${pct(h.bot_catch_rate_range)}</td>`;
        tbody.appendChild(row);
      });
  }

  document.getElementById("btn-run-test").addEventListener("click", async () => {
    const payload = buildSessionPayload(null);
    clearBuffers();
    const data = await postJSON("/api/score", payload);
    if (data.error) {
      setScore(null, data.error);
      return;
    }
    setScore(data.score, "Your session");
    addLogRow("you (live)", "-", data.score);
  });

  document.getElementById("btn-reset").addEventListener("click", () => {
    clearBuffers();
    setScore(null, "Session reset");
  });

  document.getElementById("btn-record").addEventListener("click", async () => {
    const label = document.querySelector('input[name="label"]:checked').value === "true";
    const payload = buildSessionPayload(label);
    clearBuffers();
    const retrainResEl = document.getElementById("retrain-result");

    const rec = await postJSON("/api/record", payload);
    if (rec.error) {
      retrainResEl.textContent = rec.error;
      return;
    }
    retrainResEl.textContent = `Recorded. Pending until next auto-retrain: ${rec.pending}`;

    const retrainData = await postJSON("/api/retrain", { force: false });
    if (retrainData.retrained) {
      retrainResEl.textContent =
        `Retrained on ${retrainData.n_sessions} sessions (${retrainData.n_new_sessions} new)! ` +
        `Accuracy ${pct(retrainData.accuracy_range)}, ` +
        `human pass ${pct(retrainData.human_pass_rate_range)}, ` +
        `bot catch ${pct(retrainData.bot_catch_rate_range)}.`;
    }
    refreshStatus();
  });

  document.getElementById("btn-force-retrain").addEventListener("click", async () => {
    const retrainResEl = document.getElementById("retrain-result");
    retrainResEl.textContent = "Retraining... this can take a little while.";
    const data = await postJSON("/api/retrain", { force: true });
    if (!data.retrained) {
      retrainResEl.textContent = "Not enough sessions recorded yet (need at least 4).";
      return;
    }
    retrainResEl.textContent =
      `Retrained on ${data.n_sessions} sessions (${data.n_new_sessions} new)! ` +
      `Accuracy ${pct(data.accuracy_range)}, ` +
      `human pass ${pct(data.human_pass_rate_range)}, ` +
      `bot catch ${pct(data.bot_catch_rate_range)}.`;
    refreshStatus();
  });

  document.querySelectorAll("#simulate-buttons button").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const archetype = btn.dataset.archetype;
      const res = await fetch(`/api/simulate/${archetype}`);
      const data = await res.json();
      if (data.error) return;
      setScore(data.score, `Simulated: ${archetype}`);
      addLogRow(`simulated: ${archetype}`, archetype === "human" ? "human" : "bot", data.score);
    });
  });

  refreshStatus();
  setInterval(refreshStatus, 5000);
})();
