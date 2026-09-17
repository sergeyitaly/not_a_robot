/*
 * not-a-robot.js -- reference client-side telemetry collector.
 *
 * Not part of the not-a-robot PyPI package (which only defines the
 * InteractionSession schema and feature math -- see the main README's
 * "Scope"). This is a plain, dependency-free reference implementation of
 * the capture side, matching that schema exactly. Copy it into your own
 * static assets and serve it yourself; it's small enough (~150 lines) to
 * read end to end and adapt rather than treat as a black box.
 *
 * Usage:
 *
 *   <script src="/not-a-robot.js"></script>
 *   <script>
 *     var collector = new NotARobot.Collector({ endpoint: "/telemetry" });
 *     collector.attachToForm("#signup-form");
 *   </script>
 *
 * Or, without a form (score on your own trigger):
 *
 *   var collector = new NotARobot.Collector();
 *   // ... later ...
 *   var session = collector.finish();  // { mouse_events: [...], ... }
 *   fetch("/telemetry", { method: "POST", body: JSON.stringify(session) });
 *
 * Field shapes match not_a_robot.schema exactly, as plain arrays (not
 * objects) to keep the wire payload small:
 *   mouse_events:  [[x, y, t], ...]
 *   key_events:    [[t_down, t_up], ...]
 *   scroll_events: [[t, delta_y], ...]
 *   click_events:  [[x, y, t], ...]
 *   focus_events:  [[t, focused], ...]
 *   paste_events:  [[t, length], ...]
 * A server-side InteractionSession constructor should unpack these
 * positionally -- see examples/integrations/ for both Flask and FastAPI.
 */
(function (global) {
  "use strict";

  function Collector(options) {
    options = options || {};
    this.endpoint = options.endpoint || null;
    // Mousemove fires far more often than any other event; cap it so a
    // long-lived page (or a bot deliberately flooding events) can't grow
    // this buffer without bound. This changes the feature distribution
    // for sessions long/fast enough to hit the cap -- know that if you
    // raise or lower it.
    this.maxMouseEvents = options.maxMouseEvents || 500;
    this._pageLoadPerf = performance.now();
    this._keyDownAt = {};
    this._windowScrollTop = window.scrollY || 0;
    this.session = {
      mouse_events: [],
      key_events: [],
      scroll_events: [],
      click_events: [],
      focus_events: [],
      paste_events: [],
      page_load_t: 0,
      submit_t: null,
    };
    this._bind();
  }

  Collector.prototype._now = function () {
    return performance.now() - this._pageLoadPerf;
  };

  Collector.prototype._bind = function () {
    var self = this;

    document.addEventListener("mousemove", function (e) {
      if (self.session.mouse_events.length >= self.maxMouseEvents) return;
      self.session.mouse_events.push([e.clientX, e.clientY, self._now()]);
    });

    document.addEventListener("click", function (e) {
      self.session.click_events.push([e.clientX, e.clientY, self._now()]);
    });

    window.addEventListener("blur", function () {
      self.session.focus_events.push([self._now(), false]);
    });
    window.addEventListener("focus", function () {
      self.session.focus_events.push([self._now(), true]);
    });

    document.addEventListener("keydown", function (e) {
      var key = e.code || e.key;
      if (self._keyDownAt[key] === undefined) {
        self._keyDownAt[key] = self._now();
      }
    });
    document.addEventListener("keyup", function (e) {
      var key = e.code || e.key;
      var down = self._keyDownAt[key];
      if (down !== undefined) {
        self.session.key_events.push([down, self._now()]);
        delete self._keyDownAt[key];
      }
    });

    document.addEventListener("paste", function (e) {
      var clip = e.clipboardData || global.clipboardData;
      var text = clip ? clip.getData("text") : "";
      self.session.paste_events.push([self._now(), text.length]);
    });

    window.addEventListener("scroll", function () {
      var top = window.scrollY;
      var delta = top - self._windowScrollTop;
      self._windowScrollTop = top;
      self.session.scroll_events.push([self._now(), delta]);
    });
  };

  // Marks submit_t and returns the captured session as a plain object.
  // Safe to call more than once (e.g. a failed submit retried) --
  // subsequent events keep accumulating and submit_t updates each time.
  Collector.prototype.finish = function () {
    this.session.submit_t = this._now();
    return this.session;
  };

  // POSTs the finished session as JSON to `endpoint`. `keepalive: true`
  // so the request has a chance to complete even if it's fired from a
  // submit/unload handler that navigates away immediately after.
  Collector.prototype.send = function () {
    if (!this.endpoint) {
      throw new Error(
        "NotARobot.Collector: no endpoint configured -- pass " +
          "{ endpoint: '/telemetry' }, or call finish() and send the " +
          "payload yourself"
      );
    }
    var payload = this.finish();
    return fetch(this.endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      keepalive: true,
    });
  };

  // Wires this collector to a form's submit event. Fire-and-forget by
  // default: the telemetry POST goes out alongside the real submit,
  // without blocking or delaying it -- a telemetry failure should never
  // stop a real user from submitting the form. Pass `blocking: true` to
  // wait for the POST before letting the form submit (with `onSent`/
  // `onError` callbacks); even then, a failed POST calls `form.submit()`
  // by default rather than trapping the user, unless you pass `onError`
  // yourself and take responsibility for that decision.
  Collector.prototype.attachToForm = function (selector, options) {
    options = options || {};
    var self = this;
    var form =
      typeof selector === "string" ? document.querySelector(selector) : selector;
    if (!form) {
      throw new Error(
        "NotARobot.Collector.attachToForm: no element matches " + selector
      );
    }

    form.addEventListener("submit", function (e) {
      if (!options.blocking) {
        self.send().catch(function () {
          /* best-effort; never block the real submit on this */
        });
        return;
      }

      e.preventDefault();
      self
        .send()
        .then(function () {
          if (options.onSent) options.onSent();
          else form.submit();
        })
        .catch(function (err) {
          if (options.onError) options.onError(err);
          else form.submit();
        });
    });

    return this;
  };

  global.NotARobot = { Collector: Collector };
})(typeof window !== "undefined" ? window : this);
