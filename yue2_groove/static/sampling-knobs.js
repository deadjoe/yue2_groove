/* Bearbone ADVANCED // SAMPLING dual view: SVG knobs over the Gradio sliders.
 *
 * The 14 gr.Slider inputs stay the single source of truth — component state,
 * the BUDGET PRESET dropdown, RESET DEFAULTS and generation all keep reading
 * them. A knob paints the current value (head dot + remaining arc, the
 * travelled arc hidden) and writes back through the native number + range
 * inputs, so both views always show the same numbers.
 *
 * Gradio 6 slider DOM: root `#bb-abc-temp` contains
 *   input[data-testid="number-input"] and input[data-testid="range-input"].
 * Knobs are the default view; the toggle button and localStorage key
 * `bb-sampling-view` remember the user's choice per browser.
 *
 * Drag vertically: ~120 px = full range, Shift = 6x finer (same step grid).
 * Double-click restores that phase's default; arrows / Home / End also work.
 */
(function () {
  'use strict';

  var KEY = 'bb-sampling-view';
  var START_DEG = 135;   /* bottom-left gap edge, same as the mockup        */
  var SWEEP_DEG = 270;   /* to 45° past east; the bottom gap stays hidden   */
  var SIZE = 88;
  var CX = SIZE / 2;
  var CY = SIZE / 2;
  var R = 32;
  var DOT_R = 5;
  var PX_FULL = 120;     /* vertical px for a full min→max sweep            */
  var PX_FINE = 720;     /* Shift: same step grid, 6x more px per step      */
  var NS = 'http://www.w3.org/2000/svg';

  var PARAMS = [
    { id: 'bb-abc-temp', key: 'temperature', label: 'TEMPERATURE', phase: 'abc' },
    { id: 'bb-abc-p', key: 'top_p', label: 'TOP_P', phase: 'abc' },
    { id: 'bb-abc-k', key: 'top_k', label: 'TOP_K', phase: 'abc' },
    { id: 'bb-abc-rep', key: 'repetition_penalty', label: 'REPETITION_PENALTY', phase: 'abc' },
    { id: 'bb-abc-win', key: 'penalty_window', label: 'PENALTY_WINDOW', phase: 'abc' },
    { id: 'bb-abc-min', key: 'min_tokens', label: 'MIN_TOKENS', phase: 'abc' },
    { id: 'bb-abc-max', key: 'max_tokens', label: 'MAX_TOKENS', phase: 'abc' },
    { id: 'bb-sem-temp', key: 'temperature', label: 'TEMPERATURE', phase: 'sem' },
    { id: 'bb-sem-p', key: 'top_p', label: 'TOP_P', phase: 'sem' },
    { id: 'bb-sem-k', key: 'top_k', label: 'TOP_K', phase: 'sem' },
    { id: 'bb-sem-rep', key: 'repetition_penalty', label: 'REPETITION_PENALTY', phase: 'sem' },
    { id: 'bb-sem-win', key: 'penalty_window', label: 'PENALTY_WINDOW', phase: 'sem' },
    { id: 'bb-sem-min', key: 'min_tokens', label: 'MIN_TOKENS', phase: 'sem' },
    { id: 'bb-sem-max', key: 'max_tokens', label: 'MAX_TOKENS', phase: 'sem' }
  ];

  function panel() { return document.getElementById('bb-sampling-panel'); }

  function isSlidersView() {
    var p = panel();
    return !!(p && p.classList.contains('bb-view-sliders'));
  }

  function viewBtn() {
    var wrap = document.getElementById('bb-sampling-view-btn');
    if (!wrap) return null;
    return wrap.tagName === 'BUTTON' ? wrap : wrap.querySelector('button');
  }

  function savedView() {
    try {
      var v = localStorage.getItem(KEY);
      if (v === 'sliders' || v === 'knobs') return v;
    } catch (e) {}
    return 'knobs';
  }

  function applyView(view) {
    var p = panel();
    if (!p) return;
    var knobs = view !== 'sliders';
    p.classList.toggle('bb-view-knobs', knobs);
    p.classList.toggle('bb-view-sliders', !knobs);
    try { localStorage.setItem(KEY, knobs ? 'knobs' : 'sliders'); } catch (e) {}
    var btn = viewBtn();
    if (btn) {
      var label = knobs ? 'Knobs view — switch to sliders'
                        : 'Sliders view — switch to knobs';
      btn.title = label;
      btn.setAttribute('aria-label', label);
      btn.classList.toggle('bb-showing-knobs', knobs);
      btn.classList.toggle('bb-showing-sliders', !knobs);
    }
  }

  function defaultsFor(phase) {
    var pack = window.__BB_SAMPLING_DEFAULTS__ || {};
    return (phase === 'abc' ? pack.abc : pack.sem) || {};
  }

  /* ── arc geometry ─────────────────────────────────────────────────────── */

  function polar(deg) {
    var rad = deg * Math.PI / 180;
    return [CX + R * Math.cos(rad), CY + R * Math.sin(rad)];
  }

  function headDeg(t) { return START_DEG + Math.max(0, Math.min(1, t)) * SWEEP_DEG; }

  /* remaining arc: from the head (dot) clockwise to the end of the sweep */
  function arcD(t0, t1) {
    if (t1 - t0 < 0.0005) return '';
    var a0 = headDeg(t0), a1 = headDeg(t1);
    var p0 = polar(a0), p1 = polar(a1);
    var large = ((a1 - a0) % 360) > 180 ? 1 : 0;
    return 'M ' + p0[0].toFixed(2) + ' ' + p0[1].toFixed(2) +
      ' A ' + R + ' ' + R + ' 0 ' + large + ' 1 ' +
      p1[0].toFixed(2) + ' ' + p1[1].toFixed(2);
  }

  /* ── native slider bridge ─────────────────────────────────────────────── */

  function findInputs(root) {
    if (!root) return { number: null, range: null };
    return {
      number: root.querySelector('input[data-testid="number-input"]') ||
              root.querySelector('input[type="number"]'),
      range: root.querySelector('input[data-testid="range-input"]') ||
             root.querySelector('input[type="range"]')
    };
  }

  /* set .value through the native setter so Svelte/Gradio see the change */
  function setNative(el, value) {
    if (!el) return;
    var setter = Object.getOwnPropertyDescriptor(
      window.HTMLInputElement.prototype, 'value').set;
    setter.call(el, value);
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }

  function readMeta(inputs) {
    var el = inputs.range || inputs.number;
    var min = el ? parseFloat(el.min) : 0;
    var max = el ? parseFloat(el.max) : 1;
    var step = el && el.step && el.step !== 'any' ? parseFloat(el.step) : 1;
    if (!isFinite(min)) min = 0;
    if (!isFinite(max)) max = 1;
    if (!isFinite(step) || step <= 0) step = 1;
    return { min: min, max: max, step: step };
  }

  function readValue(inputs, meta) {
    var raw = inputs.number ? inputs.number.value
      : (inputs.range ? inputs.range.value : meta.min);
    var v = parseFloat(raw);
    return isFinite(v) ? v : meta.min;
  }

  function decimalsFor(step) {
    var s = String(step);
    var i = s.indexOf('.');
    return i < 0 ? 0 : s.length - i - 1;
  }

  function formatValue(v, step) {
    var d = decimalsFor(step);
    if (d === 0) return String(Math.round(v));
    return Number(v).toFixed(d);
  }

  function snap(v, meta) {
    var n = Math.round((v - meta.min) / meta.step) * meta.step + meta.min;
    n = Math.max(meta.min, Math.min(meta.max, n));
    var d = decimalsFor(meta.step);
    if (d === 0) return Math.round(n);
    return Number(n.toFixed(d));
  }

  function writeValue(inputs, v) {
    var s = String(v);
    if (inputs.number && inputs.number.value !== s) setNative(inputs.number, s);
    if (inputs.range && inputs.range.value !== s) setNative(inputs.range, s);
  }

  /* ── knob construction / painting ─────────────────────────────────────── */

  function svgEl(name, attrs) {
    var node = document.createElementNS(NS, name);
    if (attrs) {
      Object.keys(attrs).forEach(function (k) { node.setAttribute(k, attrs[k]); });
    }
    return node;
  }

  function buildKnob(param) {
    var host = document.createElement('div');
    host.className = 'bb-knob';
    host.setAttribute('data-bb-slider', param.id);
    host.setAttribute('role', 'slider');
    host.setAttribute('aria-label', param.label);
    host.setAttribute('title', param.label + ' — drag to change, double-click to reset');
    host.tabIndex = 0;

    var label = document.createElement('div');
    label.className = 'bb-knob-label';
    label.textContent = param.label;

    var body = document.createElement('div');
    body.className = 'bb-knob-body';

    var svg = svgEl('svg', {
      class: 'bb-knob-svg',
      viewBox: '0 0 ' + SIZE + ' ' + SIZE,
      width: String(SIZE),
      height: String(SIZE),
      'aria-hidden': 'true'
    });
    var track = svgEl('path', { class: 'bb-knob-arc', d: '', fill: 'none' });
    var dot = svgEl('circle', {
      class: 'bb-knob-dot',
      r: String(DOT_R),
      cx: String(CX),
      cy: String(CY)
    });
    svg.appendChild(track);
    svg.appendChild(dot);

    var valueEl = document.createElement('div');
    valueEl.className = 'bb-knob-value';
    valueEl.textContent = '—';

    body.appendChild(svg);
    body.appendChild(valueEl);
    host.appendChild(label);
    host.appendChild(body);

    return { host: host, track: track, dot: dot, valueEl: valueEl, param: param };
  }

  function paint(knob, t, display) {
    var d = arcD(t, 1);
    if (knob.track.getAttribute('d') !== d) knob.track.setAttribute('d', d);
    var p = polar(headDeg(t));
    var cx = p[0].toFixed(2), cy = p[1].toFixed(2);
    if (knob.dot.getAttribute('cx') !== cx) knob.dot.setAttribute('cx', cx);
    if (knob.dot.getAttribute('cy') !== cy) knob.dot.setAttribute('cy', cy);
    if (knob.valueEl.textContent !== display) knob.valueEl.textContent = display;
    if (knob.host.getAttribute('aria-valuenow') !== display) {
      knob.host.setAttribute('aria-valuenow', display);
      knob.host.setAttribute('aria-valuetext', display);
    }
  }

  /* ── knob ↔ slider binding ────────────────────────────────────────────── */

  function bindKnob(knob) {
    var root = document.getElementById(knob.param.id);
    if (!root) return false;

    var stale = root.querySelector('.bb-knob');
    if (stale && stale !== knob.host) stale.remove();
    if (!knob.host.isConnected) root.insertBefore(knob.host, root.firstChild);
    root.classList.add('bb-sampling-slider');

    var drag = null;

    function current() {
      var inputs = findInputs(root);
      var meta = readMeta(inputs);
      return { inputs: inputs, meta: meta, value: readValue(inputs, meta) };
    }

    function show(c, value) {
      var span = c.meta.max - c.meta.min;
      var t = span === 0 ? 0 : (value - c.meta.min) / span;
      paint(knob, t, formatValue(value, c.meta.step));
    }

    function syncFromSlider() {
      var c = current();
      if (!c.inputs.number && !c.inputs.range) return;
      show(c, c.value);
      knob.host.setAttribute('aria-valuemin', String(c.meta.min));
      knob.host.setAttribute('aria-valuemax', String(c.meta.max));
    }

    function onPointerDown(ev) {
      if (isSlidersView()) return;                   /* knobs-only gesture   */
      if (ev.button != null && ev.button !== 0) return;
      ev.preventDefault();                           /* no text select / scroll */
      var c = current();
      drag = { y: ev.clientY, value: c.value, meta: c.meta, id: ev.pointerId };
      try { knob.host.setPointerCapture(ev.pointerId); } catch (e) {}
      knob.host.classList.add('bb-knob-active');
      document.documentElement.classList.add('bb-knob-dragging');
    }

    function onPointerMove(ev) {
      if (!drag) return;
      ev.preventDefault();
      var meta = drag.meta;
      var span = meta.max - meta.min;
      var px = ev.shiftKey ? PX_FINE : PX_FULL;
      var raw = drag.value + (drag.y - ev.clientY) / px * span;
      var v = snap(raw, meta);
      writeValue(findInputs(root), v);
      show({ meta: meta }, v);
    }

    function onPointerUp(ev) {
      if (!drag) return;
      if (ev) ev.preventDefault();
      drag = null;
      knob.host.classList.remove('bb-knob-active');
      document.documentElement.classList.remove('bb-knob-dragging');
      try {
        if (ev && ev.pointerId != null) knob.host.releasePointerCapture(ev.pointerId);
      } catch (e) {}
      syncFromSlider();
    }

    function onDoubleClick(ev) {
      if (isSlidersView()) return;
      var def = defaultsFor(knob.param.phase)[knob.param.key];
      if (def == null) return;
      ev.preventDefault();
      var c = current();
      writeValue(c.inputs, snap(def, c.meta));
      syncFromSlider();
    }

    function onKeyDown(ev) {
      if (isSlidersView()) return;
      var c = current();
      var v;
      if (ev.key === 'ArrowUp' || ev.key === 'ArrowRight') v = c.value + c.meta.step;
      else if (ev.key === 'ArrowDown' || ev.key === 'ArrowLeft') v = c.value - c.meta.step;
      else if (ev.key === 'Home') v = c.meta.min;
      else if (ev.key === 'End') v = c.meta.max;
      else return;
      ev.preventDefault();
      v = snap(v, c.meta);
      writeValue(c.inputs, v);
      show(c, v);
    }

    knob.host.addEventListener('pointerdown', onPointerDown);
    knob.host.addEventListener('pointermove', onPointerMove);
    knob.host.addEventListener('pointerup', onPointerUp);
    knob.host.addEventListener('pointercancel', onPointerUp);
    knob.host.addEventListener('lostpointercapture', onPointerUp);
    knob.host.addEventListener('dblclick', onDoubleClick);
    knob.host.addEventListener('keydown', onKeyDown);

    /* slider → knob (BUDGET PRESET, RESET DEFAULTS, typing in sliders view) */
    if (knob._wiredRoot !== root) {
      root.addEventListener('input', syncFromSlider, true);
      root.addEventListener('change', syncFromSlider, true);
      knob._wiredRoot = root;
    }

    knob.sync = syncFromSlider;
    syncFromSlider();
    return true;
  }

  /* ── boot: the panel is created lazily when the accordion opens ───────── */

  var knobs = [];
  var booted = false;

  function boot() {
    var p = panel();
    if (!p) return;
    if (!booted) {
      booted = true;
      PARAMS.forEach(function (param) { knobs.push(buildKnob(param)); });
      applyView(savedView());
    }
    knobs.forEach(function (knob) {
      if (knob.host && !knob.host.isConnected) knob._bound = false;
      if (!knob._bound) knob._bound = bindKnob(knob);
      else if (knob.sync) knob.sync();
    });
  }

  window.__bbToggleSamplingView = function () {
    var p = panel();
    if (!p) return;
    applyView(p.classList.contains('bb-view-knobs') ? 'sliders' : 'knobs');
    knobs.forEach(function (knob) { if (knob.sync) knob.sync(); });
  };

  var scheduled = null;
  function scheduleBoot() {
    if (scheduled) return;
    scheduled = setTimeout(function () { scheduled = null; boot(); }, 80);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', scheduleBoot);
  } else {
    scheduleBoot();
  }
  window.addEventListener('load', scheduleBoot);
  if (window.MutationObserver) {
    new MutationObserver(scheduleBoot).observe(document.documentElement, {
      childList: true,
      subtree: true
    });
  }
})();
