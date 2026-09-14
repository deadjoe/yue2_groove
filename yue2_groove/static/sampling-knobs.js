/* Bearbone ADVANCED // SAMPLING dual view: SVG knobs bound to Gradio 6 sliders.
 * Gradio 6 DOM: #elem_id contains input[data-testid=number-input] and
 * input[data-testid=range-input]. Knobs are the default view; toggle + localStorage
 * `bb-sampling-view` switch to native sliders. Vertical drag; Shift = fine;
 * double-click resets to ABC_DEFAULTS / SEM_DEFAULTS for that param.
 */
(function () {
  var KEY = 'bb-sampling-view';
  var START_DEG = 135;
  var SWEEP_DEG = 270;
  var SIZE = 88;
  var CX = SIZE / 2;
  var CY = SIZE / 2;
  var R = 32;
  var DOT_R = 4.5;
  var NS = 'http://www.w3.org/2000/svg';

  var PARAMS = [
    { id: 'bb-abc-temp', key: 'temperature', phase: 'abc' },
    { id: 'bb-abc-p', key: 'top_p', phase: 'abc' },
    { id: 'bb-abc-k', key: 'top_k', phase: 'abc' },
    { id: 'bb-abc-rep', key: 'repetition_penalty', phase: 'abc' },
    { id: 'bb-abc-win', key: 'penalty_window', phase: 'abc' },
    { id: 'bb-abc-min', key: 'min_tokens', phase: 'abc' },
    { id: 'bb-abc-max', key: 'max_tokens', phase: 'abc' },
    { id: 'bb-sem-temp', key: 'temperature', phase: 'sem' },
    { id: 'bb-sem-p', key: 'top_p', phase: 'sem' },
    { id: 'bb-sem-k', key: 'top_k', phase: 'sem' },
    { id: 'bb-sem-rep', key: 'repetition_penalty', phase: 'sem' },
    { id: 'bb-sem-win', key: 'penalty_window', phase: 'sem' },
    { id: 'bb-sem-min', key: 'min_tokens', phase: 'sem' },
    { id: 'bb-sem-max', key: 'max_tokens', phase: 'sem' }
  ];

  function defaultsFor(phase) {
    var pack = window.__BB_SAMPLING_DEFAULTS__ || {};
    return (phase === 'abc' ? pack.abc : pack.sem) || {};
  }

  function panel() { return document.getElementById('bb-sampling-panel'); }

  function viewBtn() {
    var wrap = document.getElementById('bb-sampling-view-btn');
    return wrap && (wrap.tagName === 'BUTTON' ? wrap : wrap.querySelector('button'));
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
      var label = knobs ? 'Switch to sliders view' : 'Switch to knobs view';
      btn.title = label;
      btn.setAttribute('aria-label', label);
      btn.classList.toggle('bb-showing-knobs', knobs);
      btn.classList.toggle('bb-showing-sliders', !knobs);
    }
  }

  function polar(deg) {
    var rad = deg * Math.PI / 180;
    return [CX + R * Math.cos(rad), CY + R * Math.sin(rad)];
  }

  function headDeg(t) { return START_DEG + Math.max(0, Math.min(1, t)) * SWEEP_DEG; }

  function arcD(t0, t1) {
    if (t1 - t0 < 0.0005) return '';
    var a0 = headDeg(t0), a1 = headDeg(t1);
    var p0 = polar(a0), p1 = polar(a1);
    var large = ((a1 - a0) % 360) > 180 ? 1 : 0;
    return 'M ' + p0[0].toFixed(2) + ' ' + p0[1].toFixed(2) +
      ' A ' + R + ' ' + R + ' 0 ' + large + ' 1 ' +
      p1[0].toFixed(2) + ' ' + p1[1].toFixed(2);
  }

  function findInputs(root) {
    if (!root) return { number: null, range: null };
    return {
      number: root.querySelector('input[data-testid="number-input"]') ||
              root.querySelector('input[type="number"]'),
      range: root.querySelector('input[data-testid="range-input"]') ||
             root.querySelector('input[type="range"]')
    };
  }

  function setNative(el, value) {
    if (!el) return;
    var proto = window.HTMLInputElement.prototype;
    var setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
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

  function el(name, attrs) {
    var node = document.createElementNS(NS, name);
    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        node.setAttribute(k, attrs[k]);
      });
    }
    return node;
  }

  function buildKnob(param) {
    var host = document.createElement('div');
    host.className = 'bb-knob';
    host.setAttribute('data-bb-slider', param.id);
    host.setAttribute('role', 'slider');
    host.tabIndex = 0;

    var label = document.createElement('div');
    label.className = 'bb-knob-label';
    label.textContent = param.key.toUpperCase();

    var body = document.createElement('div');
    body.className = 'bb-knob-body';

    var svg = el('svg', {
      class: 'bb-knob-svg',
      viewBox: '0 0 ' + SIZE + ' ' + SIZE,
      width: String(SIZE),
      height: String(SIZE),
      'aria-hidden': 'true'
    });
    var track = el('path', { class: 'bb-knob-arc', d: '', fill: 'none' });
    var dot = el('circle', {
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
    if (d) {
      knob.track.setAttribute('d', d);
      knob.track.style.display = '';
    } else {
      knob.track.setAttribute('d', '');
      knob.track.style.display = 'none';
    }
    var p = polar(headDeg(t));
    knob.dot.setAttribute('cx', p[0].toFixed(2));
    knob.dot.setAttribute('cy', p[1].toFixed(2));
    knob.valueEl.textContent = display;
    knob.host.setAttribute('aria-valuenow', display);
  }

  function bindKnob(knob) {
    var root = document.getElementById(knob.param.id);
    if (!root) return false;
    var existing = root.querySelector('.bb-knob');
    if (existing && existing !== knob.host) existing.remove();
    if (!knob.host.isConnected) {
      root.insertBefore(knob.host, root.firstChild);
    }
    root.classList.add('bb-sampling-slider');

    var drag = null;

    function syncFromSlider() {
      var inputs = findInputs(root);
      if (!inputs.number && !inputs.range) return;
      var meta = readMeta(inputs);
      var v = readValue(inputs, meta);
      var t = meta.max === meta.min ? 0 : (v - meta.min) / (meta.max - meta.min);
      paint(knob, t, formatValue(v, meta.step));
      knob.host.setAttribute('aria-valuemin', String(meta.min));
      knob.host.setAttribute('aria-valuemax', String(meta.max));
      knob._meta = meta;
      knob._inputs = inputs;
    }

    function onPointerDown(ev) {
      // Knobs-only interaction; ignore when sliders view is active.
      var panelEl = panel();
      if (panelEl && panelEl.classList.contains('bb-view-sliders')) return;
      if (ev.button != null && ev.button !== 0) return;
      ev.preventDefault();
      ev.stopPropagation();
      var inputs = findInputs(root);
      var meta = readMeta(inputs);
      var v = readValue(inputs, meta);
      drag = {
        y: ev.clientY,
        v: v,
        meta: meta,
        fine: !!ev.shiftKey,
        pointerId: ev.pointerId
      };
      try { knob.host.setPointerCapture(ev.pointerId); } catch (e) {}
      knob.host.classList.add('bb-knob-active');
      document.documentElement.classList.add('bb-knob-dragging');
    }

    function onPointerMove(ev) {
      if (!drag) return;
      ev.preventDefault();
      var meta = drag.meta;
      var span = meta.max - meta.min;
      var fine = drag.fine || ev.shiftKey;
      // ~120px vertical = full min→max; Shift ≈ 6× slower.
      var px = fine ? 720 : 120;
      var dy = drag.y - ev.clientY; // up increases
      var raw = drag.v + (span === 0 ? 0 : (dy / px) * span);
      var step = meta.step;
      if (fine) {
        var d = decimalsFor(meta.step) + 1;
        step = Math.max(meta.step / 10, Math.pow(10, -d));
      }
      var v = snap(raw, { min: meta.min, max: meta.max, step: step });
      var inputs = findInputs(root);
      writeValue(inputs, v);
      paint(knob, span === 0 ? 0 : (v - meta.min) / span, formatValue(v, step));
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
    }

    function onDblClick(ev) {
      var panelEl = panel();
      if (panelEl && panelEl.classList.contains('bb-view-sliders')) return;
      ev.preventDefault();
      ev.stopPropagation();
      var def = defaultsFor(knob.param.phase)[knob.param.key];
      if (def == null) return;
      var inputs = findInputs(root);
      var meta = readMeta(inputs);
      var v = snap(def, meta);
      writeValue(inputs, v);
      syncFromSlider();
    }

    function onKey(ev) {
      var panelEl = panel();
      if (panelEl && panelEl.classList.contains('bb-view-sliders')) return;
      var inputs = findInputs(root);
      var meta = readMeta(inputs);
      var v = readValue(inputs, meta);
      var step = ev.shiftKey ? Math.max(meta.step / 10, Math.pow(10, -(decimalsFor(meta.step) + 1))) : meta.step;
      if (ev.key === 'ArrowUp' || ev.key === 'ArrowRight') {
        ev.preventDefault();
        writeValue(inputs, snap(v + step, meta));
        syncFromSlider();
      } else if (ev.key === 'ArrowDown' || ev.key === 'ArrowLeft') {
        ev.preventDefault();
        writeValue(inputs, snap(v - step, meta));
        syncFromSlider();
      } else if (ev.key === 'Home') {
        ev.preventDefault();
        writeValue(inputs, meta.min);
        syncFromSlider();
      } else if (ev.key === 'End') {
        ev.preventDefault();
        writeValue(inputs, meta.max);
        syncFromSlider();
      }
    }

    knob.host.addEventListener('pointerdown', onPointerDown);
    knob.host.addEventListener('pointermove', onPointerMove);
    knob.host.addEventListener('pointerup', onPointerUp);
    knob.host.addEventListener('pointercancel', onPointerUp);
    knob.host.addEventListener('dblclick', onDblClick);
    knob.host.addEventListener('keydown', onKey);

    // Slider → knob (Gradio reset / preset / number typing)
    root.addEventListener('input', syncFromSlider, true);
    root.addEventListener('change', syncFromSlider, true);

    knob.sync = syncFromSlider;
    syncFromSlider();
    return true;
  }

  var knobs = [];
  var booted = false;

  function boot() {
    var p = panel();
    if (!p) return;
    if (!booted) {
      booted = true;
      applyView(savedView());
      PARAMS.forEach(function (param) {
        var knob = buildKnob(param);
        knobs.push(knob);
      });
      window.__bbToggleSamplingView = function () {
        var el = panel();
        if (!el) return;
        applyView(el.classList.contains('bb-view-knobs') ? 'sliders' : 'knobs');
      };
    }
    knobs.forEach(function (knob) {
      if (knob.host && !knob.host.isConnected) knob._bound = false;
      if (!knob._bound) {
        knob._bound = bindKnob(knob);
      } else if (knob.sync) {
        knob.sync();
      }
    });
  }

  setInterval(boot, 500);
  [200, 800, 2000, 4000].forEach(function (t) { setTimeout(boot, t); });
})();
