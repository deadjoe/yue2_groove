"""ADVANCED // SAMPLING knobs view is a CSS/JS layer over the same 14 sliders.

Three layers are pinned here: the Python wiring (build_ui, preset/reset event
graph, the defaults blob), the JS ↔ Python contract (PARAMS ids and keys), and
the JS numeric core itself, which runs under node through the read-only
`window.__BB_KNOB_INTERNALS__` export (skipped when node is not on PATH).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest

gr = pytest.importorskip("gradio")

webui = pytest.importorskip("yue2_groove.webui")

SLIDER_IDS = [
    "bb-abc-temp", "bb-abc-p", "bb-abc-k", "bb-abc-rep", "bb-abc-win",
    "bb-abc-min", "bb-abc-max",
    "bb-sem-temp", "bb-sem-p", "bb-sem-k", "bb-sem-rep", "bb-sem-win",
    "bb-sem-min", "bb-sem-max",
]
PHASE_DEFAULTS = {"abc": webui.ABC_DEFAULTS, "sem": webui.SEM_DEFAULTS}

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node not on PATH (runs the knob JS core)")

# Just enough window/document for the IIFE's boot wiring to run headless; nothing
# renders, and the script's timers/observers are stubbed out.
_NODE_SHIM = r"""
'use strict';
const fs = require('fs');
const noop = () => {};
global.window = global;
global.document = {
  readyState: 'complete', getElementById: () => null, addEventListener: noop,
  documentElement: { classList: { add: noop, remove: noop, contains: () => false } },
};
global.localStorage = { getItem: () => null, setItem: noop };
global.MutationObserver = function () { return { observe: noop }; };
global.addEventListener = noop;
global.setTimeout = noop;
global.setInterval = noop;
new Function(fs.readFileSync(process.argv[1], 'utf8'))();
const req = JSON.parse(process.argv[2]);
const result = new Function('I', 'args', req.body)(window.__BB_KNOB_INTERNALS__, req.args);
process.stdout.write(JSON.stringify(result));
"""


def _knob_js(body: str, **args):
    """Evaluate a JS function body over `I` (the knob internals) and `args` under node."""
    proc = subprocess.run(
        [NODE, "-e", _NODE_SHIM, str(webui.SAMPLING_KNOBS_FILE),
         json.dumps({"body": body, "args": args})],
        capture_output=True, text=True, check=True)
    return json.loads(proc.stdout)


def _params_from_js() -> list[dict]:
    """The PARAMS table, parsed from the script so the contract test needs no node."""
    js = webui.SAMPLING_KNOBS_FILE.read_text(encoding="utf-8")
    rows = re.findall(
        r"\{ id: '([^']+)', key: '([^']+)', label: '([^']+)', phase: '(abc|sem)' \}", js)
    return [{"id": i, "key": k, "label": lbl, "phase": ph} for i, k, lbl, ph in rows]


def _decimals(step: float) -> int:
    s = str(step)
    return len(s) - s.index(".") - 1 if "." in s else 0


@pytest.fixture
def demo(monkeypatch, tmp_path):
    monkeypatch.setattr(webui, "RUNS", tmp_path / "runs")
    (tmp_path / "runs").mkdir()
    return webui.build_ui({"device": "cpu", "dtype": "float32", "model": "m-a-p/YuE2-3B",
                           "vae": "standard", "tab": 0, "status": ""})


def _by_id(demo):
    return {getattr(c, "elem_id", None): c for c in demo.blocks.values()}


# ── Python wiring ──────────────────────────────────────────────────────────

def test_sampling_knobs_assets_and_defaults_are_wired() -> None:
    assert webui.SAMPLING_KNOBS_FILE.name == "sampling-knobs.js"
    assert webui.SAMPLING_KNOBS_FILE.is_file()
    assert "bb-sampling-panel" in webui.BASE_CSS
    assert "bb-view-knobs" in webui.BASE_CSS
    assert "ns-resize" in webui.BASE_CSS
    assert "bb-knob-dragging" in webui.BASE_CSS
    assert "__bbToggleSamplingView" in webui.SAMPLING_VIEW_TOGGLE_JS
    assert "file={SAMPLING_KNOBS_FILE}" in webui.HEAD_HTML or str(
        webui.SAMPLING_KNOBS_FILE) in webui.HEAD_HTML
    # the double-click reset contract: the blob is exactly the Python defaults,
    # and it is defined before the script that reads it
    blob = re.search(r"__BB_SAMPLING_DEFAULTS__ = (\{.*?\});</script>", webui.HEAD_HTML)
    assert blob is not None
    assert json.loads(blob.group(1)) == {"abc": webui.ABC_DEFAULTS, "sem": webui.SEM_DEFAULTS}
    assert webui.HEAD_HTML.index("__BB_SAMPLING_DEFAULTS__") < webui.HEAD_HTML.index(
        "sampling-knobs.js")
    js = webui.SAMPLING_KNOBS_FILE.read_text(encoding="utf-8")
    assert "bb-sampling-view" in js
    assert "preventDefault" in js
    assert "__bbToggleSamplingView" in js
    assert "MutationObserver" in js        # rebind when the accordion re-renders
    assert "ensureView" in js               # the script, not the server, picks the view
    assert "aria-valuenow" in js
    assert "dblclick" in js                 # double-click restores the default
    # DOM-level guards from the numeric-control audit (the numeric core itself
    # is exercised under node below):
    assert "activeDrag" in js                # window pointerup safety net
    assert "setInterval" in js               # heartbeat resync for preset/reset values
    assert "inputs.number.value !== s" in js  # number field is the written value
    assert "inputs.range.value !== s" in js   # range only when no number field exists
    assert "focus({ preventScroll: true })" in js  # arrows work right after a click


def test_sampling_knobs_css_targets_the_gradio6_dom() -> None:
    """Gradio 6 renders a phase as `.bb-sampling-phase > .styler > (.block, .form)`;
    a regression here once squeezed the knobs into a 27px-wide column."""
    css = webui.BASE_CSS
    assert ".bb-sampling-phase .form" in css
    assert "grid-template-columns: repeat(7, minmax(0, 1fr)) !important" in css
    # in knobs mode a slider block is reduced to its knob child
    assert ".bb-sampling-slider > :not(.bb-knob)" in css
    # mockup palette: stroke2 arc, fg2 label, primary dot on both scenes
    assert "var(--bb-line2)" in css
    assert ".bb-knob-dot { fill: var(--bb-primary-bg); stroke: none; }" in css
    assert "grid-template-columns: repeat(4, minmax(0, 1fr)) !important" in css
    assert "grid-template-columns: repeat(2, minmax(0, 1fr)) !important" in css
    # the view switch rides the duration line above the phases, not the preset row
    assert "#bb-sampling-panel .bb-sampling-viewbar" in css
    # phase headings are filled tags in knobs view (separation without extra space)
    assert "background: var(--bb-line) !important" in css
    # the theme blanks *:focus-visible, so the knob draws its own keyboard focus
    assert ".bb-knob:focus-visible .bb-knob-arc" in css
    # an un-armed toggle (script never ran) is an empty square: keep it hidden
    assert "#bb-sampling-view-btn:not(.bb-showing-knobs):not(.bb-showing-sliders)" in css


def test_sampling_sliders_remain_gradio_sliders_with_stable_ids(demo) -> None:
    by_id = _by_id(demo)
    assert "bb-sampling-panel" in by_id
    assert "bb-sampling-view-btn" in by_id
    assert by_id["bb-sampling-view-btn"].value == ""
    assert "bb-reset-sampling" in by_id
    for eid in SLIDER_IDS:
        assert isinstance(by_id[eid], gr.Slider), eid
    # defaults still match ABC_DEFAULTS / SEM_DEFAULTS
    assert by_id["bb-abc-temp"].value == webui.ABC_DEFAULTS["temperature"]
    assert by_id["bb-sem-max"].value == webui.SEM_DEFAULTS["max_tokens"]
    # concise phase labels over the two knob rows
    headings = [str(getattr(c, "value", "")) for c in demo.blocks.values()
                if isinstance(c, gr.Markdown)]
    assert any("ABC PHASE · SCORE PLAN" in h for h in headings)
    assert any("SEMANTIC PHASE · AUDIO" in h for h in headings)


def test_sampling_panel_is_plain_sliders_until_the_script_runs(demo) -> None:
    """Every knobs-mode rule hangs off `.bb-view-knobs`, which hides the native
    slider chrome; only the script may set it, or a failed script load would
    leave the accordion empty with no way to edit the sampling parameters."""
    classes = _by_id(demo)["bb-sampling-panel"].elem_classes or []
    assert "bb-view-knobs" not in classes
    assert "bb-view-sliders" not in classes


def test_sampling_view_toggle_is_a_js_only_click(demo) -> None:
    btn = _by_id(demo)["bb-sampling-view-btn"]
    deps = [f for f in demo.fns.values() if any(t[0] == btn._id for t in f.targets)]
    assert len(deps) == 1
    dep = deps[0]
    assert dep.targets == [(btn._id, "click")]
    assert dep.fn is None and dep.inputs == [] and dep.outputs == []
    assert "__bbToggleSamplingView" in (dep.js or "")


def test_preset_and_reset_still_drive_the_knob_sliders(demo) -> None:
    """The knobs repaint from these outputs (heartbeat), so the server-side
    writers must keep targeting the same slider blocks in the same order."""
    by_id = _by_id(demo)
    reset = next(f for f in demo.fns.values() if f.fn is webui._reset_sampling_values)
    ids = [getattr(o, "elem_id", None) for o in reset.outputs]
    assert ids[:14] == SLIDER_IDS
    assert isinstance(reset.outputs[14], gr.Dropdown)
    assert reset.outputs[14].label == "BUDGET PRESET"
    assert len(reset.outputs) == 15
    preset = next(f for f in demo.fns.values()
                  if getattr(f.fn, "__name__", "") == "apply_preset")
    touched = {o.elem_id for o in preset.outputs}
    assert touched <= set(SLIDER_IDS)
    assert {"bb-abc-max", "bb-sem-max"} <= touched
    assert by_id["bb-sem-max"] in preset.outputs


def test_reset_sampling_values_unchanged() -> None:
    vals = webui._reset_sampling_values()
    assert len(vals) == 15  # 14 params + preset label
    assert vals[0] == webui.ABC_DEFAULTS["temperature"]
    assert vals[13] == webui.SEM_DEFAULTS["max_tokens"]
    assert vals[14] == "Protocol defaults (full)"


# ── JS ↔ Python contract ──────────────────────────────────────────────────

def test_knob_params_match_the_python_side() -> None:
    """A drifted id leaves a slider without a knob; a drifted key makes the
    double-click reset a silent no-op (`defaultsFor(phase)[key]` is undefined)."""
    params = _params_from_js()
    assert [p["id"] for p in params] == SLIDER_IDS
    for phase, defaults in PHASE_DEFAULTS.items():
        keys = [p["key"] for p in params if p["phase"] == phase]
        assert keys == list(defaults), phase   # same order as _reset_sampling_values
    for p in params:
        assert p["id"].startswith(f"bb-{p['phase']}-"), p
        assert p["label"] == p["key"].upper(), p


# ── JS numeric core (node) ────────────────────────────────────────────────

@needs_node
def test_knob_numeric_core_matches_the_slider_grids(demo) -> None:
    by_id = _by_id(demo)
    params = []
    for p in _params_from_js():
        s = by_id[p["id"]]
        params.append({"id": p["id"], "min": s.minimum, "max": s.maximum, "step": s.step,
                       "default": PHASE_DEFAULTS[p["phase"]][p["key"]]})
    report = _knob_js("""
      return args.params.map(function (p) {
        var m = { min: p.min, max: p.max, step: p.step };
        var badGrid = 0, badFmt = 0;
        for (var n = 0; ; n++) {
          var v = Number((m.min + n * m.step).toFixed(3));
          if (v > m.max) break;
          if (I.snap(v, m) !== v) badGrid++;
          if (/e|\\d{7,}/.test(I.formatValue(v, m.step))) badFmt++;
        }
        return {
          id: p.id, def: I.snap(p.default, m),
          min: I.snap(m.min, m), max: I.snap(m.max, m),
          over: I.snap(m.max + 1e6, m), under: I.snap(m.min - 1e6, m),
          up: I.snap(p.default + m.step, m), down: I.snap(p.default - m.step, m),
          fmt: I.formatValue(p.default, m.step), decimals: I.decimalsFor(m.step),
          badGrid: badGrid, badFmt: badFmt
        };
      });
    """, params=params)
    assert [r["id"] for r in report] == SLIDER_IDS
    for p, r in zip(params, report, strict=True):
        lo, hi, step, default = p["min"], p["max"], p["step"], p["default"]
        # double-click reset lands exactly on the default
        assert r["def"] == default, p
        # ends are fixed points and overshoot clamps to them
        assert (r["min"], r["max"], r["over"], r["under"]) == (lo, hi, hi, lo), p
        # every grid point survives a round trip and prints without float noise
        assert r["badGrid"] == 0 and r["badFmt"] == 0, p
        # arrows: one step up / down, onto the grid, never past an end
        assert r["up"] == pytest.approx(min(hi, default + step)), p
        on_grid = abs(round((default - lo) / step) * step + lo - default) < 1e-9
        if on_grid:
            assert r["down"] == pytest.approx(max(lo, default - step)), p
        else:   # sem_max 9000 sits off the 64-grid: ArrowDown reaches the nearest point
            assert lo <= r["down"] < default and default - r["down"] <= step, p
        # the readout prints with the step's own precision
        assert r["decimals"] == _decimals(step), p
        expected = f"{default:.{_decimals(step)}f}" if _decimals(step) else str(round(default))
        assert r["fmt"] == expected, p


@needs_node
def test_knob_drag_accumulator_clamps_at_the_ends() -> None:
    """Regression: an unclamped accumulator kept climbing past max, so after an
    overshoot the knob sat dead for that many pixels on the way back."""
    r = _knob_js("""
      var m = { min: 0, max: 5, step: 0.05 };
      var raw = I.accumulate(4, 100, I.PX_FULL, m);       /* 100px up from 4: past 5 */
      var atMax = raw;
      raw = I.accumulate(raw, 200, I.PX_FULL, m);         /* keep pushing past the end */
      var back = I.accumulate(raw, -1, I.PX_FULL, m);     /* one pixel back */
      return {
        atMax: atMax, stillMax: raw, back: back, backShown: I.snap(back, m),
        coarse: I.accumulate(2.5, 12, I.PX_FULL, m) - 2.5,
        fine: I.accumulate(2.5, 12, I.PX_FINE, m) - 2.5,
        floor: I.accumulate(0.1, -300, I.PX_FULL, m),
        pxFull: I.PX_FULL, pxFine: I.PX_FINE
      };
    """)
    assert r["atMax"] == 5 and r["stillMax"] == 5
    assert r["back"] < 5 and r["backShown"] == 4.95      # moves on the first pixel back
    assert r["floor"] == 0
    # drag feel: 120px = full sweep, Shift is 6x finer on the same grid
    assert (r["pxFull"], r["pxFine"]) == (120, 720)
    assert r["coarse"] == pytest.approx(12 / 120 * 5)
    assert r["fine"] == pytest.approx(r["coarse"] / 6)


@needs_node
def test_knob_arc_geometry_keeps_the_bottom_gap() -> None:
    r = _knob_js("""
      return {
        head0: I.headDeg(0), head1: I.headDeg(1),
        clampLo: I.headDeg(-1), clampHi: I.headDeg(2),
        full: I.arcD(0, 1), half: I.arcD(0.5, 1), none: I.arcD(1, 1), tiny: I.arcD(0.9999, 1),
        start: I.polar(I.headDeg(0)), end: I.polar(I.headDeg(1)), center: I.polar(-90)
      };
    """)
    assert (r["head0"], r["head1"]) == (135, 405)          # 270° sweep from bottom-left
    assert (r["clampLo"], r["clampHi"]) == (135, 405)
    assert " 1 1 " in r["full"]                            # remaining arc > 180°: large-arc
    assert " 0 1 " in r["half"]
    assert r["none"] == "" and r["tiny"] == ""             # nothing left to draw at max
    # both ends sit below the centre (SVG y grows downward): the gap is at the bottom
    cx = cy = 44
    assert r["start"][0] < cx and r["start"][1] > cy
    assert r["end"][0] > cx and r["end"][1] > cy
    assert r["center"][1] < cy
