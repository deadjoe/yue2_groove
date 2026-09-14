"""ADVANCED // SAMPLING knobs view is a CSS/JS layer over the same 14 sliders."""
from __future__ import annotations

import pytest

gr = pytest.importorskip("gradio")

webui = pytest.importorskip("yue2_groove.webui")

SLIDER_IDS = [
    "bb-abc-temp", "bb-abc-p", "bb-abc-k", "bb-abc-rep", "bb-abc-win",
    "bb-abc-min", "bb-abc-max",
    "bb-sem-temp", "bb-sem-p", "bb-sem-k", "bb-sem-rep", "bb-sem-win",
    "bb-sem-min", "bb-sem-max",
]


def test_sampling_knobs_assets_and_defaults_are_wired() -> None:
    assert webui.SAMPLING_KNOBS_FILE.name == "sampling-knobs.js"
    assert webui.SAMPLING_KNOBS_FILE.is_file()
    assert "bb-sampling-panel" in webui.BASE_CSS
    assert "bb-view-knobs" in webui.BASE_CSS
    assert "ns-resize" in webui.BASE_CSS
    assert "bb-knob-dragging" in webui.BASE_CSS
    assert "__bbToggleSamplingView" in webui.SAMPLING_VIEW_TOGGLE_JS
    assert "__BB_SAMPLING_DEFAULTS__" in webui.HEAD_HTML
    assert "file={SAMPLING_KNOBS_FILE}" in webui.HEAD_HTML or str(
        webui.SAMPLING_KNOBS_FILE) in webui.HEAD_HTML
    js = webui.SAMPLING_KNOBS_FILE.read_text(encoding="utf-8")
    for eid in ("bb-abc-temp", "bb-sem-max"):
        assert eid in js
    assert "bb-sampling-view" in js
    assert "SWEEP_DEG" in js
    # vertical drag feel: ~120px full travel, Shift 720px (same step grid)
    assert "120" in js and "720" in js
    assert "preventDefault" in js
    assert "__bbToggleSamplingView" in js
    assert "MutationObserver" in js        # rebind when the accordion re-renders
    assert "aria-valuenow" in js
    assert "dblclick" in js                 # double-click restores the default
    # functional guards added after the numeric-control audit:
    assert "drag.raw" in js                  # incremental drag (Shift mid-drag, no jump)
    assert "drag.lastY" in js
    assert "activeDrag" in js                # window pointerup safety net
    assert "setInterval" in js               # heartbeat resync for preset/reset values
    assert "inputs.number.value !== s" in js  # number field is the written value
    assert "inputs.range.value !== s" in js   # range only when no number field exists


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


def test_sampling_sliders_remain_gradio_sliders_with_stable_ids(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(webui, "RUNS", tmp_path / "runs")
    (tmp_path / "runs").mkdir()
    demo = webui.build_ui({"device": "cpu", "dtype": "float32", "model": "m-a-p/YuE2-3B",
                           "vae": "standard", "tab": 0, "status": ""})
    by_id = {getattr(c, "elem_id", None): c for c in demo.blocks.values()}
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


def test_reset_sampling_values_unchanged() -> None:
    vals = webui._reset_sampling_values()
    assert len(vals) == 15  # 14 params + preset label
    assert vals[0] == webui.ABC_DEFAULTS["temperature"]
    assert vals[13] == webui.SEM_DEFAULTS["max_tokens"]
    assert vals[14] == "Protocol defaults (full)"
