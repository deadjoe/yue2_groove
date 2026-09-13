"""The theme toggle is a CSS-drawn icon button (Bearbone grammar), not a text button."""
from __future__ import annotations

import pytest

gr = pytest.importorskip("gradio")

webui = pytest.importorskip("yue2_groove.webui")


def test_theme_toggle_is_drawn_with_css_not_text() -> None:
    css = webui.BASE_CSS
    assert "#bb-theme-btn::before" in css
    bright = css.split("#bb-theme-btn.bb-bright::before", 1)[1].split("}", 1)[0]
    assert "box-shadow" in bright and "border: 1px solid currentColor" in bright    # sun ring + rays
    assert "THEME //" not in webui.THEME_TOGGLE_JS
    assert "THEME //" not in webui.HEAD_HTML
    # the icon is labelled for screen readers and tooltips, in both directions
    assert "Switch to the dark scene" in webui.THEME_TOGGLE_JS
    assert "Switch to the bright scene" in webui.HEAD_HTML
    assert "aria-label" in webui.THEME_TOGGLE_JS and "aria-label" in webui.HEAD_HTML
    assert "classList.toggle('bb-bright'" in webui.THEME_TOGGLE_JS  # scope-safe variant


def test_theme_button_has_no_text_label(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(webui, "RUNS", tmp_path / "runs")
    (tmp_path / "runs").mkdir()
    demo = webui.build_ui({"device": "cpu", "dtype": "float32", "model": "m-a-p/YuE2-3B",
                           "vae": "standard", "tab": 0, "status": ""})
    button = next(c for c in demo.blocks.values()
                  if getattr(c, "elem_id", None) == "bb-theme-btn")
    assert button.value == ""
