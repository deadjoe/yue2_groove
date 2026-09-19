"""The theme toggle is a CSS-drawn icon button (Bearbone grammar), not a text button."""
from __future__ import annotations

import pytest

gr = pytest.importorskip("gradio")

webui = pytest.importorskip("yue2_groove.webui")


def test_theme_toggle_is_drawn_with_css_not_text() -> None:
    css = webui.theme.BASE_CSS
    assert "#bb-theme-btn::before" in css
    bright = css.split("#bb-theme-btn.bb-bright::before", 1)[1].split("}", 1)[0]
    assert "box-shadow" in bright and "border: 1px solid currentColor" in bright    # sun ring + rays
    assert "THEME //" not in webui.frontend.THEME_TOGGLE_JS
    assert "THEME //" not in webui.frontend.HEAD_HTML
    # the icon is labelled for screen readers and tooltips, in both directions
    assert "Switch to the dark scene" in webui.frontend.THEME_TOGGLE_JS
    assert "Switch to the bright scene" in webui.frontend.HEAD_HTML
    assert "aria-label" in webui.frontend.THEME_TOGGLE_JS and "aria-label" in webui.frontend.HEAD_HTML
    assert "classList.toggle('bb-bright'" in webui.frontend.THEME_TOGGLE_JS  # scope-safe variant


def test_theme_button_has_no_text_label(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(webui.runtime, "RUNS", tmp_path / "runs")
    (tmp_path / "runs").mkdir()
    demo = webui.build_ui({"device": "cpu", "dtype": "float32", "model": "m-a-p/YuE2-3B",
                           "vae": "standard", "tab": 0, "status": ""})
    button = next(c for c in demo.blocks.values()
                  if getattr(c, "elem_id", None) == "bb-theme-btn")
    assert button.value == ""


def test_bright_dark_controls_share_the_cool_black() -> None:
    """The bright scene's dark controls must not fall back to the old warm black."""
    values = webui.theme._theme_values(webui.theme.BRIGHT)
    assert webui.theme.BRIGHT["primary_fill"] == "#11141C"
    for key in ("button_primary_background_fill", "checkbox_background_color_selected",
                "checkbox_border_color_selected", "slider_color", "loader_color"):
        assert values[key] == "#11141C", key
    assert webui.theme._bb_vars(webui.theme.BRIGHT)["--bb-chip-bg"] == "#11141C"   # selected chips/rows


def test_dark_scene_is_the_warm_black_ground() -> None:
    """The dark scene is the warm near-black family again (reverted from #11141C)."""
    assert webui.theme.DARK["bg"] == "#0B0A09"
    assert webui.theme.DARK["bg_panel"] == "#12110F"
    assert webui.theme.DARK["bg_input"] == "#171512"
    assert webui.theme.DARK["bg_lift"] == "#1C1916"
    assert webui.theme.DARK["stroke"] == "#2E2B27" and webui.theme.DARK["stroke2"] == "#8C8477"
    assert webui.theme.DARK["fg2"] == "#B5AEA2" and webui.theme.DARK["primary_text"] == "#16140F"
    values = webui.theme._theme_values(webui.theme.DARK)
    assert values["background_fill_primary"] == "#0B0A09"
    assert values["block_background_fill"] == "#12110F"
    assert values["button_primary_background_fill"] == "#F1ECE2"


def test_dark_scene_unchanged_by_the_bright_controls() -> None:
    values = webui.theme._theme_values(webui.theme.DARK)
    assert webui.theme.DARK["primary_fill"] == "#F1ECE2"        # ivory on the cool ground
    assert values["button_primary_background_fill"] == "#F1ECE2"
    assert webui.theme._bb_vars(webui.theme.DARK)["--bb-chip-bg"] != "#11141C"
