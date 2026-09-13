"""The button system: Bearbone's ops scale, three tiers, one focus per bar.

The action buttons used to be Gradio's default ``lg`` (42px / 16px) while the
rest of the UI is 10–12px wide-tracked caps, so the bars read as four equal
slabs.  These tests pin the replacement: a 12/11px scale, Primary filled,
Secondary outlined, Cancel ghost, Tool quiet and content-width.
"""
from __future__ import annotations

import pytest

gr = pytest.importorskip("gradio")

webui = pytest.importorskip("yue2_groove.webui")


def build():
    return webui.build_ui({"device": "cpu", "dtype": "float32", "model": "m",
                           "vae": "standard", "tab": 0, "status": ""})


def by_id(demo, elem_id):
    return next(c for c in demo.blocks.values() if getattr(c, "elem_id", None) == elem_id)


def by_value(demo, label):
    return next(c for c in demo.blocks.values() if getattr(c, "value", None) == label)


def test_theme_uses_the_ops_scale() -> None:
    values = webui._theme_values(webui.DARK)
    assert values["button_large_text_size"] == "12px"     # was Gradio's 16px
    assert values["button_small_text_size"] == "11px"     # was 12px
    assert values["button_large_text_weight"] == "600"
    assert values["button_large_padding"] == "8px 18px"
    assert values["button_small_padding"] == "4px 10px"


def test_css_defines_three_tiers_and_context_bars() -> None:
    css = webui.BASE_CSS
    assert "button.lg {" in css and "height: 34px" in css
    assert "button.sm {" in css and "height: 27px" in css
    assert "white-space: nowrap" in css
    for rule in (".bb-actionbar", ".bb-actionbar .bb-push-right", ".bb-tools",
                 "bb-danger-solid", "#bb-song-cards .bb-song-card"):
        assert rule in css, rule
    assert "flex-wrap: wrap !important" in css            # phones wrap, not clip


def test_action_bar_is_one_focus_plus_lighter_alternatives() -> None:
    demo = build()
    generate = by_id(demo, "bb-run")
    assert generate.size == "lg" and generate.variant == "primary"
    for elem_id in ("bb-plan", "bb-allmodes"):
        alternative = by_id(demo, elem_id)
        assert alternative.size == "sm"                   # demoted off lg
        assert alternative.variant == "secondary"
    cancel = by_id(demo, "bb-cancel")
    assert cancel.size == "sm" and cancel.variant == "stop"
    assert "bb-push-right" in cancel.elem_classes


def test_02_keeps_one_solid_primary_in_the_generate_area() -> None:
    demo = build()
    sends = [c for c in demo.blocks.values() if getattr(c, "value", None) == "SEND TO GENERATE"]
    assert {c.size for c in sends} == {"sm", "lg"}        # SONG action + COVER primary
    assert any(c.size == "lg" and c.variant == "primary" for c in sends)
    generate_cover = by_id(demo, "bb-cover-generate")
    assert generate_cover.size == "sm" and generate_cover.variant == "secondary"
    # outline Secondary does not depend on living inside an action bar
    assert "bb-secondary" in generate_cover.elem_classes


def test_destructive_confirm_is_the_only_solid_danger() -> None:
    demo = build()
    confirm = by_id(demo, "bb-lib-confirm-delete")
    assert confirm.size == "sm" and "bb-danger-solid" in confirm.elem_classes
    assert by_id(demo, "bb-lib-delete").size == "sm"


def test_action_bars_and_tool_rows_exist() -> None:
    demo = build()
    classes = [c.elem_classes or [] for c in demo.blocks.values()
               if getattr(c, "elem_classes", None)]
    assert sum("bb-actionbar" in c for c in classes) >= 4   # 01 / 02 / 03 / 06
    assert sum("bb-tools" in c for c in classes) >= 5


def test_touch_zoom_hack_is_scoped_to_editable_fields() -> None:
    css = webui.BASE_CSS
    # disabled output/status boxes must not be forced to 16px on iOS/iPadOS
    assert "input:not([disabled]):not([readonly])" in css
    assert "input, textarea, select { font-size: 16px" not in css


def test_upload_drop_zone_uses_the_hint_scale() -> None:
    css = webui.BASE_CSS
    assert '[data-testid="upload-text"] { font-size: 12px !important' in css
    assert '[data-testid="upload-text"] .or { font-size: 11px' in css
    assert '[data-testid="upload-icon"] svg { width: 20px' in css


def test_song_starters_are_cards_with_small_buttons() -> None:
    demo = build()
    starts = [c for c in demo.blocks.values() if getattr(c, "value", None) == "START"]
    assert len(starts) == 3 and all(c.size == "sm" for c in starts)
    htmls = [c.value for c in demo.blocks.values()
             if isinstance(getattr(c, "value", None), str)]
    assert any("bb-card-title" in h and "NEW SONG" in h for h in htmls)
    assert not any(getattr(c, "value", None) == "NEW SONG"
                   for c in demo.blocks.values())            # no longer a button label
