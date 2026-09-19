"""The responsive frame under Gradio 6's CSS scoping.

Gradio 6 rewrites ``css=`` through ``prefix_css``: a top-level rule is emitted
as written plus a copy prefixed with ``.gradio-container… .contain``; a rule
inside ``@media`` keeps only the prefixed copy; an ``@container`` block is
dropped.  Two things follow, and both bit the phone layout (a 390px iPhone
kept 28% of its width as dead margin, a 1180px iPad landscape lost 105px a
side and three tabs): the outer frame can only be styled by top-level rules,
and the tab strip's container query has to ride in the ``<head>``.  These
tests pin that shape so the next breakpoint does not silently do nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")

webui = pytest.importorskip("yue2_groove.webui")

# Everything outside <main class="contain">: a prefixed selector can never match it.
OUTER_FRAME = ("html", "body", "gradio-app", ".gradio-container", ".main", ".app", "#bb-tip")


def _media_blocks(css: str) -> list[str]:
    """The body of every top-level @media { … } block, braces balanced."""
    blocks, i = [], 0
    while True:
        i = css.find("@media", i)
        if i < 0:
            return blocks
        start = css.index("{", i) + 1
        depth, j = 1, start
        while depth:
            depth += {"{": 1, "}": -1}.get(css[j], 0)
            j += 1
        blocks.append(css[start : j - 1])
        i = j


def _selectors(block: str) -> list[str]:
    out = []
    for rule in block.split("}"):
        head = rule.split("{", 1)[0]
        out.extend(s.strip() for s in head.split(",") if s.strip())
    return out


def _build():
    return webui.build_ui(
        {
            "device": "cpu",
            "dtype": "float32",
            "model": "m-a-p/YuE2-3B",
            "vae": "standard",
            "tab": 0,
            "status": "",
        }
    )


@pytest.fixture(autouse=True)
def isolated_runs(tmp_path: Path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(webui.runtime, "RUNS", runs)


def test_media_queries_never_target_the_outer_frame() -> None:
    blocks = _media_blocks(webui.theme.BEARBONE_CSS)
    assert blocks, "the phone / tablet / touch rules are expected to be @media blocks"
    for block in blocks:
        for selector in _selectors(block):
            first = selector.split()[0]
            assert not any(
                first == tok or first.startswith((tok + ".", tok + ":")) for tok in OUTER_FRAME
            ), (
                f"{selector!r} inside @media only exists prefixed with "
                "`.gradio-container… .contain` after prefix_css, so it never matches"
            )


def test_frame_scales_with_clamp_instead_of_a_breakpoint() -> None:
    css = webui.theme.BASE_CSS
    frame = css.split(".gradio-container {", 1)[1].split("}", 1)[0]
    assert "clamp(" in frame and "padding:" in frame
    assert "max-width: 1400px !important" in frame
    # Gradio's own .app frame (16px 32px, 640..1920px steps) folds into ours
    assert ".gradio-container .main.app { padding: 0 !important; }" in css


def test_blocks_fill_width_removes_gradios_width_steps() -> None:
    assert _build().fill_width is True


def test_tab_strip_container_query_rides_in_the_head() -> None:
    # prefix_css drops @container from css=; the head is mounted verbatim, after it
    assert "@container (" not in webui.theme.BEARBONE_CSS
    head = webui.frontend.LAYOUT_HEAD_CSS
    assert head.startswith('<style id="bb-layout">') and head.rstrip().endswith("</style>")
    assert "#bb-main { container-type: inline-size; }" in head
    assert "@container (width <= 900px)" in head and "@container (width <= 660px)" in head
    assert ".tabs .overflow-dropdown { display: contents !important; }" in head
    assert head in webui.frontend.head_html("song")
    # the viewport copies are gone: one rule set, keyed on the column that holds the tabs
    for block in _media_blocks(webui.theme.BEARBONE_CSS):
        assert ".tab-wrapper" not in block and ".overflow-menu" not in block


def test_tabs_live_in_the_queried_column() -> None:
    demo = _build()
    main = [b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "bb-main"]
    assert len(main) == 1 and isinstance(main[0], gr.Column)
    tabs = [b for b in demo.blocks.values() if isinstance(b, gr.Tabs)]
    assert len(tabs) == 1 and tabs[0].parent is main[0]


def test_empty_chrome_placeholders_collapse() -> None:
    # gr.HTML pads an empty .html-container 12px top and bottom; two of them
    # plus the column gaps stacked to a blank 88px band under the header
    css = webui.theme.BASE_CSS
    assert (
        "#bb-current-band-wrap .html-container, #bb-busy-wrap .html-container "
        "{ padding: 0 !important; }"
    ) in css
