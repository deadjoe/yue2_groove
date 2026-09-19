"""SONG / STUDIO view switch: default view, SONG container safety, action wiring.

The browser preflight (docs/VIEW_SWITCH_PREFLIGHT.md) showed Gradio's ``visible``
toggle destroys and remounts a container, so the two roots stay mounted and the
view is a class on ``<html>``.  These tests pin the parts that do not need a
browser: the resolved default, the absence of editable state inside SONG, and
the fact that SONG actions reuse the existing handlers.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")

webui = pytest.importorskip("yue2_groove.webui")
config = pytest.importorskip("yue2_groove.config")

ABC = ("X:1\nT:\nM:4/4\nL:1/32\nQ:1/4=88\n"
       'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\n'
       'V: Ins clef=treble name="Ins Melody" snm="Inst."\nK:C\n'
       "V: Vocal\nE4E4G4G4E8z8|D4D4F4F4D8z8|\nV: Ins\nZ2|\n")

# Every Gradio component that would create editable state inside SONG.
FORBIDDEN = (gr.Textbox, gr.Radio, gr.Slider, gr.Number, gr.Checkbox, gr.Dropdown,
             gr.CheckboxGroup, gr.ColorPicker, gr.DateTime, gr.ImageEditor, gr.Dataframe,
             gr.UploadButton, gr.File)


@pytest.fixture(autouse=True)
def isolated_runs(tmp_path: Path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(webui, "RUNS", runs)
    return runs


def defaults(**overrides):
    base = {"device": "cpu", "dtype": "float32", "model": "m-a-p/YuE2-3B",
            "vae": "standard", "tab": 0, "status": ""}
    base.update(overrides)
    return base


def build(**overrides):
    return webui.build_ui(defaults(**overrides))


def make_song(root: Path, name: str = "20260913-120000-source") -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "request.json").write_text(json.dumps({
        "id": "source", "style": "English piano pop", "lyrics": "[Verse]\nla",
        "cot": "full", "seed": 5}), encoding="utf-8")
    (directory / "result.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    (directory / "audio.flac").write_bytes(b"fLaC")
    (directory / "score.abc").write_text(ABC, encoding="utf-8")
    return directory


def make_plan(root: Path, name: str = "20260913-130000-plan") -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "plan.json").write_text(json.dumps({"request": {
        "id": "plan", "style": "jazz", "lyrics": "la", "cot": "full"}}), encoding="utf-8")
    (directory / "score.abc").write_text(ABC, encoding="utf-8")
    return directory


def make_transcription(root: Path, name: str = "transcriptions/20260913-140000-ref") -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "request.json").write_text(json.dumps({
        "style": "soul", "lyrics": "la", "seed": 1}), encoding="utf-8")
    (directory / "result.json").write_text(json.dumps({"task": "melody-full"}), encoding="utf-8")
    (directory / "score.abc").write_text(ABC, encoding="utf-8")
    return directory


# ── view resolution ───────────────────────────────────────────────────────

def test_resolve_view_precedence() -> None:
    assert webui.resolve_view() == ("auto", 0)
    assert webui.resolve_view("song") == ("song", 0)
    assert webui.resolve_view("studio") == ("studio", 0)
    assert webui.resolve_view(None, 3) == ("studio", 3)
    # an explicit --tab forces Studio and its tab, even next to --view song
    assert webui.resolve_view("song", 2) == ("studio", 2)
    assert webui.resolve_view(None, None, "studio") == ("studio", 0)
    # the CLI wins over the environment
    assert webui.resolve_view("song", None, "studio") == ("song", 0)
    assert webui.resolve_view(None, None, "bogus") == ("auto", 0)


def test_default_view_is_song() -> None:
    demo = build()
    assert demo.bb_view_mode == "auto"
    assert 'var mode = "auto";' in webui._head_html("auto")
    assert 'var mode = "studio";' in webui._head_html("studio")
    # regression: the JSON placeholder must not clobber the window property name
    assert 'window.__BB_VIEW_MODE__ = mode;' in webui._head_html("studio")
    assert 'window."studio"' not in webui._head_html("studio")
    # both roots are always mounted: the switch is an <html> class, not visible=
    song_root = next(c for c in demo.blocks.values() if getattr(c, "elem_id", None) == "bb-song-root")
    studio_root = next(c for c in demo.blocks.values()
                       if getattr(c, "elem_id", None) == "bb-studio-root")
    assert song_root.visible is True and studio_root.visible is True
    assert "html.bb-view-song #bb-studio-root" in webui.BASE_CSS
    assert "html.bb-view-studio #bb-song-root" in webui.BASE_CSS
    assert "getElementById('bb-view')" in config.static_text("view.js")  # the hidden bridge is read


def test_view_bridge_is_hidden_by_css() -> None:
    """Regression: the view bridge is a Textbox and must never be visible chrome.

    The SONG tree walk cannot see it (it lives outside the SONG root), so pin the
    CSS that hides it, like #bb-current-work.
    """
    assert "#bb-view { display: none !important; }" in webui.BASE_CSS


def test_view_toggle_is_flat_compact_chrome() -> None:
    """Regression: a Row inside the chrome Row stretched and wrapped the buttons.

    Gradio gives a nested Row's children the 160px min-width and lets it wrap, so
    the toggle now sits flat next to the rail/theme buttons and the CSS targets
    the <button> elements (where the elem_id actually lands).
    """
    demo = build()
    ids = {getattr(c, "elem_id", None) for c in demo.blocks.values()}
    assert "bb-view-toggle" not in ids            # no Row inside the chrome Row
    assert {"bb-view-song-btn", "bb-view-studio-btn"} <= ids
    css = webui.BASE_CSS
    assert "#bb-view-song-btn button" not in css   # the id is on the button itself
    assert "#bb-view-song-btn, #bb-view-studio-btn {" in css
    assert "#bb-topbtns { flex-wrap: nowrap !important; }" in css


def test_rail_toggle_is_disabled_in_song_view() -> None:
    """The settings rail lives inside STUDIO, so its toggle is dead in SONG."""
    assert "rail.disabled = songView" in config.static_text("view.js")
    assert "__bbApplyRail" in config.static_text("view.js") and "__bbApplyRail" in webui.HEAD_HTML
    assert "Settings live in STUDIO" in config.static_text("view.js")


def test_forced_view_does_not_write_the_remembered_choice() -> None:
    boot = webui._head_html("studio")
    assert "window.__BB_VIEW_FORCED__ = (mode === 'song' || mode === 'studio');" in boot
    # the polling mirror only persists a choice when the launch was not forced
    assert "if (!window.__BB_VIEW_FORCED__)" in config.static_text("view.js")
    assert "__bbSetView" in config.static_text("view.js")   # a real click still remembers


def test_explicit_view_mode_is_baked_into_the_page() -> None:
    demo = build(view_mode="studio")
    assert demo.bb_view_mode == "studio"
    assert 'var mode = "studio";' in demo.bb_head


# ── SONG container safety ─────────────────────────────────────────────────

def walk(block):
    for child in getattr(block, "children", []) or []:
        yield child
        yield from walk(child)


def test_song_container_forbids_editable_components() -> None:
    demo = build()
    song_root = next(c for c in demo.blocks.values() if getattr(c, "elem_id", None) == "bb-song-root")
    offenders = [type(child).__name__ for child in walk(song_root)
                 if isinstance(child, FORBIDDEN)]
    assert offenders == [], f"SONG must hold no editable state, found: {offenders}"
    # and it does hold the pieces the view needs
    kinds = {type(child).__name__ for child in walk(song_root)}
    assert {"Button", "HTML", "Audio"} <= kinds


def test_song_score_is_read_only_html() -> None:
    html = webui._score_panel("SONG SCORE", "none", abc=ABC)
    assert 'data-bb-abc-text="' in html and "X:1" in html
    # the editable panels still resolve their textarea, with no inline text
    editable = webui._score_panel("ABC SCORE", "none")
    assert "data-bb-abc-text" not in editable


# ── render_song ───────────────────────────────────────────────────────────

def test_render_song_empty_state(isolated_runs: Path) -> None:
    out = webui.render_song("")
    assert len(out) == 16
    assert out[0]["visible"] is True and out[1]["visible"] is False   # empty shown, work hidden
    assert out[5]["visible"] is False                                  # player hidden
    assert "No current work" in out[6]
    assert all(part["visible"] is False for part in out[7:14])         # no actions


def test_render_song_with_a_song_shows_stage_score_player(isolated_runs: Path) -> None:
    make_song(isolated_runs)
    path = str((isolated_runs / "20260913-120000-source").resolve())
    out = webui.render_song(path)
    assert out[0]["visible"] is False and out[1]["visible"] is True
    assert "source" in out[2] and "SONG" in out[2]
    assert 'class="bb-stage bb-stage-current">AUDIO<' in out[3]
    assert out[3].count("bb-stage-current") == 1
    assert out[5]["visible"] is True and out[5]["value"].endswith("audio.flac")
    assert 'data-bb-abc-text="' in out[6] and "X:1" in out[6]
    visible = {key: part["visible"] for key, part in zip(webui._SONG_ACTIONS, out[7:14])}
    assert visible == {"listen": True, "render": False, "edit": True, "retry": True,
                       "check": False, "send": False, "library": True}
    assert out[10]["value"] == "TRY SEED 6"
    assert out[14]["visible"] is True     # OPEN IN STUDIO is always offered


def test_render_song_plan_offers_render(isolated_runs: Path) -> None:
    make_plan(isolated_runs)
    out = webui.render_song(str((isolated_runs / "20260913-130000-plan").resolve()))
    visible = {key: part["visible"] for key, part in zip(webui._SONG_ACTIONS, out[7:14])}
    assert visible["render"] is True and visible["listen"] is False
    assert out[5]["visible"] is False      # a plan has no audio yet


def test_render_song_transcription_offers_send(isolated_runs: Path) -> None:
    make_transcription(isolated_runs)
    out = webui.render_song("transcriptions/20260913-140000-ref")
    visible = {key: part["visible"] for key, part in zip(webui._SONG_ACTIONS, out[7:14])}
    assert visible["render"] is True and visible["send"] is True
    assert "TRANSCRIPTION" in out[2]


# ── action handlers reuse the existing behaviour ──────────────────────────

def test_song_retry_prefills_generate_with_seed_plus_one(isolated_runs: Path) -> None:
    make_song(isolated_runs)
    style, lyrics, cot, seed, _current, tabs, view = webui.song_retry(
        str((isolated_runs / "20260913-120000-source").resolve()))
    assert style["value"] == "English piano pop" and lyrics["value"] == "[Verse]\nla"
    assert cot["value"] == "full" and seed["value"] == 6
    assert view["value"] == "studio" and tabs["selected"] == "gen"


def test_song_render_action_loads_a_plan_into_generate(isolated_runs: Path) -> None:
    make_plan(isolated_runs)
    out = webui.song_render_action("20260913-130000-plan")
    assert len(out) == 12
    assert out[0]["value"].startswith("X:1") and out[1]["open"] is True
    assert out[9].endswith("20260913-130000-plan")   # current work kept
    assert out[10]["selected"] == "gen" and out[11]["value"] == "studio"


def test_song_render_action_sends_a_transcription_to_cover(isolated_runs: Path) -> None:
    make_transcription(isolated_runs)
    out = webui.song_render_action("transcriptions/20260913-140000-ref")
    assert len(out) == 12
    assert out[4]["value"].startswith("X:1")                  # cover_abc loaded
    assert out[10]["selected"] == "cover" and out[11]["value"] == "studio"


def test_song_send_hands_the_score_to_generate_without_current_bridge(
        isolated_runs: Path, monkeypatch) -> None:
    make_transcription(isolated_runs)
    captured = {}

    def fake_send(abc_text, task, style, lyrics, keep_voice):
        captured.update(abc=abc_text, task=task, keep_voice=keep_voice)
        return (gr.update(value=abc_text), gr.update(value="melody"), gr.update(value=style),
                gr.update(value=lyrics), gr.update(open=True), gr.update(selected="gen"),
                "sent")

    monkeypatch.setattr(webui, "cover_send_to_generate", fake_send)
    out = webui.song_send("transcriptions/20260913-140000-ref")
    assert len(out) == 8
    assert captured["task"] == "melody-full" and captured["keep_voice"] == "both"
    assert out[0]["value"].startswith("X:1") and out[5]["selected"] == "gen"
    assert out[7]["value"] == "studio"


def test_song_edit_and_library_wrappers_open_studio(isolated_runs: Path) -> None:
    make_song(isolated_runs)
    edit = webui.song_open_edit("20260913-120000-source")
    assert len(edit) == 13 and edit[-1]["value"] == "studio" and edit[-2]["selected"] == "edit"
    library = webui.song_open_library("20260913-120000-source")
    assert len(library) == 12 and library[-1]["value"] == "studio"
    assert library[8]["selected"] == "library"
    gen_tab, view = webui.song_open_studio("20260913-120000-source")
    assert gen_tab["selected"] == "library" and view["value"] == "studio"


def test_song_compare_stays_in_song(isolated_runs: Path, monkeypatch) -> None:
    make_song(isolated_runs, "20260913-120000-source")
    edited = make_song(isolated_runs, "20260913-130000-edit")
    (edited / "edit_manifest.json").write_text(json.dumps(
        {"source": {"rel": "20260913-120000-source"}}), encoding="utf-8")
    monkeypatch.setattr(webui, "make_comparison",
                        lambda _paths: ("/tmp/c.html", "<a>OPEN</a>", "ready"))
    link, status = webui.song_compare("20260913-130000-edit")
    assert (link, status) == ("<a>OPEN</a>", "ready")
    with pytest.raises(gr.Error, match="No baseline or source"):
        webui.song_compare("20260913-120000-source")


def test_family_heading_does_not_oversell_exact_matches() -> None:
    html = webui._song_family([{"title": "Twin", "rel": "r", "path": "/p",
                                "relations": ["same request id"], "confidence": "exact"}])
    assert ">FAMILY<" in html and "POSSIBLY RELATED" not in html
    assert "[exact]" in html
    assert "data-bb-run=\"/p\"" in html


# ── wiring ────────────────────────────────────────────────────────────────

def _events(demo):
    return {getattr(f.fn, "__name__", ""): f for f in demo.fns.values()}


def test_song_action_wiring_matches_the_handlers() -> None:
    demo = build()
    events = _events(demo)
    assert len(events["render_song"].outputs) == 16
    assert len(events["song_render_action"].outputs) == 12
    assert len(events["song_retry"].outputs) == 7
    assert len(events["song_send"].outputs) == 8
    assert len(events["song_open_edit"].outputs) == 13
    assert len(events["song_open_library"].outputs) == 12
    assert len(events["song_open_studio"].outputs) == 2
    assert len(events["song_compare"].outputs) == 2   # link + status, no view switch
    # both SEND TO GENERATE paths stay free of the current-work bridge
    current = next(c for c in demo.blocks.values()
                   if getattr(c, "elem_id", None) == "bb-current-work")
    assert current not in events["cover_send_to_generate"].outputs
    assert current not in events["song_send"].outputs
    # BUILD COMPARISON must not hide SONG before the user can click the link
    view = next(c for c in demo.blocks.values() if getattr(c, "elem_id", None) == "bb-view")
    assert view not in events["song_compare"].outputs


def test_backend_dropdown_offers_vllm_only_when_importable() -> None:
    """vLLM is upstream's optional Linux/CUDA extra; a plain install (and every
    macOS / Windows one) must not list a backend that ends in ImportError."""
    assert webui.BACKEND_CHOICES[:2] == ["torch", "torch-eager"]
    assert ("vllm" in webui.BACKEND_CHOICES) == (importlib.util.find_spec("vllm") is not None)
    demo = webui.build_ui({"device": "cpu", "dtype": "float32", "model": "m-a-p/YuE2-3B",
                           "vae": "standard", "tab": 0, "status": ""})
    backend = next(c for c in demo.blocks.values()
                   if isinstance(c, gr.Dropdown) and c.label == "BACKEND")
    assert [value for _, value in backend.choices] == webui.BACKEND_CHOICES
