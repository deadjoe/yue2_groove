"""``build_ui``: the Blocks tree of both views and the event wiring."""
from __future__ import annotations

from pathlib import Path

import gradio as gr

from .. import __version__, config, library
from . import (
    cover_tab,
    edit_tab,
    frontend,
    generate_tab,
    library_tab,
    runtime,
    song_view,
    tools_tab,
)


def build_ui(defaults):
    header = """
<div id="bb-header">
  <h1>YUE2<span class="bb-slash">//</span>GROOVE</h1>
</div>"""
    footer = f"""
<div id="bb-footer">
  <span>YUE2-INFER 0.1.6 · GROOVE {__version__} · MODEL WEIGHTS CC BY-NC 4.0 (NON-COMMERCIAL)</span>
  <span>Developed by DEADJOE@GITHUB(<a href="https://github.com/deadjoe/yue2_groove" target="_blank" rel="noopener">yue2_groove</a>)</span>
</div>"""

    # fill_width: Gradio would otherwise cap the content at 640..1920px steps
    # inside our 1400px frame (see the outer-frame note in BASE_CSS).
    with gr.Blocks(title="YUE2 // GROOVE", fill_width=True) as demo:
        gr.HTML(header)
        current_band = gr.HTML("", elem_id="bb-current-band-wrap")
        current_bridge = gr.Textbox(value="", elem_id="bb-current-work",
                                    elem_classes=["bb-output"])
        # the view is applied client-side (html class) so the roots never remount;
        # this hidden box lets a server handler ask for the Studio view.
        view_mode = defaults.get("view_mode", "auto")
        initial_view = view_mode if view_mode in song_view.VIEW_CHOICES else "song"
        view_bridge = gr.Textbox(value=initial_view, elem_id="bb-view",
                                 elem_classes=["bb-output"])
        busy_out = gr.HTML("", elem_id="bb-busy-wrap")
        with gr.Row(elem_id="bb-topbtns"):
            gr.HTML('<span class="bb-view-label">VIEW //</span>')
            view_song_btn = gr.Button("SONG", size="sm", elem_id="bb-view-song-btn")
            view_studio_btn = gr.Button("STUDIO", size="sm", elem_id="bb-view-studio-btn")
            rail_btn = gr.Button("", size="sm", elem_id="bb-rail-btn")
            theme_btn = gr.Button("", size="sm", elem_id="bb-theme-btn")

        # ═══════════════════════ SONG view ═══════════════════════
        # The director: identity, stage, the next actions, a player and a
        # read-only score.  No editable component lives here — every knob is
        # one OPEN IN STUDIO away, with the current work already set.
        with gr.Column(elem_id="bb-song-root"):
            with gr.Column(elem_id="bb-song-empty") as song_empty:
                gr.HTML('<div class="bb-song-lead">NO CURRENT WORK</div>'
                        '<h2>Start or pick a work.</h2>'
                        '<p class="bb-song-sub">SONG follows one work at a time. Pick a start, '
                        'then SONG shows its stage and the next step. Knobs live in STUDIO.</p>')
                with gr.Row(elem_id="bb-song-cards"):
                    with gr.Column(elem_classes=["bb-song-card"]):
                        gr.HTML('<div class="bb-card-title">NEW SONG</div>'
                                '<div class="bb-card-sub">Style + lyrics → a new work</div>')
                        song_new_btn = gr.Button("START", size="sm")
                    with gr.Column(elem_classes=["bb-song-card"]):
                        gr.HTML('<div class="bb-card-title">COVER A RECORDING</div>'
                                '<div class="bb-card-sub">Audio → ABC → a new song</div>')
                        song_cover_start_btn = gr.Button("START", size="sm")
                    with gr.Column(elem_classes=["bb-song-card"]):
                        gr.HTML('<div class="bb-card-title">EDIT A WORK</div>'
                                '<div class="bb-card-sub">Load a saved work and revise it</div>')
                        song_edit_start_btn = gr.Button("START", size="sm")
            with gr.Column(elem_id="bb-song-work", visible=False) as song_work:
                song_identity = gr.HTML("")
                song_stage = gr.HTML("")
                with gr.Row(elem_id="bb-song-actions"):
                    song_listen_btn = gr.Button("LISTEN", size="sm", visible=False)
                    song_render_btn = gr.Button("RENDER IN STUDIO", size="sm", visible=False)
                    song_edit_btn = gr.Button("EDIT WORK", size="sm", visible=False)
                    song_retry_btn = gr.Button("TRY ANOTHER SEED", size="sm", visible=False)
                    song_check_btn = gr.Button("OPEN CHECK IN STUDIO", size="sm", visible=False)
                    song_send_btn = gr.Button("SEND TO GENERATE", size="sm", variant="primary",
                                              visible=False)
                    song_library_btn = gr.Button("OPEN IN LIBRARY", size="sm", visible=False)
                song_player = gr.Audio(label="AUDIO", interactive=False, visible=False,
                                       elem_id="bb-song-player")
                with gr.Column(elem_id="bb-song-score"):
                    gr.HTML('<div class="bb-score-title">SCORE VIEW</div>')
                    song_score = gr.HTML(frontend.score_panel("SONG SCORE", "No current work.", abc=""),
                                         elem_id="bb-song-score-panel")
                song_family = gr.HTML("")
                with gr.Row(elem_id="bb-song-compare"):
                    song_studio_btn = gr.Button("OPEN IN STUDIO", size="sm", visible=False)
                    song_compare_btn = gr.Button("BUILD COMPARISON", size="sm", visible=False)
                    song_compare_link = gr.HTML("", elem_id="bb-song-compare-link")
                song_compare_status = gr.HTML("", elem_id="bb-song-compare-status")

        # ═══════════════════════ STUDIO view ═══════════════════════
        # The existing 7-tab power UI and the runtime rail.  Never destroyed:
        # the view switch only toggles an <html> class (see the preflight doc).
        with gr.Row(equal_height=False, elem_id="bb-studio-root",
                    elem_classes=["bb-workspace"]):
            # ═══════════ main work area ═══════════
            # bb-main is the container the tab strip queries (LAYOUT_HEAD_CSS)
            with gr.Column(scale=5, min_width=520, elem_id="bb-main"):
                with gr.Tabs(selected=("gen", "cover", "edit", "library", "tools", "decode", "batch")
                               [int(defaults.get("tab", 0)) % 7]) as tabs:
                    # ───── 01 GENERATE ─────
                    with gr.Tab("01 // GENERATE", id="gen"):
                        style = gr.Textbox(label="STYLE", lines=3, placeholder=runtime.EXAMPLE_STYLE)
                        lyrics = gr.Textbox(label="LYRICS", lines=8, placeholder=runtime.EXAMPLE_LYRICS)
                        with gr.Row():
                            lyrics_file = gr.UploadButton("UPLOAD LYRICS .TXT", size="sm",
                                                          file_count="single", type="filepath",
                                                          file_types=[".txt", ".lrc"], scale=0)
                            abc_file = gr.UploadButton("UPLOAD ABC", size="sm",
                                                       file_count="single", type="filepath",
                                                       file_types=[".abc", ".txt"], scale=0)
                        with gr.Row():
                            cot = gr.Radio(choices=[("FULL", "full"),
                                                    ("MELODY", "melody"),
                                                    ("OFF", "off")],
                                           value="full", label="PLAN MODE", scale=2)
                            seed = gr.Number(value=831001, label="SEED", precision=0, scale=1)
                            cfg = gr.Number(value=0, label="CFG SCALE", scale=1)
                            out_id = gr.Textbox(value="", label="OUTPUT ID", max_lines=1, scale=1)
                            request_reset_btn = gr.Button("RESET", size="sm", scale=1,
                                                          elem_id="bb-reset-request")
                        with gr.Row(elem_classes=["bb-actionbar"]):
                            run_btn = gr.Button("GENERATE", variant="primary", size="lg",
                                                elem_id="bb-run")
                            plan_btn = gr.Button("PLAN ONLY", size="sm", elem_id="bb-plan")
                            allmodes_btn = gr.Button("ALL MODES", size="sm",
                                                     elem_id="bb-allmodes")
                            cancel_btn = gr.Button("CANCEL", variant="stop", size="sm",
                                                   elem_id="bb-cancel",
                                                   elem_classes=["bb-push-right"])
                        allmodes_link = gr.HTML(elem_id="bb-allmodes-link")
                        with gr.Accordion("SCORE INPUT (optional)",
                                          open=False) as score_input_accordion:
                            abc = gr.Textbox(label="ABC SCORE", lines=8,
                                             placeholder="Leave empty to let the model plan")
                        with gr.Accordion("ADVANCED // SAMPLING", open=False):
                            # no view class here: sampling-knobs.js adds bb-view-knobs /
                            # bb-view-sliders when the panel appears, so without the
                            # script the native sliders stay visible and usable
                            with gr.Column(elem_id="bb-sampling-panel"):
                                with gr.Row():
                                    preset = gr.Dropdown(choices=["Protocol defaults (full)",
                                                                  "Preview (~1–1.5 min song)",
                                                                  "Quick test (~20 s)"],
                                                         value="Protocol defaults (full)",
                                                         label="BUDGET PRESET", scale=3)
                                    sampling_reset_btn = gr.Button(
                                        "RESET DEFAULTS", size="sm", scale=1,
                                        elem_id="bb-reset-sampling")
                                # the view switch rides the duration line right above the
                                # phases it controls, so the preset row keeps main's
                                # [BUDGET PRESET][RESET DEFAULTS] layout
                                with gr.Row(elem_classes=["bb-sampling-viewbar"]):
                                    duration_md = gr.Markdown(
                                        generate_tab.duration_text(runtime.SEM_DEFAULTS["max_tokens"]), scale=3)
                                    sampling_view_btn = gr.Button(
                                        "", size="sm", scale=0,
                                        elem_id="bb-sampling-view-btn")
                                with gr.Row(elem_classes=["bb-sampling-phases"]):
                                    with gr.Group(elem_classes=["bb-group",
                                                                "bb-sampling-phase"]):
                                        gr.Markdown("**ABC PHASE · SCORE PLAN**")
                                        abc_temp = gr.Slider(
                                            0, 5, value=runtime.ABC_DEFAULTS["temperature"],
                                            step=.05, label="temperature",
                                            elem_id="bb-abc-temp",
                                            elem_classes=["bb-sampling-slider"])
                                        abc_p = gr.Slider(
                                            .05, 1, value=runtime.ABC_DEFAULTS["top_p"],
                                            step=.01, label="top_p",
                                            elem_id="bb-abc-p",
                                            elem_classes=["bb-sampling-slider"])
                                        abc_k = gr.Slider(
                                            1, 1000, value=runtime.ABC_DEFAULTS["top_k"],
                                            step=1, label="top_k",
                                            elem_id="bb-abc-k",
                                            elem_classes=["bb-sampling-slider"])
                                        abc_rep = gr.Slider(
                                            1, 2, value=runtime.ABC_DEFAULTS["repetition_penalty"],
                                            step=.005, label="repetition_penalty",
                                            elem_id="bb-abc-rep",
                                            elem_classes=["bb-sampling-slider"])
                                        abc_win = gr.Slider(
                                            1, 100, value=runtime.ABC_DEFAULTS["penalty_window"],
                                            step=1, label="penalty_window",
                                            elem_id="bb-abc-win",
                                            elem_classes=["bb-sampling-slider"])
                                        abc_min = gr.Slider(
                                            0, 4096, value=runtime.ABC_DEFAULTS["min_tokens"],
                                            step=8, label="min_tokens",
                                            elem_id="bb-abc-min",
                                            elem_classes=["bb-sampling-slider"])
                                        abc_max = gr.Slider(
                                            64, 4096, value=runtime.ABC_DEFAULTS["max_tokens"],
                                            step=32, label="max_tokens",
                                            elem_id="bb-abc-max",
                                            elem_classes=["bb-sampling-slider"])
                                    with gr.Group(elem_classes=["bb-group",
                                                                "bb-sampling-phase"]):
                                        gr.Markdown("**SEMANTIC PHASE · AUDIO**")
                                        sem_temp = gr.Slider(
                                            0, 5, value=runtime.SEM_DEFAULTS["temperature"],
                                            step=.05, label="temperature",
                                            elem_id="bb-sem-temp",
                                            elem_classes=["bb-sampling-slider"])
                                        sem_p = gr.Slider(
                                            .05, 1, value=runtime.SEM_DEFAULTS["top_p"],
                                            step=.01, label="top_p",
                                            elem_id="bb-sem-p",
                                            elem_classes=["bb-sampling-slider"])
                                        sem_k = gr.Slider(
                                            1, 1000, value=runtime.SEM_DEFAULTS["top_k"],
                                            step=1, label="top_k",
                                            elem_id="bb-sem-k",
                                            elem_classes=["bb-sampling-slider"])
                                        sem_rep = gr.Slider(
                                            1, 2, value=runtime.SEM_DEFAULTS["repetition_penalty"],
                                            step=.005, label="repetition_penalty",
                                            elem_id="bb-sem-rep",
                                            elem_classes=["bb-sampling-slider"])
                                        sem_win = gr.Slider(
                                            1, 100, value=runtime.SEM_DEFAULTS["penalty_window"],
                                            step=1, label="penalty_window",
                                            elem_id="bb-sem-win",
                                            elem_classes=["bb-sampling-slider"])
                                        sem_min = gr.Slider(
                                            0, 9000, value=runtime.SEM_DEFAULTS["min_tokens"],
                                            step=8, label="min_tokens",
                                            elem_id="bb-sem-min",
                                            elem_classes=["bb-sampling-slider"])
                                        sem_max = gr.Slider(
                                            64, 9000, value=runtime.SEM_DEFAULTS["max_tokens"],
                                            step=64, label="max_tokens",
                                            elem_id="bb-sem-max",
                                            elem_classes=["bb-sampling-slider"])
                                gr.Markdown("ODE method and context are fixed by the protocol "
                                            "(midpoint / 24576), same as upstream.",
                                            elem_classes=["bb-note"])
                        audio_out = gr.Audio(label="RESULT", type="filepath")
                        score_out = gr.Textbox(label="ABC SCORE", lines=8, max_lines=24,
                                               elem_id="bb-abc-source", elem_classes=["bb-output"])
                        gr.HTML('<div class="bb-score-title">SCORE VIEW</div>'
                                + frontend.score_panel("ABC SCORE",
                                               "No score yet — generate with PLAN MODE = FULL / "
                                               "MELODY, or run PLAN ONLY."),
                                elem_id="bb-score-panel")
                        with gr.Accordion("ARTIFACTS", open=False):
                            files_out = gr.File(label="FILES", file_count="multiple", height=120,
                                                elem_id="bb-files")
                        with gr.Row(elem_classes=["bb-tools"]):
                            open_library_btn = gr.Button("OPEN IN LIBRARY", size="sm")
                            gen_edit_btn = gr.Button("EDIT THIS RUN", size="sm")
                        gen_status = gr.Textbox(label="STATUS", lines=6, interactive=False)
                        gen_last_run = gr.State("")

                    # ───── 02 COVER ─────
                    with gr.Tab("02 // COVER", id="cover") as cover_tab_ui:
                        gr.Markdown(
                            "**Audio → ABC → cover.** Transcribe a recording with SheetSage2, "
                            "polish the score, then generate it in a new style — right here or in "
                            "**01 GENERATE**. SheetSage2 runs in its own environment (README, "
                            "⌜Cover from audio⌝); CHECK ENVIRONMENT says whether it is ready.",
                            elem_classes=["bb-note"])
                        cover_audio = gr.Audio(label="SOURCE AUDIO", sources=["upload"],
                                               type="filepath", elem_id="bb-cover-audio")
                        with gr.Row():
                            cover_task = gr.Radio(
                                choices=[("MELODY // VOCAL", "melody-vocal"),
                                         ("MELODY // VOCAL+INST", "melody-full"),
                                         ("FULL SCORE // + CHORDS", "full")],
                                value="melody-full", label="TRANSCRIPTION TASK", scale=3)
                            cover_max_seconds = gr.Number(value=0,
                                                          label="MAX SECONDS (0 = WHOLE FILE)",
                                                          precision=0, scale=1)
                        with gr.Accordion("SHEETSAGE2 OPTIONS", open=False):
                            cover_model = gr.Textbox(value=config.default_sheetsage_model(),
                                                     label="SHEETSAGE2 MODEL / DIR")
                            cover_base_model = gr.Textbox(
                                value=config.default_sheetsage_base_model(),
                                label="BASE MODEL / MERT SNAPSHOT",
                                placeholder="Optional local MERT-v2-FullSong path for offline loads")
                            with gr.Row():
                                cover_device = gr.Dropdown(choices=["auto", "cuda", "mps", "cpu"],
                                                           value=config.default_sheetsage_device(),
                                                           label="DEVICE", scale=1)
                                cover_dtype = gr.Dropdown(
                                    choices=[("auto", "auto"), ("bf16", "bf16"),
                                             ("fp32", "fp32")],
                                    value="auto", label="DTYPE", scale=1)
                                cover_revision = gr.Textbox(label="MODEL REVISION", max_lines=1,
                                                            scale=1)
                                cover_offline = gr.Checkbox(value=False, label="OFFLINE", scale=1)
                        with gr.Row(elem_classes=["bb-actionbar"]):
                            cover_btn = gr.Button("TRANSCRIBE", variant="primary", size="lg",
                                                  elem_id="bb-cover-run")
                            cover_cancel_btn = gr.Button("CANCEL", variant="stop", size="sm",
                                                         elem_classes=["bb-push-right"])
                        with gr.Row(elem_classes=["bb-tools"]):
                            cover_keep_warm = gr.Checkbox(
                                value=config.sheetsage_keep_warm(), label="KEEP SHEETSAGE2 WARM",
                                info="Reuse one resident model process between transcriptions until "
                                     "UNLOAD or the idle timeout")
                            cover_env_btn = gr.Button("CHECK ENVIRONMENT", size="sm")
                            cover_detect_btn = gr.Button("AUTO-DETECT VENV", size="sm")
                            cover_unload_btn = gr.Button("UNLOAD SHEETSAGE2", size="sm")
                        with gr.Row():
                            cover_source = gr.Dropdown(
                                label="SOURCE WORK", scale=4,
                                choices=[rel for _label, rel in library_tab.library_choices(
                                    library_tab.library_mode("time", "desc"))[1]],
                                info="A saved work or transcription with a score.abc",
                                interactive=True)
                        with gr.Row(elem_classes=["bb-tools"]):
                            cover_source_refresh = gr.Button("REFRESH", size="sm")
                            cover_load_btn = gr.Button("LOAD ABC", size="sm")
                            cover_send_edit_btn = gr.Button("SEND TO EDIT", size="sm")
                        cover_abc = gr.Textbox(label="COVER ABC", lines=10, max_lines=24,
                                               elem_id="bb-cover-abc")
                        gr.HTML('<div class="bb-score-title">SCORE VIEW</div>'
                                + frontend.score_panel("COVER ABC",
                                               "No transcription yet — upload audio and "
                                               "press TRANSCRIBE."),
                                elem_id="bb-cover-score-panel")
                        with gr.Row():
                            cover_keep = gr.Dropdown(
                                choices=["both", "Vocal", "Ins"], value="both",
                                label="MELODY VOICES", scale=2,
                                info="Which melodies survive STRIP and the cover generation")
                            cover_strip_btn = gr.Button("STRIP CHORDS", size="sm", scale=0)
                            cover_send_btn = gr.Button("SEND TO GENERATE", variant="primary",
                                                       size="lg", scale=0)
                        with gr.Accordion("GENERATE COVER // direct from this score",
                                          open=False) as cover_generate_accordion:
                            gr.Markdown(
                                "Score-conditioned generation with the target style and lyrics; "
                                "the submitted score appears as RESULT ABC below. "
                                "**SEND TO GENERATE** hands the score to 01 for fine control "
                                "(sampling, model, ALL MODES); **GENERATE COVER** stays here for "
                                "a quick take with the same shared sampling settings.",
                                elem_classes=["bb-note"])
                            cover_style = gr.Textbox(label="STYLE", lines=2,
                                                     placeholder=runtime.EXAMPLE_STYLE)
                            cover_lyrics = gr.Textbox(label="LYRICS", lines=5,
                                                      placeholder=runtime.EXAMPLE_LYRICS)
                            with gr.Row():
                                cover_seed = gr.Number(value=831001, label="SEED", precision=0,
                                                       scale=1)
                                cover_cfg = gr.Number(value=0, label="CFG SCALE", scale=1)
                                cover_generate_btn = gr.Button("GENERATE COVER", size="sm",
                                                               scale=2,
                                                               elem_classes=["bb-secondary"],
                                                               elem_id="bb-cover-generate")
                                cover_open_library_btn = gr.Button("OPEN IN LIBRARY", size="sm",
                                                                   scale=1)
                            cover_sampling_note = gr.Textbox(
                                label="SAMPLING // FROM 01 GENERATE", lines=2,
                                interactive=False, elem_id="bb-cover-sampling")
                            gr.Markdown("Sampling parameters are shared with **01 GENERATE → "
                                        "ADVANCED // SAMPLING**; change them there.",
                                        elem_classes=["bb-note"])
                            cover_result_audio = gr.Audio(label="RESULT", type="filepath")
                            cover_result_abc = gr.Textbox(label="RESULT ABC", lines=6, max_lines=18,
                                                          interactive=False,
                                                          elem_classes=["bb-output"])
                            gr.HTML('<div class="bb-score-title">SCORE VIEW // RESULT</div>'
                                    + frontend.score_panel("RESULT ABC", "No cover generated yet."),
                                    elem_id="bb-cover-result-score-panel")
                            with gr.Accordion("GENERATED FILES", open=False):
                                cover_gen_files = gr.File(label="FILES", file_count="multiple",
                                                          height=120)
                            cover_last_run = gr.State("")
                        with gr.Accordion("TRANSCRIPTION ARTIFACTS", open=False):
                            cover_files = gr.File(label="FILES", file_count="multiple", height=120,
                                                  elem_id="bb-cover-files")
                        cover_status = gr.Textbox(label="STATUS", lines=5, interactive=False)

                    # ───── 03 EDIT ─────
                    with gr.Tab("03 // EDIT", id="edit") as edit_tab_ui:
                        gr.Markdown(
                            "**Load → FREEZE BASELINE → edit → CHECK INVARIANTS → GENERATE EDITED "
                            "→ compare.** The edited ABC is always submitted explicitly, so an edit "
                            "can never silently degrade into a fresh plan; chord-only edits pass the "
                            "exact note/meter check. FREEZE and CHECK are always required — "
                            "ALLOW MELODY/RHYTHM CHANGES only lets a *failing* check through for "
                            "intentional adaptations.",
                            elem_classes=["bb-note"])
                        with gr.Row():
                            edit_source = gr.Dropdown(
                                label="SOURCE WORK",
                                choices=[rel for _label, rel in library_tab.library_choices(
                                    library_tab.library_mode("time", "desc"))[1]],
                                interactive=True, scale=4)
                            edit_refresh_btn = gr.Button("REFRESH", size="sm", scale=1)
                            edit_load_btn = gr.Button("LOAD", size="sm", scale=1)
                        with gr.Row():
                            edit_freeze_btn = gr.Button("FREEZE BASELINE", size="sm", scale=1)
                            edit_baseline_info = gr.Textbox(label="BASELINE", lines=3,
                                                            interactive=False, scale=3)
                        edit_style = gr.Textbox(label="STYLE", lines=2)
                        edit_lyrics = gr.Textbox(label="LYRICS", lines=5)
                        edit_abc = gr.Textbox(label="EDITED ABC", lines=10, max_lines=24,
                                              elem_id="bb-edit-abc")
                        gr.HTML('<div class="bb-score-title">SCORE VIEW</div>'
                                + frontend.score_panel("EDITED ABC",
                                               "Load a source work to start editing."),
                                elem_id="bb-edit-score-panel")
                        with gr.Row():
                            edit_cot = gr.Radio(choices=[("FULL // melody+chords", "full"),
                                                         ("MELODY // chord-free", "melody")],
                                                value="full", label="PLAN MODE", scale=2)
                            edit_seed = gr.Number(value=831001, label="SEED", precision=0, scale=1)
                            edit_cfg = gr.Number(value=0, label="CFG SCALE", scale=1)
                        with gr.Accordion("INVARIANT CHECK", open=True):
                            edit_contract = gr.Radio(
                                choices=[("EXACT // notes + meter", "exact"),
                                         ("PITCH // rhythm free", "pitch"),
                                         ("FREE // report only", "free")],
                                value="exact", label="CONTRACT", scale=3)
                            with gr.Row():
                                edit_voice = gr.Dropdown(choices=["both", "Vocal", "Ins"],
                                                         value="both", label="COMPARE VOICES",
                                                         scale=1)
                                edit_allow_tempo = gr.Checkbox(value=False,
                                                               label="ALLOW TEMPO CHANGE", scale=1)
                                edit_allow_meter = gr.Checkbox(value=False,
                                                               label="ALLOW METER CHANGE", scale=1)
                                edit_allow_changes = gr.Checkbox(
                                    value=False, label="ALLOW MELODY/RHYTHM CHANGES", scale=1)
                                edit_check_btn = gr.Button("CHECK INVARIANTS", size="sm", scale=1)
                            edit_check_out = gr.Textbox(label="CHECK RESULT", lines=7,
                                                        interactive=False)
                            gr.Markdown("**Symbolic check only.** A passing contract says nothing "
                                        "about how the generated audio sounds — confirm with the "
                                        "listening comparison below (whole song and a passage "
                                        "around the edit).", elem_classes=["bb-note"])
                        with gr.Row(elem_classes=["bb-actionbar"]):
                            edit_run_btn = gr.Button("GENERATE EDITED", variant="primary",
                                                     size="lg", elem_id="bb-edit-run")
                            edit_cancel_btn = gr.Button("CANCEL", variant="stop", size="sm",
                                                        elem_classes=["bb-push-right"])
                        edit_sampling_note = gr.Textbox(label="SAMPLING // FROM 01 GENERATE",
                                                        lines=2, interactive=False,
                                                        elem_id="bb-edit-sampling")
                        gr.Markdown("Sampling parameters are shared with **01 GENERATE → ADVANCED "
                                    "// SAMPLING**; change them there.", elem_classes=["bb-note"])
                        edit_audio = gr.Audio(label="RESULT", type="filepath")
                        edit_result_abc = gr.Textbox(label="RESULT ABC", lines=8, max_lines=24,
                                                     interactive=False, elem_classes=["bb-output"],
                                                     elem_id="bb-edit-result-abc")
                        gr.HTML('<div class="bb-score-title">SCORE VIEW // RESULT</div>'
                                + frontend.score_panel("RESULT ABC", "No edit generated yet."),
                                elem_id="bb-edit-result-score-panel")
                        with gr.Accordion("ARTIFACTS", open=False):
                            edit_files = gr.File(label="FILES", file_count="multiple", height=120)
                        with gr.Row():
                            edit_compare_btn = gr.Button("BUILD COMPARISON // baseline vs edit",
                                                         size="sm", scale=2)
                            edit_library_btn = gr.Button("OPEN IN LIBRARY", size="sm", scale=1)
                            edit_compare_file = gr.File(label="COMPARISON HTML",
                                                        file_types=[".html"], type="filepath",
                                                        scale=2)
                        edit_compare_link = gr.HTML(elem_id="bb-edit-compare-link")
                        edit_compare_status = gr.Textbox(label="COMPARISON", lines=3,
                                                         interactive=False)
                        edit_status = gr.Textbox(label="STATUS", lines=6, interactive=False)
                        edit_baseline_abc = gr.State("")
                        edit_source_rel = gr.State("")
                        edit_check_state = gr.State({})
                        edit_baseline_state = gr.State(None)
                        edit_last_run = gr.State("")
                    # ───── 04 LIBRARY ─────
                    with gr.Tab("04 // LIBRARY", id="library") as library_tab_ui:
                        with gr.Row():
                            with gr.Column(scale=2, min_width=260):
                                lib_sort_key = gr.Radio(
                                    choices=[("TIME", "time"), ("NAME", "name")],
                                    value="time", label="SORT", elem_id="bb-lib-sort")
                                lib_sort_dir = gr.Radio(
                                    choices=[("DESC", "desc"), ("ASC", "asc")],
                                    value="desc", label="ORDER", elem_id="bb-lib-order")
                                with gr.Row(elem_classes=["bb-tools"]):
                                    lib_refresh_btn = gr.Button("REFRESH", size="sm", elem_id="bb-lib-refresh")
                                lib_list = gr.CheckboxGroup(choices=[], value=[], label="WORKS",
                                                            interactive=True, elem_id="bb-lib-list")
                                with gr.Row():
                                    lib_rename_box = gr.Textbox(label="RENAME TO", max_lines=1,
                                                                scale=3, elem_id="bb-lib-rename-box")
                                    lib_rename_btn = gr.Button("RENAME", size="sm", scale=1,
                                                               interactive=False, elem_id="bb-lib-rename")
                                with gr.Row(elem_classes=["bb-tools"]):
                                    lib_edit_btn = gr.Button("OPEN IN 03 EDIT", size="sm",
                                                             elem_id="bb-lib-edit")
                                    lib_cover_btn = gr.Button("USE IN 02 COVER", size="sm",
                                                              elem_id="bb-lib-cover")
                                with gr.Row(elem_classes=["bb-tools"]):
                                    lib_delete_btn = gr.Button("DELETE SELECTED", size="sm",
                                                               elem_id="bb-lib-delete")
                                lib_confirm = gr.HTML("", elem_id="bb-lib-confirm")
                                with gr.Row(elem_classes=["bb-tools"]):
                                    lib_confirm_btn = gr.Button("CONFIRM DELETE", variant="stop", size="sm",
                                                                interactive=False,
                                                                elem_classes=["bb-danger-solid"],
                                                                elem_id="bb-lib-confirm-delete")
                                    lib_cancel_btn = gr.Button("CANCEL", size="sm",
                                                               elem_id="bb-lib-cancel-delete")
                                lib_pending = gr.State([])
                                # hidden bridge: row clicks set this to the work being viewed
                                lib_active = gr.Textbox(value="", elem_id="bb-lib-active",
                                                        elem_classes=["bb-output"])
                                lib_status = gr.Textbox(label="LIBRARY STATUS", lines=2,
                                                        interactive=False, elem_id="bb-lib-status")
                            with gr.Column(scale=3, min_width=320):
                                lib_info = gr.HTML(library.render_empty_html("Loading…"),
                                                   elem_id="bb-lib-info")
                                with gr.Accordion("STYLE", open=True):
                                    lib_style = gr.Textbox(value="", lines=4, interactive=False,
                                                           buttons=["copy"], elem_id="bb-lib-style")
                                with gr.Accordion("LYRICS", open=True):
                                    lib_lyrics = gr.Textbox(value="", lines=8, interactive=False,
                                                            buttons=["copy"], elem_id="bb-lib-lyrics")
                                with gr.Accordion("ABC SCORE (source)", open=False):
                                    lib_abc = gr.Textbox(value="", lines=8, interactive=False,
                                                         buttons=["copy"], elem_id="bb-lib-abc")
                                gr.HTML('<div class="bb-score-title">SCORE VIEW</div>'
                                        '<div id="bb-lib-score"><div id="bb-lib-score-inner">'
                                        '<div class="bb-score-empty">Select a work to view its score.</div>'
                                        '</div></div>', elem_id="bb-lib-score-panel")

                    # ───── 05 TOOLS ─────
                    with gr.Tab("05 // TOOLS", id="tools"):
                        with gr.Accordion("ABC TOOLS", open=True):
                            abc_tool_text = gr.Textbox(label="ABC", lines=6,
                                                       value=(config.EXAMPLES_DIR / "melody.abc").read_text(encoding="utf-8")
                                                       if (config.EXAMPLES_DIR / "melody.abc").exists() else "")
                            with gr.Row(elem_classes=["bb-tools"]):
                                inspect_btn = gr.Button("VALIDATE / EXPORT EVENTS", size="sm")
                                strip_btn = gr.Button("STRIP CHORDS (cover melody)", size="sm")
                                strip_voice = gr.Dropdown(choices=["both", "Vocal", "Ins"], value="both",
                                                          label="KEEP VOICES", scale=0)
                            abc_result = gr.Textbox(label="RESULT", lines=10)
                            gr.Markdown("**Edit invariant check** — confirm the sounding notes and "
                                        "meter are unchanged after editing.",
                                        elem_classes=["bb-note"])
                            abc_after = gr.Textbox(label="AFTER // EDITED ABC", lines=6)
                            with gr.Row(elem_classes=["bb-tools"]):
                                compare_voice = gr.Dropdown(choices=["both", "Vocal", "Ins"], value="both",
                                                            label="COMPARE VOICES", scale=1)
                                allow_tempo = gr.Checkbox(value=False, label="ALLOW TEMPO CHANGE", scale=1)
                                compare_abc_btn = gr.Button("COMPARE BEFORE/AFTER", size="sm", scale=1)
                            abc_compare_out = gr.Textbox(label="COMPARE RESULT", lines=6)
                        with gr.Accordion("LISTENING COMPARISON (static HTML)", open=False):
                            with gr.Row():
                                batch_pick = gr.Dropdown(label="FILL FROM GROUP RUN (BATCH / ALL MODES)",
                                                         choices=[c for c, _ in generate_tab.scan_batches()],
                                                         interactive=True, scale=4,
                                                         elem_id="bb-batch-pick")
                                batch_refresh = gr.Button("REFRESH LIST", size="sm", scale=1,
                                                          elem_id="bb-refresh-batches")
                            fill_btn = gr.Button("▾ FILL RUN DIRECTORIES FROM BATCH", size="sm",
                                                 elem_id="bb-fill-batch")
                            compare_paths = gr.Textbox(label="RUN DIRECTORIES", lines=3,
                                                       elem_id="bb-compare-paths")
                            compare_btn = gr.Button("BUILD COMPARISON", size="sm",
                                                    elem_id="bb-build-compare")
                            compare_file = gr.File(label="COMPARISON HTML", file_types=[".html"],
                                                   type="filepath")
                            compare_link = gr.HTML(elem_id="bb-compare-link")
                            compare_status = gr.Textbox(label="STATUS", lines=4, interactive=False,
                                                        elem_id="bb-compare-status")
                        with gr.Accordion("DOCTOR // ENVIRONMENT", open=False):
                            verify_hashes = gr.Checkbox(value=False, label="VERIFY WEIGHT HASHES")
                            doctor_btn = gr.Button("RUN DOCTOR", size="sm")
                            doctor_out = gr.Textbox(label="REPORT", lines=14)

                    # ───── 06 DECODE ─────
                    with gr.Tab("06 // DECODE", id="decode"):
                        gr.Markdown(
                            "**Evaluation / reproduction path** — re-decode a saved **latent.npy** "
                            "without generating again (same as the upstream skill script "
                            "`run_yue2.py decode`). Typical use: compare `standard` and `legacy` "
                            "decoders on the same song. Not part of the song-writing flow.",
                            elem_classes=["bb-note"],
                        )
                        with gr.Row():
                            source_dir = gr.Dropdown(label="RUN DIRECTORY", choices=runtime.scan_runs(),
                                                     interactive=True, scale=4)
                            refresh_btn = gr.Button("RELOAD", size="sm", scale=1)
                        with gr.Row():
                            latent_upload = gr.UploadButton("UPLOAD LATENT .NPY", size="sm",
                                                            file_count="single", type="filepath",
                                                            file_types=[".npy"], scale=0)
                            dec_vae_choice = gr.Radio(choices=[("SOURCE", "keep"),
                                                               ("STANDARD", "standard"),
                                                               ("LEGACY", "legacy"),
                                                               ("CUSTOM", "custom")],
                                                      value="standard", label="DECODER VAE", scale=3)
                        with gr.Accordion("ADVANCED // DECODE OPTIONS", open=False):
                            with gr.Row():
                                dec_vae_custom = gr.Textbox(label="CUSTOM VAE PATH / HF ID", max_lines=1)
                                dec_vae_revision = gr.Textbox(label="VAE REVISION", max_lines=1)
                            with gr.Row():
                                full_decode = gr.Checkbox(value=False, label="FULL DECODE")
                                decode_reset_btn = gr.Button("RESET", size="sm", scale=0,
                                                             elem_id="bb-reset-decode")
                        with gr.Row(elem_classes=["bb-actionbar"]):
                            decode_btn = gr.Button("RE-DECODE", variant="primary", size="lg",
                                                   elem_id="bb-decode")
                        decode_audio = gr.Audio(label="RESULT", type="filepath")
                        decode_status = gr.Textbox(label="STATUS", lines=6, interactive=False)

                    # ───── 07 BATCH ─────
                    with gr.Tab("07 // BATCH", id="batch"):
                        gr.Markdown(
                            "One JSON request per line (same as `yue2 batch`): "
                            "`id` (required, unique), `style`/`tags`, `lyrics`, `cot`, `seed`, "
                            "`cfg_scale`, `abc`, `abc_path` (relative to the uploaded file), optional "
                            "`abc_sampling` / `semantic_sampling` overrides. Fields you omit use the "
                            "sampling settings above. One request runs at a time.",
                            elem_classes=["bb-note"],
                        )
                        batch_file = gr.UploadButton("UPLOAD .JSONL", size="sm", file_count="single",
                                                     type="filepath", file_types=[".jsonl", ".txt"])
                        batch_text = gr.Textbox(label="JSONL REQUESTS", lines=8,
                                                placeholder='{"id":"pop1","style":"English piano pop","lyrics":"...","cot":"full"}\n'
                                                            '{"id":"jazz1","style":"English jazz","lyrics":"...","cot":"melody","seed":7}')
                        with gr.Row():
                            batch_id = gr.Textbox(label="OUTPUT NAME", value="batch", max_lines=1, scale=2)
                            batch_btn = gr.Button("RUN BATCH", variant="primary", size="lg", scale=0,
                                                  elem_id="bb-batch")
                        batch_table = gr.Dataframe(headers=["id", "status", "audio", "seconds", "artifacts"],
                                                   label="RESULTS", wrap=True)
                        batch_status = gr.Textbox(label="STATUS", lines=4, interactive=False)


            # ═══════════ runtime rail ═══════════
            with gr.Column(scale=2, min_width=300, elem_id="bb-rail"):
                with gr.Accordion("RUNTIME", open=False):
                    device = gr.Dropdown(choices=["auto", "mps", "cpu", "cuda"],
                                         value=defaults["device"], label="DEVICE")
                    dtype = gr.Dropdown(choices=runtime.DTYPE_CHOICES, value=defaults["dtype"], label="DTYPE")
                    model = gr.Textbox(value=defaults["model"], label="MODEL ID / LOCAL DIR")
                    vae_choice = gr.Radio(choices=[("STANDARD", "standard"),
                                                   ("LEGACY", "legacy"),
                                                   ("CUSTOM", "custom")],
                                          value=defaults.get("vae", "standard"),
                                          label="DEFAULT VAE")
                    vae_custom = gr.Textbox(label="CUSTOM VAE", max_lines=1)
                    with gr.Row():
                        revision = gr.Textbox(label="MODEL REVISION", max_lines=1)
                        vae_revision = gr.Textbox(label="VAE REVISION", max_lines=1)
                    offline = gr.Checkbox(value=False, label="OFFLINE")
                    backend = gr.Dropdown(choices=runtime.BACKEND_CHOICES, value="torch", label="BACKEND")
                    quantization = gr.Dropdown(choices=["none", "fp8"], value="none",
                                               label="QUANTIZATION")
                    offload_ar = gr.Checkbox(value=False, label="OFFLOAD AR WEIGHTS")
                    budget = gr.Number(value=24, label="MEMORY BUDGET")
                    ode_steps = gr.Slider(4, 64, value=32, step=4, label="ODE STEPS")
                    vae_core_frames = gr.Dropdown(choices=["auto", "512", "1024"], value="auto",
                                                  label="VAE CORE FRAMES")
                    with gr.Row():
                        load_btn = gr.Button("LOAD / APPLY", size="sm", scale=1)
                        unload_btn = gr.Button("UNLOAD MODEL", size="sm", scale=1)
                    runtime_reset_btn = gr.Button("RESET DEFAULTS", size="sm",
                                                  elem_id="bb-reset-runtime")
                env_status = gr.Textbox(value=defaults.get("status", ""), label="STATUS",
                                        lines=4, interactive=False, elem_id="bb-env-status")
                gr.Markdown(
                    "Apple Silicon: MPS runs bfloat16 with torch >= 2.11 (install with the "
                    "override file; torch 2.10 corrupts MPS attention past 1024 tokens). "
                    "vLLM and FP8 need NVIDIA CUDA.",
                    elem_id="bb-runtime-note",
                    elem_classes=["bb-note"],
                )

        # ───── event wiring ─────
        model_args = [device, dtype, backend, quantization, offload_ar, budget, ode_steps,
                      vae_core_frames, model, vae_choice, vae_custom, revision, vae_revision, offline]
        gen_common = [style, lyrics, cot, seed, cfg, abc, out_id, preset,
                      abc_temp, abc_p, abc_k, abc_rep, abc_win, abc_min, abc_max,
                      sem_temp, sem_p, sem_k, sem_rep, sem_win, sem_min, sem_max] + model_args
        run_btn.click(generate_tab.generate, inputs=gen_common,
                      outputs=[audio_out, score_out, gen_status, files_out,
                               run_btn, plan_btn, gen_last_run])
        library_outputs_for_flow = [lib_list, lib_info, lib_style, lib_lyrics, lib_abc,
                                    lib_rename_box, lib_rename_btn, lib_status, tabs, lib_active,
                                    current_bridge]
        edit_outputs_for_flow = [edit_style, edit_lyrics, edit_abc, edit_baseline_abc,
                                 edit_source_rel, edit_check_state, edit_baseline_state,
                                 edit_baseline_info, edit_status, edit_source, current_bridge,
                                 tabs]
        song_outputs = [song_empty, song_work, song_identity, song_stage, song_family,
                        song_player, song_score,
                        song_listen_btn, song_render_btn, song_edit_btn, song_retry_btn,
                        song_check_btn, song_send_btn, song_library_btn,
                        song_studio_btn, song_compare_btn]
        # SONG follows the current work: every success path / handoff lights the band
        # and (re)renders the director view, which stays mounted while hidden.
        current_bridge.change(song_view.render_song, inputs=[current_bridge], outputs=song_outputs)
        open_library_btn.click(library_tab.open_last_in_library, inputs=[gen_last_run],
                               outputs=library_outputs_for_flow)
        gen_last_run.change(song_view.mirror_current, inputs=[gen_last_run], outputs=[current_bridge])
        edit_last_run.change(song_view.mirror_current, inputs=[edit_last_run], outputs=[current_bridge])
        cover_last_run.change(song_view.mirror_current, inputs=[cover_last_run], outputs=[current_bridge])
        gen_edit_btn.click(library_tab.library_open_in_edit, inputs=[gen_last_run],
                           outputs=edit_outputs_for_flow)
        plan_btn.click(generate_tab.plan_only,
                       inputs=[style, lyrics, cot, seed, cfg, out_id,
                               abc_temp, abc_p, abc_k, abc_rep, abc_win, abc_min, abc_max]
                              + model_args,
                       outputs=[score_out, gen_status, files_out, run_btn, plan_btn,
                                current_bridge])
        allmodes_btn.click(generate_tab.generate_all_modes,
                           inputs=[style, lyrics, seed, cfg, abc, out_id,
                                   abc_temp, abc_p, abc_k, abc_rep, abc_win, abc_min, abc_max,
                                   sem_temp, sem_p, sem_k, sem_rep, sem_win, sem_min, sem_max]
                                  + model_args,
                           outputs=[gen_status, files_out, allmodes_link,
                                    run_btn, plan_btn, allmodes_btn, current_bridge])
        # Cooperative cancel only: do NOT use cancels=[...] here, because Gradio would
        # tear down the running generator event and drop its final "re-enable buttons" yield.
        cancel_btn.click(runtime.cancel_run, outputs=gen_status)
        preset.change(generate_tab.apply_preset, inputs=preset,
                      outputs=[abc_min, abc_max, sem_min, sem_max])
        sem_max.change(generate_tab.duration_text, inputs=sem_max, outputs=duration_md)
        sampling_reset_btn.click(generate_tab.reset_sampling_values,
                                 outputs=[abc_temp, abc_p, abc_k, abc_rep, abc_win, abc_min, abc_max,
                                          sem_temp, sem_p, sem_k, sem_rep, sem_win, sem_min, sem_max,
                                          preset])
        # View layer only: knobs ↔ sliders; Gradio slider values / event graph unchanged.
        sampling_view_btn.click(fn=None, js=frontend.SAMPLING_VIEW_TOGGLE_JS)
        request_reset_btn.click(lambda: ("", "", "full", 831001, 0, ""),
                                outputs=[style, lyrics, cot, seed, cfg, out_id])
        runtime_reset_btn.click(
            lambda: ("auto", defaults["dtype"], defaults["model"], defaults.get("vae", "standard"),
                     "", "", "", False, "torch", "none", False, 24, 32, "auto"),
            outputs=[device, dtype, model, vae_choice, vae_custom, revision, vae_revision,
                     offline, backend, quantization, offload_ar, budget, ode_steps, vae_core_frames])

        def _load_text(path):
            return Path(path).read_text(encoding="utf-8", errors="replace") if path else gr.update()

        lyrics_file.upload(_load_text, lyrics_file, lyrics)
        abc_file.upload(_load_text, abc_file, abc)

        refresh_btn.click(generate_tab.update_run_choices, outputs=source_dir)
        decode_btn.click(generate_tab.decode_run,
                         inputs=[source_dir, latent_upload, dec_vae_choice, dec_vae_custom,
                                 dec_vae_revision, full_decode] + model_args,
                         outputs=[decode_audio, decode_status, decode_btn, current_bridge])
        decode_reset_btn.click(lambda: ("standard", "", "", False),
                               outputs=[dec_vae_choice, dec_vae_custom, dec_vae_revision, full_decode])

        batch_btn.click(generate_tab.batch_generate,
                        inputs=[batch_text, batch_file, batch_id,
                                abc_temp, abc_p, abc_k, abc_rep, abc_win, abc_min, abc_max,
                                sem_temp, sem_p, sem_k, sem_rep, sem_win, sem_min, sem_max]
                              + model_args,
                        outputs=[batch_table, batch_status, batch_btn, current_bridge])
        cancel_btn.click(runtime.cancel_run, outputs=batch_status)

        inspect_btn.click(tools_tab.abc_inspect, inputs=abc_tool_text, outputs=abc_result)
        strip_btn.click(tools_tab.abc_strip_chords, inputs=[abc_tool_text, strip_voice], outputs=abc_result)
        compare_abc_btn.click(tools_tab.abc_compare, inputs=[abc_tool_text, abc_after, compare_voice, allow_tempo],
                              outputs=abc_compare_out)
        compare_btn.click(tools_tab.make_comparison, inputs=compare_paths,
                          outputs=[compare_file, compare_link, compare_status])
        fill_btn.click(generate_tab.fill_from_batch, inputs=batch_pick, outputs=compare_paths)
        batch_refresh.click(generate_tab.update_batch_choices, outputs=batch_pick)
        doctor_btn.click(tools_tab.run_doctor,
                         inputs=[model, vae_choice, vae_custom, revision, vae_revision, offline,
                                 verify_hashes],
                         outputs=doctor_out)
        load_btn.click(lambda *rail: runtime.load_pipeline(runtime.RuntimeSettings(*rail))[1],
                       inputs=model_args, outputs=env_status)
        unload_btn.click(lambda: (runtime.unload_pipeline(), "Model unloaded")[1], outputs=env_status)
        current_bridge.change(song_view.publish_current, inputs=[current_bridge],
                              outputs=[current_band, current_bridge])
        # ───── SONG wiring: producer actions reuse the existing handlers ─────
        # (or open Studio on the right tab); no action duplicates editable state.
        view_song_btn.click(fn=None, js=frontend.VIEW_SET_JS("song"), outputs=view_song_btn)
        view_studio_btn.click(fn=None, js=frontend.VIEW_SET_JS("studio"), outputs=view_studio_btn)
        song_new_btn.click(lambda: (gr.update(selected="gen"), gr.update(value="studio")),
                           outputs=[tabs, view_bridge])
        song_cover_start_btn.click(
            lambda: (gr.update(selected="cover"), gr.update(value="studio")),
            outputs=[tabs, view_bridge])
        song_edit_start_btn.click(
            lambda: (gr.update(selected="library"), gr.update(value="studio")),
            outputs=[tabs, view_bridge])
        song_listen_btn.click(fn=None, js=frontend.SONG_LISTEN_JS, outputs=song_listen_btn)
        song_render_btn.click(
            song_view.song_render_action, inputs=[current_bridge],
            outputs=[abc, score_input_accordion, style, lyrics,
                     cover_abc, cover_style, cover_lyrics, cover_status, cover_source,
                     current_bridge, tabs, view_bridge])
        song_edit_btn.click(song_view.song_open_edit, inputs=[current_bridge],
                            outputs=edit_outputs_for_flow + [view_bridge])
        song_check_btn.click(song_view.song_open_edit, inputs=[current_bridge],
                             outputs=edit_outputs_for_flow + [view_bridge])
        song_retry_btn.click(
            song_view.song_retry, inputs=[current_bridge],
            outputs=[style, lyrics, cot, seed, current_bridge, tabs, view_bridge])
        song_send_btn.click(
            song_view.song_send, inputs=[current_bridge],
            outputs=[abc, cot, style, lyrics, score_input_accordion, tabs, cover_status,
                     view_bridge])
        song_library_btn.click(song_view.song_open_library, inputs=[current_bridge],
                               outputs=library_outputs_for_flow + [view_bridge])
        song_studio_btn.click(song_view.song_open_studio, inputs=[current_bridge],
                              outputs=[tabs, view_bridge])
        song_compare_btn.click(song_view.song_compare, inputs=[current_bridge],
                               outputs=[song_compare_link, song_compare_status])
        cover_detect_btn.click(cover_tab.cover_detect_python, outputs=cover_status)
        busy_timer = gr.Timer(2.0)
        busy_timer.tick(frontend.busy_banner, outputs=busy_out)
        theme_btn.click(fn=None, js=frontend.THEME_TOGGLE_JS, outputs=theme_btn)
        rail_btn.click(fn=None, js=frontend.RAIL_TOGGLE_JS, outputs=rail_btn)

        # ───── cover wiring (SheetSage2 lives in sheetsage_adapter.py) ─────
        cover_controls = [cover_btn, cover_strip_btn, cover_send_btn, cover_generate_btn,
                          cover_env_btn, cover_unload_btn, cover_keep_warm]
        cover_btn.click(cover_tab.cover_transcribe,
                        inputs=[cover_audio, cover_task, cover_max_seconds, cover_model,
                                cover_device, cover_dtype, cover_revision, cover_base_model,
                                cover_keep_warm, cover_offline],
                        outputs=[cover_abc, cover_files, cover_status, *cover_controls,
                                 cover_generate_accordion, cover_source, current_bridge])
        cover_generate_btn.click(
            cover_tab.cover_generate,
            inputs=[cover_style, cover_lyrics, cover_abc, cover_task, cover_keep, cover_seed,
                    cover_cfg,
                    abc_temp, abc_p, abc_k, abc_rep, abc_win, abc_min, abc_max,
                    sem_temp, sem_p, sem_k, sem_rep, sem_win, sem_min, sem_max] + model_args,
            outputs=[cover_status, cover_result_audio, cover_result_abc, cover_gen_files,
                     *cover_controls, cover_last_run])
        cover_open_library_btn.click(library_tab.open_last_in_library, inputs=[cover_last_run],
                                     outputs=library_outputs_for_flow)
        cover_cancel_btn.click(runtime.cancel_run, outputs=cover_status)
        cover_env_btn.click(cover_tab.cover_check_environment, outputs=cover_status)
        cover_unload_btn.click(cover_tab.cover_unload_worker, outputs=cover_status)
        cover_source_refresh.click(cover_tab.cover_choices, outputs=cover_source)
        cover_load_btn.click(cover_tab.cover_load, inputs=[cover_source],
                             outputs=[cover_abc, cover_style, cover_lyrics, cover_status])
        cover_send_edit_btn.click(
            cover_tab.cover_send_to_edit,
            inputs=[cover_abc, cover_source, cover_style, cover_lyrics],
            outputs=[edit_abc, edit_baseline_abc, edit_style, edit_lyrics, edit_source_rel,
                     edit_check_state, edit_baseline_state, edit_baseline_info, edit_status,
                     edit_source, current_bridge, tabs])
        cover_strip_btn.click(cover_tab.cover_strip, inputs=[cover_abc, cover_keep],
                              outputs=[cover_abc, cover_status])
        cover_send_btn.click(cover_tab.cover_send_to_generate,
                             inputs=[cover_abc, cover_task, cover_style, cover_lyrics, cover_keep],
                             outputs=[abc, cot, style, lyrics, score_input_accordion, tabs,
                                      cover_status])

        # ───── edit wiring (pure logic in edit_flow.py) ─────
        edit_tab_ui.select(edit_tab.edit_choices, outputs=edit_source)
        edit_refresh_btn.click(edit_tab.edit_choices, outputs=edit_source)
        edit_tab_ui.select(cover_tab.sampling_summary,
                        inputs=[abc_temp, abc_p, abc_k, abc_rep, abc_win, abc_min, abc_max,
                                sem_temp, sem_p, sem_k, sem_rep, sem_win, sem_min, sem_max],
                        outputs=edit_sampling_note)
        cover_tab_ui.select(cover_tab.sampling_summary,
                         inputs=[abc_temp, abc_p, abc_k, abc_rep, abc_win, abc_min, abc_max,
                                 sem_temp, sem_p, sem_k, sem_rep, sem_win, sem_min, sem_max],
                         outputs=cover_sampling_note)
        cover_tab_ui.select(cover_tab.cover_choices, outputs=cover_source)
        # the mirror lives in a closed accordion (not mounted until expanded)
        cover_generate_accordion.expand(
            cover_tab.sampling_summary,
            inputs=[abc_temp, abc_p, abc_k, abc_rep, abc_win, abc_min, abc_max,
                    sem_temp, sem_p, sem_k, sem_rep, sem_win, sem_min, sem_max],
            outputs=cover_sampling_note)
        edit_load_btn.click(
            edit_tab.edit_load, inputs=[edit_source],
            outputs=[edit_style, edit_lyrics, edit_abc, edit_baseline_abc, edit_source_rel,
                     edit_check_state, edit_baseline_state, edit_baseline_info, edit_status])
        edit_freeze_btn.click(edit_tab.edit_freeze, inputs=[edit_source_rel, edit_source],
                              outputs=[edit_baseline_state, edit_baseline_info, edit_status])
        edit_check_btn.click(
            edit_tab.edit_check,
            inputs=[edit_baseline_abc, edit_abc, edit_voice, edit_allow_tempo,
                    edit_baseline_state, edit_contract, edit_allow_meter],
            outputs=[edit_check_out, edit_check_state])
        edit_run_btn.click(
            edit_tab.edit_generate,
            inputs=[edit_style, edit_lyrics, edit_cot, edit_seed, edit_cfg, edit_abc,
                    edit_baseline_abc, edit_source_rel, edit_check_state, edit_allow_changes,
                    edit_baseline_state,
                    abc_temp, abc_p, abc_k, abc_rep, abc_win, abc_min, abc_max,
                    sem_temp, sem_p, sem_k, sem_rep, sem_win, sem_min, sem_max] + model_args,
            outputs=[edit_audio, edit_result_abc, edit_files, edit_status,
                     edit_run_btn, edit_check_btn, edit_freeze_btn, edit_load_btn,
                     edit_refresh_btn, edit_compare_btn, edit_last_run])
        edit_cancel_btn.click(runtime.cancel_run, outputs=edit_status)
        edit_library_btn.click(library_tab.open_last_in_library, inputs=[edit_last_run],
                               outputs=library_outputs_for_flow)
        edit_compare_btn.click(edit_tab.edit_compare, inputs=[edit_source_rel, edit_last_run],
                               outputs=[edit_compare_file, edit_compare_link, edit_compare_status])

        # ───── library wiring (toolkit in library.py) ─────
        library_outputs = [lib_list, lib_info, lib_style, lib_lyrics, lib_abc,
                           lib_rename_box, lib_rename_btn, lib_status]
        library_tab_ui.select(library_tab.library_refresh,
                           inputs=[lib_sort_key, lib_sort_dir, lib_list, lib_active],
                           outputs=library_outputs)
        lib_refresh_btn.click(library_tab.library_refresh,
                              inputs=[lib_sort_key, lib_sort_dir, lib_list, lib_active],
                              outputs=library_outputs)
        lib_sort_key.change(library_tab.library_refresh,
                            inputs=[lib_sort_key, lib_sort_dir, lib_list, lib_active],
                            outputs=library_outputs)
        lib_sort_dir.change(library_tab.library_refresh,
                            inputs=[lib_sort_key, lib_sort_dir, lib_list, lib_active],
                            outputs=library_outputs)
        # row click = view/play only (JS sets the hidden bb-lib-active box)
        lib_active.change(library_tab.library_view, inputs=[lib_active],
                          outputs=[lib_info, lib_style, lib_lyrics, lib_abc,
                                   lib_rename_box, lib_rename_btn])
        lib_rename_btn.click(library_tab.library_rename,
                             inputs=[lib_active, lib_rename_box, lib_sort_key, lib_sort_dir, lib_list],
                             outputs=[lib_list, lib_info, lib_style, lib_lyrics, lib_abc,
                                      lib_rename_box, lib_rename_btn, lib_status, lib_active])
        lib_edit_btn.click(library_tab.library_open_in_edit, inputs=[lib_active],
                           outputs=edit_outputs_for_flow)
        lib_cover_btn.click(library_tab.library_use_in_cover, inputs=[lib_active],
                            outputs=[cover_abc, cover_style, cover_lyrics, cover_status,
                                     cover_source, current_bridge, tabs])
        lib_delete_btn.click(library_tab.library_delete_prepare,
                             inputs=[lib_list, lib_sort_key, lib_sort_dir],
                             outputs=[lib_confirm, lib_pending, lib_confirm_btn, lib_status])
        # note: delete-confirm returns (…, confirm, pending, button, status, active)
        lib_confirm_btn.click(library_tab.library_delete_confirm,
                              inputs=[lib_pending, lib_sort_key, lib_sort_dir, lib_active],
                              outputs=[lib_list, lib_info, lib_style, lib_lyrics, lib_abc,
                                       lib_rename_box, lib_rename_btn,
                                       lib_confirm, lib_pending, lib_confirm_btn, lib_status,
                                       lib_active])
        lib_cancel_btn.click(library_tab.library_delete_cancel,
                             outputs=[lib_confirm, lib_pending, lib_confirm_btn, lib_status])
        demo.load(library_tab.library_refresh, inputs=[lib_sort_key, lib_sort_dir, lib_list, lib_active],
                  outputs=library_outputs)
        demo.load(edit_tab.edit_choices, outputs=edit_source)
        demo.load(cover_tab.sampling_summary_pair,
                  inputs=[abc_temp, abc_p, abc_k, abc_rep, abc_win, abc_min, abc_max,
                          sem_temp, sem_p, sem_k, sem_rep, sem_win, sem_min, sem_max],
                  outputs=[edit_sampling_note, cover_sampling_note])

        gr.HTML(footer)

    demo.bb_head = frontend.head_html(view_mode)
    demo.bb_view_mode = view_mode
    return demo
