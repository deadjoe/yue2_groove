"""UI handler tests for 06 COVER and 07 EDIT — Gradio updates, no model.

The handlers are plain functions, so they can be called with values directly.
Model work is mocked at the ``adapter``/``sheetsage_adapter`` boundary; these
tests verify the state machine: guards, yields, button re-enabling, states.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")

from yue2_groove import edit_flow, library, webui  # noqa: E402

BASE_ABC = textwrap.dedent("""\
    X:1
    T:
    M:4/4
    L:1/16
    Q:1/4=88
    V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
    V: Ins clef=treble name="Ins Melody" snm="Inst."
    K:C
    % verse
    V: Vocal
    "C"E2G2A2G2E2D2C4|"Am"D2E2G2E2D2C2D4|
    V: Ins
    Z2|
""")
CHORD_FREE = BASE_ABC.replace('"C"', "").replace('"Am"', "")


@pytest.fixture(autouse=True)
def isolated_runs(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(webui, "RUNS", tmp_path / "runs")
    (tmp_path / "runs").mkdir()
    return tmp_path / "runs"


def make_work(root: Path, name: str = "20260901-120000-source") -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "score.abc").write_text(BASE_ABC, encoding="utf-8")
    (directory / "request.json").write_text(json.dumps({
        "id": "source", "style": "English piano pop", "lyrics": "[Verse]\nla",
        "cot": "full", "seed": 5}), encoding="utf-8")
    (directory / "result.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    return directory


def updates(values):
    return [v for v in values]


# ── cover ────────────────────────────────────────────────────────────────

def test_score_panel_marks_the_textarea_label() -> None:
    html = webui._score_panel("COVER ABC", "nothing here")
    assert 'data-bb-abc="COVER ABC"' in html and 'bb-score-inner' in html
    assert "nothing here" in html


def test_cover_strip_and_send_flow() -> None:
    stripped, status = webui.cover_strip(BASE_ABC, "both")
    assert '"C"' not in stripped and "verified" in status

    score_update, cot_update, tab_update, status = webui.cover_send_to_generate(BASE_ABC, "melody-full")
    assert '"C"' not in score_update["value"] and score_update["value"].startswith("X:1")
    assert cot_update["value"] == "melody" and tab_update["selected"] == "gen"
    assert "GENERATE" in status

    score_update, cot_update, _tab, _status = webui.cover_send_to_generate(BASE_ABC, "full")
    assert '"C"' in score_update["value"] and cot_update["value"] == "full"

    with pytest.raises(gr.Error, match="Cannot send"):
        webui.cover_send_to_generate("   ", "melody-full")


def test_cover_check_environment_reports_missing_configuration(monkeypatch) -> None:
    monkeypatch.delenv("YUE2_GROOVE_SHEETSAGE_PYTHON", raising=False)
    message = webui.cover_check_environment()
    assert "not ready" in message and "YUE2_GROOVE_SHEETSAGE_PYTHON" in message


def test_cover_transcribe_yields_result_and_frees_the_lock(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "ref.wav"
    audio.write_bytes(b"RIFF")
    transcript = tmp_path / "transcript"
    transcript.mkdir()
    (transcript / "score.abc").write_text(CHORD_FREE, encoding="utf-8")
    seen = {}

    def fake_transcribe(audio_path, **kwargs):
        seen.update(kwargs)
        return {"abc": CHORD_FREE, "warnings": ["low confidence"], "task": "melody-full",
                "melody_only": True, "output_dir": str(transcript), "device": "cpu",
                "dtype": "fp32"}

    monkeypatch.setattr(webui.sheetsage_adapter, "transcribe", fake_transcribe)
    yields = list(webui.cover_transcribe(str(audio), "melody-full", 0, "m-a-p/SheetSage2",
                                         "auto", "auto", "", "/mert-snapshot", False, False))
    assert all(len(chunk) == 10 for chunk in yields)         # matches the 10 wired outputs
    abc, files, status, *buttons = yields[-1]
    assert abc == CHORD_FREE and files and str(transcript / "score.abc") in files
    assert "low confidence" in status and len(buttons) == 7
    assert all(button["interactive"] is True for button in buttons)   # all actions re-enabled
    assert seen["cancelled"] is not None and seen["base_model"] == "/mert-snapshot"
    assert seen["keep_warm"] is False
    assert seen["output_dir"].parent == webui.RUNS / "transcriptions"
    # the lock was released: a second transcription can start
    assert list(webui.cover_transcribe(str(audio), "melody-full", 0, "m", "auto", "auto", "",
                                       "", False, False))


def test_cover_transcribe_honours_the_transcriptions_override(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "ref.wav"
    audio.write_bytes(b"RIFF")
    transcript = tmp_path / "transcript"
    transcript.mkdir()
    seen = {}
    monkeypatch.setenv("YUE2_GROOVE_TRANSCRIPTIONS", str(tmp_path / "custom-place"))

    def fake_transcribe(audio_path, **kwargs):
        seen.update(kwargs)
        return {"abc": CHORD_FREE, "warnings": [], "task": "melody-full", "melody_only": True,
                "output_dir": str(transcript), "device": "cpu", "dtype": "fp32"}

    monkeypatch.setattr(webui.sheetsage_adapter, "transcribe", fake_transcribe)
    list(webui.cover_transcribe(str(audio), "melody-full", 0, "m", "auto", "auto", "", "", False,
                                    False))
    assert Path(seen["output_dir"]).parent == tmp_path / "custom-place"


def test_cover_transcribe_refuses_without_audio_and_when_busy() -> None:
    with pytest.raises(gr.Error, match="Upload a source audio"):
        next(webui.cover_transcribe("", "melody-full", 0, "", "auto", "auto", "", "", False, False))
    webui._RUNNING.acquire()
    try:
        yields = list(webui.cover_transcribe("/tmp/x.wav", "melody-full", 0, "", "auto",
                                             "auto", "", "", False, False))
        assert "already running" in yields[0][2]
    finally:
        webui._RUNNING.release()


# ── edit ─────────────────────────────────────────────────────────────────

def test_edit_load_fills_the_baseline(isolated_runs: Path) -> None:
    make_work(isolated_runs)
    (style, lyrics, abc, baseline_abc, rel, check_state, baseline_state, info, status) = \
        webui.edit_load("20260901-120000-source")
    assert style["value"].startswith("English") and lyrics["value"].startswith("[Verse]")
    assert abc["value"] == BASE_ABC.strip() and baseline_abc == BASE_ABC.strip()
    assert rel == "20260901-120000-source" and check_state == {} and baseline_state is None
    assert "not frozen yet" in info and "Loaded" in status
    with pytest.raises(gr.Error, match="Pick a source work"):
        webui.edit_load("")


def test_edit_freeze_and_check_guards(isolated_runs: Path) -> None:
    make_work(isolated_runs)
    record, info, status = webui.edit_freeze("20260901-120000-source")
    assert record["schema"] == "yue2-groove-baseline-v1" and "untouched" in info
    assert "frozen" in status
    with pytest.raises(gr.Error, match="Load a source work"):
        webui.edit_freeze("")

    with pytest.raises(gr.Error, match="Freeze the baseline"):
        webui.edit_check(BASE_ABC, BASE_ABC, "both", False, None)
    output, state = webui.edit_check(BASE_ABC, BASE_ABC, "both", False, record)
    assert json.loads(output)["match"] is True and state["sha256"]
    output, state = webui.edit_check(BASE_ABC, CHORD_FREE, "both", False, record)
    assert json.loads(output)["match"] is True  # chord-only edits pass
    with pytest.raises(gr.Error, match="Load a source work"):
        webui.edit_check("", BASE_ABC, "both", False, record)


class FakeSong:
    abc = BASE_ABC
    timing = {"nar_seconds": 1.0, "vae_seconds": 0.5,
              "semantic": {"output_tps": 9.0}, "abc": {"output_tokens": 10}}


def fake_run_generation(pipe, request, outdir, *args, **kwargs):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    (Path(outdir) / "audio.flac").write_bytes(b"fLaC")
    manifest = kwargs.get("extra_manifest") or {}
    if manifest:
        (Path(outdir) / "edit_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return FakeSong(), {"audio_seconds": 12.5, "truncated": False}, 2.0


def edit_generate_args(**overrides):
    args = dict(style="jazz", lyrics="la", cot="full", seed=831001, cfg_scale=0,
                abc_text=BASE_ABC, baseline_abc=BASE_ABC, source_rel="20260901-120000-source",
                check_state={"sha256": edit_flow.sha256_text(edit_flow.clean_abc(BASE_ABC)),
                             "match": True, "result": {"match": True}, "voices": "both",
                             "allow_tempo_change": False},
                allow_changes=False,
                baseline_state={"schema": "yue2-groove-baseline-v1",
                                "baseline": {"path": "/tmp/baseline"}},
                abc_temp=.7, abc_p=.9, abc_k=30, abc_rep=1.005, abc_win=100, abc_min=32,
                abc_max=4096, sem_temp=1.0, sem_p=.95, sem_k=100, sem_rep=1.2, sem_win=50,
                sem_min=200, sem_max=9000, device="cpu", dtype="float32", backend="torch",
                quantization="none", offload_ar=False, budget=24, ode_steps=32,
                vae_core_frames="auto", model="m-a-p/YuE2-3B", vae_choice="standard",
                vae_custom="", revision="", vae_revision="", offline=False)
    args.update(overrides)
    return [args[name] for name in (
        "style", "lyrics", "cot", "seed", "cfg_scale", "abc_text", "baseline_abc", "source_rel",
        "check_state", "allow_changes", "baseline_state",
        "abc_temp", "abc_p", "abc_k", "abc_rep", "abc_win", "abc_min", "abc_max",
        "sem_temp", "sem_p", "sem_k", "sem_rep", "sem_win", "sem_min", "sem_max",
        "device", "dtype", "backend", "quantization", "offload_ar", "budget", "ode_steps",
        "vae_core_frames", "model", "vae_choice", "vae_custom", "revision", "vae_revision",
        "offline")]


def test_edit_generate_guards(isolated_runs: Path, monkeypatch) -> None:
    with pytest.raises(gr.Error, match="cannot be used"):
        next(webui.edit_generate(*edit_generate_args(abc_text="   ")))
    with pytest.raises(gr.Error, match="Run CHECK INVARIANTS"):
        next(webui.edit_generate(*edit_generate_args(check_state={})))
    with pytest.raises(gr.Error, match="did not pass"):
        next(webui.edit_generate(*edit_generate_args(
            check_state={"sha256": edit_flow.sha256_text(edit_flow.clean_abc(BASE_ABC)),
                         "match": False, "result": {"differences": ["Vocal: sounding notes differ"]},
                         "voices": "both", "allow_tempo_change": False})))
    with pytest.raises(gr.Error, match="Freeze the baseline"):
        next(webui.edit_generate(*edit_generate_args(baseline_state=None)))
    with pytest.raises(gr.Error, match="Load a source work"):
        next(webui.edit_generate(*edit_generate_args(baseline_abc="", baseline_state=None)))
    # the override only lets a *failing* check through; freeze and the check stay mandatory
    with pytest.raises(gr.Error, match="Load a source work"):
        next(webui.edit_generate(*edit_generate_args(baseline_abc="", baseline_state=None,
                                                     allow_changes=True)))
    with pytest.raises(gr.Error, match="Freeze the baseline"):
        next(webui.edit_generate(*edit_generate_args(baseline_state=None, allow_changes=True)))
    with pytest.raises(gr.Error, match="Run CHECK INVARIANTS"):
        next(webui.edit_generate(*edit_generate_args(check_state={}, allow_changes=True)))
    monkeypatch.setattr(webui, "_get_pipe", lambda *a, **k: (object(), "note"))
    monkeypatch.setattr(webui, "_run_generation", fake_run_generation)
    failing = {"sha256": edit_flow.sha256_text(edit_flow.clean_abc(BASE_ABC)),
               "match": False,
               "result": {"match": False, "differences": ["Vocal: sounding notes differ"]},
               "voices": "both", "allow_tempo_change": False}
    yields = list(webui.edit_generate(*edit_generate_args(check_state=failing, allow_changes=True)))
    assert all(len(chunk) == 11 for chunk in yields)          # matches the 11 wired outputs
    audio, abc, files, status, *rest = yields[-1]
    assert len(rest) == 7                                      # 6 idle buttons + last run
    assert audio.endswith("audio.flac") and rest[0]["interactive"] is True
    last_run = rest[-1]
    assert Path(last_run, "edit_manifest.json").is_file()
    manifest = json.loads(Path(last_run, "edit_manifest.json").read_text(encoding="utf-8"))
    assert manifest["permitted"]["changes_override"] is True
    assert manifest["source"]["frozen"] is True               # never null on this path
    assert manifest["invariants"]["match"] is False           # the permitted difference is recorded
    assert manifest["abc"]["after_sha256"]


def test_edit_generate_happy_path_writes_a_manifest(isolated_runs: Path, monkeypatch) -> None:
    monkeypatch.setattr(webui, "_get_pipe", lambda *a, **k: (object(), "note"))
    monkeypatch.setattr(webui, "_run_generation", fake_run_generation)
    yields = list(webui.edit_generate(*edit_generate_args()))
    assert all(len(chunk) == 11 for chunk in yields)
    audio, result_abc, files, status, *rest = yields[-1]
    # the editor is not an output: only the separate RESULT ABC box receives song.abc
    assert audio.endswith("audio.flac") and result_abc == FakeSong.abc
    assert all(button["interactive"] is True for button in rest[:-1])   # 6 idle buttons
    assert "edit_manifest.json" in status
    last_run = rest[-1]
    manifest = json.loads(Path(last_run, "edit_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"]["rel"] == "20260901-120000-source"
    assert manifest["invariants"]["match"] is True
    assert manifest["abc"]["after_sha256"] == edit_flow.sha256_text(edit_flow.clean_abc(BASE_ABC))


def test_edit_generate_event_targets_the_result_box_not_the_editor(isolated_runs: Path) -> None:
    """Regression: GENERATE EDITED must never write back into EDITED ABC."""
    demo = webui.build_ui({"device": "cpu", "dtype": "float32", "model": "m-a-p/YuE2-3B",
                           "vae": "standard", "tab": 6, "status": ""})
    components = {c.elem_id: c for c in demo.blocks.values()
                  if getattr(c, "elem_id", None) in ("bb-edit-abc", "bb-edit-result-abc")}
    events = [f for f in demo.fns.values()
              if getattr(f.fn, "__name__", "") == "edit_generate"]
    assert len(events) == 1
    assert components["bb-edit-result-abc"] in events[0].outputs
    assert components["bb-edit-abc"] not in events[0].outputs


def test_edit_compare_requires_both_sides(isolated_runs: Path) -> None:
    with pytest.raises(gr.Error, match="Load a source work"):
        webui.edit_compare("", "/tmp/x")
    with pytest.raises(gr.Error, match="Generate an edited version"):
        webui.edit_compare("20260901-120000-source", "")


def test_edit_source_choices_skip_non_works(isolated_runs: Path) -> None:
    make_work(isolated_runs)
    (isolated_runs / "transcriptions").mkdir()
    (isolated_runs / "baselines").mkdir()
    (isolated_runs / "20260902-000000-run").mkdir()
    (isolated_runs / "20260902-000000-run" / "latent.npy").write_bytes(b"npy")
    scanned = {Path(p).name for p in webui._scan_runs()}
    assert scanned == {"20260902-000000-run", "20260901-120000-source"}
    assert "transcriptions" not in scanned and "baselines" not in scanned
    choices = webui.edit_choices()["choices"]
    assert any(rel == "20260901-120000-source" for _label, rel in choices)


# ── helpers / layout ─────────────────────────────────────────────────────

def test_score_panel_escapes_data_attributes() -> None:
    html_text = webui._score_panel('OP"EN', 'quote " and <tag> & more', elem_id="bb-x")
    assert 'data-bb-abc="OP&quot;EN"' in html_text
    assert 'data-bb-empty="quote &quot; and &lt;tag&gt; &amp; more"' in html_text
    assert "<tag>" not in html_text


def test_sampling_summary_mirrors_the_shared_sliders() -> None:
    text = webui._sampling_summary(.7, .9, 30, 1.005, 100, 32, 4096,
                                   1.0, .95, 100, 1.2, 50, 200, 9000)
    assert text.splitlines()[0].startswith("ABC")
    assert "max=4096" in text and text.splitlines()[1].startswith("SEM") and "max=9000" in text


# ── direct cover generation ──────────────────────────────────────────────

def cover_generate_args(**overrides):
    args = dict(style="English jazz", lyrics="[Verse]\nla", abc_text=BASE_ABC,
                task="melody-full", seed=831001, cfg_scale=0,
                abc_temp=.7, abc_p=.9, abc_k=30, abc_rep=1.005, abc_win=100, abc_min=32,
                abc_max=4096, sem_temp=1.0, sem_p=.95, sem_k=100, sem_rep=1.2, sem_win=50,
                sem_min=200, sem_max=9000, device="cpu", dtype="float32", backend="torch",
                quantization="none", offload_ar=False, budget=24, ode_steps=32,
                vae_core_frames="auto", model="m-a-p/YuE2-3B", vae_choice="standard",
                vae_custom="", revision="", vae_revision="", offline=False)
    args.update(overrides)
    return [args[name] for name in (
        "style", "lyrics", "abc_text", "task", "seed", "cfg_scale",
        "abc_temp", "abc_p", "abc_k", "abc_rep", "abc_win", "abc_min", "abc_max",
        "sem_temp", "sem_p", "sem_k", "sem_rep", "sem_win", "sem_min", "sem_max",
        "device", "dtype", "backend", "quantization", "offload_ar", "budget", "ode_steps",
        "vae_core_frames", "model", "vae_choice", "vae_custom", "revision", "vae_revision",
        "offline")]


def test_cover_generate_runs_directly_from_the_cover_tab(isolated_runs: Path,
                                                         monkeypatch) -> None:
    captured = {}

    def fake_run_generation(pipe, request, outdir, **kwargs):
        captured["cot"] = request.cot
        captured["abc"] = request.abc
        Path(outdir).mkdir(parents=True, exist_ok=True)
        (Path(outdir) / "audio.flac").write_bytes(b"fLaC")
        return FakeSong(), {"audio_seconds": 12.5, "truncated": False}, 1.0

    monkeypatch.setattr(webui, "_get_pipe", lambda *a, **k: (object(), "note"))
    monkeypatch.setattr(webui, "_run_generation", fake_run_generation)

    yields = list(webui.cover_generate(*cover_generate_args()))
    assert all(len(chunk) == 11 for chunk in yields)  # status + audio + result abc + files + 7 controls
    status, audio, result_abc, files, *buttons = yields[-1]
    assert captured["cot"] == "melody" and '"C"' not in captured["abc"]
    assert audio.endswith("audio.flac") and result_abc == FakeSong.abc
    assert len(buttons) == 7 and all(b["interactive"] is True for b in buttons)
    assert "run directory" in status

    # a full-score transcription keeps its chord symbols and uses cot=full
    list(webui.cover_generate(*cover_generate_args(task="full")))
    assert captured["cot"] == "full" and '"C"' in captured["abc"]

    with pytest.raises(gr.Error, match="score-conditioned"):
        next(webui.cover_generate(*cover_generate_args(abc_text="   ")))


# ── comparison builders ──────────────────────────────────────────────────

def test_edit_compare_feeds_the_source_and_last_run_to_the_builder(isolated_runs: Path,
                                                                   monkeypatch) -> None:
    make_work(isolated_runs)
    captured = {}

    def fake_make_comparison(paths_text, progress=None):
        captured["paths"] = paths_text
        return "/tmp/index.html", "<a>link</a>", "page: /tmp/index.html"

    monkeypatch.setattr(webui, "make_comparison", fake_make_comparison)
    result = webui.edit_compare("20260901-120000-source", "/tmp/runs/edit-1")
    source = str((isolated_runs / "20260901-120000-source").resolve())
    assert captured["paths"] == f"{source}\n/tmp/runs/edit-1"
    assert result == ("/tmp/index.html", "<a>link</a>", "page: /tmp/index.html")


def test_make_comparison_builds_a_real_listening_bundle(isolated_runs: Path) -> None:
    sources = []
    for name in ("20260901-120000-a", "20260901-120001-b"):
        d = isolated_runs / name
        d.mkdir()
        (d / "result.json").write_text(json.dumps({
            "status": "complete", "truncated": False, "sample_rate": 48000,
            "audio_seconds": 1.0}), encoding="utf-8")
        (d / "request.json").write_text(json.dumps({
            "id": name, "style": "test", "lyrics": "la", "cot": "full", "seed": 1}),
            encoding="utf-8")
        (d / "audio.flac").write_bytes(b"fLaC" + b"\x00" * 64)
        sources.append(str(d))

    html_path, link, status = webui.make_comparison("\n".join(sources))
    outdir = Path(html_path).parent
    assert (outdir / "index.html").is_file() and (outdir / "manifest.json").is_file()
    assert (outdir / "case-001").is_dir() and (outdir / "case-002").is_dir()
    assert "gradio_api/file=" in link and "cases" in status


def test_cover_transcribe_cancel_explains_the_stopped_worker(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "ref.wav"
    audio.write_bytes(b"RIFF")

    def fake_transcribe(audio_path, **kwargs):
        raise InterruptedError("Transcription cancelled")

    monkeypatch.setattr(webui.sheetsage_adapter, "transcribe", fake_transcribe)
    yields = list(webui.cover_transcribe(str(audio), "melody-full", 0, "m", "auto", "auto", "",
                                         "", True, False))
    status = yields[-1][2]
    assert "Cancelled" in status and "worker was stopped" in status
    assert list(webui.cover_transcribe(str(audio), "melody-full", 0, "m", "auto", "auto", "",
                                       "", False, False))[-1][2].count("worker was stopped") == 0
