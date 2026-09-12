"""01 GENERATE → ALL MODES: text-only full/melody/off run + comparison bundle.

The model and the generation core are mocked; the handler's state machine is real:
mode order and ids, per-mode directories with input.json, run.json summary, retained
failures, cancel behavior and the automatic listening comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")

from yue2_groove import webui  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_runs(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(webui, "RUNS", tmp_path / "runs")
    (tmp_path / "runs").mkdir()
    return tmp_path / "runs"


class FakeSong:
    abc = "X:1"


def all_modes_args(**overrides):
    args = dict(style="English piano pop", lyrics="[Verse]\nla", seed=831001, cfg_scale=0,
                abc_text="", out_id="modes",
                abc_temp=.7, abc_p=.9, abc_k=30, abc_rep=1.005, abc_win=100, abc_min=32,
                abc_max=4096, sem_temp=1.0, sem_p=.95, sem_k=100, sem_rep=1.2, sem_win=50,
                sem_min=200, sem_max=9000, device="cpu", dtype="float32", backend="torch",
                quantization="none", offload_ar=False, budget=24, ode_steps=32,
                vae_core_frames="auto", model="m-a-p/YuE2-3B", vae_choice="standard",
                vae_custom="", revision="", vae_revision="", offline=False)
    args.update(overrides)
    return [args[name] for name in (
        "style", "lyrics", "seed", "cfg_scale", "abc_text", "out_id",
        "abc_temp", "abc_p", "abc_k", "abc_rep", "abc_win", "abc_min", "abc_max",
        "sem_temp", "sem_p", "sem_k", "sem_rep", "sem_win", "sem_min", "sem_max",
        "device", "dtype", "backend", "quantization", "offload_ar", "budget", "ode_steps",
        "vae_core_frames", "model", "vae_choice", "vae_custom", "revision", "vae_revision",
        "offline")]


def install_fakes(monkeypatch, state):
    def fake_run_generation(pipe, request, outdir, *, abc_sampling, semantic_sampling,
                            progress, note, extra_manifest=None):
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        state["calls"].append({"cot": request.cot, "id": request.id, "dir": str(outdir)})
        if progress:
            progress(0.5)
        if state.get("cancel_after") is not None and len(state["calls"]) >= state["cancel_after"]:
            webui._CANCEL.set()
        if state.get("fail_mode") == request.cot:
            raise RuntimeError(f"{request.cot} exploded")
        (outdir / "audio.flac").write_bytes(b"fLaC")
        (outdir / "result.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
        return FakeSong(), {"audio_seconds": 12.5, "truncated": False}, 1.5

    def fake_comparison(paths_text, progress=None):
        state["comparison_sources"] = [p for p in paths_text.splitlines() if p.strip()]
        return "/tmp/index.html", '<a href="/x">OPEN COMPARISON PAGE ↗</a>', \
            "page: /tmp/index.html\nneeds_review: 0"

    monkeypatch.setattr(webui, "_get_pipe", lambda *a, **k: (object(), "note"))
    monkeypatch.setattr(webui, "_run_generation", fake_run_generation)
    monkeypatch.setattr(webui, "make_comparison", fake_comparison)


def test_all_modes_runs_full_melody_off_and_builds_the_comparison(monkeypatch) -> None:
    state = {"calls": []}
    install_fakes(monkeypatch, state)

    yields = list(webui.generate_all_modes(*all_modes_args()))
    assert all(len(chunk) == 6 for chunk in yields)          # matches the 6 wired outputs
    status, files, link, *_idle = yields[-1]

    assert [call["cot"] for call in state["calls"]] == ["full", "melody", "off"]
    assert [call["id"] for call in state["calls"]] == ["modes_full", "modes_melody", "modes_off"]
    root = Path(state["calls"][0]["dir"]).parent
    assert root.name.endswith("-allmodes-modes")
    assert {Path(call["dir"]).name for call in state["calls"]} == {"full", "melody", "off"}
    assert len([f for f in files if f.endswith("audio.flac")]) == 3
    assert link == '<a href="/x">OPEN COMPARISON PAGE ↗</a>'
    assert "full" in status and "melody" in status and "off" in status
    assert "needs_review" in status
    assert state["comparison_sources"] == [call["dir"] for call in state["calls"]]

    summary = json.loads((root / "run.json").read_text(encoding="utf-8"))
    assert summary["action"] == "all-modes"
    assert [r["status"] for r in summary["results"]] == ["complete"] * 3
    for mode in ("full", "melody", "off"):
        assert json.loads((root / mode / "input.json").read_text(encoding="utf-8"))["cot"] == mode


def test_all_modes_retains_a_failed_mode_and_compares_the_rest(monkeypatch) -> None:
    state = {"calls": [], "fail_mode": "melody"}
    install_fakes(monkeypatch, state)

    yields = list(webui.generate_all_modes(*all_modes_args()))
    assert all(len(chunk) == 6 for chunk in yields)
    status, files, _link, *_idle = yields[-1]

    root = Path(state["calls"][0]["dir"]).parent
    summary = json.loads((root / "run.json").read_text(encoding="utf-8"))
    assert [r["status"] for r in summary["results"]] == ["complete", "failed", "complete"]
    failure = json.loads((root / "melody" / "failure.json").read_text(encoding="utf-8"))
    assert failure["type"] == "RuntimeError" and "exploded" in failure["error"]
    assert "melody" in status and "failed" in status and "exploded" in status
    assert state["comparison_sources"] == [str(root / "full"), str(root / "off")]


def test_all_modes_cancel_stops_between_modes(monkeypatch) -> None:
    state = {"calls": [], "cancel_after": 1}
    install_fakes(monkeypatch, state)

    yields = list(webui.generate_all_modes(*all_modes_args()))
    status, *_rest = yields[-1]

    assert [call["cot"] for call in state["calls"]] == ["full"]
    root = Path(state["calls"][0]["dir"]).parent
    summary = json.loads((root / "run.json").read_text(encoding="utf-8"))
    assert [r["mode"] for r in summary["results"]] == ["full", "melody"]
    assert [r["status"] for r in summary["results"]] == ["complete", "cancelled"]
    assert "cancelled" in status
    assert state.get("comparison_sources") is None


def test_all_modes_refuses_an_abc_input() -> None:
    with pytest.raises(gr.Error, match="text-only"):
        next(webui.generate_all_modes(*all_modes_args(abc_text="X:1\nT:\nrest")))


def test_all_modes_respects_the_running_lock() -> None:
    webui._RUNNING.acquire()
    try:
        yields = list(webui.generate_all_modes(*all_modes_args()))
        assert all(len(chunk) == 6 for chunk in yields)
        assert "already running" in yields[0][0]
    finally:
        webui._RUNNING.release()


def test_scan_batches_includes_all_modes_groups(isolated_runs: Path) -> None:
    group = isolated_runs / "20260901-120000-allmodes-modes"
    for mode in ("full", "melody", "off"):
        (group / mode).mkdir(parents=True)
        (group / mode / "result.json").write_text("{}", encoding="utf-8")
    (group / "run.json").write_text("{}", encoding="utf-8")
    choices = webui._scan_batches()
    assert [label for label, _path in choices] and "-allmodes-" in choices[0][0]
    filled = webui._fill_from_batch(choices[0][0])
    assert filled.splitlines() == [str(group / "full"), str(group / "melody"), str(group / "off")]
