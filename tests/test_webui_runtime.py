"""The webui kernel: settings as one value, the sampling pair, the job slot."""

from __future__ import annotations

from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")

webui = pytest.importorskip("yue2_groove.webui")
runtime = webui.runtime

RAIL = (
    "auto",
    "bfloat16",
    "torch",
    "none",
    False,
    24,
    32,
    "auto",
    "m-a-p/YuE2-3B",
    "standard",
    "",
    "",
    "",
    False,
)
SAMPLING = (0.7, 0.9, 30, 1.005, 100, 32, 4096, 1.0, 0.95, 100, 1.2, 50, 200, 9000)


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(runtime, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(runtime, "pick_device", lambda device: "cpu")
    monkeypatch.setattr(runtime, "resolve_vae", lambda choice, custom: (f"/vae/{choice}", choice))


def test_sampling_pair_knows_the_slider_order() -> None:
    abc, sem = runtime.sampling_pair(*SAMPLING)
    assert (abc.temperature, abc.top_k, abc.max_tokens) == (0.7, 30, 4096)
    assert (sem.temperature, sem.top_k, sem.max_tokens) == (1.0, 100, 9000)
    with pytest.raises(gr.Error, match="14 sampling values"):
        runtime.sampling_pair(*SAMPLING[:7])
    with pytest.raises(gr.Error, match="semantic phase"):
        runtime.sampling_pair(*SAMPLING[:7], "x", *SAMPLING[8:])


def test_settings_key_identifies_a_loaded_pipeline() -> None:
    base = runtime.RuntimeSettings(*RAIL)
    same = runtime.RuntimeSettings(*RAIL)
    assert base.key == same.key and base.cores is None
    assert runtime.RuntimeSettings(*RAIL[:7], "512", *RAIL[8:]).cores == 512
    # a rail change that matters changes the key; whitespace-only revisions do not
    assert runtime.RuntimeSettings(*RAIL[:6], 16, *RAIL[7:]).key != base.key
    assert runtime.RuntimeSettings(*RAIL[:9], "legacy", *RAIL[10:]).key != base.key
    assert runtime.RuntimeSettings(*RAIL[:11], "", "", False).key == base.key


def test_get_pipe_reuses_the_cache_for_equal_settings(monkeypatch) -> None:
    loads = []
    monkeypatch.setattr(
        runtime,
        "load_pipeline",
        lambda settings, progress=None: (loads.append(settings) or "pipe", "Loaded"),
    )
    monkeypatch.setattr(runtime, "_PIPE", None)
    monkeypatch.setattr(runtime, "_PIPE_KEY", None)
    settings = runtime.RuntimeSettings(*RAIL)
    assert runtime.get_pipe(settings) == ("pipe", "Loaded") and len(loads) == 1
    monkeypatch.setattr(runtime, "_PIPE", "pipe")
    monkeypatch.setattr(runtime, "_PIPE_KEY", settings.key)
    assert runtime.get_pipe(runtime.RuntimeSettings(*RAIL)) == ("pipe", "Model ready")
    assert len(loads) == 1
    runtime.get_pipe(runtime.RuntimeSettings(*RAIL[:6], 16, *RAIL[7:]))
    assert len(loads) == 2


def test_job_slot_is_exclusive_and_clears_a_stale_cancel() -> None:
    runtime.CANCEL.set()
    assert runtime.try_start_job() is True
    try:
        assert not runtime.CANCEL.is_set()
        assert runtime.try_start_job() is False  # a second job is refused, not queued
        assert "JOB RUNNING" in webui.frontend.busy_banner()
    finally:
        runtime.end_job()
    assert webui.frontend.busy_banner() == ""
    assert runtime.failure_text(InterruptedError("stopped")) == "Cancelled: stopped"
    assert runtime.failure_text(ValueError("bad"), "Decode") == "Decode failed: ValueError: bad"


def test_a_closed_batch_generator_releases_the_job_slot(monkeypatch) -> None:
    """Regression: the first yield sat between acquiring the slot and the try/finally."""
    monkeypatch.setattr(runtime, "get_pipe", lambda settings, progress=None: ("pipe", "ready"))
    gen = webui.generate_tab.batch_generate(
        '{"id": "a", "style": "s", "lyrics": "l"}',
        None,
        "",
        *SAMPLING,
        *RAIL,
        progress=lambda *a, **k: None,
    )
    first = next(gen)
    assert first[1] == "Starting batch…"
    gen.close()
    assert runtime.try_start_job(), "the slot stayed locked after the generator was closed"
    runtime.end_job()
