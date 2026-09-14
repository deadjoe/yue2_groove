"""Run durability: the pending marker, the fsync, and visible unfinished runs.

The 2026-09-13 panics could lose a run that had already been listened to because
``save_artifacts`` never fsynced and the service truncated its log on restart.
These tests pin the replacements: a run is marked pending until its artifacts are
flushed, a failed/cancelled run keeps the marker, and the Library shows it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")

library = pytest.importorskip("yue2_groove.library")
webui = pytest.importorskip("yue2_groove.webui")


class _Sampling:
    max_tokens = 64


class _Request:
    cot = "off"
    seed = 1


class _Song:
    def __init__(self, fail=None):
        self._fail = fail

    def save_artifacts(self, directory):
        directory = Path(directory)
        (directory / "audio.flac").write_bytes(b"fLaC")
        if self._fail:
            raise self._fail
        (directory / "result.json").write_text(
            json.dumps({"status": "complete", "audio_seconds": 1.0}), encoding="utf-8")
        return {"audio_seconds": 1.0, "truncated": False}


def _run(monkeypatch, tmp_path, *, fail=None):
    outdir = tmp_path / "runs" / "20260913-120000-Test"
    monkeypatch.setattr(webui.adapter, "generate", lambda *a, **k: _Song(fail))
    return webui._run_generation(
        object(), _Request(), outdir, abc_sampling=_Sampling(), semantic_sampling=_Sampling(),
        progress=lambda *a, **k: None, note=""), outdir


def _pending(outdir: Path) -> dict:
    return json.loads((outdir / webui.PENDING_FILE).read_text(encoding="utf-8"))


def test_a_finished_run_clears_its_pending_marker(tmp_path, monkeypatch) -> None:
    _result, outdir = _run(monkeypatch, tmp_path)
    assert (outdir / "audio.flac").is_file() and (outdir / "result.json").is_file()
    assert not (outdir / webui.PENDING_FILE).exists()   # cleared only after the flush


def test_a_failed_run_keeps_an_incomplete_marker(tmp_path, monkeypatch) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        _run(monkeypatch, tmp_path, fail=RuntimeError("boom"))
    pending = _pending(tmp_path / "runs" / "20260913-120000-Test")
    assert pending["status"] == "failed" and "boom" in pending["error"]
    assert pending["schema"] == "yue2-groove-pending-v1"


def test_a_cancelled_run_keeps_a_cancelled_marker(tmp_path, monkeypatch) -> None:
    with pytest.raises(InterruptedError):
        _run(monkeypatch, tmp_path, fail=InterruptedError("stopped"))
    assert _pending(tmp_path / "runs" / "20260913-120000-Test")["status"] == "cancelled"


def test_library_lists_and_labels_an_incomplete_run(tmp_path) -> None:
    runs = tmp_path / "runs"
    run = runs / "20260913-120000-Grand-piano"
    run.mkdir(parents=True)
    (run / "audio.flac").write_bytes(b"fLaC")
    (run / webui.PENDING_FILE).write_text(
        json.dumps({"status": "failed", "error": "MPS backend out of memory"}),
        encoding="utf-8")

    items = library.scan(runs)
    assert [item["rel"] for item in items] == ["20260913-120000-Grand-piano"]
    item = items[0]
    assert item["pending"] is True and item["status"] == "failed"
    assert item["kind"] == "incomplete" and "INCOMPLETE" in library.label(item)

    loaded_item, det = library.load(runs, item["rel"])
    html_text = library.render_info_html(loaded_item, det)
    assert "INCOMPLETE" in html_text and "MPS backend out of memory" in html_text


def test_library_lists_a_run_that_only_has_the_marker(tmp_path) -> None:
    runs = tmp_path / "runs"
    run = runs / "20260913-120000-Grand-piano"
    run.mkdir(parents=True)
    (run / webui.PENDING_FILE).write_text('{"status": "running"}', encoding="utf-8")
    items = library.scan(runs)
    assert len(items) == 1 and items[0]["pending"] is True
    assert "INCOMPLETE" in library.label(items[0])


def test_a_cancelled_pending_only_run_can_be_deleted(tmp_path) -> None:
    runs = tmp_path / "runs"
    run = runs / "20260914-013606-Grand_Piano_CFG15"
    run.mkdir(parents=True)
    (run / webui.PENDING_FILE).write_text('{"status": "cancelled"}', encoding="utf-8")
    assert [item["rel"] for item in library.scan(runs)] == ["20260914-013606-Grand_Piano_CFG15"]

    ok, message = library.delete(runs, ["20260914-013606-Grand_Piano_CFG15"])
    assert ok is True and "Deleted 1" in message, message
    assert not run.exists()


def test_delete_still_refuses_a_directory_without_artifacts(tmp_path) -> None:
    runs = tmp_path / "runs"
    (runs / "junk").mkdir(parents=True)
    ok, message = library.delete(runs, ["junk"])
    assert ok is False and "no known artifacts" in message
    assert (runs / "junk").is_dir()


def test_edit_and_cover_choices_skip_incomplete_runs(tmp_path, monkeypatch) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setattr(webui, "RUNS", runs)
    good = runs / "20260913-120000-Good"
    good.mkdir(parents=True)
    (good / "score.abc").write_text("X:1\nK:C\nC4|\n", encoding="utf-8")
    bad = runs / "20260913-130000-Bad"
    bad.mkdir(parents=True)
    (bad / webui.PENDING_FILE).write_text('{"status": "failed"}', encoding="utf-8")

    assert [value for _label, value in webui.edit_choices()["choices"]] == \
        ["20260913-120000-Good"]
    assert [value for _label, value in webui.cover_choices()["choices"]] == \
        ["20260913-120000-Good"]
    # the Library itself still shows the incomplete run
    assert {item["rel"] for item in library.scan(runs)} == \
        {"20260913-120000-Good", "20260913-130000-Bad"}


def test_fsync_fd_without_fcntl(tmp_path, monkeypatch):
    """Windows has no fcntl; durability must still fsync via os.fsync alone."""
    import os

    monkeypatch.setattr(webui, "fcntl", None)
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"ok")
    fd = os.open(path, os.O_RDONLY)
    try:
        webui._fsync_fd(fd)  # must not raise when fcntl is missing
    finally:
        os.close(fd)
