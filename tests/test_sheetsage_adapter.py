"""``yue2_groove.sheetsage_adapter`` and ``sheetsage_driver``, without SheetSage2.

The adapter never imports ``transformers``; it builds a command line and runs the
driver in the separate SheetSage2 environment.  The tests here substitute a tiny
real subprocess for the driver, so the Popen plumbing (progress parsing, cancel,
failure records) is exercised for real.
"""
from __future__ import annotations

import ast
import json
import os
import stat
import sys
import textwrap
from pathlib import Path

import pytest

from yue2_groove import config, sheetsage_adapter as adapter
from yue2_groove import sheetsage_driver as driver

PACKAGE = Path(adapter.__file__).resolve().parent


def write_stub(tmp_path: Path, body: str) -> Path:
    stub = tmp_path / "driver_stub.py"
    stub.write_text(textwrap.dedent(body), encoding="utf-8")
    return stub


def patch_command(monkeypatch, stub: Path, captured: dict):
    def fake_build_command(audio_path, output_dir, **kwargs):
        captured.update({"audio": str(audio_path), "output": str(output_dir), **kwargs})
        return [sys.executable, str(stub), str(output_dir)]
    monkeypatch.setattr(adapter, "build_command", fake_build_command)


SUCCESS_STUB = """
    import json, pathlib, sys
    out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    print("@@PROGRESS " + json.dumps({"done": 1, "total": 4}), flush=True)
    print("@@PROGRESS " + json.dumps({"done": 3, "total": 4}), flush=True)
    (out / "result.json").write_text(json.dumps({
        "status": "complete", "abc": "X:1", "warnings": ["low confidence"],
        "abc_error": None, "melody_only": True, "score_path": str(out / "score.abc")}))
    print("Saved score.abc")
"""


def test_build_command_shape(monkeypatch, tmp_path: Path) -> None:
    fake_python = tmp_path / "python"
    fake_python.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("YUE2_GROOVE_SHEETSAGE_PYTHON", str(fake_python))
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"RIFF")

    cmd = adapter.build_command(
        audio, tmp_path / "out", task="melody-vocal", model="m-a-p/SheetSage2",
        revision="abc123", base_model="/mert", offline=True, device="cpu", dtype="fp32",
        preset="paper", max_seconds=30, threads=2)
    assert cmd[0] == str(fake_python)
    assert cmd[1] == str(adapter.DRIVER) and adapter.DRIVER.is_file()
    assert "--task" in cmd and cmd[cmd.index("--task") + 1] == "melody-vocal"
    for flag, value in (("--model", "m-a-p/SheetSage2"), ("--revision", "abc123"),
                        ("--base-model", "/mert"), ("--device", "cpu"), ("--dtype", "fp32"),
                        ("--preset", "paper"), ("--max-seconds", "30.0"), ("--threads", "2")):
        assert cmd[cmd.index(flag) + 1] == value
    assert "--offline" in cmd
    with pytest.raises(ValueError, match="task"):
        adapter.build_command(audio, tmp_path / "out", task="nope")


def test_resolve_python_reports_configuration_errors(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("YUE2_GROOVE_SHEETSAGE_PYTHON", raising=False)
    with pytest.raises(adapter.SheetsageNotConfigured, match="YUE2_GROOVE_SHEETSAGE_PYTHON"):
        adapter.resolve_python()
    with pytest.raises(adapter.SheetsageNotConfigured, match="not found"):
        adapter.resolve_python(str(tmp_path / "missing-python"))
    plain = tmp_path / "not-executable"
    plain.write_text("", encoding="utf-8")
    plain.chmod(0o644)
    if os.access(plain, os.X_OK):  # running as root ignores the mode bit
        pytest.skip("cannot test a non-executable file as this user")
    with pytest.raises(adapter.SheetsageNotConfigured, match="not executable"):
        adapter.resolve_python(str(plain))


def test_transcribe_streams_progress_and_reads_result(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "reference.wav"
    audio.write_bytes(b"RIFF")
    captured: dict = {}
    patch_command(monkeypatch, write_stub(tmp_path, SUCCESS_STUB), captured)
    seen: list[tuple] = []

    record = adapter.transcribe(audio, output_dir=tmp_path / "out", task="melody-full",
                                progress=lambda value, text: seen.append((value, text)),
                                model="/local/SheetSage2", device="cpu", dtype="fp32")

    assert record["abc"] == "X:1" and record["warnings"] == ["low confidence"]
    assert record["task"] == "melody-full" and record["melody_only"] is True
    assert record["output_dir"] == str(tmp_path / "out")
    assert [v for v, _ in seen if v is not None] == [1 / 4, 3 / 4]
    assert captured["task"] == "melody-full" and captured["device"] == "cpu"


def test_melody_only_boolean_maps_onto_the_task(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "reference.wav"
    audio.write_bytes(b"RIFF")
    captured: dict = {}
    patch_command(monkeypatch, write_stub(tmp_path, SUCCESS_STUB), captured)

    adapter.transcribe(audio, output_dir=tmp_path / "a", task="melody-full", melody_only=False)
    assert captured["task"] == "full"
    adapter.transcribe(audio, output_dir=tmp_path / "b", task="full", melody_only=True)
    assert captured["task"] == "melody-full"


def test_transcribe_failure_uses_failure_record(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "reference.wav"
    audio.write_bytes(b"RIFF")
    stub = write_stub(tmp_path, """
        import json, pathlib, sys
        out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
        (out / "failure.json").write_text(json.dumps({
            "status": "failed", "type": "DriverError",
            "error": "Transcription produced no usable ABC: no decoded beats",
            "warnings": ["short input"]}))
        print("boom", file=sys.stderr)
        sys.exit(2)
    """)
    captured: dict = {}
    patch_command(monkeypatch, stub, captured)

    with pytest.raises(adapter.SheetsageFailed) as excinfo:
        adapter.transcribe(audio, output_dir=tmp_path / "out")
    assert "no usable ABC" in str(excinfo.value) and "short input" in str(excinfo.value)


def test_transcribe_without_result_json_fails_clearly(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "reference.wav"
    audio.write_bytes(b"RIFF")
    captured: dict = {}
    patch_command(monkeypatch, write_stub(tmp_path, "import sys; sys.exit(0)"), captured)
    with pytest.raises(adapter.SheetsageFailed, match="result.json"):
        adapter.transcribe(audio, output_dir=tmp_path / "out")


def test_transcribe_missing_audio_is_refused_before_spawning(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(adapter, "build_command",
                        lambda *a, **k: pytest.fail("must not build a command for a missing file"))
    with pytest.raises(adapter.SheetsageFailed, match="Audio file not found"):
        adapter.transcribe(tmp_path / "nope.wav")


def test_cancel_terminates_the_driver(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "reference.wav"
    audio.write_bytes(b"RIFF")
    stub = write_stub(tmp_path, """
        import time
        print("@@PROGRESS " + '{"done": 1, "total": 10}', flush=True)
        time.sleep(60)
    """)
    captured: dict = {}
    patch_command(monkeypatch, stub, captured)
    checks = {"count": 0}

    def cancelled() -> bool:
        checks["count"] += 1
        return checks["count"] > 1

    with pytest.raises(InterruptedError, match="cancelled"):
        adapter.transcribe(audio, output_dir=tmp_path / "out", cancelled=cancelled)
    assert checks["count"] >= 2


def test_output_directory_is_never_reused(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "reference.wav"
    audio.write_bytes(b"RIFF")
    captured: dict = {}
    patch_command(monkeypatch, write_stub(tmp_path, SUCCESS_STUB), captured)
    existing = tmp_path / "out"
    existing.mkdir()
    (existing / "keep.txt").write_text("keep", encoding="utf-8")
    adapter.transcribe(audio, output_dir=existing)
    assert captured["output"].endswith("out-2")
    assert (existing / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_probe_and_format(monkeypatch, tmp_path: Path) -> None:
    fake_python = tmp_path / "python"
    fake_python.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)
    payload = {"python": "3.11.9", "executable": str(fake_python),
               "packages": {"torch": True, "transformers": True, "huggingface-hub": True},
               "versions": {"torch": "2.8.0", "transformers": "4.45.2", "huggingface-hub": "0.36.0"}}

    class Completed:
        returncode = 0
        stdout = json.dumps(payload) + "\n"
        stderr = ""

    monkeypatch.setattr(adapter.subprocess, "run", lambda *a, **k: Completed())
    monkeypatch.setattr(adapter.shutil, "which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)
    info = adapter.probe(str(fake_python))
    assert info["ok"] is True and info["ffmpeg"] == "/usr/bin/ffmpeg"
    text = adapter.format_probe(info)
    assert "transformers: ok 4.45.2" in text and "huggingface_hub: ok 0.36.0" in text
    assert "ffmpeg" in text and text.endswith("ready")

    class Broken:
        returncode = 1
        stdout = ""
        stderr = "ImportError: no torch"

    monkeypatch.setattr(adapter.subprocess, "run", lambda *a, **k: Broken())
    with pytest.raises(adapter.SheetsageFailed, match="no torch"):
        adapter.probe(str(fake_python))


def test_driver_task_settings_and_melody_only_interface() -> None:
    prompts, melody_only = driver.task_settings("melody-vocal")
    assert prompts[-1] == "melody_vocal" and melody_only is True
    prompts, melody_only = driver.task_settings("melody-full")
    assert prompts[-1] == "melody_full" and melody_only is True
    prompts, melody_only = driver.task_settings("full")
    assert prompts[-2:] == ["chord_full", "melody_full"] and melody_only is False
    with pytest.raises(driver.DriverError, match="Unknown transcription task"):
        driver.task_settings("other")

    class Missing:
        def transcribe(self, audio, **kwargs): ...
    with pytest.raises(driver.DriverError, match="does not expose melody_only"):
        driver.check_melody_only_interface(Missing())

    class Positional:
        def transcribe(self, audio, melody_only, /): ...
    with pytest.raises(driver.DriverError, match="does not expose melody_only"):
        driver.check_melody_only_interface(Positional())

    class Ok:
        def transcribe(self, audio, *, melody_only=False): ...
    driver.check_melody_only_interface(Ok())  # must not raise


def test_driver_progress_callback_prints_events(capsys) -> None:
    callback = driver.progress_callback()
    callback(2, 5)
    callback(done=5, total=5)
    callback("metadata", 1, 3)
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("@@PROGRESS ")]
    events = [json.loads(line[len("@@PROGRESS "):]) for line in lines]
    assert {"done": 2, "total": 5} in events and {"done": 5, "total": 5} in events
    assert {"done": 1, "total": 3, "stage": "metadata"} in events


def test_transformers_and_torch_are_imported_lazily() -> None:
    """Architecture guard: only the driver, and only inside ``run``."""
    for name, expected_inside in (("sheetsage_adapter.py", False), ("sheetsage_driver.py", True)):
        tree = ast.parse((PACKAGE / name).read_text(encoding="utf-8"))
        top_level = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                top_level.add((node.module or "").split(".")[0])
        assert "transformers" not in top_level, name
        assert "torch" not in top_level, name
        if not expected_inside:
            assert "yue2" not in top_level, name


def _imported_modules(path: Path) -> set[str]:
    modules = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module or "")
    return modules


def test_yue2_is_imported_only_by_adapter_py() -> None:
    """The UI/skill modules must go through ``adapter.py`` for every yue2 call."""
    offenders = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        names = {name for name in _imported_modules(path)
                 if name == "yue2" or name.startswith("yue2.")}
        if names:
            offenders[path.name] = names
    assert set(offenders) == {"adapter.py"}, offenders


def test_webui_never_imports_transformers() -> None:
    assert not any(name == "transformers" or name.startswith("transformers.")
                   for name in _imported_modules(PACKAGE / "webui.py"))


def test_sheetsage_env_does_not_leak_into_yue2_config(monkeypatch) -> None:
    monkeypatch.setenv("YUE2_GROOVE_SHEETSAGE_MODEL", "/models/SheetSage2")
    assert config.default_sheetsage_model() == "/models/SheetSage2"
    monkeypatch.delenv("YUE2_GROOVE_SHEETSAGE_MODEL", raising=False)
    monkeypatch.setenv("YUE2_GROOVE_MODELS", "/models")
    assert config.default_sheetsage_model() == "m-a-p/SheetSage2"  # subdir absent → hub id
