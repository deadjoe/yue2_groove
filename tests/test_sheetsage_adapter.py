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
import time
from pathlib import Path

import pytest

from yue2_groove import config
from yue2_groove import sheetsage_adapter as adapter
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


def test_unconfigured_environment_leaves_no_output_directory(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("YUE2_GROOVE_SHEETSAGE_PYTHON", raising=False)
    audio = tmp_path / "reference.wav"
    audio.write_bytes(b"RIFF")
    with pytest.raises(adapter.SheetsageNotConfigured):
        adapter.transcribe(audio, output_dir=tmp_path / "nested" / "out")
    assert not (tmp_path / "nested").exists()


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


def test_transcriptions_dir_follows_the_runs_override_and_the_env(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("YUE2_GROOVE_TRANSCRIPTIONS", raising=False)
    assert config.transcriptions_dir(tmp_path / "runs") == tmp_path / "runs" / "transcriptions"
    monkeypatch.setenv("YUE2_GROOVE_TRANSCRIPTIONS", str(tmp_path / "custom"))
    assert config.transcriptions_dir(tmp_path / "runs") == (tmp_path / "custom").resolve()


# ── resident worker (--serve) ────────────────────────────────────────────

SERVE_STUB = """
    import json, os, pathlib, sys, time
    pid_file = os.environ.get("STUB_PID_FILE")
    if pid_file:
        pathlib.Path(pid_file).write_text(str(os.getpid()))
    print("@@READY " + json.dumps({"model": "stub", "device": "cpu", "dtype": "fp32"}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        op = request.get("op")
        if op == "stop":
            break
        if op == "ping":
            print("@@RESULT " + json.dumps({"ok": True, "pong": True}), flush=True)
            continue
        if os.environ.get("STUB_CRASH") == "1":
            print("boom", file=sys.stderr)
            sys.exit(3)
        if os.environ.get("STUB_SLEEP") == "1":
            time.sleep(60)
        out = pathlib.Path(request["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        print("@@PROGRESS " + json.dumps({"done": 1, "total": 2}), flush=True)
        (out / "result.json").write_text(json.dumps({
            "status": "complete", "task": request.get("task", "melody-full"),
            "melody_only": request.get("task") != "full", "abc": "X:1",
            "warnings": [], "output_dir": str(out)}))
        print("@@RESULT " + json.dumps({"ok": True, "output_dir": str(out), "abc": "X:1"}),
              flush=True)
    sys.exit(0)
"""


@pytest.fixture(autouse=True)
def _no_resident_worker():
    yield
    adapter.stop_worker()
    adapter._REAPER = None   # next warm start gets a fresh reaper with the current tick


def enable_warm(monkeypatch, tmp_path: Path, stub_body: str = SERVE_STUB) -> Path:
    stub = write_stub(tmp_path, stub_body)

    def fake_serve_command(*, python=None, **kwargs):
        return [sys.executable, str(stub)]
    monkeypatch.setattr(adapter, "build_serve_command", fake_serve_command)
    monkeypatch.setenv("YUE2_GROOVE_SHEETSAGE_KEEP_WARM", "1")
    return stub


def test_build_serve_command_shape(monkeypatch, tmp_path: Path) -> None:
    fake_python = tmp_path / "python"
    fake_python.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("YUE2_GROOVE_SHEETSAGE_PYTHON", str(fake_python))
    cmd = adapter.build_serve_command(model="m-a-p/SheetSage2", device="cpu", dtype="fp32",
                                      offline=True)
    assert cmd[:3] == [str(fake_python), str(adapter.DRIVER), "--serve"]
    assert "--task" not in cmd                       # tasks travel in each request
    assert cmd[cmd.index("--model") + 1] == "m-a-p/SheetSage2" and "--offline" in cmd


def test_warm_worker_is_reused_and_stoppable(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "ref.wav"
    audio.write_bytes(b"RIFF")
    enable_warm(monkeypatch, tmp_path)
    pid_file = tmp_path / "worker.pid"
    monkeypatch.setenv("STUB_PID_FILE", str(pid_file))
    progress = []

    first = adapter.transcribe(audio, output_dir=tmp_path / "one", task="melody-full",
                               progress=lambda value, text: progress.append(value))
    status = adapter.worker_status()
    assert first["abc"] == "X:1" and first["worker"]["resident"] is True
    assert status is not None and status["pid"] == int(pid_file.read_text())
    assert [value for value in progress if value is not None] == [0.5]

    second = adapter.transcribe(audio, output_dir=tmp_path / "two", task="melody-full")
    assert adapter.worker_status()["pid"] == status["pid"]        # same resident process
    assert second["task"] == "melody-full" and second["output_dir"].endswith("two")

    assert adapter.stop_worker() is True
    assert adapter.worker_status() is None
    assert adapter.stop_worker() is False


def test_warm_worker_idle_expiry_restarts(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "ref.wav"
    audio.write_bytes(b"RIFF")
    enable_warm(monkeypatch, tmp_path)
    pid_file = tmp_path / "worker.pid"
    monkeypatch.setenv("STUB_PID_FILE", str(pid_file))
    monkeypatch.setenv("YUE2_GROOVE_SHEETSAGE_IDLE_SECONDS", "0.01")

    adapter.transcribe(audio, output_dir=tmp_path / "one")
    first_pid = adapter.worker_status()["pid"]
    time.sleep(0.05)
    adapter.transcribe(audio, output_dir=tmp_path / "two")
    assert adapter.worker_status()["pid"] != first_pid


def test_warm_worker_cancel_terminates_it(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "ref.wav"
    audio.write_bytes(b"RIFF")
    enable_warm(monkeypatch, tmp_path)
    monkeypatch.setenv("STUB_SLEEP", "1")
    checks = {"count": 0}

    def cancelled() -> bool:
        checks["count"] += 1
        return checks["count"] > 1

    with pytest.raises(InterruptedError, match="cancelled"):
        adapter.transcribe(audio, output_dir=tmp_path / "out", cancelled=cancelled)
    assert adapter.worker_status() is None


def test_warm_worker_crash_falls_back_cleanly(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "ref.wav"
    audio.write_bytes(b"RIFF")
    enable_warm(monkeypatch, tmp_path)
    monkeypatch.setenv("STUB_CRASH", "1")
    with pytest.raises(adapter.SheetsageFailed, match="worker"):
        adapter.transcribe(audio, output_dir=tmp_path / "out")
    assert adapter.worker_status() is None


def test_warm_off_uses_the_one_shot_path(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "ref.wav"
    audio.write_bytes(b"RIFF")
    monkeypatch.delenv("YUE2_GROOVE_SHEETSAGE_KEEP_WARM", raising=False)
    monkeypatch.setattr(adapter, "build_serve_command",
                        lambda **kwargs: pytest.fail("warm worker must not start"))
    captured: dict = {}
    patch_command(monkeypatch, write_stub(tmp_path, SUCCESS_STUB), captured)
    record = adapter.transcribe(audio, output_dir=tmp_path / "out")
    assert record["abc"] == "X:1" and adapter.worker_status() is None


def test_driver_serve_loop(monkeypatch, tmp_path: Path, capsys) -> None:
    import io
    import json as jsonlib
    from types import SimpleNamespace

    calls = []
    monkeypatch.setattr(driver, "load_model", lambda args: ("MODEL", "cpu", "fp32"))
    monkeypatch.setattr(driver, "provenance", lambda args, model, device, dtype: {"model": "m"})

    def fake_transcribe(model, args, device, dtype, provenance, request):
        calls.append(request)
        out = Path(request["output_dir"])
        out.mkdir(parents=True)
        (out / "result.json").write_text(jsonlib.dumps({
            "abc": "X:1", "status": "complete", "task": request.get("task"),
            "melody_only": True, "output_dir": str(out)}), encoding="utf-8")
        return {"abc": "X:1", "status": "complete", "output_dir": str(out),
                "task": request.get("task"), "melody_only": True, "warnings": []}

    monkeypatch.setattr(driver, "run_transcription", fake_transcribe)
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    payload = "\n".join([
        jsonlib.dumps({"op": "ping"}),
        jsonlib.dumps({"op": "transcribe", "audio": str(audio),
                       "output_dir": str(tmp_path / "out"), "task": "melody-vocal"}),
        jsonlib.dumps({"op": "stop"}),
    ]) + "\n"
    monkeypatch.setattr(driver.sys, "stdin", io.StringIO(payload))
    args = SimpleNamespace(model="m", revision=None, base_model=None, offline=False,
                           device="cpu", dtype="fp32", preset="default", threads=4,
                           task="melody-full", max_seconds=None)

    assert driver.serve(args) == 0
    lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith("@@READY ") for line in lines)
    replies = [jsonlib.loads(line[len("@@RESULT "):]) for line in lines
               if line.startswith("@@RESULT ")]
    assert replies[0]["pong"] is True and replies[1]["ok"] is True
    assert calls == [{"op": "transcribe", "audio": str(audio),
                      "output_dir": str(tmp_path / "out"), "task": "melody-vocal"}]


def test_warm_config_flags(monkeypatch) -> None:
    monkeypatch.delenv("YUE2_GROOVE_SHEETSAGE_KEEP_WARM", raising=False)
    assert config.sheetsage_keep_warm() is False
    monkeypatch.setenv("YUE2_GROOVE_SHEETSAGE_KEEP_WARM", "true")
    assert config.sheetsage_keep_warm() is True
    monkeypatch.setenv("YUE2_GROOVE_SHEETSAGE_KEEP_WARM", "0")
    assert config.sheetsage_keep_warm() is False
    monkeypatch.delenv("YUE2_GROOVE_SHEETSAGE_IDLE_SECONDS", raising=False)
    assert config.sheetsage_idle_seconds() == 900.0
    monkeypatch.setenv("YUE2_GROOVE_SHEETSAGE_IDLE_SECONDS", "no")
    assert config.sheetsage_idle_seconds() == 900.0
    monkeypatch.setenv("YUE2_GROOVE_SHEETSAGE_IDLE_SECONDS", "30")
    assert config.sheetsage_idle_seconds() == 30.0


def test_idle_worker_is_reaped_in_the_background(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "ref.wav"
    audio.write_bytes(b"RIFF")
    enable_warm(monkeypatch, tmp_path)
    monkeypatch.setenv("YUE2_GROOVE_SHEETSAGE_IDLE_SECONDS", "0.05")
    monkeypatch.setattr(adapter, "_REAPER_TICK", 0.02)

    adapter.transcribe(audio, output_dir=tmp_path / "one")
    assert adapter.worker_status() is not None
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and adapter.worker_status() is not None:
        time.sleep(0.02)
    assert adapter.worker_status() is None            # freed without another transcription


def test_reaper_never_stops_a_busy_worker(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "ref.wav"
    audio.write_bytes(b"RIFF")
    enable_warm(monkeypatch, tmp_path)
    adapter.transcribe(audio, output_dir=tmp_path / "one")
    worker = adapter._WORKER
    assert worker is not None
    monkeypatch.setenv("YUE2_GROOVE_SHEETSAGE_IDLE_SECONDS", "0.001")
    with worker._lock:
        time.sleep(0.01)
        assert adapter._reap_idle_worker() is False
    assert adapter.worker_status() is not None
