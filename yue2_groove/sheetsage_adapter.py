"""The only groove module that talks to SheetSage2.

SheetSage2 is *not* a pip package: it is a Transformers model with remote code
(``AutoModel.from_pretrained("m-a-p/SheetSage2", trust_remote_code=True)``) whose
pinned torch/transformers versions collide with YuE2's.  It therefore runs in its
own virtual environment and this module talks to it over a subprocess boundary:
``config.sheetsage_python()`` selects the interpreter, ``sheetsage_driver.py`` is
the script executed inside it, and audio/ABC/results are exchanged through files
and stdout progress lines.  No ``transformers`` import ever happens in the groove
process, so an unconfigured SheetSage2 cannot break the rest of the UI.

Kept deliberately narrow — the adapter counterpart of ``adapter.py``:

* ``resolve_python`` / ``probe``  — is the second environment usable?
* ``build_command`` / ``transcribe`` — run one transcription, with progress and
  cooperative cancel; returns the driver's ``result.json`` (``abc``, ``warnings``,
  ``abc_error``, ``output_dir``, ...).

The model *load* is part of the subprocess lifecycle: every transcription starts a
fresh interpreter, loads the model (weights stay in the shared Hugging Face cache)
and frees it on exit.  A long-lived server process is a later optimization; it
would change this module only.
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

from . import config

DRIVER = Path(__file__).resolve().parent / "sheetsage_driver.py"
PROGRESS_PREFIX = "@@PROGRESS "

TASKS = ("melody-vocal", "melody-full", "full")

Progress = Callable[[float | None, str], None]
Cancelled = Callable[[], bool]


class SheetsageNotConfigured(RuntimeError):
    """The SheetSage2 virtual environment is missing or misconfigured."""


class SheetsageFailed(RuntimeError):
    """The transcription subprocess failed."""


def resolve_python(explicit: str | None = None) -> str:
    """Absolute path of the SheetSage2 interpreter, or a clear configuration error."""
    value = (explicit or config.sheetsage_python()).strip()
    if not value:
        raise SheetsageNotConfigured(
            "SheetSage2 is not configured: create its separate virtual environment "
            "(see README, 'Cover from audio') and set "
            "YUE2_GROOVE_SHEETSAGE_PYTHON=/path/to/.venv-sheetsage2/bin/python")
    path = Path(value).expanduser()
    if not path.is_file():
        raise SheetsageNotConfigured(f"SheetSage2 python not found: {path}")
    if not os.access(path, os.X_OK):
        raise SheetsageNotConfigured(f"SheetSage2 python is not executable: {path}")
    return str(path)


def _python_fragment() -> str:
    return (
        "import json, importlib.metadata, importlib.util, sys;"
        "mods={n: importlib.util.find_spec(n) is not None for n in "
        "('torch','transformers','huggingface_hub','torchaudio')};"
        "versions={};"
        "[versions.__setitem__(n, importlib.metadata.version(n)) "
        "for n in ('torch','transformers','huggingface-hub') "
        "if mods.get(n)];"
        "print(json.dumps({'python': sys.version.split()[0], 'executable': sys.executable,"
        " 'packages': mods, 'versions': versions}))"
    )


def probe(python: str | None = None, *, timeout: float = 180) -> dict:
    """Check the SheetSage2 environment without loading any model.

    Returns ``{"python", "packages", "versions", "ffmpeg", "ok"}``; raises
    ``SheetsageNotConfigured`` when the interpreter itself is missing.
    """
    interpreter = resolve_python(python)
    try:
        result = subprocess.run([interpreter, "-c", _python_fragment()],
                                capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SheetsageFailed(f"Could not run the SheetSage2 python ({interpreter}): {exc}") from exc
    if result.returncode != 0:
        raise SheetsageFailed(f"SheetSage2 environment check failed: {result.stderr.strip()[-800:]}")
    try:
        info = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise SheetsageFailed(f"Unexpected check output from {interpreter}: {result.stdout[-300:]!r}") from exc
    packages = info.get("packages") or {}
    info["ffmpeg"] = shutil.which("ffmpeg")
    info["ok"] = bool(packages.get("torch") and packages.get("transformers"))
    return info


def format_probe(info: dict) -> str:
    lines = [f"python: {info.get('python')}  ({info.get('executable')})"]
    versions = info.get("versions") or {}
    packages = info.get("packages") or {}
    for name in ("torch", "transformers", "huggingface_hub", "torchaudio"):
        mark = "ok" if packages.get(name) else "missing"
        version = versions.get(name) or versions.get(name.replace("_", "-")) or ""
        lines.append(f"  {name}: {mark}{(' ' + version) if version else ''}")
    lines.append(f"ffmpeg: {info.get('ffmpeg') or 'not on PATH (SheetSage2 may need it)'}")
    lines.append("ready" if info.get("ok") else "not ready — install the SheetSage2 requirements first")
    return "\n".join(lines)


def build_command(audio_path, output_dir, *, python=None, task: str = "melody-full",
                  model: str | None = None, revision: str | None = None,
                  base_model: str | None = None, offline: bool = False,
                  device: str = "auto", dtype: str = "auto", preset: str = "default",
                  max_seconds: float | None = None, threads: int = 4) -> list[str]:
    """argv for one transcription; the driver resolves ``melody_only`` from ``task``."""
    if task not in TASKS:
        raise ValueError(f"task must be one of {', '.join(TASKS)}, got {task!r}")
    cmd = [resolve_python(python), str(DRIVER), str(Path(audio_path).expanduser()),
           "--output", str(Path(output_dir).expanduser()), "--task", task,
           "--model", model or config.default_sheetsage_model(),
           "--device", device, "--dtype", dtype, "--preset", preset]
    if revision:
        cmd += ["--revision", revision]
    if base_model:
        cmd += ["--base-model", base_model]
    if offline:
        cmd += ["--offline"]
    if max_seconds is not None:
        cmd += ["--max-seconds", str(float(max_seconds))]
    if threads:
        cmd += ["--threads", str(int(threads))]
    return cmd


def _unique_dir(directory: Path) -> Path:
    if not directory.exists():
        return directory
    for index in range(2, 1000):
        candidate = directory.with_name(f"{directory.name}-{index}")
        if not candidate.exists():
            return candidate
    raise SheetsageFailed(f"Could not find a free directory next to {directory}")


def _default_output_dir(audio: Path) -> Path:
    stem = "".join(c if c.isalnum() or c in "-_" else "-" for c in audio.stem)[:40].strip("-") or "audio"
    return config.transcriptions_dir() / f"{time.strftime('%Y%m%d-%H%M%S')}-{stem}"


def _terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        else:  # pragma: no cover - Windows
            process.terminate()
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        else:  # pragma: no cover - Windows
            process.kill()
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _read_failure(output_dir: Path, stderr_tail: str) -> str:
    record = {}
    try:
        record = json.loads((output_dir / "failure.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    message = str(record.get("error") or "").strip()
    warnings = record.get("warnings") or []
    if warnings:
        message += f" (warnings: {'; '.join(str(w) for w in warnings[:3])})"
    if not message:
        message = stderr_tail.strip()[-1200:] or "no error output"
    return message


def transcribe(audio_path, *, output_dir=None, task: str = "melody-full",
               melody_only: bool | None = None, model: str | None = None,
               revision: str | None = None, base_model: str | None = None,
               offline: bool = False, device: str = "auto", dtype: str = "auto",
               preset: str = "default", max_seconds: float | None = None,
               threads: int = 4, python: str | None = None,
               cancelled: Cancelled | None = None, progress: Progress | None = None,
               timeout: float | None = None) -> dict:
    """Transcribe one audio file and return the driver's result record.

    ``task`` selects vocal-melody / full-melody / full-score transcription; the
    optional ``melody_only`` flips between the melody tasks and ``full`` so callers
    that only think in the boolean can still be explicit.  ``progress`` receives
    ``(fraction_or_None, description)`` events; ``cancelled`` is polled while the
    subprocess runs and terminates it (raising ``InterruptedError``).
    """
    audio = Path(audio_path).expanduser()
    if not audio.is_file():
        raise SheetsageFailed(f"Audio file not found: {audio}")
    task = task if task in TASKS else "melody-full"
    if melody_only is True and task == "full":
        task = "melody-full"
    elif melody_only is False and task != "full":
        task = "full"

    output = Path(output_dir).expanduser() if output_dir else _default_output_dir(audio)
    output = _unique_dir(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_command(audio, output, python=python, task=task, model=model,
                        revision=revision, base_model=base_model, offline=offline,
                        device=device, dtype=dtype, preset=preset,
                        max_seconds=max_seconds, threads=threads)

    def report(fraction, description):
        if progress is not None:
            try:
                progress(fraction, description)
            except Exception:  # noqa: BLE001 — a UI callback must never kill the job
                pass

    report(None, "Starting the SheetSage2 environment…")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, bufsize=1, stdin=subprocess.DEVNULL,
                               start_new_session=(os.name == "posix"))
    lines: queue.Queue[str | None] = queue.Queue()
    stderr_lines: list[str] = []

    def pump_stdout():
        try:
            for line in process.stdout:
                lines.put(line)
        except (OSError, ValueError):
            pass
        finally:
            lines.put(None)

    def pump_stderr():
        try:
            for line in process.stderr:
                stderr_lines.append(line)
        except (OSError, ValueError):
            pass

    stdout_thread = threading.Thread(target=pump_stdout, daemon=True)
    stderr_thread = threading.Thread(target=pump_stderr, daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    deadline = time.monotonic() + timeout if timeout else None
    cancelled_at = None
    stream_done = False
    while True:
        try:
            line = lines.get(timeout=0.25)
        except queue.Empty:
            line = ""
        if line is None:
            stream_done = True
        elif line.startswith(PROGRESS_PREFIX):
            try:
                event = json.loads(line[len(PROGRESS_PREFIX):])
            except ValueError:
                event = {}
            done, total = event.get("done"), event.get("total")
            fraction = (done / total) if (isinstance(done, (int, float)) and total) else None
            report(fraction, "Transcribing…" + (f" {done}/{total}" if total else ""))
        if cancelled is not None and process.poll() is None and cancelled():
            cancelled_at = time.monotonic()
            _terminate(process)
            break
        if deadline is not None and time.monotonic() > deadline:
            _terminate(process)
            raise SheetsageFailed(
                f"Transcription timed out after {timeout:.0f}s; the audio may be too long for this machine")
        if process.poll() is not None and (stream_done or lines.empty()):
            break

    process.wait()
    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)
    if cancelled_at is not None:
        raise InterruptedError("Transcription cancelled")
    if process.returncode != 0:
        raise SheetsageFailed(
            f"SheetSage2 transcription failed (exit {process.returncode}): "
            f"{_read_failure(output, ''.join(stderr_lines))}")
    try:
        record = json.loads((output / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SheetsageFailed(
            f"SheetSage2 finished but wrote no readable result.json in {output}: "
            f"{''.join(stderr_lines).strip()[-600:]}") from exc
    record.setdefault("output_dir", str(output))
    record["task"], record["melody_only"] = task, bool(record.get("melody_only", task != "full"))
    record["command"] = cmd
    return record
