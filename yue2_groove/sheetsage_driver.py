#!/usr/bin/env python3
"""Standalone SheetSage2 transcription driver (runs in the SheetSage2 env).

This file is executed by the *SheetSage2* virtual environment's Python, which
pins different torch/transformers versions than YuE2.  ``yue2_groove`` never
imports ``transformers`` in its own process; ``sheetsage_adapter.py`` builds the
command line and runs this script with ``subprocess``.  Keep the two modules in
sync — together they are the SheetSage2↔groove boundary (the counterpart of
``adapter.py`` for ``yue2``).

Only the standard library is imported at module load, so tests can import this
file in the YuE2 environment without touching the model code.

Interface
---------
stdout: one progress event per line, ``@@PROGRESS {"done": .., "total": ..}``;
        everything else on stdout/stderr is diagnostic text.
output: ``<output_dir>/result.json`` — the contract read by the adapter::

    {"task", "melody_only", "prompts", "abc", "abc_error", "warnings",
     "num_events", "score_path", "output_dir", "device", "dtype", ...}

Failure: ``<output_dir>/failure.json`` plus a nonzero exit status.  The adapter
turns that record into a user-facing error message.

Adapted from the YuE2 skill helper ``skills/yue2-music/scripts/transcribe.py``
(Apache 2.0, Copyright (c) 2026 the YuE2 authors); see NOTICE.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import re
import sys
from pathlib import Path

PROGRESS_PREFIX = "@@PROGRESS "

TASKS = ("melody-vocal", "melody-full", "full")
BASE_PROMPTS = ("timestamp", "downbeat_meter", "structure", "key")

# Best-effort import of the vendored portable checker when this file is run from
# the package directory (python yue2_groove/sheetsage_driver.py).  Absent when
# the driver is copied elsewhere; the checks that need it are skipped then.
try:  # pragma: no cover - exercised only in the SheetSage2 environment
    from vendor.abc_tools import parse_abc  # type: ignore
except ImportError:  # pragma: no cover
    try:
        from yue2_groove.vendor.abc_tools import parse_abc  # type: ignore
    except ImportError:
        parse_abc = None


class DriverError(RuntimeError):
    """A transcription failure the adapter can present verbatim."""


def task_settings(task: str) -> tuple[list[str], bool]:
    """(prompts, melody_only) for a task name.

    ``melody-vocal`` keeps only the vocal melody, ``melody-full`` keeps the vocal
    and instrumental melodies (both without chord symbols / chord playback), and
    ``full`` is the default chord + melody transcription.
    """
    if task == "full":
        return list(BASE_PROMPTS) + ["chord_full", "melody_full"], False
    if task == "melody-full":
        return list(BASE_PROMPTS) + ["melody_full"], True
    if task == "melody-vocal":
        return list(BASE_PROMPTS) + ["melody_vocal"], True
    raise DriverError(f"Unknown transcription task {task!r}; use one of {', '.join(TASKS)}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def fresh_directory(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def progress_callback():
    """A callback safe for any signature the model uses; prints one stdout line."""
    def report(*args, **kwargs):
        numbers, stage = [], None
        for value in list(args) + list(kwargs.values()):
            if isinstance(value, bool):
                continue
            if isinstance(value, str):
                if stage is None and len(value) <= 40:
                    stage = value
                continue
            if isinstance(value, (int, float)):
                numbers.append(int(value) if float(value).is_integer() else round(float(value), 4))
        event = {"done": numbers[0] if numbers else None}
        if len(numbers) > 1:
            event["total"] = numbers[1]
        if stage:
            event["stage"] = stage
        print(PROGRESS_PREFIX + json.dumps(event, ensure_ascii=False), flush=True)
    return report


def check_melody_only_interface(model) -> None:
    """Refuse to guess: ``melody_only`` must exist as a keyword on ``transcribe``."""
    try:
        parameter = inspect.signature(model.transcribe).parameters.get("melody_only")
    except (TypeError, ValueError) as exc:
        raise DriverError("Cannot verify SheetSage2's melody_only interface; refresh the model "
                          "code to a reviewed revision exposing melody_only explicitly") from exc
    if parameter is None or parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
        raise DriverError("This SheetSage2 revision does not expose melody_only; refresh the model "
                          "and remote code to a reviewed revision supporting melody_only=True")


def verify_melody_only_abc(abc: str) -> None:
    """A melody-only transcription must not contain quoted chord symbols."""
    if parse_abc is None:
        return
    try:
        score = parse_abc(abc)
    except ValueError as exc:
        raise DriverError(f"Melody-only transcription did not parse as native ABC: {exc}") from exc
    if any(voice.chords for voice in score.voices.values()):
        raise DriverError("Melody transcription contains unexpected chord symbols")


def snapshot_hashes(snapshot: Path) -> dict:
    files = {}
    if snapshot.is_dir():
        for path in sorted(snapshot.iterdir()):
            if path.is_file() and (path.suffix in {".py", ".safetensors"} or path.name == "config.json"):
                files[path.name] = sha256(path)
    return files


def run(args) -> dict:
    audio = Path(args.audio).expanduser()
    if not audio.is_file():
        raise DriverError(f"Audio file not found: {audio}")
    if args.max_seconds is not None and args.max_seconds <= 0:
        raise DriverError("--max-seconds must be positive and explicitly crops the input")
    prompts, melody_only = task_settings(args.task)
    output = fresh_directory(Path(args.output).expanduser())

    write_json(output / "input.json", {
        "source_name": audio.name, "source_audio_sha256": sha256(audio),
        "model": args.model, "revision": args.revision, "offline": args.offline,
        "base_model_path": args.base_model, "prompts": prompts, "task": args.task,
        "melody_only": melody_only, "preset": args.preset, "max_seconds": args.max_seconds,
        "device": args.device, "dtype": args.dtype,
    })
    result_record = {
        "status": "failed", "task": args.task, "melody_only": melody_only, "prompts": prompts,
        "abc": None, "abc_error": None, "warnings": [], "num_events": None,
        "output_dir": str(output), "source_audio": str(audio), "score_path": None,
    }
    try:
        import torch  # noqa: PLC0415  (deliberately late: this is the SheetSage2 env)
        from transformers import AutoModel  # noqa: PLC0415

        device = args.device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        dtype = args.dtype
        if dtype == "auto":
            dtype = "bf16" if device == "cuda" else "fp32"
        result_record["device"], result_record["dtype"] = device, dtype
        if args.threads:
            torch.set_num_threads(args.threads)

        loader = dict(trust_remote_code=True, local_files_only=bool(args.offline))
        if args.revision:
            loader.update(revision=args.revision, code_revision=args.revision)
        if args.base_model:
            loader["base_model_path"] = args.base_model
        model = AutoModel.from_pretrained(args.model, **loader).eval().to(device)
        if melody_only:
            check_melody_only_interface(model)
        snapshot = Path(getattr(model, "_source_snapshot", args.model))
        write_json(output / "model_provenance.json", {
            "model": args.model, "requested_revision": args.revision,
            "config": model.config.to_dict(), "snapshot_sha256": snapshot_hashes(snapshot),
            "packages": {name: importlib.metadata.version(name)
                         for name in ("torch", "transformers", "huggingface-hub")},
        })

        options = {"melody_only": True} if melody_only else {}
        result = model.transcribe(
            str(audio), output_dir=str(output), prompts=prompts,
            dtype=dtype, preset=args.preset, max_seconds=args.max_seconds,
            progress=progress_callback(), **options,
        )
        if not isinstance(result, dict):
            raise DriverError(f"SheetSage2 returned {type(result).__name__}, expected a dict")
        abc = result.get("abc")
        abc_error = result.get("abc_error")
        warnings = list(result.get("warnings") or [])
        result_record.update({"abc": abc, "abc_error": abc_error, "warnings": warnings,
                              "num_events": result.get("num_events")})
        if abc_error or not abc:
            write_json(output / "failure.json", {
                "status": "failed", "type": "DriverError",
                "error": f"Transcription produced no usable ABC: {abc_error}",
                "abc_error": abc_error, "warnings": warnings})
            write_json(output / "result.json", result_record)
            raise DriverError(f"Transcription produced no usable ABC: {abc_error}")
        if melody_only:
            verify_melody_only_abc(abc)
        score_path = output / "score.abc"
        if not score_path.is_file():
            # Keep the contract even when the model implementation only returned
            # the text: write the score the adapter is going to display.
            score_path.write_text(abc, encoding="utf-8")
        result_record.update({"status": "complete", "score_path": str(score_path)})
        write_json(output / "result.json", result_record)
        return result_record
    except Exception as exc:
        if not (output / "failure.json").exists():
            write_json(output / "failure.json", {
                "status": "failed", "type": type(exc).__name__, "error": str(exc),
                "abc_error": result_record.get("abc_error"),
                "warnings": result_record.get("warnings", [])})
        if not (output / "result.json").exists():
            write_json(output / "result.json", result_record)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory")
    parser.add_argument("--task", choices=TASKS, default="melody-full")
    parser.add_argument("--model", default="m-a-p/SheetSage2")
    parser.add_argument("--revision", help="Pin model and remote code to the same commit")
    parser.add_argument("--base-model", help="Verified MERT-v2-FullSong snapshot for an offline adapter load")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--dtype", default="auto", choices=("auto", "bf16", "fp32"))
    parser.add_argument("--preset", choices=("default", "paper"), default="default")
    parser.add_argument("--max-seconds", type=float, help="Explicitly crop the input; omitted processes it whole")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    try:
        record = run(args)
        print(f"Saved {record['score_path']}; warnings: {record['warnings']}")
        return 0
    except Exception as exc:  # noqa: BLE001 - the CLI must report every failure clearly
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
