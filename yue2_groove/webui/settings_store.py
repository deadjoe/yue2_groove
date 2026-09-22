"""The settings rail on disk: one JSON file per install, restored at every launch and page load.

The rail holds what depends on the machine (device, backend, ODE steps …), not on the song, so
a change is checked and saved as soon as the user makes it.  A value this machine cannot honour
— an engine that is not installed, a device it does not have — falls back to its automatic
choice on its own, field by field, and STATUS says why; the check runs again at launch, for a
file carried over from another setup.  Precedence at launch: an explicit ``--flag`` /
environment variable, then this file, then the automatic rules.  RESET DEFAULTS deletes it.

The file also records the app version and what the machine and the automatic rules resolved
to, so a user who reports a problem can send it as it is.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import tempfile
import threading
import time
from functools import cache
from pathlib import Path
from typing import Any

import gradio as gr
import torch

from .. import __version__, config, gguf_engine
from . import runtime

log = logging.getLogger("yue2_groove")

FILE_NAME = "yue2_groove_settings.json"
SCHEMA = 1

# the rail, in the order its components are wired (RuntimeSettings is the one place that knows it)
FIELDS = tuple(f.name for f in dataclasses.fields(runtime.RuntimeSettings))
LABELS = {
    "device": "DEVICE",
    "dtype": "DTYPE",
    "backend": "BACKEND",
    "quantization": "QUANTIZATION",
    "offload_ar": "OFFLOAD AR WEIGHTS",
    "budget": "MEMORY BUDGET",
    "ode_steps": "ODE STEPS",
    "vae_core_frames": "VAE CORE FRAMES",
    "model": "MODEL ID / LOCAL DIR",
    "vae_choice": "DEFAULT VAE",
    "vae_custom": "CUSTOM VAE",
    "revision": "MODEL REVISION",
    "vae_revision": "VAE REVISION",
    "offline": "OFFLINE",
}
DEVICES = ("auto", "mps", "cpu", "cuda")
DTYPES = tuple(value for _, value in runtime.DTYPE_CHOICES)
QUANTIZATIONS = ("none", "fp8")
CORE_FRAMES = ("auto", "512", "1024")
VAE_CHOICES = ("standard", "legacy", "custom")
ODE_RANGE = (4, 64, 4)  # the ODE STEPS slider: min, max, step
# what the rail shows when nothing else decides (the machine-dependent fields are resolved)
FIXED_DEFAULTS = {
    "quantization": "none",
    "offload_ar": False,
    "budget": 24,
    "ode_steps": 32,
    "vae_core_frames": "auto",
    "vae_custom": "",
    "revision": "",
    "vae_revision": "",
    "offline": False,
}

# The rail as the server last saved it: page loads show this rather than the values the
# Blocks were built with, which are the launch values and go stale after the first change.
_CURRENT: dict[str, Any] | None = None
_FACTORY: dict[str, Any] | None = None
# the save handlers run off the queue, so two can overlap (a pick and a blur): one at a time
_SAVE_LOCK = threading.Lock()


def path() -> Path:
    """``$YUE2_GROOVE_SETTINGS``, else next to the works directory (the app directory under
    Pinokio, the ``/data`` volume in Docker) — never inside it, where every entry is a run."""
    override = os.environ.get("YUE2_GROOVE_SETTINGS")
    if override:
        return Path(override).expanduser().resolve()
    return runtime.RUNS.parent / FILE_NAME


# ─────────────────────────── the machine ───────────────────────────


def _cuda() -> bool:
    return bool(torch.cuda.is_available())


def _mps() -> bool:
    return bool(torch.backends.mps.is_available())


@cache
def _vram_gib() -> float | None:
    return gguf_engine.cuda_total_vram_gib()


@cache
def _gpu_name() -> str | None:
    """The first NVIDIA card's name from ``nvidia-smi`` (no CUDA context, see gguf_engine)."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.run(
            [smi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            timeout=15,
            check=False,
            **config.SUBPROCESS_TEXT,
        )
        lines = out.stdout.strip().splitlines() if out.returncode == 0 else []
        return lines[0].strip() if lines else None
    except Exception:  # noqa: BLE001 — a name for the record only
        return None


def _torch_at_least(major: int, minor: int) -> bool:
    found = re.match(r"(\d+)\.(\d+)", torch.__version__)
    return bool(found) and (int(found[1]), int(found[2])) >= (major, minor)


def _auto_backend(device: str, device_explicit: bool) -> str:
    return runtime.resolve_backend("auto", device, device_explicit=device_explicit)[0]


def _looks_local(value: str) -> bool:
    """A filesystem path rather than a Hugging Face id (``org/name``, checked at load only)."""
    return value.startswith(("/", "~", ".", "\\")) or bool(re.match(r"[A-Za-z]:[\\/]", value))


# ─────────────────────────── values ───────────────────────────


def factory(
    *,
    device: str = "auto",
    dtype: str = "auto",
    backend: str = "auto",
    model: str | None = None,
    vae: str = "standard",
) -> tuple[dict[str, Any], str]:
    """The rail this launch would show without a saved file, and the automatic-backend note."""
    picked = runtime.pick_device(device)
    if dtype == "auto":
        dtype = "bfloat16" if picked in ("cuda", "mps") else "float32"
    backend, note = runtime.resolve_backend(backend, picked, device_explicit=device != "auto")
    values = {
        "device": picked,
        "dtype": dtype,
        "backend": backend,
        "model": model or config.default_model(),
        "vae_choice": vae,
        **FIXED_DEFAULTS,
    }
    return {field: values[field] for field in FIELDS}, note


def _coerce(field: str, value: Any) -> Any:
    """One value in the type its component holds; ``ValueError`` when it cannot be."""
    if field in ("offload_ar", "offline"):
        if isinstance(value, bool):
            return value
        raise ValueError(value)
    if field == "budget":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(value)
        number = float(value)
        return int(number) if number.is_integer() else number
    if field == "ode_steps":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(value)
        if not float(value).is_integer():
            raise ValueError(value)
        return int(value)
    if not isinstance(value, str):
        raise ValueError(value)
    return value.strip() if field in ("model", "vae_custom", "revision", "vae_revision") else value


def check(values: dict[str, Any], fallback: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """``(clean, notes)``: every field this machine can honour, the rest back to automatic.

    Each field falls back on its own, so a missing engine never costs the user their ODE
    steps.  *notes* are the lines STATUS shows — fallbacks and the warnings that only a
    load can settle (a card that may be too small, a torch that corrupts MPS bf16).
    """
    clean: dict[str, Any] = {}
    notes: list[str] = []

    def fall(field: str, why: str, value: Any) -> None:
        notes.append(f"{LABELS[field]} {clean[field]!s}: {why} → {value}")
        clean[field] = value

    for field in FIELDS:
        try:
            clean[field] = _coerce(field, values.get(field, fallback[field]))
        except ValueError:
            clean[field] = fallback[field]
            if field in values:
                notes.append(
                    f"{LABELS[field]} {values[field]!r}: not a valid value → {clean[field]}"
                )

    if clean["device"] not in DEVICES:
        fall("device", "not a device", "auto")
    elif clean["device"] == "cuda" and not _cuda():
        fall("device", "no CUDA on this machine", "auto")
    elif clean["device"] == "mps" and not _mps():
        fall("device", "no Apple GPU (MPS) on this machine", "auto")
    device = runtime.pick_device(clean["device"])
    explicit_device = clean["device"] != "auto"

    if clean["dtype"] not in DTYPES:
        fall("dtype", "not a dtype", "bfloat16" if device in ("cuda", "mps") else "float32")
    elif (
        device == "mps"
        and clean["dtype"] == "bfloat16"
        and clean["backend"] != "gguf"  # DTYPE does not apply to the GGUF engine
        and not _torch_at_least(2, 11)
    ):
        notes.append(
            f"DTYPE bfloat16 on MPS needs torch >= 2.11 (this is {torch.__version__}); "
            "install with the override file or pick float32"
        )

    backend = clean["backend"]
    if backend not in runtime.BACKEND_CHOICES:
        why = "not installed on this machine" if backend == "vllm" else "not an engine"
        fall("backend", why, _auto_backend(device, explicit_device))
    elif backend == "vllm" and device != "cuda":
        fall("backend", "needs NVIDIA CUDA", _auto_backend(device, explicit_device))
    elif backend == "gguf" and not gguf_engine.available():
        fall(
            "backend",
            "the yue2.cpp binaries are not installed",
            _auto_backend(device, explicit_device),
        )
    elif backend != "gguf" and device == "cuda":
        recommended = _auto_backend(device, explicit_device)
        if recommended == "gguf":
            vram = _vram_gib()
            size = f"{round(vram)} GB card" if vram else "this card"
            notes.append(
                f"BACKEND {backend}: {size} is below the "
                f"{gguf_engine.vram_threshold_gib():.0f} GB the reference engine is sized for; "
                "auto would pick gguf"
            )

    if clean["quantization"] not in QUANTIZATIONS:
        fall("quantization", "not a quantization", "none")
    elif clean["quantization"] == "fp8" and device != "cuda":
        fall("quantization", "needs NVIDIA CUDA", "none")

    if not clean["budget"] > 0:
        fall("budget", "must be above 0", FIXED_DEFAULTS["budget"])
    low, high, step = ODE_RANGE
    if not (low <= clean["ode_steps"] <= high and clean["ode_steps"] % step == 0):
        fall("ode_steps", f"outside {low}..{high} in steps of {step}", FIXED_DEFAULTS["ode_steps"])
    if clean["vae_core_frames"] not in CORE_FRAMES:
        fall("vae_core_frames", "not a choice", "auto")

    model = clean["model"]
    if not model:
        fall("model", "empty", fallback["model"])
    elif _looks_local(model) and not Path(model).expanduser().exists():
        fall("model", "no such directory", fallback["model"])

    if clean["vae_choice"] not in VAE_CHOICES:
        fall("vae_choice", "not a choice", "standard")
    elif clean["vae_choice"] == "custom":
        custom = clean["vae_custom"]
        if not custom:
            fall("vae_choice", "CUSTOM VAE is empty", "standard")
        elif _looks_local(custom) and not Path(custom).expanduser().exists():
            fall("vae_choice", f"no such directory {custom}", "standard")
    if clean["backend"] == "gguf" and clean["vae_choice"] != "standard":
        notes.append("BACKEND gguf runs the standard VAE only; pick STANDARD or the torch backend")
    return clean, notes


def load() -> dict[str, Any]:
    """The fields to restore (raw, unchecked), or ``{}`` without a readable file.

    Only the ones the user set away from the launch defaults (``changed``): a field left at
    its default keeps following the automatic rules, so a BACKEND nobody picked is chosen
    afresh at every launch (a GGUF engine installed later is still picked up).  The file
    holds every field anyway, for a bug report; one without ``changed`` restores them all."""
    target = path()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        log.warning("settings: ignoring %s (%s)", target, exc)
        return {}
    settings = data.get("settings") if isinstance(data, dict) else None
    if not isinstance(settings, dict):
        log.warning("settings: ignoring %s (no settings object)", target)
        return {}
    changed = data.get("changed")
    keep = set(changed) if isinstance(changed, list) else set(FIELDS)
    return {field: settings[field] for field in FIELDS if field in settings and field in keep}


def record(values: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    """The file's content: the rail, which fields differ from this launch's *defaults*, and
    what it resolved to on this machine."""
    device = runtime.pick_device(values["device"])
    backend, _ = runtime.effective_backend(device, values["backend"])
    vram = _vram_gib()
    return {
        "schema": SCHEMA,
        "app": "yue2_groove",
        "version": __version__,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "settings": {field: values[field] for field in FIELDS},
        "changed": [field for field in FIELDS if values[field] != defaults.get(field)],
        "resolved": {"device": device, "backend": backend},
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": _cuda(),
            "mps": _mps(),
            "gpu": _gpu_name(),
            "vram_gib": round(vram, 1) if vram else None,
            "gguf_installed": gguf_engine.available(),
        },
    }


def write(values: dict[str, Any], defaults: dict[str, Any]) -> Path:
    """Save atomically (a torn file would cost the user every setting at the next launch)."""
    target = path()
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record(values, defaults), ensure_ascii=False, indent=2) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return target


# ─────────────────────────── launch ───────────────────────────


def startup(
    *,
    device: str | None = None,
    dtype: str | None = None,
    backend: str | None = None,
    model: str | None = None,
    vae: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """``(rail, factory, notes)`` for this launch.

    The keyword arguments are what the command line or the environment set explicitly
    (``None`` / ``"auto"`` for "decide for me"); they win over the file, which wins over the
    automatic rules.  *notes* are for the log and STATUS.  Also primes the page-load state.
    """
    global _CURRENT, _FACTORY
    explicit = {
        k: v
        for k, v in {
            "device": device,
            "dtype": dtype,
            "backend": backend,
            "model": model,
            "vae_choice": vae,
        }.items()
        if v not in (None, "auto")
    }
    base, auto_note = factory(
        device=device or "auto",
        dtype=dtype or "auto",
        backend=backend or "auto",
        model=model,
        vae=vae or "standard",
    )
    saved = {k: v for k, v in load().items() if k not in explicit}
    notes: list[str] = []
    rail = base
    if saved:
        rail, checked = check({**base, **saved}, base)
        for field in explicit:  # a flag the user typed stands even where the check disagrees
            rail[field] = base[field]
        notes += [n for n in checked if not n.startswith(tuple(LABELS[f] for f in explicit))]
        notes.insert(0, f"Settings restored from {path()}")
    if auto_note and rail["backend"] == base["backend"] and "backend" not in saved:
        notes.append(auto_note)
    _CURRENT, _FACTORY = dict(rail), dict(base)
    return rail, base, notes


# ─────────────────────────── handlers ───────────────────────────


def _pending(values: dict[str, Any]) -> str:
    """A line when a model is loaded that these settings would replace."""
    if runtime._PIPE is None:
        return ""
    try:
        changed = runtime.RuntimeSettings(**values).key != runtime._PIPE_KEY
    except Exception:  # noqa: BLE001 — a key that cannot be built is settled by the load
        return ""
    return "Takes effect at the next run (or LOAD / APPLY)." if changed else ""


def save_from_rail(*values):
    """The rail's change handler: check, save, and put back what had to fall back."""
    with _SAVE_LOCK:
        return _save(dict(zip(FIELDS, values, strict=True)))


def _save(raw: dict[str, Any]) -> tuple:
    global _CURRENT
    fallback = _FACTORY or factory(model=str(raw.get("model") or "") or None)[0]
    # compare what the user submitted, not what the check made of it: a pick that falls
    # back to the current value is still an edit and its note must be shown
    if _CURRENT is not None and all(raw[f] == _CURRENT[f] for f in FIELDS):
        return (*[gr.update() for _ in FIELDS], gr.update())  # a blur without an edit
    clean, notes = check(raw, fallback)
    _CURRENT = dict(clean)  # a refresh shows it even when the file cannot be written
    try:
        target = write(clean, fallback)
        lines = [f"Settings saved to {target}"]
    except OSError as exc:
        lines = [f"Settings NOT saved ({exc}); they apply to this session only"]
    lines += notes
    pending = _pending(clean)
    if pending:
        lines.append(pending)
    updates = [gr.update(value=clean[f]) if clean[f] != raw[f] else gr.update() for f in FIELDS]
    return (*updates, "\n".join(lines))


def reset(values: dict[str, Any]) -> tuple:
    """RESET DEFAULTS: the launch defaults back on the rail and the saved file gone."""
    global _CURRENT
    with _SAVE_LOCK:
        _CURRENT = dict(values)
        return _reset(values)


def _reset(values: dict[str, Any]) -> tuple:
    try:
        path().unlink(missing_ok=True)
        status = "Settings reset to this machine's defaults; the saved file was removed."
    except OSError as exc:
        status = f"Settings reset for this session, but {path()} could not be removed ({exc})"
    return (*[values[f] for f in FIELDS], status)


def rail_values():
    """Page load: the rail as last saved, so a refresh never shows stale launch values."""
    if _CURRENT is None:
        return tuple(gr.update() for _ in FIELDS)
    return tuple(_CURRENT[f] for f in FIELDS)
