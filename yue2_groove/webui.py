"""YUE2 // GROOVE — unofficial Gradio web UI for YuE2 song generation.

Built on top of the official ``yue2`` CLI/Python API; every human-operable control of
the CLI/API is reachable from the browser so you never have to fall back to a terminal
for one parameter.

  01 GENERATE   style + lyrics (+ optional ABC) → editable score plan → 48 kHz stereo song;
                all sampling parameters of both the ABC and the semantic phase; plan mode
                full/melody/off, seed, cfg_scale, custom id, cancel, live progress; ALL MODES
                runs the same text request as full + melody + off and compares them
  02 COVER      source audio → SheetSage2 transcription (separate venv) → editable ABC →
                chord strip → SEND TO GENERATE or generate right on the tab
  03 EDIT       load a work as a frozen baseline, edit its ABC, check exact melody/meter
                invariants, regenerate from the edited score, compare baseline vs edit
  04 LIBRARY    every generated work: sort, select, rename, confirmed delete; details with a
                player (spectrum + transport), style / lyrics / ABC / score and run tables
  05 TOOLS      ABC validation / event export, chord stripping (cover melodies), edit
                invariant check, environment doctor, listening-comparison page
  06 DECODE     re-decode a saved latent.npy (source / standard / legacy / custom VAE,
                full or tiled) without generating again; single .npy upload supported
  07 BATCH      one JSON request per line (the equivalent of ``yue2 batch``), run in order
  Settings rail model & runtime: device / dtype / backend / quantization / offload_ar /
                memory budget / ODE steps / VAE core frames / revisions / offline; load & unload

Two views, one kernel.  **SONG** is the producer-facing director: it follows the
current work, states its stage (DRAFT / SCORE / AUDIO / REVISE / DONE) and offers
the next actions, all of which reuse the Studio handlers; its score is read-only
and it holds no editable component.  **STUDIO** is the full 7-tab gear room
above.  The switch is a class on ``<html>`` and both roots stay mounted, so the
score SVG and the player survive it (see ``docs/VIEW_SWITCH_PREFLIGHT.md``).

SheetSage2 (COVER) runs in its own virtual environment; set ``YUE2_GROOVE_SHEETSAGE_PYTHON``
to its interpreter (see README, 'Cover from audio').  Nothing in this process imports
``transformers``; the subprocess boundary lives in ``sheetsage_adapter.py``.

Apple Silicon (MPS): bfloat16 works with torch >= 2.11.  The upstream pin (torch 2.10.0)
hits pytorch/pytorch#174861 — the single-query SDPA kernel corrupts once the KV cache
passes 1024 tokens — so install with the override file in ``overrides/`` (see README) and
run ``scripts/mps_sdpa_check.py`` to verify.  vLLM / FP8 need NVIDIA CUDA.

Usage:
  python -m yue2_groove --port 7860             # opens the browser, SONG view
  python -m yue2_groove --view studio           # force the STUDIO view
  python -m yue2_groove --tab 1                 # force STUDIO on tab 0..6
  python -m yue2_groove --host 0.0.0.0 --auth user:pass   # LAN access (set a password)
  bash scripts/serve.sh start|stop|restart|status|log     # background service

The default view is SONG; ``YUE2_GROOVE_VIEW=song|studio`` or ``--view`` forces
it for a launch, and the last choice is remembered per browser otherwise.

Visual language: Bearbone Design System v0.2 (warm near-black ground family + ivory ink,
1px strokes, no shadows/gradients, monospace), with a dark and a bright scene.

Model weights are CC BY-NC 4.0 (non-commercial); this UI is not affiliated with the
YuE2 authors.
"""
from __future__ import annotations

import argparse
import atexit

try:
    import fcntl  # Unix only; optional macOS F_FULLFSYNC in _fsync_fd
except ImportError:  # Windows (and any host without the module)
    fcntl = None
import html
import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from . import __version__, adapter, config, cover, edit_flow, library, sheetsage_adapter, workflow
from .vendor import abc_tools

# Where generated works are stored; main() may override it with --runs.
RUNS = config.runs_dir()

_PIPE = None
_PIPE_KEY = None
_LOCK = threading.Lock()
_RUNNING = threading.Lock()
_CANCEL = threading.Event()

DTYPE_CHOICES = [
    ("bfloat16 (checkpoint dtype; default on CUDA/MPS)", "bfloat16"),
    ("float32 (cast at load; slower, 2x memory)", "float32"),
]

ABC_DEFAULTS = {"temperature": .7, "top_p": .9, "top_k": 30, "repetition_penalty": 1.005,
                "penalty_window": 100, "min_tokens": 32, "max_tokens": 4096}
SEM_DEFAULTS = {"temperature": 1.0, "top_p": .95, "top_k": 100, "repetition_penalty": 1.2,
                "penalty_window": 50, "min_tokens": 200, "max_tokens": 9000}


# ─────────────────────────── helpers ───────────────────────────

def _pick_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_vae(choice: str, custom: str) -> tuple[str, str]:
    """Return (path_or_hub_id, display_name)."""
    if choice == "standard":
        return config.default_vae(), "standard (YuE2-Vae)"
    if choice == "legacy":
        return config.default_vae_legacy(), "legacy (benchmark)"
    if not (custom or "").strip():
        raise gr.Error("Custom VAE requires a path or Hugging Face ID")
    return custom.strip(), "custom"


def _sampling(temp, top_p, top_k, rep, window, min_tokens, max_tokens, label):
    try:
        return adapter.sampling(temperature=float(temp), top_p=float(top_p), top_k=int(top_k),
                                repetition_penalty=float(rep), penalty_window=int(window),
                                min_tokens=int(min_tokens), max_tokens=int(max_tokens))
    except (ValueError, TypeError) as exc:
        raise gr.Error(f"{label} sampling parameters invalid: {exc}") from exc


def _pipe_key(device, dtype, backend, quantization, offload_ar, budget, ode_steps,
              vae_core_frames, model, vae_path, revision, vae_revision, offline):
    return (_pick_device(device), dtype, backend, quantization, bool(offload_ar),
            float(budget), int(ode_steps), vae_core_frames, model, vae_path,
            revision or "", vae_revision or "", bool(offline))


def load_pipeline(device, dtype, backend, quantization, offload_ar, budget, ode_steps,
                  vae_core_frames, model, vae_choice, vae_custom, revision, vae_revision,
                  offline, progress=gr.Progress()):
    """(Re)load the pipeline. Reuses the existing one when settings are unchanged."""
    global _PIPE, _PIPE_KEY
    device = _pick_device(device)
    vae_path, vae_name = resolve_vae(vae_choice, vae_custom)
    cores = None if vae_core_frames == "auto" else int(vae_core_frames)
    key = _pipe_key(device, dtype, backend, quantization, offload_ar, budget, ode_steps,
                    cores, model, vae_path, revision, vae_revision, offline)
    if _PIPE is not None and _PIPE_KEY == key:
        return _PIPE, f"Model ready: device={device} dtype={dtype} backend={backend} vae={vae_name}"
    if backend == "vllm" and device != "cuda":
        raise gr.Error("vLLM backend requires NVIDIA CUDA; use torch here (MPS falls back to eager)")
    if quantization == "fp8" and device != "cuda":
        raise gr.Error("FP8 quantization requires NVIDIA CUDA (sm89+)")

    unload_pipeline()
    if progress is not None:
        progress(0.05, desc="Loading model (first run downloads ~7.3 GB)…")
    with _LOCK:
        pipe, used_dtype = adapter.load_pipeline(
            model, vae=vae_path, device=device, dtype=dtype, backend=backend,
            quantization=quantization, offload_ar=offload_ar, memory_budget_gib=budget,
            ode_steps=ode_steps, vae_core_frames=cores, revision=revision,
            vae_revision=vae_revision, local_files_only=offline)
        _PIPE, _PIPE_KEY = pipe, key
    note = (f"Loaded: device={device} dtype={used_dtype} backend={backend} "
            f"vae={vae_name} ode_steps={ode_steps} cores={cores or 'auto'}")
    return _PIPE, note


def unload_pipeline():
    global _PIPE, _PIPE_KEY
    with _LOCK:
        if _PIPE is not None:
            try:
                adapter.close_pipeline(_PIPE)
            except Exception:  # noqa: BLE001, S110 — closing must never raise
                pass
        _PIPE, _PIPE_KEY = None, None
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _get_pipe(device, dtype, backend, quantization, offload_ar, budget, ode_steps,
              vae_core_frames, model, vae_choice, vae_custom, revision, vae_revision,
              offline, progress):
    vae_path, _ = resolve_vae(vae_choice, vae_custom)
    cores = None if vae_core_frames == "auto" else int(vae_core_frames)
    key = _pipe_key(_pick_device(device), dtype, backend, quantization, offload_ar, budget,
                    ode_steps, cores, model, vae_path, revision, vae_revision, offline)
    if _PIPE is None or _PIPE_KEY != key:
        return load_pipeline(device, dtype, backend, quantization, offload_ar, budget,
                             ode_steps, vae_core_frames, model, vae_choice, vae_custom,
                             revision, vae_revision, offline, progress)
    return _PIPE, "Model ready"


def _write_local_env(directory: Path, pipe, note: str = "") -> None:
    """Record what actually ran in this run directory (``local_env.json``).

    Upstream's ``config.json`` hardcodes ``"model_dtype": "bfloat16"``, so an
    explicit float32 cast from this UI (or any future override) would otherwise
    be misreported.  Best-effort only: a sidecar, never a rewrite of upstream
    artifacts.  The Library shows it when it disagrees with ``config.json``.
    """
    try:
        payload = {
            "tool": "yue2_groove",
            "dtype": adapter.model_dtype(pipe),
            "device": str(getattr(pipe, "device", "")),
            "torch": torch.__version__,
            "yue2": adapter.yue2_version(),
            "note": note or None,
        }
        (Path(directory) / "local_env.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception:  # noqa: BLE001, S110 — provenance must never fail a run
        pass


def _slug(text: str) -> str:
    out = "".join(c if c.isalnum() or c in "-_" else "-" for c in (text or "")[:40].strip())
    return out.strip("-") or "song"


def _artifact_files(directory: Path, score: bool):
    names = ["audio.flac", "request.json", "config.json", "result.json",
             "latent.npy", "semantic.npy"] + (["score.abc"] if score else [])
    return [str(directory / n) for n in names if (directory / n).exists()]


def _looks_like_run(path: Path) -> bool:
    return any((path / name).exists()
               for name in ("latent.npy", "result.json", "plan.json", "decode.json"))


def _scan_runs():
    if not RUNS.is_dir():
        return []
    return [str(p) for p in sorted(RUNS.iterdir(), reverse=True)
            if p.is_dir() and not p.name.startswith(".") and _looks_like_run(p)]


def _update_run_choices():
    return gr.update(choices=_scan_runs())


def _scan_batches():
    """[(label, batch_dir)] for batch or all-modes groups with saved songs."""
    out = []
    if not RUNS.is_dir():
        return out
    for d in sorted(RUNS.iterdir(), reverse=True):
        if not d.is_dir() or not ("batch" in d.name or "-allmodes-" in d.name):
            continue
        songs = [p for p in sorted(d.iterdir()) if p.is_dir() and (p / "result.json").is_file()]
        if songs:
            out.append((f"{d.name}  ({len(songs)} songs)", str(d)))
    return out


def _update_batch_choices():
    return gr.update(choices=[c for c, _ in _scan_batches()])


def _fill_from_batch(label):
    """Fill the comparison input with every saved song of the chosen batch run."""
    if not label:
        raise gr.Error("Pick a batch run first")
    for choice, path in _scan_batches():
        if choice == label:
            songs = [str(p) for p in sorted(Path(path).iterdir())
                     if p.is_dir() and (p / "result.json").is_file()]
            return "\n".join(songs)
    raise gr.Error("That batch directory is gone; press REFRESH LIST")


def cancel_run():
    _CANCEL.set()
    return "Cancel requested — will stop after the current token / ODE step"


def _reset_sampling_values():
    a, s = ABC_DEFAULTS, SEM_DEFAULTS
    return (a["temperature"], a["top_p"], a["top_k"], a["repetition_penalty"],
            a["penalty_window"], a["min_tokens"], a["max_tokens"],
            s["temperature"], s["top_p"], s["top_k"], s["repetition_penalty"],
            s["penalty_window"], s["min_tokens"], s["max_tokens"],
            "Protocol defaults (full)")


def duration_text(tokens):
    seconds = max(0, int(tokens)) / 25.0
    minutes = seconds / 60.0
    return (f"**Estimated audio length:** ≈ {seconds:.0f} s "
            f"({minutes:.1f} min) at {int(tokens)} semantic tokens")


# ───────────────────── run durability (panic-safe writes) ─────────────────────
# save_artifacts() closes the files but never fsyncs, and the whole point of the
# 2026-09-13 panics was that a run could be listened to and still vanish when the
# kernel died before APFS flushed it.  A run is only "done" once its files and
# directory are flushed; ".pending" is the marker the Library shows when it is not.
PENDING_FILE = "pending.json"


def _fsync_fd(fd: int) -> None:
    try:
        os.fsync(fd)
    except OSError:
        pass
    # macOS: fsync only reaches the drive cache, F_FULLFSYNC reaches the media.
    # fcntl is absent on Windows; skip the extra flush there.
    if fcntl is not None and hasattr(fcntl, "F_FULLFSYNC"):
        try:
            fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
        except OSError:
            pass


def _fsync_path(path) -> None:
    try:
        fd = os.open(Path(path), os.O_RDONLY)
    except OSError:
        return
    try:
        _fsync_fd(fd)
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _fsync_tree(directory) -> None:
    """Flush every file in *directory* and the directory itself."""
    directory = Path(directory)
    try:
        entries = list(directory.iterdir())
    except OSError:
        return
    for entry in entries:
        if entry.is_file():
            _fsync_path(entry)
    _fsync_path(directory)


def _write_pending(directory, status: str, error: str = "") -> None:
    """Mark *directory* as an unfinished run, durably.

    Written before generation starts and removed only after every artifact is
    flushed, so a run that never finishes stays visible in the Library instead of
    disappearing silently.
    """
    directory = Path(directory)
    payload = {"schema": "yue2-groove-pending-v1", "status": status,
               "pid": os.getpid(), "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if error:
        payload["error"] = error[:500]
    path = directory / PENDING_FILE
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
        _fsync_path(path)
        _fsync_path(directory)
    except OSError:
        pass


def _clear_pending(directory) -> None:
    directory = Path(directory)
    try:
        (directory / PENDING_FILE).unlink()
        _fsync_path(directory)
    except OSError:
        pass


def _run_generation(pipe, request, outdir, *, abc_sampling, semantic_sampling, progress, note,
                    extra_manifest=None):
    """Generate one song into *outdir* and save its artifacts.

    Shared by 01 GENERATE and 03 EDIT so both flows report progress identically.
    ``extra_manifest`` (a dict) is written next to the run as ``edit_manifest.json``.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _write_pending(outdir, "running")
    print(f"[yue2_groove] run start: {outdir}", flush=True)
    counts = {"abc": 0, "semantic": 0}
    abc_budget = abc_sampling.max_tokens if request.cot != "off" else 0
    sem_budget = semantic_sampling.max_tokens

    def on_token(phase, token):
        counts[phase] = counts.get(phase, 0) + 1
        if request.cot == "off":
            frac = 0.55 * min(1.0, counts["semantic"] / max(1, sem_budget))
        else:
            frac = (0.15 * min(1.0, counts["abc"] / max(1, abc_budget))
                    + 0.40 * min(1.0, counts["semantic"] / max(1, sem_budget)))
        progress(min(0.55, frac), desc=f"Generating {phase}: {counts[phase]} tokens")

    def on_progress(stage, done, total):
        if stage == "nar":
            progress(0.55 + 0.40 * min(1.0, done / max(1, total)),
                     desc=f"Synthesizing audio: step {done}/{total}")
        elif stage == "vae":
            progress(0.95 + 0.05 * min(1.0, done / max(1, total)),
                     desc=f"Decoding audio: chunk {done}/{total}")

    t0 = time.perf_counter()
    try:
        song = adapter.generate(pipe, request, abc_sampling=abc_sampling,
                                semantic_sampling=semantic_sampling, cancelled=_CANCEL.is_set,
                                on_token=on_token, on_progress=on_progress)
        progress(1.0, desc="Saving artifacts…")
        result = song.save_artifacts(outdir)
        if extra_manifest is not None:
            (Path(outdir) / "edit_manifest.json").write_text(
                json.dumps(extra_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _write_local_env(outdir, pipe, note)
    except Exception as exc:
        _write_pending(outdir, "cancelled" if isinstance(exc, InterruptedError) else "failed",
                       error=f"{type(exc).__name__}: {exc}")
        print(f"[yue2_groove] run unfinished: {outdir} — {type(exc).__name__}: {exc}", flush=True)
        raise
    # flush the artifacts first, then remove the marker: the run only stops being
    # "pending" once it is actually on disk
    _fsync_tree(outdir)
    _clear_pending(outdir)
    print(f"[yue2_groove] run done: {outdir}", flush=True)
    return song, result, time.perf_counter() - t0


def _generation_status(song, result, outdir, elapsed, request, note):
    return (f"Done: {result['audio_seconds']:.1f}s audio in {elapsed:.0f}s\n"
            f"truncated={result['truncated']}  seed={request.seed}  cfg={request.guidance}\n"
            f"NAR={song.timing['nar_seconds']:.0f}s  VAE={song.timing['vae_seconds']:.0f}s  "
            f"semantic={song.timing['semantic'].get('output_tps', 0):.1f} tok/s  "
            f"ABC={song.timing['abc'].get('output_tokens', 0)} tokens\n"
            f"run directory: {outdir}\n{note}")


def generate(style, lyrics, cot, seed, cfg_scale, abc_text, out_id, preset,
             abc_temp, abc_p, abc_k, abc_rep, abc_win, abc_min, abc_max,
             sem_temp, sem_p, sem_k, sem_rep, sem_win, sem_min, sem_max,
             device, dtype, backend, quantization, offload_ar, budget, ode_steps,
             vae_core_frames, model, vae_choice, vae_custom, revision, vae_revision, offline,
             progress=gr.Progress()):
    """Generator: disables the action buttons until the run finishes."""
    style, lyrics = _request_texts(style, lyrics)   # empty input is an error
    abc_sampling = _sampling(abc_temp, abc_p, abc_k, abc_rep, abc_win, abc_min, abc_max, "ABC phase")
    sem_sampling = _sampling(sem_temp, sem_p, sem_k, sem_rep, sem_win, sem_min, sem_max, "semantic phase")
    kwargs = {}
    if (out_id or "").strip():
        kwargs["id"] = out_id.strip()
    if cfg_scale:
        kwargs["cfg_scale"] = float(cfg_scale)
    if (abc_text or "").strip():
        if cot == "off":
            raise gr.Error("An ABC score requires cot=full or cot=melody")
        # Pasted scores often carry the chat/TUI code-block indentation; the model
        # expects clean ABC lines ("X:1", "V: Vocal", ...). Strip the common indent.
        kwargs["abc"] = textwrap.dedent(abc_text).strip()
    try:
        request = adapter.song_request(style=style, lyrics=lyrics, cot=cot, seed=int(seed), **kwargs)
    except (ValueError, TypeError) as exc:
        raise gr.Error(f"Invalid request: {exc}") from exc

    _CANCEL.clear()
    if not _RUNNING.acquire(blocking=False):
        # A fast double-click can queue a second run before the button disables.
        # Keep the buttons as they are (the running job owns them).
        yield gr.update(), gr.update(), "Another job is already running — wait for it to finish", \
            gr.update(), gr.update(), gr.update(), gr.skip()
        return
    busy = (gr.update(interactive=False), gr.update(interactive=False))
    idle = (gr.update(interactive=True), gr.update(interactive=True))
    try:
        yield gr.update(), gr.update(), "Starting generation…", gr.update(), *busy, gr.skip()
        pipe, note = _get_pipe(device, dtype, backend, quantization, offload_ar, budget,
                               ode_steps, vae_core_frames, model, vae_choice, vae_custom,
                               revision, vae_revision, offline, progress)
        progress(0.02, desc="Starting generation…")
        outdir = RUNS / (f"{time.strftime('%Y%m%d-%H%M%S')}-"
                         f"{request.id if request.id != 'song' else _slug(style)}")
        song, result, elapsed = _run_generation(
            pipe, request, outdir, abc_sampling=abc_sampling, semantic_sampling=sem_sampling,
            progress=progress, note=note)
        yield str(outdir / "audio.flac"), (song.abc or ""), \
            _generation_status(song, result, outdir, elapsed, request, note), \
            _artifact_files(outdir, bool(song.abc)), *idle, str(outdir)
    except InterruptedError as exc:
        yield gr.update(), gr.update(), f"Cancelled: {exc}", gr.update(), *idle, gr.skip()
    except Exception as exc:  # noqa: BLE001
        yield gr.update(), gr.update(), \
            f"Generation failed: {type(exc).__name__}: {exc}", gr.update(), *idle, gr.skip()
    finally:
        _RUNNING.release()


ALL_MODES = ("full", "melody", "off")


def generate_all_modes(style, lyrics, seed, cfg_scale, abc_text, out_id,
                       abc_temp, abc_p, abc_k, abc_rep, abc_win, abc_min, abc_max,
                       sem_temp, sem_p, sem_k, sem_rep, sem_win, sem_min, sem_max,
                       device, dtype, backend, quantization, offload_ar, budget, ode_steps,
                       vae_core_frames, model, vae_choice, vae_custom, revision, vae_revision,
                       offline, progress=gr.Progress()):
    """Generator: the same text request as full / melody / off, then a comparison.

    Mirrors upstream's ``all-modes``: text-only input (``cot=off`` cannot accept an
    ABC), one fresh directory per mode, a ``run.json`` summary at the group root,
    failures retained as ``failure.json``, and a listening bundle of the modes that
    completed.  Buttons are disabled while running; CANCEL stops after the current
    mode.
    """
    if (abc_text or "").strip():
        raise gr.Error("ALL MODES is text-only: cot=off cannot accept an ABC score. Clear "
                       "ABC SCORE, or generate the modes one by one with their own ABC.")
    style, lyrics = _request_texts(style, lyrics)
    base_id = (out_id or "").strip() or _slug(style)
    try:
        adapter.song_request(style=style, lyrics=lyrics, cot="full", seed=int(seed), id=base_id)
    except (ValueError, TypeError) as exc:
        raise gr.Error(f"Invalid request: {exc}") from exc
    abc_sampling = _sampling(abc_temp, abc_p, abc_k, abc_rep, abc_win, abc_min, abc_max, "ABC phase")
    sem_sampling = _sampling(sem_temp, sem_p, sem_k, sem_rep, sem_win, sem_min, sem_max,
                             "semantic phase")

    _CANCEL.clear()
    if not _RUNNING.acquire(blocking=False):
        yield ("Another job is already running — wait for it to finish",
               gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.skip())
        return
    busy = (gr.update(interactive=False),) * 3
    idle = (gr.update(interactive=True),) * 3
    try:
        yield ("Starting ALL MODES (full → melody → off)…",
               gr.update(), gr.update(), *busy, gr.skip())
        pipe, note = _get_pipe(device, dtype, backend, quantization, offload_ar, budget,
                               ode_steps, vae_core_frames, model, vae_choice, vae_custom,
                               revision, vae_revision, offline, progress)
        root = RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}-allmodes-{_slug(base_id)}"
        results, files, done_dirs = [], [], []
        for position, mode in enumerate(ALL_MODES):
            mode_dir = root / mode
            if _CANCEL.is_set():
                results.append({"mode": mode, "status": "cancelled"})
                break
            kwargs = {"style": style, "lyrics": lyrics, "cot": mode, "seed": int(seed),
                      "id": f"{base_id}_{mode}"}
            if cfg_scale:
                kwargs["cfg_scale"] = float(cfg_scale)
            request = adapter.song_request(**kwargs)

            def scoped(value, desc=None, position=position, mode=mode):
                fraction = position + (0.0 if value is None else min(1.0, float(value)))
                progress(min(1.0, fraction / len(ALL_MODES)),
                         desc=f"[{mode}] {desc}" if desc else f"[{mode}]")

            try:
                mode_dir.mkdir(parents=True, exist_ok=True)
                (mode_dir / "input.json").write_text(
                    json.dumps(request.to_dict(), ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
                song, receipt, elapsed = _run_generation(
                    pipe, request, mode_dir, abc_sampling=abc_sampling,
                    semantic_sampling=sem_sampling, progress=scoped, note=note)
                results.append({"mode": mode, "status": "complete",
                                "audio_seconds": receipt["audio_seconds"],
                                "truncated": receipt["truncated"],
                                "seconds": round(elapsed, 2)})
                files += _artifact_files(mode_dir, bool(song.abc))
                done_dirs.append(str(mode_dir))
            except InterruptedError as exc:
                record = {"mode": mode, "status": "cancelled", "error": str(exc)}
                (mode_dir / "failure.json").write_text(
                    json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                results.append(record)
                break
            except Exception as exc:  # noqa: BLE001 — keep going; the failure is retained
                record = {"mode": mode, "status": "failed", "type": type(exc).__name__,
                          "error": str(exc)}
                mode_dir.mkdir(parents=True, exist_ok=True)
                (mode_dir / "failure.json").write_text(
                    json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                results.append(record)
            progress((position + 1) / len(ALL_MODES), desc=f"{mode} finished")

        root.mkdir(parents=True, exist_ok=True)
        summary = {"action": "all-modes", "request": {"style": style, "lyrics": lyrics,
                   "seed": int(seed), "cfg_scale": float(cfg_scale) if cfg_scale else None,
                   "id": base_id}, "results": results}
        (root / "run.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        link, comparison = "", ""
        if len(done_dirs) >= 2:
            progress(0.98, desc="Building listening comparison…")
            try:
                _html, link, comparison = make_comparison("\n".join(done_dirs))
            except gr.Error as exc:
                comparison = f"comparison failed: {exc}"

        lines = [f"ALL MODES finished: {root}", ""]
        for record in results:
            if record["status"] == "complete":
                lines.append(f"  {record['mode']:<7} complete  {record['audio_seconds']:.1f}s "
                             f"audio in {record['seconds']:.0f}s  truncated={record['truncated']}")
            else:
                lines.append(f"  {record['mode']:<7} {record['status']}  "
                             f"{record.get('error', '')}")
        lines += ["", note]
        if comparison:
            lines.append(comparison)
        if len(done_dirs) < 2:
            lines.append("(the comparison needs at least two completed modes)")
        yield "\n".join(lines), files, link, *idle, str(root)
    except Exception as exc:  # noqa: BLE001
        yield (f"ALL MODES failed: {type(exc).__name__}: {exc}",
               gr.update(), gr.update(), *idle, gr.skip())
    finally:
        _RUNNING.release()


def plan_only(style, lyrics, cot, seed, cfg_scale, out_id,
              abc_temp, abc_p, abc_k, abc_rep, abc_win, abc_min, abc_max,
              device, dtype, backend, quantization, offload_ar, budget, ode_steps,
              vae_core_frames, model, vae_choice, vae_custom, revision, vae_revision, offline,
              progress=gr.Progress()):
    """Generator: disables the action buttons until planning finishes."""
    style, lyrics = _request_texts(style, lyrics)   # empty input is an error
    abc_sampling = _sampling(abc_temp, abc_p, abc_k, abc_rep, abc_win, abc_min, abc_max, "ABC phase")
    kwargs = {}
    if (out_id or "").strip():
        kwargs["id"] = out_id.strip()
    if cfg_scale:
        kwargs["cfg_scale"] = float(cfg_scale)
    try:
        request = adapter.song_request(style=style, lyrics=lyrics, cot=cot, seed=int(seed), **kwargs)
    except (ValueError, TypeError) as exc:
        raise gr.Error(f"Invalid request: {exc}") from exc

    _CANCEL.clear()
    if not _RUNNING.acquire(blocking=False):
        yield gr.update(), "Another job is already running — wait for it to finish", \
            gr.update(), gr.update(), gr.update(), gr.skip()
        return
    busy = (gr.update(interactive=False), gr.update(interactive=False))
    idle = (gr.update(interactive=True), gr.update(interactive=True))
    try:
        yield gr.update(), "Planning score…", gr.update(), *busy, gr.skip()
        pipe, note = _get_pipe(device, dtype, backend, quantization, offload_ar, budget,
                               ode_steps, vae_core_frames, model, vae_choice, vae_custom,
                               revision, vae_revision, offline, progress)
        progress(0.05, desc="Planning score…")
        plan = adapter.plan(pipe, request, abc_sampling=abc_sampling, cancelled=_CANCEL.is_set)
        outdir = RUNS / (f"{time.strftime('%Y%m%d-%H%M%S')}-"
                         f"{request.id if request.id != 'song' else _slug(style)}-plan")
        plan.save(outdir)
        files = [str(outdir / n) for n in ("score.abc", "plan.json", "abc_tokens.npy", "prefix.npy")
                 if (outdir / n).exists()]
        yield (plan.abc or ""), f"Plan saved: {outdir}\n{note}", files, *idle, str(outdir)
    except InterruptedError as exc:
        yield gr.update(), f"Cancelled: {exc}", gr.update(), *idle, gr.skip()
    except Exception as exc:  # noqa: BLE001
        yield gr.update(), f"Planning failed: {type(exc).__name__}: {exc}", gr.update(), *idle, \
            gr.skip()
    finally:
        _RUNNING.release()


def decode_run(source_dir, latent_file, dec_vae_choice, dec_vae_custom, dec_vae_revision,
               full_decode, device, dtype, backend, quantization, offload_ar,
               budget, ode_steps, vae_core_frames, model, gen_vae_choice, gen_vae_custom,
               revision, gen_vae_revision, offline, progress=gr.Progress()):
    """Generator: re-decodes saved latents; disables the decode button while running."""
    path = None
    if (latent_file or "").strip():
        path = Path(latent_file.strip())
    elif (source_dir or "").strip():
        path = Path(source_dir.strip()) / "latent.npy"
    if path is None or not path.is_file():
        raise gr.Error("Provide latent.npy (upload a file or point at a saved run directory)")
    latents = np.load(path, allow_pickle=False)
    if latents.ndim != 2 or latents.shape[1] != 64:
        raise gr.Error(f"Latent shape must be [T,64], got {latents.shape}")

    if not _RUNNING.acquire(blocking=False):
        yield gr.update(), "Another job is already running — wait for it to finish", \
            gr.update(), gr.skip()
        return
    try:
        yield gr.update(), "Decoding…", gr.update(interactive=False), gr.skip()
        pipe, note = _get_pipe(device, dtype, backend, quantization, offload_ar, budget,
                               ode_steps, vae_core_frames, model, gen_vae_choice, gen_vae_custom,
                               revision, gen_vae_revision, offline, progress)
        override = None
        if dec_vae_choice != "keep":
            override, vae_name = resolve_vae(dec_vae_choice, dec_vae_custom)
        else:
            vae_name = "source-generation VAE"
        if dec_vae_choice != "keep" and dec_vae_revision:
            override = str(adapter.resolve_model(override, revision=dec_vae_revision,
                                                 local_files_only=bool(offline)))
        progress(0.1, desc=f"Decoding {latents.shape[0]} frames ({vae_name})…")
        t0 = time.perf_counter()

        def on_progress(done, total):
            progress(0.1 + 0.85 * min(1.0, done / max(1, total)),
                     desc=f"Decoding audio: chunk {done}/{total}")

        audio = adapter.decode(pipe, latents, full=bool(full_decode), vae=override,
                               on_progress=on_progress)
        seconds = time.perf_counter() - t0
        outdir = RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}-decode-{_slug(vae_name)}"
        outdir.mkdir(parents=True, exist_ok=True)
        sf.write(outdir / "audio.flac", audio, 48000, subtype="PCM_24")
        np.save(outdir / "latent.npy", latents.astype(np.float32))
        (outdir / "decode.json").write_text(json.dumps({
            "operation": "decode_cached_latents", "source_latent": str(path),
            "vae": override or str(pipe.vae_dir), "full_decode": bool(full_decode),
            "core_frames": None if full_decode else pipe.vae_core_frames,
            "seconds": seconds, "device": str(pipe.device),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _write_local_env(outdir, pipe, note)
        status = (f"Decoded {len(audio) / 48000:.1f}s audio in {seconds:.0f}s\n"
                  f"VAE={vae_name}  mode={'full' if full_decode else 'tiled'}  "
                  f"source={path}\nrun directory: {outdir}\n{note}")
        yield str(outdir / "audio.flac"), status, gr.update(interactive=True), str(outdir)
    except Exception as exc:  # noqa: BLE001
        yield gr.update(), f"Decode failed: {type(exc).__name__}: {exc}", \
            gr.update(interactive=True), gr.skip()
    finally:
        _RUNNING.release()


def batch_generate(jsonl_text, jsonl_file, out_id,
                   abc_temp, abc_p, abc_k, abc_rep, abc_win, abc_min, abc_max,
                   sem_temp, sem_p, sem_k, sem_rep, sem_win, sem_min, sem_max,
                   device, dtype, backend, quantization, offload_ar, budget, ode_steps,
                   vae_core_frames, model, vae_choice, vae_custom, revision, vae_revision,
                   offline, progress=gr.Progress()):
    """Generator: runs a JSONL queue; disables the batch button while running."""
    text = ""
    base = Path.cwd()
    if jsonl_file:
        base = Path(jsonl_file).parent
        text = Path(jsonl_file).read_text(encoding="utf-8")
    elif jsonl_text.strip():
        text = jsonl_text
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows:
        raise gr.Error("Provide JSONL (one request per line) or upload a .jsonl file")
    ids = [r.get("id") for r in rows]
    if any(x is None for x in ids) or len(set(ids)) != len(ids):
        raise gr.Error("Every line needs a unique id")
    abc_sampling = _sampling(abc_temp, abc_p, abc_k, abc_rep, abc_win, abc_min, abc_max, "ABC phase")
    sem_sampling = _sampling(sem_temp, sem_p, sem_k, sem_rep, sem_win, sem_min, sem_max, "semantic phase")

    _CANCEL.clear()
    if not _RUNNING.acquire(blocking=False):
        yield gr.update(), "Another job is already running — wait for it to finish", \
            gr.update(), gr.skip()
        return
    yield gr.update(), "Starting batch…", gr.update(interactive=False), gr.skip()
    try:
        pipe, note = _get_pipe(device, dtype, backend, quantization, offload_ar, budget,
                               ode_steps, vae_core_frames, model, vae_choice, vae_custom,
                               revision, vae_revision, offline, progress)
        outdir = RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}-batch-{_slug(out_id or 'batch')}"
        outdir.mkdir(parents=True, exist_ok=True)
        allowed = {"style", "tags", "lyrics", "cot", "seed", "abc", "cfg_scale", "id"}
        results, failures = [], 0
        batch_start = time.perf_counter()
        for index, row in enumerate(rows, 1):
            if _CANCEL.is_set():
                results.append([row.get("id"), "cancelled", "", "", ""])
                break
            row_start = time.perf_counter()
            kwargs = {k: v for k, v in row.items() if k in allowed and k != "id"}
            if "abc_path" in row:
                kwargs["abc"] = (base / row["abc_path"]).read_text(encoding="utf-8")
            try:
                request = adapter.song_request(id=row["id"], **kwargs)
            except (ValueError, TypeError) as exc:
                results.append([row["id"], f"invalid: {exc}", "", "", ""])
                failures += 1
                continue
            a_s = _sampling(*((row.get("abc_sampling") or {}).get(k, getattr(abc_sampling, k))
                              for k in adapter.sampling_fields(abc_sampling)), "ABC phase")
            s_s = _sampling(*((row.get("semantic_sampling") or {}).get(k, getattr(sem_sampling, k))
                              for k in adapter.sampling_fields(sem_sampling)), "semantic phase")

            def on_token(phase, token, index=index):
                progress((index - 1 + 0.5) / len(rows), desc=f"Song {index}/{len(rows)}: {phase}")

            def on_progress(stage, done, total, index=index):
                span = 1.0 / len(rows)
                if stage == "nar":
                    inner = 0.55 + 0.40 * min(1.0, done / max(1, total))
                    desc = f"Song {index}/{len(rows)}: synthesizing {done}/{total}"
                else:
                    inner = 0.95 + 0.05 * min(1.0, done / max(1, total))
                    desc = f"Song {index}/{len(rows)}: decoding {done}/{total}"
                progress(min(1.0, (index - 1) * span + span * inner), desc=desc)

            try:
                song = adapter.generate(pipe, request, abc_sampling=a_s, semantic_sampling=s_s,
                                        cancelled=_CANCEL.is_set, on_token=on_token,
                                        on_progress=on_progress)
                receipt = song.save_artifacts(outdir / row["id"])
                _write_local_env(outdir / row["id"], pipe, note)
                results.append([row["id"], "complete", f"{receipt['audio_seconds']:.1f}s",
                                f"{time.perf_counter() - row_start:.0f}s", str(outdir / row["id"])])
            except InterruptedError:
                results.append([row["id"], "cancelled", "",
                                f"{time.perf_counter() - row_start:.0f}s", ""])
                break
            except Exception as exc:  # noqa: BLE001
                failures += 1
                results.append([row["id"], f"failed: {type(exc).__name__}: {exc}", "",
                                f"{time.perf_counter() - row_start:.0f}s", ""])
            progress(index / len(rows), desc=f"Completed {index}/{len(rows)}")
        total = time.perf_counter() - batch_start
        status = (f"Batch finished: {len(results)} rows, {failures} failed in {total:.0f}s "
                  f"({total / max(1, len(results)):.0f}s per song)\n"
                  f"run directory: {outdir}\n{note}")
        yield results, status, gr.update(interactive=True), str(outdir)
    except Exception as exc:  # noqa: BLE001
        yield gr.update(), f"Batch failed: {type(exc).__name__}: {exc}", \
            gr.update(interactive=True), gr.skip()
    finally:
        _RUNNING.release()


# ─────────────────────────── tools ───────────────────────────

def abc_inspect(text):
    if not (text or "").strip():
        raise gr.Error("Paste an ABC score first")
    try:
        tools = abc_tools
        report = tools.report(tools.parse_abc(text))
    except ValueError as exc:
        raise gr.Error(f"Invalid ABC: {exc}") from exc
    return json.dumps(report, ensure_ascii=False, indent=2, default=tools.json_value)


def abc_strip_chords(text, keep_voice):
    if not (text or "").strip():
        raise gr.Error("Paste an ABC score first")
    try:
        return abc_tools.strip_chords(text, keep_voice=keep_voice)
    except ValueError as exc:
        raise gr.Error(f"Processing failed: {exc}") from exc


def abc_compare(before, after, voices, allow_tempo):
    tools = abc_tools
    if not (before or "").strip() or not (after or "").strip():
        raise gr.Error("Provide both the original and edited ABC")
    try:
        result = tools.compare(tools.parse_abc(before), tools.parse_abc(after),
                               names=tools.VOICES if voices == "both" else (voices,),
                               allow_tempo_change=bool(allow_tempo))
    except ValueError as exc:
        raise gr.Error(f"Comparison failed: {exc}") from exc
    return json.dumps(result, ensure_ascii=False, indent=2)


def run_doctor(model, vae_choice, vae_custom, revision, vae_revision, offline, verify):
    vae_path, _ = resolve_vae(vae_choice, vae_custom)
    cmd = adapter.doctor_command(model, vae_path, revision=revision, vae_revision=vae_revision,
                                 offline=bool(offline), verify=bool(verify))
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, check=False)
    return res.stdout.strip() or res.stderr.strip()


def make_comparison(paths_text, progress=gr.Progress()):
    sources = [p.strip() for p in (paths_text or "").splitlines() if p.strip()]
    if not sources:
        raise gr.Error("List one saved run directory per line")
    for src in sources:
        if not (Path(src) / "result.json").is_file():
            raise gr.Error(f"Not a valid YuE2 run directory (result.json missing): {src}")
    outdir = RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}-comparison"
    progress(0.2, desc="Building listening comparison…")
    cmd = [sys.executable, "-m", "yue2_groove.vendor.listen", *sources, "--output", str(outdir)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, check=False)
    if res.returncode not in (0, 1):
        raise gr.Error(f"Build failed: {res.stderr.strip()}")
    html_path = outdir / "index.html"
    served = f"/gradio_api/file={html_path}"
    link = (f'<a href="{served}" target="_blank" rel="noopener">'
            f'OPEN COMPARISON PAGE ↗</a>')
    status = (f"{res.stdout.strip()}\n"
              f"Click the link above to open it in a new tab, or paste one of:\n"
              f"  {served}\n"
              f"  file://{html_path}")
    return str(html_path), link, status


# ───────────────── cover tab (see sheetsage_adapter.py / cover.py) ─────────────────

def _score_panel(prefix: str, empty: str, elem_id: str = "", abc: str | None = None) -> str:
    """HTML for one abcjs score panel; SCORE_JS fills the .bb-score-inner div.

    When *abc* is given the score is rendered straight from that text (a read-only
    panel, e.g. SONG).  Otherwise SCORE_JS finds the textarea whose label starts
    with *prefix* — the editable Studio panels keep working that way.
    """
    panel_id = f' id="{html.escape(elem_id, quote=True)}"' if elem_id else ""
    message = html.escape(empty)
    inline = (f' data-bb-abc-text="{html.escape(abc, quote=True)}"'
              if abc is not None else "")
    return (f'<div class="bb-score-panel" data-bb-abc="{html.escape(prefix, quote=True)}" '
            f'data-bb-empty="{message}"{inline}{panel_id}>'
            f'<div class="bb-score-inner"><div class="bb-score-empty">{message}</div></div></div>')


def _sampling_summary_pair(*args):
    """``demo.load`` wrapper: one summary string per mirror output.

    Gradio requires exactly one return value per output component, and the page
    load updates both the EDIT and the COVER mirror.
    """
    text = _sampling_summary(*args)
    return text, text


def _sampling_summary(abc_temp, abc_p, abc_k, abc_rep, abc_win, abc_min, abc_max,
                      sem_temp, sem_p, sem_k, sem_rep, sem_win, sem_min, sem_max) -> str:
    """One-line view of the sampling parameters shared with 01 GENERATE."""
    return (f"ABC  t={float(abc_temp):g}  p={float(abc_p):g}  k={int(abc_k)}  "
            f"rep={float(abc_rep):g}  win={int(abc_win)}  min={int(abc_min)}  max={int(abc_max)}\n"
            f"SEM  t={float(sem_temp):g}  p={float(sem_p):g}  k={int(sem_k)}  "
            f"rep={float(sem_rep):g}  win={int(sem_win)}  min={int(sem_min)}  max={int(sem_max)}")


def _transcription_files(directory) -> list[str]:
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return [str(p) for p in sorted(directory.rglob("*")) if p.is_file()]


def cover_check_environment():
    """CHECK ENVIRONMENT: probe the SheetSage2 venv without loading any model."""
    try:
        return sheetsage_adapter.format_probe(sheetsage_adapter.probe())
    except (sheetsage_adapter.SheetsageNotConfigured, sheetsage_adapter.SheetsageFailed) as exc:
        return f"SheetSage2 environment not ready: {exc}"


def cover_unload_worker():
    """Drop the resident SheetSage2 worker (freeing its memory) if there is one."""
    stopped = sheetsage_adapter.stop_worker()
    return ("SheetSage2 worker unloaded; the next transcription loads the model again."
            if stopped else "No resident SheetSage2 worker.")


def cover_transcribe(audio_path, task, max_seconds, model, device, dtype, revision, base_model,
                     keep_warm, offline, progress=gr.Progress()):
    """Generator: transcription disables the TRANSCRIBE button while running."""
    if not (audio_path or "").strip():
        raise gr.Error("Upload a source audio file first")
    if task not in sheetsage_adapter.TASKS:
        raise gr.Error(f"Unknown transcription task: {task}")
    _CANCEL.clear()
    if not _RUNNING.acquire(blocking=False):
        yield (gr.update(), gr.update(),
               "Another job is already running — wait for it to finish",
               *((gr.update(),) * 7), gr.update(), gr.update(), gr.skip())
        return
    controls = (gr.update(interactive=False),) * 7
    idle = (gr.update(interactive=True),) * 7
    try:
        yield gr.update(), gr.update(), "Starting SheetSage2 transcription…", *controls, \
            gr.update(), gr.update(), gr.skip()
        outdir = config.transcriptions_dir(RUNS) / \
            f"{time.strftime('%Y%m%d-%H%M%S')}-{_slug(Path(audio_path).stem)}"

        def on_progress(value, text):
            progress(value, desc=text)

        record = sheetsage_adapter.transcribe(
            audio_path, output_dir=outdir, task=task,
            model=(model or "").strip() or None, revision=(revision or "").strip() or None,
            base_model=(base_model or "").strip() or None, keep_warm=bool(keep_warm),
            offline=bool(offline), device=device, dtype=dtype,
            max_seconds=float(max_seconds) if max_seconds else None,
            cancelled=_CANCEL.is_set, progress=on_progress)
        abc = record.get("abc") or ""
        warnings = record.get("warnings") or []
        lines = [f"Transcribed (task={record.get('task')}) → {record['output_dir']}",
                 (f"device={record.get('device')} dtype={record.get('dtype')}  "
                  f"melody_only={record.get('melody_only')}"),
                 "Review the ABC before covering; transcription can contain musical errors."]
        if warnings:
            lines.append("warnings: " + "; ".join(str(w) for w in warnings))
        worker = sheetsage_adapter.worker_status()
        if worker:
            lines.append(f"SheetSage2 worker resident (pid {worker['pid']}) — reused by the next "
                         f"transcription until UNLOAD or the idle timeout.")
        # surface the direct-generation path now that there is a score to use,
        # and select the new transcription as the reusable source
        _items, choices = _library_choices(_library_mode("time", "desc"))
        known = {value for _label, value in choices}
        try:
            rel = Path(record["output_dir"]).resolve().relative_to(RUNS.resolve()).as_posix()
        except (ValueError, OSError):
            rel = None
        source_update = (gr.update(choices=choices, value=rel) if rel in known
                         else gr.update(choices=choices))
        yield (abc, _transcription_files(record["output_dir"]), "\n".join(lines),
               *idle, gr.update(open=True), source_update, record["output_dir"])
    except InterruptedError as exc:
        note = (" The resident SheetSage2 worker was stopped; the next transcription reloads it."
                if keep_warm else "")
        yield gr.update(), gr.update(), f"Cancelled: {exc}.{note}", *idle, gr.update(), \
            gr.update(), gr.skip()
    except Exception as exc:  # noqa: BLE001
        yield gr.update(), gr.update(), \
            f"Transcription failed: {type(exc).__name__}: {exc}", *idle, gr.update(), \
            gr.update(), gr.skip()
    finally:
        _RUNNING.release()


def cover_choices():
    """Choices for the COVER source dropdown (any saved work or transcription with an ABC)."""
    _items, choices = _library_choices(_library_mode("time", "desc"), include_pending=False)
    return gr.update(choices=choices)


def cover_load(rel):
    """Fill COVER ABC (and STYLE/LYRICS when the source has them) from a saved work."""
    if not (rel or "").strip():
        raise gr.Error("Pick a SOURCE WORK first (press REFRESH if the list is empty)")
    item, det = library.load(RUNS, rel)
    if item is None or det is None:
        raise gr.Error("That work no longer exists — press REFRESH")
    abc = (det.get("abc") or "").strip()
    if not abc:
        raise gr.Error(f"{item['name']} has no ABC score to load")
    request = det.get("request") or {}
    style = request.get("style") or request.get("tags") or ""
    lyrics = request.get("lyrics") or ""
    return (gr.update(value=abc),
            gr.update(value=style) if style.strip() else gr.update(),
            gr.update(value=lyrics) if lyrics.strip() else gr.update(),
            (f"Loaded {item['name']} ({item['kind']}) — {len(abc)} chars. Continue here or "
             f"press SEND TO EDIT."))


def cover_send_to_edit(abc_text, rel, style, lyrics):
    """Hand the current COVER score to 03 EDIT, keeping the source as the freeze target."""
    if not (abc_text or "").strip():
        raise gr.Error("Transcribe or load a score first")
    if not (rel or "").strip():
        raise gr.Error("LOAD a source work (or transcribe) first — 03 EDIT freezes that source "
                       "before it can check the edit")
    item, det = library.load(RUNS, rel)
    if item is None or det is None:
        raise gr.Error("That source work no longer exists — refresh 02 COVER")
    source_abc = (det.get("abc") or "").strip()
    if not source_abc:
        raise gr.Error(f"{item['name']} has no ABC to freeze as the edit baseline")
    try:
        current = edit_flow.validate_edited_abc(abc_text)
    except ValueError as exc:
        raise gr.Error(f"Cannot send this score to 03 EDIT: {exc}") from exc
    request = det.get("request") or {}
    style_value = (style or "").strip() or request.get("style") or request.get("tags") or ""
    lyrics_value = (lyrics or "").strip() or request.get("lyrics") or ""
    return (gr.update(value=current),          # EDITED ABC (what you see in COVER)
            source_abc,                        # baseline ABC for CHECK INVARIANTS
            gr.update(value=style_value) if style_value else gr.update(),
            gr.update(value=lyrics_value) if lyrics_value else gr.update(),
            rel,                               # SOURCE WORK in 03 EDIT (state)
            {},                                # check state reset
            None,                              # not frozen yet: FREEZE BASELINE is required
            f"From 02 COVER: baseline = {rel}. Press FREEZE BASELINE, then CHECK INVARIANTS.",
            "Loaded from 02 COVER — FREEZE BASELINE is required before CHECK / GENERATE EDITED.",
            gr.update(choices=[value for _label, value in
                               _library_choices(_library_mode("time", "desc"))[1]], value=rel),
            str((RUNS / rel).resolve()), gr.update(selected="edit"))


def cover_generate(style, lyrics, abc_text, task, keep_voice, seed, cfg_scale,
                   abc_temp, abc_p, abc_k, abc_rep, abc_win, abc_min, abc_max,
                   sem_temp, sem_p, sem_k, sem_rep, sem_win, sem_min, sem_max,
                   device, dtype, backend, quantization, offload_ar, budget, ode_steps,
                   vae_core_frames, model, vae_choice, vae_custom, revision, vae_revision, offline,
                   progress=gr.Progress()):
    """Generator: generate directly from the score on 02 COVER (no tab detour).

    Uses the same ``cover.build_cover_request`` as SEND TO GENERATE (melody tasks get a
    chord-free score, full tasks keep the harmony) and the shared generation core.
    """
    if not (abc_text or "").strip():
        raise gr.Error("Transcribe (or paste) an ABC score first — GENERATE COVER is "
                       "score-conditioned and never plans a fresh melody")
    style, lyrics = _request_texts(style, lyrics)
    try:
        request = cover.build_cover_request(style, lyrics, abc_text, task=task, seed=int(seed),
                                            cfg_scale=cfg_scale, keep_voice=keep_voice,
                                            request_factory=adapter.song_request)
    except (ValueError, TypeError) as exc:
        raise gr.Error(f"Invalid cover request: {exc}") from exc
    abc_sampling = _sampling(abc_temp, abc_p, abc_k, abc_rep, abc_win, abc_min, abc_max, "ABC phase")
    sem_sampling = _sampling(sem_temp, sem_p, sem_k, sem_rep, sem_win, sem_min, sem_max,
                             "semantic phase")

    _CANCEL.clear()
    if not _RUNNING.acquire(blocking=False):
        yield ("Another job is already running — wait for it to finish",
               gr.update(), gr.update(), gr.update(), *((gr.update(),) * 7), gr.skip())
        return
    controls = (gr.update(interactive=False),) * 7
    idle = (gr.update(interactive=True),) * 7
    try:
        yield ("Starting cover generation…", gr.update(), gr.update(), gr.update(), *controls,
               gr.skip())
        pipe, note = _get_pipe(device, dtype, backend, quantization, offload_ar, budget,
                               ode_steps, vae_core_frames, model, vae_choice, vae_custom,
                               revision, vae_revision, offline, progress)
        outdir = RUNS / (f"{time.strftime('%Y%m%d-%H%M%S')}-cover-"
                         f"{_slug(request.id if request.id != 'song' else style)}")
        song, result, elapsed = _run_generation(
            pipe, request, outdir, abc_sampling=abc_sampling, semantic_sampling=sem_sampling,
            progress=progress, note=note)
        yield (_generation_status(song, result, outdir, elapsed, request, note),
               str(outdir / "audio.flac"), (song.abc or ""),
               _artifact_files(outdir, bool(song.abc)), *idle, str(outdir))
    except InterruptedError as exc:
        yield f"Cancelled: {exc}", gr.update(), gr.update(), gr.update(), *idle, gr.skip()
    except Exception as exc:  # noqa: BLE001
        yield (f"Generation failed: {type(exc).__name__}: {exc}",
               gr.update(), gr.update(), gr.update(), *idle, gr.skip())
    finally:
        _RUNNING.release()


def cover_strip(text, keep_voice):
    try:
        stripped = cover.prepare_cover_abc(text, keep_voice=keep_voice)
    except ValueError as exc:
        raise gr.Error(f"Chord strip failed: {exc}") from exc
    return stripped, (f"Chords removed (kept: {keep_voice}). Melody, meter and tempo were "
                      f"verified unchanged.")


def cover_send_to_generate(text, task, style, lyrics, keep_voice):
    """One click into 01 GENERATE: ABC, plan mode, and any target style/lyrics typed here."""
    try:
        cot = cover.cot_for_task(task)
        if cot == "melody":
            prepared = cover.prepare_cover_abc(text, keep_voice=keep_voice)
        else:
            cover.inspect_abc(text)              # validate before handing it over
            prepared = cover.clean_abc(text)
        if not prepared.strip():
            raise ValueError("There is no ABC score to send")
    except ValueError as exc:
        raise gr.Error(f"Cannot send this score to GENERATE: {exc}") from exc
    copied = []
    style_update = gr.update()
    lyrics_update = gr.update()
    if (style or "").strip():
        style_update = gr.update(value=style.strip())
        copied.append("STYLE")
    if (lyrics or "").strip():
        lyrics_update = gr.update(value=lyrics.strip())
        copied.append("LYRICS")
    what = f" and the {', '.join(copied)} typed here" if copied else ""
    detail = "chord-free (melody mode)" if cot == "melody" else "full score (with chords)"
    status = (f"Sent to 01 GENERATE: PLAN MODE={cot.upper()}, the {detail} ABC{what}. "
              f"SCORE INPUT is open so you can review it; add anything still missing, "
              f"then press GENERATE.")
    return (gr.update(value=prepared), gr.update(value=cot), style_update, lyrics_update,
            gr.update(open=True), gr.update(selected="gen"), status)


# ───────────────── edit tab (see edit_flow.py) ─────────────────

BUSY_HTML = ('<div id="bb-busy" role="status" aria-live="polite">'
             '● JOB RUNNING — a second job is refused until it finishes; use CANCEL on the '
             'running tab to stop it'
             '</div>')


def busy_banner():
    """Global busy indicator polled by a gr.Timer (no handler signature changes)."""
    return BUSY_HTML if _RUNNING.locked() else ""


def _sheetsage_python_candidates() -> list[Path]:
    repo = Path(__file__).resolve().parent.parent
    return [repo / ".venv-sheetsage2" / "bin" / "python",
            repo.parent / "YuE" / ".venv-sheetsage2" / "bin" / "python"]


def cover_detect_python():
    """Look for a SheetSage2 venv in the documented locations and use it for this session."""
    found = next((path for path in _sheetsage_python_candidates() if path.is_file()), None)
    if found is None:
        return ("No SheetSage2 venv found at ./.venv-sheetsage2 or ../YuE/.venv-sheetsage2 — "
                "install one (README, 'Cover from audio') or set YUE2_GROOVE_SHEETSAGE_PYTHON.")
    os.environ["YUE2_GROOVE_SHEETSAGE_PYTHON"] = str(found)
    return f"Using {found} for this session — add it to .env (or --sheetsage-python) to persist."


def mirror_current(value):
    """Forward a *_last_run state into the hidden current-work bridge."""
    return (value or "").strip() or gr.update()


def _abs_of_run(value) -> str:
    """Absolute path of a run given an absolute path, a Library rel or empty."""
    if not (value or "").strip():
        return ""
    return str((RUNS / _rel_of_run(value)).resolve())


def publish_current(path):
    """Band for the hidden current-work bridge; clears an invalid stored path once."""
    text = workflow.band(RUNS, path)
    if (path or "").strip() and not text:
        return "", gr.update(value="")      # (band, bridge): stale entry — clear both
    return text, gr.update()                # band rendered, bridge untouched


def _rel_of_run(value) -> str:
    """Accept a run directory (absolute) or a Library rel and return the rel."""
    text = (value or "").strip()
    if not text:
        raise gr.Error("Pick or generate a work first")
    candidate = Path(text)
    if candidate.is_absolute():
        try:
            return candidate.resolve().relative_to(RUNS.resolve()).as_posix()
        except (ValueError, OSError):
            raise gr.Error("That run is outside the runs directory") from None
    return text


def library_open_in_edit(active):
    """Load the viewed/generated work into 03 EDIT and switch there."""
    rel = _rel_of_run(active)
    (style, lyrics, abc, baseline_abc, source_rel, check_state, baseline_state, info,
     status) = edit_load(rel)
    choices = gr.update(choices=[value for _label, value in
                                 _library_choices(_library_mode("time", "desc"))[1]], value=rel)
    return (style, lyrics, abc, baseline_abc, source_rel, check_state, baseline_state, info,
            status, choices, str((RUNS / rel).resolve()), gr.update(selected="edit"))


def library_use_in_cover(active):
    """Load the viewed/generated work into 02 COVER and switch there."""
    rel = _rel_of_run(active)
    abc, style, lyrics, status = cover_load(rel)
    choices = gr.update(choices=[value for _label, value in
                                 _library_choices(_library_mode("time", "desc"))[1]], value=rel)
    return abc, style, lyrics, status, choices, str((RUNS / rel).resolve()), \
        gr.update(selected="cover")


def open_last_in_library(active):
    """Show a freshly generated run in 04 LIBRARY (list, details, player)."""
    rel = _rel_of_run(active)
    _items, choices = _library_choices(_library_mode("time", "desc"))
    info, style, lyrics, abc, rename_box, rename_btn, status = _library_details([rel])
    return (gr.update(choices=choices, value=[rel]), info, style, lyrics, abc, rename_box,
            rename_btn, status, gr.update(selected="library"), rel, str((RUNS / rel).resolve()))


# ───────────────── SONG view (director; see workflow.py for identity) ─────────────────
# The SONG view only states facts and offers actions that map to existing Studio
# handlers — it holds no editable component (no textbox / radio / slider / number /
# checkbox), so there is no second copy of STYLE / LYRICS / ABC / sampling state.
VIEW_CHOICES = ("song", "studio")
_SONG_ACTIONS = ("listen", "render", "edit", "retry", "check", "send", "library")
_SONG_STAGES = (("draft", "DRAFT"), ("score", "SCORE"), ("audio", "AUDIO"),
                ("revise", "REVISE"), ("done", "DONE"))


def resolve_view(cli_view=None, cli_tab=None, env_view=None) -> tuple[str, int]:
    """Resolve the startup view: ``(view_mode, tab)``.

    ``view_mode`` is ``auto`` (the last stored choice, else the SONG default),
    ``song`` or ``studio``.  An explicit ``--tab N`` forces the Studio view on
    that tab; ``--view`` / ``YUE2_GROOVE_VIEW`` force a view for a fresh session.
    """
    if cli_tab is not None:
        return "studio", int(cli_tab)
    cli = (cli_view or "").strip().lower()
    if cli in VIEW_CHOICES:
        return cli, 0
    env = (env_view or "").strip().lower()
    if env in VIEW_CHOICES:
        return env, 0
    return "auto", 0


def _song_work(active):
    """Identify the current work from a bridge value, or ``None``."""
    if not (active or "").strip():
        return None
    try:
        return workflow.identify(RUNS, _abs_of_run(active))
    except gr.Error:
        return None


def _song_artifact(work: dict, name: str):
    """One artifact of a work, looking into the children of a group run."""
    paths = [Path(work["path"])]
    paths += [RUNS / child for child in work.get("children") or []]
    for base in paths:
        candidate = base / name
        if candidate.is_file():
            return candidate
    return None


def _song_audio(work: dict):
    path = _song_artifact(work, "audio.flac")
    return str(path) if path is not None else None


def _song_abc(work: dict) -> str:
    path = _song_artifact(work, "score.abc")
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _song_stage_track(stage: str) -> str:
    """DRAFT → SCORE → AUDIO → REVISE → DONE, with the current stage highlighted.

    DONE is only ever reached through ``finished.json`` (``workflow.identify``),
    so the last cell stays unpainted unless the artifact says the work is done.
    """
    keys = [key for key, _label in _SONG_STAGES]
    current = keys.index(stage) if stage in keys else 0
    cells = []
    for index, (key, label) in enumerate(_SONG_STAGES):
        classes = ["bb-stage"]
        if index == current:
            classes.append("bb-stage-current")
        elif index < current:
            classes.append("bb-stage-past")
        cells.append(f'<span class="{" ".join(classes)}">{label}</span>')
    return '<div id="bb-song-stage">' + \
        '<span class="bb-stage-sep">→</span>'.join(cells) + "</div>"


def _song_identity(work: dict) -> str:
    return (
        '<div id="bb-song-identity">'
        f'<div class="bb-song-title">{html.escape(work["title"])}</div>'
        '<div class="bb-song-meta">'
        f'<span>KIND <b>{html.escape(work["kind_label"])}</b></span>'
        f'<span>STAGE <b>{html.escape(work["stage_label"])}</b></span>'
        f'<span>LAST <b>{html.escape(workflow.last_event(work))}</b></span>'
        f'<span>RUN <b>{html.escape(work["rel"])}</b></span>'
        "</div></div>"
    )


def _song_family(entries) -> str:
    if not entries:
        return ""
    rows = []
    for entry in entries:
        confidence = entry.get("confidence", "heuristic")
        tag = "exact" if confidence == "exact" else "possibly related"
        relations = html.escape(", ".join(entry.get("relations") or []))
        rows.append(
            '<div class="bb-family-item">'
            f'<b class="bb-family-hit" data-bb-run="{html.escape(entry["path"], quote=True)}">'
            f'{html.escape(entry["title"])}</b> '
            f'<span class="bb-family-rel">[{tag}] {relations}</span></div>'
        )
    return ('<div id="bb-song-family"><div class="bb-score-title">FAMILY</div>'
            '<div class="bb-family-list">'
            + "".join(rows) + "</div></div>")


def render_song(active):
    """Everything the SONG view shows for the current work, or the empty state."""
    work = _song_work(active)
    if work is None:
        return (
            gr.update(visible=True),                          # song_empty
            gr.update(visible=False),                         # song_work
            "", "", "",                                        # identity, stage, family
            gr.update(value=None, visible=False),             # song_player
            _score_panel("SONG SCORE", "No current work.", abc=""),
            *[gr.update(visible=False) for _ in _SONG_ACTIONS],
            gr.update(visible=False),                         # song_studio_btn
            gr.update(visible=False),                         # song_compare_btn
        )
    actions = workflow.next_actions(work)
    ids = {action["id"] for action in actions}
    updates = []
    for key in _SONG_ACTIONS:
        if key == "retry":
            seed = next((a.get("seed") for a in actions if a["id"] == "retry"), None)
            label = f"TRY SEED {seed}" if isinstance(seed, int) else "TRY ANOTHER SEED"
            updates.append(gr.update(visible=key in ids, value=label))
        else:
            updates.append(gr.update(visible=key in ids))
    audio = _song_audio(work)
    family = workflow.family(RUNS, work)
    comparable = bool(work.get("source_rel") or work.get("baseline_rel"))
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        _song_identity(work),
        _song_stage_track(work["stage"]),
        _song_family(family),
        gr.update(value=audio, visible=bool(audio)),
        _score_panel("SONG SCORE", "No score for this work.", abc=_song_abc(work)),
        *updates,
        gr.update(visible=True),
        gr.update(visible=comparable),
    )


def song_render_action(active):
    """RENDER: open the Studio surface that renders this work's score.

    A plan goes to 01 GENERATE with the score attached; a transcription goes to
    02 COVER, where score-conditioned generation lives.  Nothing is generated
    silently.  The two branches return the same 12 outputs (the unused Studio
    fields are left untouched) so one Gradio event can serve both kinds.
    """
    work = _song_work(active)
    if work is None:
        raise gr.Error("That work no longer exists — refresh SONG")
    noop = gr.update()
    current = str(Path(work["path"]).resolve())
    if work["kind"] == "transcription":
        abc, style, lyrics, status, choices, _current, tabs = library_use_in_cover(work["rel"])
        return (noop, noop, noop, noop, abc, style, lyrics, status, choices, current, tabs,
                gr.update(value="studio"))
    request = work.get("request") or {}
    return (
        gr.update(value=_song_abc(work)),                 # abc
        gr.update(open=True),                             # score_input_accordion
        gr.update(value=request.get("style") or ""),     # style
        gr.update(value=request.get("lyrics") or ""),    # lyrics
        noop, noop, noop, noop,                           # cover abc/style/lyrics/status
        noop,                                             # cover_source
        current,                                          # current_bridge
        gr.update(selected="gen"),                       # tabs
        gr.update(value="studio"),                       # view_bridge
    )


def song_retry(active):
    """TRY ANOTHER SEED: prefill 01 GENERATE with the same request and seed + 1."""
    work = _song_work(active)
    if work is None:
        raise gr.Error("That work no longer exists — refresh SONG")
    request = work.get("request") or {}
    seed = request.get("seed")
    next_seed = seed + 1 if isinstance(seed, int) else 831001
    cot = request.get("cot") if request.get("cot") in ("full", "melody", "off") else "full"
    return (
        gr.update(value=request.get("style") or ""),     # style
        gr.update(value=request.get("lyrics") or ""),    # lyrics
        gr.update(value=cot),                             # cot
        gr.update(value=next_seed),                       # seed
        str(Path(work["path"]).resolve()),               # current_bridge
        gr.update(selected="gen"),                       # tabs
        gr.update(value="studio"),                       # view_bridge
    )


def song_send(active):
    """SEND TO GENERATE: the existing cover → generate handoff, with no cover UI."""
    work = _song_work(active)
    if work is None:
        raise gr.Error("That work no longer exists — refresh SONG")
    abc_text = _song_abc(work)
    if not abc_text.strip():
        raise gr.Error("That transcription has no score to send")
    request = work.get("request") or {}
    task = request.get("task") or "melody-full"
    abc, cot, style, lyrics, accordion, tabs, status = cover_send_to_generate(
        abc_text, task, request.get("style") or "", request.get("lyrics") or "", "both")
    return (abc, cot, style, lyrics, accordion, tabs, status, gr.update(value="studio"))


def song_open_edit(active):
    """EDIT / CHECK: the existing library → edit handoff (baseline + invariant section)."""
    return (*library_open_in_edit(active), gr.update(value="studio"))


def song_open_library(active):
    """OPEN IN LIBRARY: the existing library detail + player."""
    return (*open_last_in_library(active), gr.update(value="studio"))


def song_open_studio(active):
    """OPEN IN STUDIO: the last surface that makes sense for this work."""
    work = _song_work(active)
    if work is None:
        return gr.update(selected="gen"), gr.update(value="studio")
    if work["kind"] == "transcription":
        tab = "cover"
    elif work["kind"] == "plan":
        tab = "gen"
    else:
        tab = "library"
    return gr.update(selected=tab), gr.update(value="studio")


def song_compare(active):
    """BUILD COMPARISON: source/baseline vs the current work, via the compare helper."""
    work = _song_work(active)
    if work is None:
        raise gr.Error("That work no longer exists — refresh SONG")
    other = work.get("source_rel") or work.get("baseline_rel")
    if not other:
        raise gr.Error("No baseline or source to compare against")
    source = (RUNS / other).resolve()
    if not source.is_dir():
        raise gr.Error("The source work for this comparison is missing")
    _path, link, status = make_comparison(f"{source}\n{Path(work['path']).resolve()}")
    # stay in SONG: the link opens in a new tab and the status is readable here
    return link, status


def edit_choices():
    _items, choices = _library_choices(_library_mode("time", "desc"), include_pending=False)
    return gr.update(choices=choices)


def edit_load(rel):
    """Load one saved work as the edit source; its ABC becomes the baseline."""
    if not (rel or "").strip():
        raise gr.Error("Pick a source work first")
    item, det = library.load(RUNS, rel)
    if item is None or det is None:
        raise gr.Error("That work no longer exists — press REFRESH")
    abc = (det.get("abc") or "").strip()
    if not abc:
        raise gr.Error(f"{item['name']} has no ABC score to edit")
    request = det.get("request") or {}
    style = request.get("style") or request.get("tags") or ""
    info = (f"{item['name']} · {rel}\n"
            "baseline not frozen yet — FREEZE BASELINE is required before CHECK INVARIANTS "
            "and GENERATE EDITED (the original run directory is never modified).")
    return (gr.update(value=style),
            gr.update(value=request.get("lyrics", "") or ""),
            gr.update(value=abc), abc, rel, {}, None, info,
            f"Loaded {item['name']} as the edit source.")


def edit_freeze(source_rel, visible_rel=""):
    if not (source_rel or "").strip():
        raise gr.Error("Load a source work first")
    if (visible_rel or "").strip() and visible_rel.strip() != source_rel.strip():
        raise gr.Error(f"SOURCE WORK now shows “{visible_rel.strip()}” but “{source_rel.strip()}” "
                       f"is loaded — press LOAD first so the baseline you freeze is the one you "
                       f"see")
    try:
        record = edit_flow.freeze_baseline(RUNS, source_rel)
    except (ValueError, OSError) as exc:
        raise gr.Error(f"Freeze failed: {exc}") from exc
    info = (f"{source_rel} → {record['baseline']['rel']}\n"
            f"recorded {len(record['hashes'])} file hash(es) + copies of score.abc/request.json; "
            "the original run directory is untouched.")
    return record, info, f"Baseline frozen: {record['baseline']['rel']}"


def edit_check(baseline_abc, abc_text, voices, allow_tempo, baseline_state, contract, allow_meter):
    if not (baseline_abc or "").strip():
        raise gr.Error("Load a source work first — the check compares against its baseline ABC")
    if not baseline_state:
        raise gr.Error("Freeze the baseline first (FREEZE BASELINE) — the check is recorded "
                       "against that frozen record")
    try:
        result = edit_flow.check_invariants(baseline_abc, abc_text, voices=voices,
                                            allow_tempo_change=bool(allow_tempo),
                                            allow_meter_change=bool(allow_meter),
                                            contract=contract or "exact")
    except ValueError as exc:
        raise gr.Error(f"Invariant check failed: {exc}") from exc
    state = {"sha256": edit_flow.sha256_text(edit_flow.clean_abc(abc_text)),
             "match": bool(result["match"]), "result": result,
             "voices": voices, "allow_tempo_change": bool(allow_tempo),
             "allow_meter_change": bool(allow_meter), "contract": result["contract"]}
    return json.dumps(result, ensure_ascii=False, indent=2), state


def edit_generate(style, lyrics, cot, seed, cfg_scale, abc_text, baseline_abc, source_rel,
                  check_state, allow_changes, baseline_state,
                  abc_temp, abc_p, abc_k, abc_rep, abc_win, abc_min, abc_max,
                  sem_temp, sem_p, sem_k, sem_rep, sem_win, sem_min, sem_max,
                  device, dtype, backend, quantization, offload_ar, budget, ode_steps,
                  vae_core_frames, model, vae_choice, vae_custom, revision, vae_revision, offline,
                  progress=gr.Progress()):
    """Generator: regenerate from the edited score; never silently drops the edit."""
    try:
        abc = edit_flow.validate_edited_abc(abc_text)
    except ValueError as exc:
        raise gr.Error(f"The edited ABC cannot be used: {exc}") from exc
    if not (baseline_abc or "").strip():
        raise gr.Error("Load a source work first — 03 EDIT regenerates a saved work from its "
                       "edited score; use 01 GENERATE for a fresh song")
    if not baseline_state:
        raise gr.Error("Freeze the baseline first (FREEZE BASELINE) — the edit manifest must "
                       "point at the frozen source")
    if not check_state or check_state.get("sha256") != edit_flow.sha256_text(abc):
        raise gr.Error("Run CHECK INVARIANTS on the current edited ABC before generating")
    if not check_state.get("match") and not allow_changes:
        differences = "; ".join((check_state.get("result") or {}).get("differences", [])[:3])
        raise gr.Error("CHECK INVARIANTS did not pass: " + (differences or "scores differ") +
                       " — enable ALLOW MELODY/RHYTHM CHANGES if the change is intentional")
    style, lyrics = _request_texts(style, lyrics)
    abc_sampling = _sampling(abc_temp, abc_p, abc_k, abc_rep, abc_win, abc_min, abc_max, "ABC phase")
    sem_sampling = _sampling(sem_temp, sem_p, sem_k, sem_rep, sem_win, sem_min, sem_max,
                             "semantic phase")
    try:
        request = edit_flow.build_edit_request(style, lyrics, abc, cot=cot, seed=int(seed),
                                               cfg_scale=cfg_scale,
                                               request_factory=adapter.song_request)
    except (ValueError, TypeError) as exc:
        raise gr.Error(f"Invalid request: {exc}") from exc

    _CANCEL.clear()
    if not _RUNNING.acquire(blocking=False):
        yield (gr.update(), gr.update(), gr.update(),
               "Another job is already running — wait for it to finish",
               *((gr.update(),) * 6), gr.update())
        return
    busy = (gr.update(interactive=False),) * 6
    idle = (gr.update(interactive=True),) * 6
    try:
        yield gr.update(), gr.update(), gr.update(), "Starting edit generation…", *busy, \
            gr.update()
        pipe, note = _get_pipe(device, dtype, backend, quantization, offload_ar, budget,
                               ode_steps, vae_core_frames, model, vae_choice, vae_custom,
                               revision, vae_revision, offline, progress)
        outdir = RUNS / (f"{time.strftime('%Y%m%d-%H%M%S')}-edit-"
                         f"{_slug(request.id if request.id != 'song' else style)}")
        manifest = edit_flow.build_edit_manifest(
            source_rel=source_rel or "", before_abc=baseline_abc or "", after_abc=abc,
            cot=cot, seed=int(seed), cfg_scale=float(cfg_scale) if cfg_scale else None,
            invariants=(check_state or {}).get("result"),
            voices=(check_state or {}).get("voices", "both"),
            allow_tempo_change=bool((check_state or {}).get("allow_tempo_change")),
            allow_meter_change=bool((check_state or {}).get("allow_meter_change")),
            contract=(check_state or {}).get("contract", "exact"),
            allow_changes=bool(allow_changes), baseline=baseline_state)
        song, result, elapsed = _run_generation(
            pipe, request, outdir, abc_sampling=abc_sampling, semantic_sampling=sem_sampling,
            progress=progress, note=note, extra_manifest=manifest)
        status = _generation_status(song, result, outdir, elapsed, request, note) + \
            "\nedit_manifest.json records the source/edit hashes and the invariant result."
        yield str(outdir / "audio.flac"), (song.abc or ""), \
            _artifact_files(outdir, bool(song.abc)), status, *idle, str(outdir)
    except InterruptedError as exc:
        yield gr.update(), gr.update(), gr.update(), f"Cancelled: {exc}", *idle, gr.update()
    except Exception as exc:  # noqa: BLE001
        yield gr.update(), gr.update(), gr.update(), \
            f"Generation failed: {type(exc).__name__}: {exc}", *idle, gr.update()
    finally:
        _RUNNING.release()


def edit_compare(source_rel, last_run):
    if not (source_rel or "").strip():
        raise gr.Error("Load a source work first")
    if not (last_run or "").strip():
        raise gr.Error("Generate an edited version first")
    _item, det = library.load(RUNS, source_rel)
    if det is None:
        raise gr.Error("The source work no longer exists")
    return make_comparison(f"{det['path']}\n{last_run}")


# ───────────────── library tab (see library.py) ─────────
def _library_mode(sort_key, sort_dir):
    key = "name" if str(sort_key) == "name" else "time"
    direction = "asc" if str(sort_dir) == "asc" else "desc"
    return f"{key}_{direction}"


def _library_choices(sort_mode, include_pending=True):
    items = library.sort_items(library.scan(RUNS), sort_mode)
    if not include_pending:
        items = [item for item in items if not item.get("pending")]
    return items, [(library.label(item), item["rel"]) for item in items]


def _library_details(selected):
    """(info html, style, lyrics, abc, rename box, rename button, status)."""
    selected = list(selected or [])
    if not selected:
        return (library.render_empty_html("Select one work to see its details."), "", "", "",
                gr.update(value="", interactive=False), gr.update(interactive=False), "")
    if len(selected) > 1:
        return (library.render_multi_html(selected), "", "", "",
                gr.update(value="", interactive=False), gr.update(interactive=False),
                f"{len(selected)} selected — pick one to view details, or delete the selection.")
    item, det = library.load(RUNS, selected[0])
    if item is None:
        return (library.render_empty_html("That work no longer exists — refresh the list."), "", "", "",
                gr.update(value="", interactive=False), gr.update(interactive=False), "Not found.")
    request = det.get("request") or {}
    return (library.render_info_html(item, det), request.get("style", "") or "",
            request.get("lyrics", "") or "", det.get("abc", "") or "",
            gr.update(value=item["name"], interactive=True), gr.update(interactive=True),
            f"{item['name']} · {library.format_seconds(item.get('duration'))}")


def library_refresh(sort_key, sort_dir, selected=(), active=""):
    items, choices = _library_choices(_library_mode(sort_key, sort_dir))
    rels = {item["rel"] for item in items}
    keep = [rel for rel in (selected or ()) if rel in rels]
    keep_active = active if active in rels else ""
    info, style, lyrics, abc, rename_box, rename_btn, _ = _library_details(
        [keep_active] if keep_active else [])
    summary = (f"{len(items)} work(s) · {library.summarize(items)}" if items
               else "No works yet — generate something, then refresh.")
    return (gr.update(choices=choices, value=keep), info, style, lyrics, abc,
            rename_box, rename_btn, summary)


def library_view(active):
    """Row click: show details only. Selection (checkboxes) is a separate state."""
    info, style, lyrics, abc, rename_box, rename_btn, _ = _library_details(
        [active] if active else [])
    return info, style, lyrics, abc, rename_box, rename_btn


def library_rename(active, new_name, sort_key, sort_dir, selected=()):
    if not active:
        info, style, lyrics, abc, rename_box, rename_btn, _ = _library_details([])
        return (gr.update(), info, style, lyrics, abc, rename_box, rename_btn,
                "Click a work first, then rename it.", gr.update())
    ok, message, new_rel = library.rename(RUNS, active, new_name)
    items, choices = _library_choices(_library_mode(sort_key, sort_dir))
    existing = {item["rel"] for item in items}
    target = new_rel if (ok and new_rel) else active
    # renaming must not disturb the checkboxes: keep the other selections,
    # and follow the renamed work if it was checked
    keep = []
    for rel in (selected or ()):
        if rel == active:
            if target in existing and target not in keep:
                keep.append(target)
        elif rel in existing and rel not in keep:
            keep.append(rel)
    info, style, lyrics, abc, rename_box, rename_btn, _ = _library_details([target])
    return (gr.update(choices=choices, value=keep), info, style, lyrics, abc,
            rename_box, rename_btn, message, gr.update(value=target))


def library_delete_prepare(selected, sort_key, sort_dir):
    items, _ = _library_choices(_library_mode(sort_key, sort_dir))
    by_rel = {item["rel"]: item for item in items}
    chosen = [by_rel[rel] for rel in (selected or []) if rel in by_rel]
    if not chosen:
        return library.render_confirm_html([]), [], gr.update(interactive=False), "Nothing selected."
    return (library.render_confirm_html(chosen), [item["rel"] for item in chosen],
            gr.update(interactive=True), f"{len(chosen)} item(s) ready to delete — confirm below.")


def library_delete_confirm(pending, sort_key, sort_dir, active=""):
    _, message = library.delete(RUNS, pending or [])
    items, choices = _library_choices(_library_mode(sort_key, sort_dir))
    rels = {item["rel"] for item in items}
    keep_active = active if (active and active in rels) else ""
    if keep_active:
        info, style, lyrics, abc, rename_box, rename_btn, _ = _library_details([keep_active])
    else:
        empty = ("Select one work to see its details." if items
                 else "No works yet — generate something, then refresh.")
        info, style, lyrics, abc = library.render_empty_html(empty), "", "", ""
        rename_box, rename_btn = gr.update(value="", interactive=False), gr.update(interactive=False)
    return (gr.update(choices=choices, value=[]), info, style, lyrics, abc,
            rename_box, rename_btn,
            library.render_confirm_html([]), [], gr.update(interactive=False), message,
            gr.update(value=keep_active))


def library_delete_cancel():
    return library.render_confirm_html([]), [], gr.update(interactive=False), "Delete cancelled."



def apply_preset(name):
    presets = {
        "Protocol defaults (full)": (32, 4096, 200, 9000),
        "Preview (~1–1.5 min song)": (32, 700, 64, 2200),
        "Quick test (~20 s)": (16, 256, 32, 512),
    }
    a_min, a_max, s_min, s_max = presets[name]
    return (gr.update(value=a_min), gr.update(value=a_max),
            gr.update(value=s_min), gr.update(value=s_max))


def _load_project_example():
    """Example request that feeds the placeholders and the "fill example" buttons."""
    return (("English, warm piano pop, expressive female voice, acoustic piano, "
             "rounded bass and light drums, 88 BPM"),
            ("[Verse]\nNeon fades along the lane\nFootsteps keep the time of rain\n\n"
             "[Chorus]\nLet the day come into view\nEvery road begins with you"))


EXAMPLE_STYLE, EXAMPLE_LYRICS = _load_project_example()


def _request_texts(style, lyrics, *, fallback: bool = False):
    """Resolve STYLE/LYRICS; empty fields are an error unless *fallback* is set.

    The repository example is never substituted silently: it stays a placeholder
    and one E-button click away, but the request that runs is what the user typed.
    """
    style, lyrics = (style or "").strip(), (lyrics or "").strip()
    if fallback:
        return style or EXAMPLE_STYLE, lyrics or EXAMPLE_LYRICS
    missing = [name for name, value in (("STYLE", style), ("LYRICS", lyrics)) if not value]
    if missing:
        raise gr.Error(f"{' and '.join(missing)} empty — type a target, or click E to fill the "
                       f"repository example")
    return style, lyrics


# ─────────────────────── Bearbone DS v0.2, two scenes ───────────────────────
# dark scene:   warm near-black #0B0A09 ground + ivory #F1ECE2 ink (never pure black/white).
# bright scene: ivory #F1ECE2 ground + warm near-black #16140F ink; dark controls
# (primary buttons, selected chips/checkboxes, sliders) share the #11141C family.
# Shared grammar: 1px strokes, 10px panels, no shadows / gradients / glow, monospace, ops footer.

FONT_STACK = ["Berkeley Mono", "Sarasa Mono SC", "JetBrains Mono", "SF Mono",
              "Noto Sans Mono CJK SC", "ui-monospace", "Menlo", "monospace"]

DARK = {
    "bg": "#0B0A09", "bg_panel": "#12110F", "bg_input": "#171512", "bg_lift": "#1C1916",
    "fg": "#F1ECE2", "fg2": "#B5AEA2", "fg3": "#7A746A", "fg4": "#57524A",
    "stroke": "#2E2B27", "stroke2": "#8C8477",
    "primary_fill": "#F1ECE2", "primary_hover": "#FBF8F2", "primary_text": "#16140F",
    "danger": "#D08A8A",
}

BRIGHT = {
    "bg": "#F1ECE2", "bg_panel": "#F7F3EB", "bg_input": "#F7F3EB", "bg_lift": "#EAE4D8",
    "fg": "#16140F", "fg2": "#4A463F", "fg3": "#7A746A", "fg4": "#A69D8D",
    "stroke": "#C4BBA8", "stroke2": "#7E7462",
    "primary_fill": "#11141C", "primary_hover": "#2B3244", "primary_text": "#F1ECE2",
    "danger": "#B23B3B",
}


def _theme_values(p):
    """Gradio theme variables (snake_case); both scenes share this mapping."""
    return {
        "body_background_fill": p["bg"],
        "body_text_color": p["fg"],
        "body_text_color_subdued": p["fg3"],
        "body_text_size": "13px",
        "background_fill_primary": p["bg"],
        "background_fill_secondary": p["bg_panel"],
        "block_background_fill": p["bg_panel"],
        "block_border_color": "transparent",
        "block_border_width": "0px",
        "block_radius": "10px",
        "block_padding": "12px",
        "block_label_background_fill": "transparent",
        "block_label_border_width": "0px",
        "block_label_text_color": p["fg3"],
        "block_label_text_size": "11px",
        "block_label_text_weight": "500",
        "block_label_padding": "0 0 6px 0",
        "block_label_margin": "0",
        "block_title_text_color": p["fg2"],
        "block_title_text_weight": "500",
        "container_radius": "12px",
        "panel_background_fill": p["bg"],
        "panel_border_color": p["stroke"],
        "panel_border_width": "1px",
        "border_color_primary": p["stroke"],
        "border_color_accent": p["stroke2"],
        "border_color_accent_subdued": p["stroke"],
        "color_accent": p["primary_fill"],
        "color_accent_soft": p["stroke"],
        "link_text_color": p["fg"],
        "link_text_color_hover": p["fg2"],
        "link_text_color_active": p["fg"],
        "link_text_color_visited": p["fg2"],
        "code_background_fill": p["bg_input"],
        "input_background_fill": p["bg_input"],
        "input_background_fill_focus": p["bg_input"],
        "input_background_fill_hover": p["bg_input"],
        "input_border_color": p["stroke"],
        "input_border_color_focus": p["stroke2"],
        "input_border_color_hover": p["stroke2"],
        "input_border_width": "1px",
        "input_radius": "8px",
        "input_placeholder_color": p["fg4"],
        "input_text_size": "13px",
        "input_shadow": "none",
        "input_shadow_focus": "none",
        "button_border_width": "1px",
        "button_large_radius": "9px",
        "button_medium_radius": "9px",
        "button_small_radius": "8px",
        "button_large_text_size": "12px",
        "button_large_text_weight": "600",
        "button_large_padding": "8px 18px",
        "button_small_text_size": "11px",
        "button_small_text_weight": "500",
        "button_small_padding": "4px 10px",
        "button_primary_background_fill": p["primary_fill"],
        "button_primary_background_fill_hover": p["primary_hover"],
        "button_primary_border_color": p["primary_fill"],
        "button_primary_border_color_hover": p["primary_hover"],
        "button_primary_text_color": p["primary_text"],
        "button_primary_text_color_hover": p["primary_text"],
        "button_primary_shadow": "none",
        "button_primary_shadow_hover": "none",
        "button_primary_shadow_active": "none",
        "button_secondary_background_fill": "transparent",
        "button_secondary_background_fill_hover": p["bg_input"],
        "button_secondary_border_color": p["stroke"],
        "button_secondary_border_color_hover": p["stroke2"],
        "button_secondary_text_color": p["fg2"],
        "button_secondary_text_color_hover": p["fg"],
        "button_secondary_shadow": "none",
        "button_secondary_shadow_hover": "none",
        "button_secondary_shadow_active": "none",
        "button_cancel_background_fill": "transparent",
        "button_cancel_background_fill_hover": p["bg_input"],
        "button_cancel_border_color": p["stroke"],
        "button_cancel_border_color_hover": p["stroke2"],
        "button_cancel_text_color": p["fg3"],
        "button_cancel_text_color_hover": p["fg"],
        "button_cancel_shadow": "none",
        "button_cancel_shadow_hover": "none",
        "button_cancel_shadow_active": "none",
        "shadow_drop": "none",
        "shadow_drop_lg": "none",
        "shadow_inset": "none",
        "checkbox_background_color": p["bg_input"],
        "checkbox_background_color_selected": p["primary_fill"],
        "checkbox_background_color_hover": p["bg_lift"],
        "checkbox_border_color": p["fg4"],
        "checkbox_border_color_selected": p["primary_fill"],
        "checkbox_border_width": "1px",
        "checkbox_border_radius": "3px",
        "checkbox_check": p["primary_text"],
        "checkbox_shadow": "none",
        "checkbox_label_background_fill": p["bg_input"],
        "checkbox_label_background_fill_hover": p["bg_lift"],
        "checkbox_label_background_fill_selected": p["primary_fill"] if p is BRIGHT else p["stroke"],
        "checkbox_label_border_color": p["stroke"],
        "checkbox_label_border_color_selected": p["primary_fill"] if p is BRIGHT else p["stroke2"],
        "checkbox_label_text_color": p["fg2"],
        "checkbox_label_text_color_selected": p["primary_text"] if p is BRIGHT else p["fg"],
        "checkbox_label_shadow": "none",
        "checkbox_label_shadow_hover": "none",
        "checkbox_label_shadow_active": "none",
        "slider_color": p["primary_fill"],
        "loader_color": p["primary_fill"],
        "stat_background_fill": p["bg_input"],
        "table_border_color": p["stroke"],
        "table_even_background_fill": p["bg_panel"],
        "table_odd_background_fill": p["bg"],
        "table_text_color": p["fg2"],
        "table_radius": "8px",
        "error_background_fill": p["bg_input"],
        "error_border_color": p["stroke2"],
        "error_text_color": p["fg"],
        "error_icon_color": p["fg2"],
        "accordion_text_color": p["fg2"],
        "section_header_text_size": "13px",
        "section_header_text_weight": "500",
        "embed_radius": "10px",
        "layout_gap": "10px",
        "form_gap_width": "10px",
    }


def _bb_vars(p):
    """Scene palette → --bb-* variables for the custom CSS."""
    return {
        "--bb-field": p["bg"], "--bb-panel": p["bg_panel"],
        "--bb-well": p["bg_input"], "--bb-lift": p["bg_lift"],
        "--bb-ink": p["fg"], "--bb-ink2": p["fg2"],
        "--bb-ink3": p["fg3"], "--bb-ink4": p["fg4"],
        "--bb-line": p["stroke"], "--bb-line2": p["stroke2"],
        "--bb-primary-bg": p["primary_fill"],
        "--bb-primary-bg-hover": p["primary_hover"],
        "--bb-primary-fg": p["primary_text"],
        "--bb-danger": p["danger"],
        "--bb-chip-bg": p["stroke"] if p is not BRIGHT else p["primary_fill"],
        "--bb-chip-fg": p["fg"] if p is not BRIGHT else p["primary_text"],
    }


def _palette_css(selector, p):
    """Write the scene palette as --bb-* variables for the custom CSS."""
    scheme = "dark" if p is DARK else "light"
    lines = [f"{selector} {{"]
    for key, value in _bb_vars(p).items():
        lines.append(f"  {key}: {value};")
    lines.append(f"  color-scheme: {scheme};")
    lines.append("}")
    return "\n".join(lines)


def _scene_css(selector, values, palette=None):
    """Override the Gradio theme variables (runtime switch to the bright scene).

    Gradio merges custom CSS per selector, so the --bb-* variables must live in the
    same rule as the theme variables or the two overwrite each other.
    """
    lines = [f"{selector} {{"]
    if palette is not None:
        for key, value in _bb_vars(palette).items():
            lines.append(f"  {key}: {value} !important;")
        lines.append(f"  color-scheme: {'dark' if palette is DARK else 'light'} !important;")
    for key, value in values.items():
        lines.append(f"  --{key.replace('_', '-')}: {value} !important;")
    lines.append("}")
    return "\n".join(lines)


BASE_CSS = """
.gradio-container {
  width: 100% !important;   /* keep the frame stable when the settings rail is hidden */
  max-width: 1400px !important;
  margin: 0 auto !important;
  padding: 26px 22px 8px !important;
  position: relative !important;
  font-family: "Berkeley Mono", "Sarasa Mono SC", "JetBrains Mono", "SF Mono",
               ui-monospace, Menlo, monospace !important;
}
html, body, .gradio-container, gradio-app {
  background: var(--bb-field) !important;
  color: var(--bb-ink) !important;
}
/* hide Gradio's own footer / API link */
footer, #footer, .built-with, .show-api, .api-links { display: none !important; }
/* header top-right controls: settings-rail toggle + scene switch */
#bb-topbtns {
  position: absolute !important; top: 46px; right: 42px;
  width: auto !important; display: flex !important; align-items: center;
  gap: 8px; z-index: 60;
}
#bb-topbtns > * { width: auto !important; flex: 0 0 auto !important; }
/* theme toggle: a CSS-drawn icon button, same grammar as the rail toggle.
   Dark scene shows a crescent moon, the bright scene a ring-and-rays sun;
   both are 1px currentColor strokes with no background dependency. */
#bb-theme-btn {
  position: relative; width: 28px !important; height: 28px; min-width: 28px !important;
  padding: 0 !important; color: var(--bb-ink3) !important; line-height: 1;
}
#bb-theme-btn:hover { color: var(--bb-ink) !important; }
/* Gradio 6 scopes custom CSS to `.gradio-container... .contain <selector>`, so a
   scene class on <html> cannot win against the scoped base rule. The JS toggles
   `bb-bright` on the button itself, which keeps the variant selector scope-safe. */
#bb-theme-btn::before {
  content: ""; position: absolute; left: 8px; top: 8px; width: 12px; height: 12px;
  border-radius: 50%;
  box-shadow: inset -3px -1px 0 0 currentColor;   /* crescent moon */
}
#bb-theme-btn.bb-bright::before {
  left: 50%; top: 50%; width: 12px; height: 12px; margin: -6px 0 0 -6px;
  border: 1px solid currentColor;
  box-shadow: 0 -8px 0 -5px currentColor, 0 8px 0 -5px currentColor,
              -8px 0 0 -5px currentColor, 8px 0 0 -5px currentColor;
}
/* settings-rail toggle: a CSS-drawn sidebar icon */
#bb-rail-btn {
  position: relative; width: 28px !important; height: 28px; min-width: 28px !important;
  padding: 0 !important; color: var(--bb-ink3) !important; line-height: 1;
}
#bb-rail-btn:hover { color: var(--bb-ink) !important; }
#bb-rail-btn.bb-on { color: var(--bb-ink) !important; }
#bb-rail-btn::before {
  content: ""; position: absolute; left: 8px; top: 8px; width: 12px; height: 12px;
  border: 1px solid currentColor; border-radius: 2px;
}
#bb-rail-btn::after {
  content: ""; position: absolute; left: 12px; top: 9px; width: 1px; height: 10px;
  background: currentColor;
}
#bb-rail-btn.bb-on::after { left: 13px; width: 6px; }
/* the settings rail starts collapsed so the workspace gets the width */
html.bb-rail-hidden #bb-rail { display: none !important; }
/* header */
#bb-header {
  border: 1px solid var(--bb-line); border-radius: 12px; background: var(--bb-panel);
  padding: 22px 26px 20px; margin-bottom: 16px;
}
#bb-header .bb-eyebrow {
  font-size: 11px; letter-spacing: .2em; color: var(--bb-ink3); text-transform: uppercase;
}
#bb-header h1 {
  margin: 12px 0 10px; font-size: 24px; line-height: 1.25; font-weight: 600;
  letter-spacing: .05em; color: var(--bb-ink);
}
#bb-header h1 .bb-slash { color: var(--bb-ink4); padding: 0 8px; }
#bb-header .bb-meta { font-size: 11.5px; letter-spacing: .08em; color: var(--bb-ink3); text-transform: uppercase; }
/* ops footer */
#bb-footer {
  border-top: 1px solid var(--bb-line); margin-top: 24px; padding: 12px 2px 4px;
  display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px 18px;
  font-size: 11px; letter-spacing: .12em; color: var(--bb-ink3); text-transform: uppercase;
}
/* tabs (Gradio 6 dropped .tab-nav; target the real markup) */
.tabs .tab-container[role="tablist"] > button,
.tabs .overflow-menu > button {
  font: inherit !important; font-size: 11.5px !important; letter-spacing: .12em !important;
  text-transform: uppercase !important; color: var(--bb-ink3) !important;
  background: transparent !important; border: none !important;
  border-bottom: 1px solid transparent !important; border-radius: 0 !important;
  padding: 10px 14px !important;
}
.tabs .tab-container[role="tablist"] > button:hover,
.tabs .overflow-menu > button:hover { color: var(--bb-ink2) !important; }
.tabs .tab-container[role="tablist"] > button.selected {
  color: var(--bb-ink) !important; border-bottom: 1px solid var(--bb-ink) !important;
}
/* panels / accordions */
.block, .form, .panel, details, .accordion { border-radius: 10px !important; }
details, .accordion { border: 1px solid var(--bb-line) !important; background: var(--bb-panel) !important; }
.label-wrap { text-transform: uppercase !important; letter-spacing: .14em !important; font-size: 11px !important; }
/* component labels (must not match the text span of radio/checkbox chips, which is a different label > span) */
span[data-testid="block-info"], .block-title, label > span.label-text {
  text-transform: uppercase; letter-spacing: .12em; font-size: 10.5px !important; color: var(--bb-ink3) !important;
}
/* inputs */
input, textarea, select { font-family: inherit !important; letter-spacing: .01em; }
/* upload drop zones (Audio / File): a hint, not a headline.  Gradio ships these
   at 16px text and a 27px icon, the only text in the UI above the 13px body. */
[data-testid="upload-text"] { font-size: 12px !important; line-height: 1.5 !important; }
[data-testid="upload-text"] .or { font-size: 11px !important; }
[data-testid="upload-icon"] { width: 24px !important; height: 24px !important; }
[data-testid="upload-icon"] svg { width: 20px !important; height: 20px !important; }
input:focus, textarea:focus, select:focus {
  border-color: var(--bb-line2) !important; box-shadow: none !important; outline: none !important;
}
*:focus, *:focus-visible { box-shadow: none !important; outline: none !important; }
/* buttons: Bearbone ops scale (11–12px, wide tracking) instead of Gradio's
   16px display scale.  Primary is the only filled action; alternatives are
   outline; utilities are quiet and content-width. */
button.lg {
  height: 34px !important; min-height: 34px !important; padding: 0 18px !important;
  font-size: 12px !important; font-weight: 600 !important; letter-spacing: .14em !important;
  text-transform: uppercase !important; border-radius: 8px !important;
  white-space: nowrap !important;
}
button.sm {
  height: 27px !important; min-height: 27px !important; padding: 0 10px !important;
  font-size: 11px !important; font-weight: 500 !important; letter-spacing: .10em !important;
  text-transform: uppercase !important; border-radius: 8px !important;
  white-space: nowrap !important;
}
button.primary { text-transform: uppercase !important; letter-spacing: .14em !important; }
button.primary:hover { background: var(--bb-primary-bg-hover) !important; }
button.stop { text-transform: uppercase !important; }
/* an action bar: one filled focus, lighter alternatives, a ghost escape */
.bb-actionbar { align-items: center !important; gap: 8px !important; flex-wrap: nowrap !important; }
.bb-actionbar > * { flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; }
.bb-actionbar .bb-push-right { margin-left: auto !important; }
.bb-actionbar button.sm.secondary,
button.sm.secondary.bb-secondary {
  height: 30px !important; min-height: 30px !important; padding: 0 14px !important;
  font-size: 11.5px !important; font-weight: 500 !important; letter-spacing: .12em !important;
  background: transparent !important; border: 1px solid var(--bb-line) !important;
  color: var(--bb-ink2) !important;
}
.bb-actionbar button.sm.secondary:hover,
button.sm.secondary.bb-secondary:hover {
  border-color: var(--bb-line2) !important; color: var(--bb-ink) !important;
  background: transparent !important;
}
.bb-actionbar button.sm.stop {
  height: 30px !important; min-height: 30px !important; padding: 0 10px !important;
  font-size: 11.5px !important; font-weight: 500 !important; letter-spacing: .10em !important;
  background: transparent !important; border: 1px solid transparent !important;
  color: var(--bb-ink3) !important; box-shadow: none !important;
}
.bb-actionbar button.sm.stop:hover {
  color: var(--bb-ink) !important; background: transparent !important;
  border-color: transparent !important;
}
/* tool rows: content-width, quiet, border on hover only */
.bb-tools { align-items: center !important; gap: 8px !important; flex-wrap: wrap !important; }
.bb-tools > * { flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; }
.bb-tools button.sm {
  background: transparent !important; border: 1px solid transparent !important;
  color: var(--bb-ink3) !important;
}
.bb-tools button.sm:hover {
  border-color: var(--bb-line) !important; color: var(--bb-ink) !important;
  background: transparent !important;
}
.bb-tools button.sm.stop { color: var(--bb-ink3) !important; }
.bb-tools button.sm.stop:hover {
  color: var(--bb-danger) !important; border-color: var(--bb-danger) !important;
}
/* the destructive confirmation is the only solid danger in the UI */
button.sm.stop.bb-danger-solid {
  background: var(--bb-danger) !important; border-color: var(--bb-danger) !important;
  color: var(--bb-primary-fg) !important;
}
button.sm.stop.bb-danger-solid:hover { filter: brightness(1.08); }
/* scrollbar / selection */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: var(--bb-line); border-radius: 6px; }
::-webkit-scrollbar-thumb:hover { background: var(--bb-line2); }
::-webkit-scrollbar-track { background: transparent; }
::selection { background: var(--bb-ink); color: var(--bb-field); }
/* flatten upload widgets into button-like controls */
button.upload-button { text-transform: uppercase !important; letter-spacing: .08em !important; }
/* tables */
table { border-color: var(--bb-line) !important; }
/* selected radio/checkbox chips: the inner span must not inherit muted meta ink */
.gradio-container label.selected {
  background: var(--bb-chip-bg) !important;
  border-color: var(--bb-chip-bg) !important;
}
.gradio-container label.selected,
.gradio-container label.selected span { color: var(--bb-chip-fg) !important; }
/* choice chips (radio/checkbox options): match the label scale of the UI */
.gradio-container label[data-testid$="-radio-label"],
.gradio-container label[data-testid$="-checkbox-label"] { padding: 5px 10px !important; }
.gradio-container label[data-testid$="-radio-label"] span,
.gradio-container label[data-testid$="-checkbox-label"] span {
  font-size: 11.5px !important; letter-spacing: .04em !important; text-transform: uppercase;
}
/* SCORE VIEW (abcjs) */
.bb-score-title {
  font-size: 10.5px; letter-spacing: .12em; text-transform: uppercase;
  color: var(--bb-ink3); margin: 2px 0 6px;
}
.bb-score-panel {
  display: block !important;
  margin-top: 2px; border: 1px solid var(--bb-line); border-radius: 8px;
  background: var(--bb-panel); padding: 14px 10px; min-height: 96px;
  max-height: 460px !important; overflow: auto !important;
}
.bb-score-inner { display: block; }
.bb-score-inner svg { max-width: 100%; }
.bb-score-panel svg { max-width: 100%; height: auto; }
.bb-score-empty, .bb-score-error {
  color: var(--bb-ink3); font-size: 11px; letter-spacing: .1em;
  text-transform: uppercase; padding: 8px 4px;
}
.bb-score-error { color: var(--bb-ink2); }
/* remove Gradio's glow / shadows */
.gradio-container * { box-shadow: none !important; }
/* inline code / pre: dark ground with a thin stroke, never a light block */
.prose code, .md code, code {
  background: var(--bb-well) !important; color: var(--bb-ink) !important;
  border: 1px solid var(--bb-line) !important; border-radius: 4px;
  padding: 1px 5px; font-size: .92em;
}
.prose pre, pre {
  background: var(--bb-well) !important; border: 1px solid var(--bb-line) !important;
  border-radius: 8px; color: var(--bb-ink2) !important;
}
/* Gradio paints .form with --border-color-primary to fake a 1px frame; make it
   transparent so the 10px layout gaps do not show a solid colour band */
.form { background: transparent !important; }
.bb-group { border: 1px solid var(--bb-line) !important; border-radius: 10px !important; }
#bb-files { max-height: 200px; overflow: auto; }
/* “E” button: drops the repository example into STYLE / LYRICS */
.bb-eg-host { position: relative !important; }
.bb-eg-host textarea { padding-right: 36px !important; }
.bb-eg { position: absolute; top: 6px; right: 6px; z-index: 5; width: 20px; height: 20px;
  display: inline-flex; align-items: center; justify-content: center;
  border: 1px solid var(--bb-ink3); border-radius: 4px; background: var(--bb-panel);
  color: var(--bb-ink3); font: inherit; font-size: 10px; letter-spacing: 0;
  line-height: 1; cursor: pointer; padding: 0; }
.bb-eg:hover { color: var(--bb-ink); border-color: var(--bb-ink); }
@media (hover: none) and (pointer: coarse) {
  .bb-eg { width: 24px; height: 24px; font-size: 11px; }
  .bb-pback, .bb-pbtn, .bb-pend { width: 26px; height: 26px; }
  .bb-pjump { width: 40px; height: 26px; font-size: 10.5px; }
  /* touch targets stay comfortable even at the ops scale */
  button.sm { height: 32px !important; min-height: 32px !important; }
}
/* static footnote-style notes (matches the sub-label / footer scale, not body text) */
.bb-note p { font-size: 11px !important; line-height: 1.55; letter-spacing: .04em;
  color: var(--bb-ink3) !important; margin: 0 0 4px; }
.bb-note code { font-size: 10.5px !important; }
.bb-note { margin-top: 2px; }
/* the environment status is a hint, not content: same scale as the note below it */
#bb-env-status textarea { font-size: 11.5px !important; line-height: 1.55 !important;
  letter-spacing: .04em; color: var(--bb-ink3) !important; }
/* current work: a factual one-line band above the view, fed by a hidden
   bridge that the client mirrors to localStorage (per browser, not per server) */
#bb-current-work { display: none !important; }
/* the view bridge is a hidden Textbox too: the view itself is an <html> class */
#bb-view { display: none !important; }
#bb-current-band-wrap { min-height: 0; }
#bb-current-band {
  display: flex; flex-wrap: wrap; gap: 4px 10px; align-items: baseline;
  border: 1px solid var(--bb-line); border-radius: 8px; background: var(--bb-panel);
  padding: 7px 12px; margin-bottom: 10px;
  font-size: 11px; letter-spacing: .1em; text-transform: uppercase; color: var(--bb-ink2);
}
#bb-current-band b { color: var(--bb-ink); font-weight: 500; }

/* ── SONG / STUDIO views ────────────────────────────────────────────────
   Both roots stay mounted in Gradio (visible=True) and one is hidden with a
   class on <html>.  Remounting them would redraw the score SVG and reset the
   player (see docs/VIEW_SWITCH_PREFLIGHT.md). */
html.bb-view-song #bb-studio-root { display: none !important; }
html.bb-view-studio #bb-song-root { display: none !important; }
#bb-song-root { display: block; min-height: 200px; }
/* the view toggle is chrome, the same weight as the theme / rail buttons.
   The buttons sit flat in #bb-topbtns (a Row inside a Row makes Gradio stretch
   them to its 160px min-width and wrap); the elem_id is on the <button>. */
#bb-topbtns { flex-wrap: nowrap !important; }
#bb-topbtns .bb-view-label {
  font-size: 10px; letter-spacing: .14em; color: var(--bb-ink4);
  text-transform: uppercase; white-space: nowrap; margin-right: 2px;
}
#bb-view-song-btn, #bb-view-studio-btn {
  min-width: 0 !important; width: auto !important; flex: 0 0 auto !important;
  padding: 4px 9px !important; font-size: 10px !important;
  letter-spacing: .12em !important; line-height: 1.4;
  color: var(--bb-ink3) !important; border-color: var(--bb-line) !important;
  background: transparent !important;
}
#bb-view-song-btn:hover, #bb-view-studio-btn:hover { color: var(--bb-ink) !important; }
#bb-view-song-btn.bb-active, #bb-view-studio-btn.bb-active {
  color: var(--bb-ink) !important; border-color: var(--bb-line2) !important;
  background: var(--bb-lift) !important;
}
/* SONG: empty state */
#bb-song-empty {
  border: 1px solid var(--bb-line); border-radius: 12px; background: var(--bb-panel);
  padding: 26px 22px 28px;
}
#bb-song-empty .bb-song-lead {
  font-size: 11px; letter-spacing: .14em; text-transform: uppercase; color: var(--bb-ink3);
}
#bb-song-empty h2 {
  margin: 12px 0 6px; font-size: 20px; font-weight: 600; letter-spacing: .04em;
  color: var(--bb-ink);
}
#bb-song-empty p.bb-song-sub { font-size: 12px; color: var(--bb-ink3); margin: 0 0 6px; }
#bb-song-cards { gap: 10px !important; align-items: stretch !important; margin-top: 18px; }
#bb-song-cards .bb-song-card {
  border: 1px solid var(--bb-line) !important; border-radius: 10px !important;
  background: var(--bb-field) !important; padding: 14px 16px 12px !important;
  display: flex !important; flex-direction: column !important; justify-content: space-between !important;
  gap: 10px !important;
}
#bb-song-cards .bb-card-title {
  font-size: 13px; font-weight: 600; letter-spacing: .06em; color: var(--bb-ink);
  text-transform: uppercase;
}
#bb-song-cards .bb-card-sub {
  font-size: 11px; letter-spacing: .04em; color: var(--bb-ink3); margin-top: 4px;
}
#bb-song-cards .bb-song-card button { width: 100% !important; }
/* SONG: current work */
#bb-song-work { gap: 12px; }
#bb-song-identity {
  border: 1px solid var(--bb-line); border-radius: 12px; background: var(--bb-panel);
  padding: 18px 20px 16px;
}
#bb-song-identity .bb-song-title {
  font-size: 18px; font-weight: 600; letter-spacing: .04em; color: var(--bb-ink);
  margin: 0 0 8px; overflow-wrap: anywhere;
}
#bb-song-identity .bb-song-meta {
  display: flex; flex-wrap: wrap; gap: 4px 12px; font-size: 11px; letter-spacing: .1em;
  text-transform: uppercase; color: var(--bb-ink3);
}
#bb-song-identity .bb-song-meta b { color: var(--bb-ink2); font-weight: 500; }
#bb-song-stage {
  display: flex; flex-wrap: wrap; align-items: center; gap: 6px;
  border: 1px solid var(--bb-line); border-radius: 10px; background: var(--bb-panel);
  padding: 10px 12px;
}
#bb-song-stage .bb-stage {
  border: 1px solid var(--bb-line); border-radius: 6px; padding: 5px 10px;
  font-size: 10.5px; letter-spacing: .12em; text-transform: uppercase; color: var(--bb-ink4);
}
#bb-song-stage .bb-stage-past { color: var(--bb-ink3); border-color: var(--bb-line); }
#bb-song-stage .bb-stage-current {
  color: var(--bb-primary-fg); background: var(--bb-primary-bg);
  border-color: var(--bb-primary-bg); font-weight: 600;
}
#bb-song-stage .bb-stage-sep { color: var(--bb-ink4); font-size: 11px; }
#bb-song-actions { gap: 8px; }
#bb-song-actions button { text-transform: uppercase !important; letter-spacing: .08em !important; }
#bb-song-player { margin-top: 2px; }
#bb-song-score { border: 1px solid var(--bb-line); border-radius: 10px; background: var(--bb-panel);
  padding: 12px 14px; }
#bb-song-family { border: 1px solid var(--bb-line); border-radius: 10px; background: var(--bb-panel);
  padding: 12px 14px; }
#bb-song-family .bb-family-list { display: flex; flex-direction: column; gap: 6px; margin-top: 6px; }
#bb-song-family .bb-family-item { font-size: 11.5px; color: var(--bb-ink2); }
#bb-song-family .bb-family-item b { color: var(--bb-ink); font-weight: 500; }
#bb-song-family .bb-family-rel { color: var(--bb-ink3); letter-spacing: .04em; }
#bb-song-family .bb-family-hit { cursor: pointer; text-decoration: underline; text-underline-offset: 3px; }
#bb-song-family .bb-family-hit.bb-family-active { color: var(--bb-ink2); }

/* global busy banner */
#bb-busy-wrap { min-height: 0; }
#bb-busy {
  border: 1px solid var(--bb-line2); border-radius: 8px; padding: 7px 12px; margin-bottom: 10px;
  font-size: 11px; letter-spacing: .1em; text-transform: uppercase; color: var(--bb-ink2);
  background: var(--bb-panel);
}

/* footer credit link */
#bb-footer a {
  color: var(--bb-ink2) !important; text-decoration: underline; text-underline-offset: 3px;
}
#bb-footer a:hover { color: var(--bb-ink) !important; }

/* listening comparison link */
#bb-compare-link a,
#bb-allmodes-link a {
  display: inline-block; margin-top: 4px; color: var(--bb-ink) !important;
  font-size: 11px; letter-spacing: .12em; text-transform: uppercase;
  text-decoration: underline; text-underline-offset: 3px;
}
/* collapsible ABC source (component stays in the DOM so SCORE VIEW can read it) */
.bb-fold {
  display: inline-flex; align-items: center; justify-content: center;
  width: 14px; height: 14px; margin-left: 6px;
  border: 1px solid var(--bb-ink3); border-radius: 3px;
  font-size: 9px; line-height: 1; color: var(--bb-ink3);
  cursor: pointer; user-select: none;
}
.bb-fold:hover { color: var(--bb-ink); border-color: var(--bb-ink); }
.bb-folded .input-container {
  max-height: 0 !important; overflow: hidden !important; min-height: 0 !important;
}
/* the ABC source keeps a scrollable body even if Gradio measured it while folded */
#bb-abc-source textarea { overflow-y: auto !important; max-height: 420px !important; }
/* disabled action buttons while a job is running */
button:disabled, button[disabled] { opacity: .4 !important; cursor: not-allowed !important; }
/* ⓘ parameter tips: icon + one global floating layer (immune to component overflow/stacking) */
.bb-i {
  display: inline-flex; align-items: center; justify-content: center;
  width: 13px; height: 13px; margin-left: 6px; vertical-align: middle;
  border: 1px solid var(--bb-ink3); border-radius: 50%;
  font-size: 9px; line-height: 1; color: var(--bb-ink3); font-style: normal;
  letter-spacing: 0; text-transform: none; cursor: help; user-select: none;
}
.bb-i:hover, .bb-i:focus-visible { color: var(--bb-ink); border-color: var(--bb-ink); outline: none; }
#bb-tip {
  position: fixed; left: 0; top: 0; z-index: 2147483000; display: none;
  max-width: min(320px, calc(100vw - 16px)); padding: 10px 12px; pointer-events: none;
  border: 1px solid var(--bb-line); border-radius: 8px;
  background: var(--bb-panel); color: var(--bb-ink2);
  font-size: 11px; line-height: 1.55; letter-spacing: .02em;
  text-transform: none; white-space: normal;
}
#bb-tip.bb-tip-show { display: block; }

/* ── ADVANCED // SAMPLING dual view (knobs default / sliders) ─────────────
   Knobs are a view layer over the same 14 gr.Slider inputs: Gradio keeps the
   number + range inputs in the DOM as the single source of truth (component
   state, BUDGET PRESET, RESET DEFAULTS, generation) and sampling-knobs.js
   paints a knob beside them. Gradio 6 renders a phase as
   `.bb-sampling-phase > .styler > (.block header, .form sliders)`, so knobs
   mode reduces each slider block to its `.bb-knob` child and lays the phase
   `.form` out as a 7-column grid. Sliders mode is the untouched native layout. */
.bb-knob {
  display: none; flex-direction: column; align-items: center; gap: 6px;
  user-select: none; -webkit-user-select: none; cursor: ns-resize;
  touch-action: none; color: var(--bb-ink);
}
#bb-sampling-panel.bb-view-knobs .bb-knob { display: flex; }
#bb-sampling-panel.bb-view-knobs .bb-sampling-phases {
  display: grid !important;
  grid-template-columns: minmax(0, 1fr) !important;
  gap: 18px !important;
}
/* open layout under the knob view: Gradio's group ground/border is only visible
   because the native sliders used to cover it, and the mockup wants the knobs
   on the page itself (sliders mode keeps the cards untouched) */
#bb-sampling-panel.bb-view-knobs .bb-sampling-phase {
  width: 100% !important;
  min-width: 0 !important;
  background: transparent !important;
  border-color: transparent !important;
}
#bb-sampling-panel.bb-view-knobs .bb-sampling-phase .styler {
  background: transparent !important;
}
/* mockup typography: phase heading reads as a quiet row label, not a bold
   group title (sliders mode keeps Gradio's group heading) */
#bb-sampling-panel.bb-view-knobs .bb-sampling-phase .prose strong {
  font-weight: 400;
  color: var(--bb-ink2) !important;
}
#bb-sampling-panel.bb-view-knobs .bb-sampling-phase .form {
  display: grid !important;
  grid-template-columns: repeat(7, minmax(0, 1fr)) !important;
  gap: 12px 8px !important;
  align-items: start !important;
}
/* in knobs mode a slider block is only a knob host; the native chrome (label,
   number field, range, reset) stays in the DOM for Gradio but paints nothing */
#bb-sampling-panel.bb-view-knobs .bb-sampling-slider {
  min-width: 0 !important; width: 100% !important;
  padding: 0 !important; border: none !important; box-shadow: none !important;
  background: transparent !important; overflow: visible !important;
}
#bb-sampling-panel.bb-view-knobs .bb-sampling-slider > :not(.bb-knob) {
  display: none !important;
}
.bb-knob-label {
  font-size: 10px; letter-spacing: .12em; text-transform: uppercase;
  color: var(--bb-ink2); text-align: center; line-height: 1.2;
  max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.bb-knob-body {
  position: relative; width: 88px; height: 88px;
  display: flex; align-items: center; justify-content: center;
}
.bb-knob-svg { display: block; width: 88px; height: 88px; overflow: visible; }
.bb-knob-arc {
  fill: none; stroke: var(--bb-line2); stroke-width: 2.5;
  stroke-linecap: round;
}
.bb-knob-dot { fill: var(--bb-primary-bg); stroke: none; }
.bb-knob-value {
  position: absolute; inset: 0; display: flex; align-items: center;
  justify-content: center; pointer-events: none;
  font-size: 12px; font-weight: 600; letter-spacing: .02em;
  color: var(--bb-ink); font-variant-numeric: tabular-nums;
}
.bb-knob:hover .bb-knob-arc { stroke: var(--bb-ink2); }
.bb-knob.bb-knob-active .bb-knob-dot { fill: var(--bb-ink); }
.bb-knob.bb-knob-active .bb-knob-arc { stroke: var(--bb-ink2); }
.bb-knob.bb-knob-active { cursor: ns-resize; }
html.bb-knob-dragging, html.bb-knob-dragging body {
  cursor: ns-resize !important;
  user-select: none !important;
  -webkit-user-select: none !important;
  overscroll-behavior: none;
}
html.bb-knob-dragging { touch-action: none; }

/* view toggle: CSS icon of the scene on screen (knob ↔ fader bars), same
   grammar as the theme/rail buttons; the title says what the click gives */
#bb-sampling-view-btn {
  position: relative; width: 28px !important; height: 27px; min-width: 28px !important;
  padding: 0 !important; color: var(--bb-ink3) !important; flex: 0 0 auto !important;
}
#bb-sampling-view-btn:hover { color: var(--bb-ink) !important; }
/* knobs showing: dial ring + head dot (click for the sliders) */
#bb-sampling-view-btn.bb-showing-knobs::before {
  content: ""; position: absolute; left: 7px; top: 7px; width: 12px; height: 12px;
  border: 1px solid currentColor; border-radius: 50%;
  background: radial-gradient(circle at 72% 28%, currentColor 0 1.6px, transparent 1.7px);
  box-shadow: none;
}
/* sliders showing: three fader bars (click for the knobs) */
#bb-sampling-view-btn.bb-showing-sliders::before {
  content: ""; position: absolute; left: 8px; top: 8px; width: 12px; height: 11px;
  background:
    linear-gradient(currentColor, currentColor) 0 0 / 100% 1px no-repeat,
    linear-gradient(currentColor, currentColor) 0 5px / 100% 1px no-repeat,
    linear-gradient(currentColor, currentColor) 0 10px / 100% 1px no-repeat;
}
@media (max-width: 1000px) {
  #bb-sampling-panel.bb-view-knobs .bb-sampling-phase .form {
    grid-template-columns: repeat(4, minmax(0, 1fr)) !important;
  }
}
@media (max-width: 620px) {
  #bb-sampling-panel.bb-view-knobs .bb-sampling-phase .form {
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
  }
}

/* ── phones ──────────────────────────────────────────────────────────────
   Gradio 6 hides overflowing tabs behind a tiny ⋯ menu and the theme button
   is absolutely positioned over the title. Below 700px: tighter frame, theme
   button parked in the header corner with reserved room, and the tabs
   wrap (the overflow containers become display:contents so every tab is
   always visible instead of hidden in the dropdown). */
@media (max-width: 700px) {
  .gradio-container { padding: 14px 10px 6px !important; }
  /* action bars and tool rows wrap instead of running off the screen */
  .bb-actionbar, .bb-tools { flex-wrap: wrap !important; }
  .bb-actionbar .bb-push-right { margin-left: 0 !important; }
  #bb-header { padding: 16px 12px 12px; contain: inline-size; }
  #bb-header h1 { font-size: 18px; margin: 8px 0 6px; overflow-wrap: anywhere; }
  /* phones: rail toggle + theme button share one row */
  #bb-topbtns { position: static !important; inset: auto !important;
                width: 100% !important; margin: 0 0 10px 0 !important; }
  #bb-topbtns > * { flex: 0 0 auto !important; }
  #bb-rail-btn { width: 46px !important; min-width: 46px !important; height: 32px; }
  #bb-theme-btn { flex: 0 0 auto !important; width: 46px !important; min-width: 46px !important;
                  height: 32px; margin: 0 !important; }
  .tabs .tab-wrapper { display: flex !important; flex-wrap: wrap !important; height: auto !important; min-height: 32px; }
  .tabs .tab-container[role="tablist"],
  .tabs .overflow-menu,
  .tabs .overflow-dropdown { display: contents !important; }
  .tabs .overflow-menu > button { display: none !important; }
  .tabs .tab-container[role="tablist"] > button,
  .tabs .overflow-dropdown > button {
    flex: 1 1 44% !important; min-height: 40px; padding: 10px 6px !important;
    font-size: 10.5px !important; letter-spacing: .08em !important;
  }
  .bb-score-panel { max-height: 60vh !important; }
}
@media (max-width: 360px) { #bb-header h1 { font-size: 16px; } }

/* ── tablets / narrow laptops ────────────────────────────────────────────
   Below ~1024px the two-column workspace would squeeze the main column; the
   settings rail wraps underneath instead, and the tabs wrap four-up so Gradio's
   overflow menu does not hide them behind a ⋯ button. */
@media (max-width: 1024px) {
  .bb-workspace { flex-wrap: wrap !important; }
  .bb-workspace #bb-rail { flex-basis: 100% !important; min-width: 0 !important; }
}
@media (min-width: 701px) and (max-width: 1024px) {
  .tabs .tab-wrapper { display: flex !important; flex-wrap: wrap !important; height: auto !important; }
  .tabs .tab-container[role="tablist"],
  .tabs .overflow-menu,
  .tabs .overflow-dropdown { display: contents !important; }
  .tabs .overflow-menu > button { display: none !important; }
  .tabs .tab-container[role="tablist"] > button,
  .tabs .overflow-dropdown > button {
    flex: 1 1 22% !important; min-height: 38px; padding: 9px 6px !important;
    font-size: 10.5px !important; letter-spacing: .08em !important;
  }
}

/* ── touch devices ────────────────────────────────────────────────────────
   16px inputs stop iOS Safari from zooming the page on focus; the ⓘ and the
   ABC fold control get real touch targets. */
@media (hover: none) and (pointer: coarse) {
  /* 16px stops iOS Safari zooming on focus — but only a field that can take
     focus needs it.  Gradio renders output/status boxes as disabled, so they
     keep the desktop 13px instead of jumping to 16px. */
  input:not([disabled]):not([readonly]),
  textarea:not([disabled]):not([readonly]),
  select:not([disabled]) { font-size: 16px !important; }
  .bb-i { width: 20px !important; height: 20px !important; }
  .bb-fold { width: 22px !important; height: 22px !important; font-size: 12px !important; }
}
"""

# Gradio adds `.dark` to <body> when the OS is in dark appearance and re-declares its
# theme variables there; an html-level override would lose to that local declaration.
# So the bright scene must be applied on every element that can carry the theme scope.
BRIGHT_SELECTOR = (
    "html.bb-bright, html.bb-bright body, html.bb-bright body.dark, "
    "html.bb-bright .dark, html.bb-bright gradio-app, html.bb-bright .gradio-container"
)

BEARBONE_CSS = "\n".join([
    _palette_css(":root", DARK),
    BASE_CSS,
    _scene_css(BRIGHT_SELECTOR, _theme_values(BRIGHT), palette=BRIGHT),
    library.LIBRARY_CSS,
])

SAMPLING_VIEW_TOGGLE_JS = """() => {
  if (window.__bbToggleSamplingView) window.__bbToggleSamplingView();
}"""

THEME_TOGGLE_JS = """() => {
  const r = document.documentElement;
  const on = r.classList.toggle('bb-bright');
  try { localStorage.setItem('bb-theme', on ? 'bright' : 'dark'); } catch (e) {}
  const wrap = document.getElementById('bb-theme-btn');
  const btn = wrap && (wrap.tagName === 'BUTTON' ? wrap : wrap.querySelector('button'));
  if (btn) {
    btn.classList.toggle('bb-bright', on);
    const label = on ? 'Switch to the dark scene' : 'Switch to the bright scene';
    btn.title = label;
    btn.setAttribute('aria-label', label);
  }
  return '';
}"""

RAIL_TOGGLE_JS = """() => {
  const r = document.documentElement;
  const hidden = r.classList.toggle('bb-rail-hidden');
  try { localStorage.setItem('bb-rail', hidden ? 'off' : 'on'); } catch (e) {}
  const wrap = document.getElementById('bb-rail-btn');
  const btn = wrap && (wrap.tagName === 'BUTTON' ? wrap : wrap.querySelector('button'));
  if (btn) {
    btn.classList.toggle('bb-on', !hidden);
    const label = hidden ? 'Show the settings rail' : 'Hide the settings rail';
    btn.title = label;
    btn.setAttribute('aria-label', label);
  }
  return '';
}"""

TIPS = {
    "STYLE": "Genre, instruments, vocal character, language and BPM. The grey text is the repository example; click E to drop it in, or paste your own.",
    "LYRICS": "Words to sing. Use section tags like [Verse] / [Chorus]; line breaks shape the phrasing. Click E to drop in the repository example.",
    "PLAN MODE": "FULL = melody plus a chord plan (editable ABC); MELODY = melody only, no chord symbols, freer arrangement (best for covers); OFF = no symbolic plan at all.",
    "SEED": "Random seed. Same seed + same settings reproduces a take; change it for a different one.",
    "CFG SCALE": "Prompt guidance strength. 0 = default (1.0, or 1.01 for off); higher follows the prompt harder, too high can sound harsh.",
    "OUTPUT ID": "Optional filename-safe name used for the run directory and the request id.",
    "ABC SCORE": "Optional ABC score used as the composition input. Leave empty to let the model plan.",
    "ABC": "Paste an ABC score to validate it, strip chords, or compare before/after edits.",
    "BUDGET PRESET": "Defaults to the upstream protocol caps (full length). The preview and quick presets are opt-in: they only lower the token caps to finish faster.",
    "temperature": "Sampling randomness. Lower = safer, more repetitive; higher = more varied but can drift.",
    "top_p": "Nucleus sampling: keep the smallest token set whose probability sums to p. Lower = tighter.",
    "top_k": "Sample only from the k most likely tokens. Smaller = safer, larger = more varied.",
    "repetition_penalty": "Penalises recently used tokens. Above 1 discourages repeats; too high hurts musicality.",
    "penalty_window": "How many recent tokens count for the repetition penalty.",
    "min_tokens": "Do not emit the end token before this many tokens (guarantees a minimum length).",
    "max_tokens": "Hard cap for this phase. Semantic tokens ≈ 25 per second of audio; the model may end earlier.",
    "RUN DIRECTORY": "A previously saved run that contains latent.npy. Re-decodes it without generating again.",
    "DECODER VAE": "SOURCE = the decoder recorded in the source run; STANDARD = normal listening decoder (YuE2-Vae); LEGACY = benchmark decoder; CUSTOM = the path or HF id below.",
    "CUSTOM VAE PATH / HF ID": "Path or Hugging Face repo id of a custom decoder.",
    "FULL DECODE": "Decode the whole latent at once (faster, more memory). Tiled chunks are safer for long songs.",
    "JSONL REQUESTS": "One JSON request per line: id, style/tags, lyrics, cot, seed, cfg_scale, abc or abc_path, optional abc_sampling / semantic_sampling overrides.",
    "OUTPUT NAME": "Folder name for this batch.",
    "KEEP VOICES": "Which voices survive the chord strip (04 TOOLS checker).",
    "MELODY VOICES": "Which melody voices a cover keeps: both (vocal + instrumental), Vocal only, or Ins only. Applies to STRIP CHORDS and to SEND / GENERATE COVER.",
    "AFTER // EDITED ABC": "The edited score; compared against the original above (05 TOOLS copy of the invariant check).",
    "COMPARE VOICES": "Which voices the invariant check compares.",
    "ALLOW TEMPO CHANGE": "Allow the quarter-note tempo to change without reporting it as a violation.",
    "RUN DIRECTORIES": "One saved run directory per line; each must contain result.json.",
    "VERIFY WEIGHT HASHES": "Re-hash the checkpoint files (about 7 GB, a few seconds) to confirm integrity.",
    "DEVICE": "auto picks CUDA → MPS → CPU.",
    "DTYPE": "bfloat16 is the checkpoint dtype and the default on CUDA/MPS; float32 casts the weights at load (slower, twice the memory). MPS bf16 needs torch >= 2.11 (the 2.10 SDPA defect is fixed there); see overrides/ in the repository.",
    "MODEL ID / LOCAL DIR": "Hugging Face repo id or a local model directory.",
    "DEFAULT VAE": "Decoder for new generations: STANDARD = YuE2-Vae (listening), LEGACY = benchmark decoder, CUSTOM = the path used for CUSTOM VAE below.",
    "CUSTOM VAE": "Used when the default VAE is set to custom.",
    "MODEL REVISION": "Pin a git revision of the model repository (optional).",
    "VAE REVISION": "Pin a revision of the VAE repository (optional).",
    "OFFLINE": "Never touch the network; use only the local Hugging Face cache.",
    "BACKEND": "torch uses CUDA graphs when available and eager elsewhere; vLLM is a CUDA-only fast path.",
    "QUANTIZATION": "fp8 shrinks the AR weights on CUDA sm89+; none everywhere else.",
    "OFFLOAD AR WEIGHTS": "Move AR weights to CPU during synthesis to save VRAM (single request only).",
    "MEMORY BUDGET": "CUDA-only memory cap in GiB; also selects VAE chunking (≤12 GiB → 512 frames).",
    "ODE STEPS": "Flow-matching steps for audio synthesis. More = higher quality but slower; 32 is the protocol default.",
    "VAE CORE FRAMES": "Chunk size for tiled VAE decode; auto = 512 at ≤12 GiB budget, otherwise 1024.",
    "WORKS": "Click a work to view and play it; tick the checkbox to select it for deletion. Both are independent.",
    "SORT": "Sort key: TIME = creation time, NAME = work name.",
    "ORDER": "Sort order: DESC = newest / Z→A first, ASC = oldest / A→Z first.",
    "RENAME TO": "New name for the work being viewed; the timestamp prefix is kept.",
    "LIBRARY STATUS": "Result of the last library action.",
    "SOURCE AUDIO": "Reference recording for the cover (wav/mp3/flac). SheetSage2 decodes it to mono 24 kHz; the melody becomes the symbolic condition.",
    "TRANSCRIPTION TASK": "MELODY // VOCAL keeps only the sung line; MELODY // VOCAL+INST keeps vocal and instrumental melodies (chord-free, best for covers); FULL SCORE adds chords for cot=full regeneration.",
    "MAX SECONDS (0 = WHOLE FILE)": "Deliberately crop the transcript to the first N seconds; 0 processes the whole file. Long files take longer and use more GPU memory.",
    "SHEETSAGE2 MODEL / DIR": "Hugging Face id (m-a-p/SheetSage2) or the path of a downloaded snapshot. MERT-v2-FullSong loads automatically as its parent encoder.",
    "BASE MODEL / MERT SNAPSHOT": "Optional local path of the MERT-v2-FullSong snapshot; passed as base_model_path so a fully offline adapter load does not need the Hub cache.",
    "KEEP SHEETSAGE2 WARM": "Keep one SheetSage2 process resident and reuse its loaded model between transcriptions (faster repeats) until UNLOAD SHEETSAGE2 or the idle timeout. Off: every transcription starts a fresh process and frees all memory on exit. Locked while a transcription runs; changes apply from the next request.",
    "COVER ABC": "The transcription, editable. Fix wrong notes/meter before covering; STRIP CHORDS removes harmony for cot=melody, SEND TO GENERATE fills 01 GENERATE and sets the plan mode.",
    "SOURCE WORK": "A saved work with a score.abc (generated plan or an earlier edit); its ABC becomes the frozen baseline.",
    "BASELINE": "Record of the frozen source: hashes plus copies of score.abc/request.json. The original run directory is never modified.",
    "EDITED ABC": "Your edit of the baseline score. It is validated and submitted explicitly — generation never falls back to a fresh plan.",
    "RESULT ABC": "The score actually submitted for the last generation (chord-stripped when the plan mode requires it). The editor above stays untouched.",
    "SAMPLING // FROM 01 GENERATE": "Read-only mirror of 01 GENERATE → ADVANCED // SAMPLING. Both flows share those sliders; change them there.",
    "CONTRACT": "What CHECK INVARIANTS must preserve: EXACT keeps notes, meter and durations (tempo/meter optional); PITCH keeps only the ordered pitch sequence, so rhythm may change; FREE records differences without gating anything.",
    "ALLOW METER CHANGE": "EXACT contract only: treat bar/time-grid differences as permitted instead of a violation.",
    "ALLOW MELODY/RHYTHM CHANGES": "Permit generating even when CHECK INVARIANTS did not pass (e.g. an intentional pitch edit under EXACT). The FREE contract already permits everything, so this override only matters for EXACT/PITCH; it never skips FREEZE BASELINE or the check itself.",
    "CHECK RESULT": "Result of CHECK INVARIANTS under the chosen CONTRACT: EXACT compares sounding notes and the meter grid (tempo/meter permissions optional), PITCH compares only the ordered pitch sequence, FREE lists the differences without gating anything. Chord-only edits pass under EXACT.",
    "COMPARISON": "Baseline vs edited render: a local listening page built from both run directories.",
}

TIP_JS = """(function () {
  var SELECTOR = 'span[data-testid=\"block-info\"], span.label-text, .block-title';
  var tip = null;
  function ensureTip() {
    if (tip && tip.isConnected) return tip;
    tip = document.createElement('div');
    tip.id = 'bb-tip';
    document.body.appendChild(tip);
    return tip;
  }
  function hide() { if (tip) { tip.className = ''; tip.removeAttribute('data-src'); } }
  function show(icon) {
    var text = icon.getAttribute('data-tip');
    if (!text) return;
    var el = ensureTip();
    el.textContent = text;
    el.className = 'bb-tip-show';
    el.setAttribute('data-src', text);
    var r = icon.getBoundingClientRect();
    var w = el.offsetWidth, h = el.offsetHeight;
    var vw = window.innerWidth, vh = window.innerHeight;
    var left = r.left;
    if (left + w > vw - 8) left = vw - w - 8;
    if (left < 8) left = 8;
    var top = r.bottom + 8;
    if (top + h > vh - 8) top = r.top - h - 8;
    if (top < 8) top = 8;
    el.style.left = left + 'px';
    el.style.top = top + 'px';
  }
  function inject() {
    var tips = window.__BB_TIPS__ || {};
    var nodes = document.querySelectorAll(SELECTOR);
    for (var i = 0; i < nodes.length; i++) {
      var el = nodes[i];
      if (el.getAttribute('data-bb-tip') === '1') continue;
      var key = (el.textContent || '').trim();
      if (!tips[key]) continue;
      el.setAttribute('data-bb-tip', '1');
      var mark = document.createElement('span');
      mark.className = 'bb-i';
      mark.setAttribute('data-tip', tips[key]);
      mark.setAttribute('tabindex', '0');
      mark.setAttribute('role', 'img');
      mark.setAttribute('aria-label', tips[key]);
      mark.textContent = 'i';
      el.appendChild(mark);
    }
  }
  document.addEventListener('mouseover', function (e) {
    var i = e.target && e.target.closest ? e.target.closest('.bb-i') : null;
    if (i) show(i);
  });
  document.addEventListener('mouseout', function (e) {
    var i = e.target && e.target.closest ? e.target.closest('.bb-i') : null;
    if (i) hide();
  });
  document.addEventListener('focusin', function (e) {
    if (e.target && e.target.classList && e.target.classList.contains('bb-i')) show(e.target);
  });
  document.addEventListener('focusout', function (e) {
    if (e.target && e.target.classList && e.target.classList.contains('bb-i')) hide();
  });
  // clicking the icon must not activate the surrounding radio/checkbox label,
  // and must *show* the tip (never toggle it away: hover already showed it)
  document.addEventListener('click', function (e) {
    var i = e.target && e.target.closest ? e.target.closest('.bb-i') : null;
    if (i) {
      e.preventDefault();
      e.stopPropagation();
      show(i);
      return;
    }
    if (tip && tip.className === 'bb-tip-show') hide();
  }, true);
  document.addEventListener('scroll', hide, true);
  window.addEventListener('resize', hide);
  setInterval(inject, 600);
})();"""

HEAD_HTML = """<meta name="color-scheme" content="dark light">
<script>
(function () {
  try {
    var q = new URLSearchParams(location.search).get('theme');
    var saved = localStorage.getItem('bb-theme');
    var bright = q ? (q === 'bright') : (saved === 'bright');
    if (bright) document.documentElement.classList.add('bb-bright');
  } catch (e) {}
  try {
    // the settings rail starts hidden; "on" is the only value that shows it
    if (localStorage.getItem('bb-rail') !== 'on') {
      document.documentElement.classList.add('bb-rail-hidden');
    }
  } catch (e) {}
  function sync() {
    var on = document.documentElement.classList.contains('bb-bright');
    var wrap = document.getElementById('bb-theme-btn');
    var btn = wrap && (wrap.tagName === 'BUTTON' ? wrap : wrap.querySelector('button'));
    if (btn) {
      btn.classList.toggle('bb-bright', on);
      var label = on ? 'Switch to the dark scene' : 'Switch to the bright scene';
      btn.title = label;
      btn.setAttribute('aria-label', label);
    }
    var railWrap = document.getElementById('bb-rail-btn');
    var railBtn = railWrap && (railWrap.tagName === 'BUTTON' ? railWrap : railWrap.querySelector('button'));
    if (railBtn) {
      // the rail lives inside STUDIO; VIEW_JS disables the toggle in SONG
      if (window.__bbApplyRail) {
        window.__bbApplyRail();
      } else {
        var hidden = document.documentElement.classList.contains('bb-rail-hidden');
        railBtn.classList.toggle('bb-on', !hidden);
        var label = hidden ? 'Show the settings rail' : 'Hide the settings rail';
        railBtn.title = label;
        railBtn.setAttribute('aria-label', label);
      }
    }
  }
  sync();
  [300, 1000, 2500, 5000].forEach(function (t) { setTimeout(sync, t); });
})();
</script>"""

HEAD_HTML += ("<script>window.__BB_TIPS__ = " + json.dumps(TIPS, ensure_ascii=False)
              + ";</script><script>" + TIP_JS + "</script>")

# ── abcjs score rendering (bundled under yue2_groove/static, served via allowed_paths) ──
ABCJS_FILE = config.STATIC_DIR / "abcjs-basic-min.js"
SAMPLING_KNOBS_FILE = config.STATIC_DIR / "sampling-knobs.js"
# ── abcjs score rendering (bundled under yue2_groove/static, served via allowed_paths) ──
# Every score panel declares data-bb-abc="<label prefix of its ABC textbox>"; the
# script finds the matching textarea and renders into the panel's .bb-score-inner.
ABCJS_FILE = config.STATIC_DIR / "abcjs-basic-min.js"
SCORE_JS = """(function () {
  // Look for the textarea whose block label starts with `prefix`, preferring the
  // panel's own tab (so identical labels in different tabs cannot cross-render).
  function findArea(prefix, panel) {
    var root = (panel && panel.closest && panel.closest('.tabitem')) || document;
    var areas = [];
    function scan(scope) {
      var labels = scope.querySelectorAll('span[data-testid="block-info"]');
      for (var i = 0; i < labels.length; i++) {
        var label = (labels[i].textContent || '').trim();
        if (label.indexOf(prefix) !== 0) continue;
        var block = labels[i].closest('.block') || labels[i].parentElement;
        var ta = block && block.querySelector('textarea');
        if (ta) areas.push(ta);
      }
    }
    scan(root);
    if (!areas.length && root !== document) scan(document);
    return areas.length ? areas[0] : null;
  }
  function renderPanel(panel) {
    var box = panel.querySelector('.bb-score-inner');
    if (!box) return;
    // read-only panels (SONG) carry the score in data-bb-abc-text and have no textarea
    var explicit = panel.getAttribute('data-bb-abc-text');
    var abc;
    if (explicit !== null) {
      abc = explicit;
    } else {
      var ta = findArea(panel.getAttribute('data-bb-abc') || 'ABC SCORE', panel);
      abc = ta ? (ta.value || '') : '';
    }
    var ink = getComputedStyle(document.documentElement).getPropertyValue('--bb-ink').trim() || '#F1ECE2';
    var key = ink + '|' + abc;
    // The key lives on the element, not in a JS cache: if Gradio re-creates the
    // tab DOM (which drops the rendered SVG), the fresh node has no key and the
    // score is drawn again instead of being skipped as "already rendered".
    var hasSvg = !!box.querySelector('svg');
    if (box.getAttribute('data-bb-key') === key && (hasSvg || !abc.trim())) return;
    box.setAttribute('data-bb-key', key);
    if (!abc.trim()) {
      var empty = panel.getAttribute('data-bb-empty') || 'No score yet.';
      box.innerHTML = '<div class=\"bb-score-empty\">' + empty + '</div>';
      return;
    }
    box.innerHTML = '';
    try {
      if (window.ABCJS && ABCJS.renderAbc) {
        // fit the staff to the panel: a fixed 900px staffwidth overflows the
        // score box on phones and gets clipped / needs sideways scrolling
        var avail = Math.max(240, Math.min(900, (box.clientWidth || 340) - 18));
        ABCJS.renderAbc(box, abc, {
          responsive: 'resize', foregroundColor: ink,
          scale: avail < 520 ? 0.95 : 1.1,
          staffwidth: avail, paddingtop: 4, paddingbottom: 4,
        });
      } else {
        box.innerHTML = '<div class=\"bb-score-error\">Score renderer not loaded.</div>';
      }
    } catch (e) {
      box.innerHTML = '<div class=\"bb-score-error\">Could not render this ABC: '
        + String(e && e.message ? e.message : e).slice(0, 180) + '</div>';
    }
  }
  function render() {
    var panels = document.querySelectorAll('[data-bb-abc]');
    for (var i = 0; i < panels.length; i++) renderPanel(panels[i]);
  }
  // re-render on rotation/resize so the staff width follows the new panel width
  var bbResizeTimer = null;
  function bbRelayout() {
    if (bbResizeTimer) clearTimeout(bbResizeTimer);
    bbResizeTimer = setTimeout(function () {
      var boxes = document.querySelectorAll('.bb-score-inner');
      for (var i = 0; i < boxes.length; i++) boxes[i].removeAttribute('data-bb-key');
    }, 250);
  }
  window.addEventListener('resize', bbRelayout);
  window.addEventListener('orientationchange', bbRelayout);
  setInterval(render, 700);
})();"""

HEAD_HTML += (f'<script src="/gradio_api/file={ABCJS_FILE}"></script>'
              "<script>" + SCORE_JS + "</script>")

ABC_FOLD_JS = """(function () {
  function init() {
    var box = document.getElementById('bb-abc-source');
    if (!box || box.getAttribute('data-bb-fold') === '1') return;
    var info = box.querySelector('span[data-testid=\"block-info\"]') || box.querySelector('.block-title');
    if (!info) return;
    box.setAttribute('data-bb-fold', '1');
    var chev = document.createElement('span');
    chev.className = 'bb-fold';
    chev.setAttribute('role', 'button');
    chev.setAttribute('tabindex', '0');
    chev.setAttribute('aria-label', 'Collapse or expand the ABC source');
    chev.textContent = '+';
    info.appendChild(chev);
    function toggle(e) {
      if (e) { e.preventDefault(); e.stopPropagation(); }
      var folded = box.classList.toggle('bb-folded');
      chev.textContent = folded ? '+' : '-';
      var ta = box.querySelector('textarea');
      if (ta && !folded) {
        // re-measure: Gradio may have sized it while the box was folded/hidden
        ta.style.height = 'auto';
        var target = Math.min(Math.max(ta.scrollHeight + 4, 120), 420);
        ta.style.height = target + 'px';
        ta.style.overflowY = ta.scrollHeight > target ? 'auto' : 'hidden';
      }
    }
    chev.addEventListener('click', toggle, true);
    chev.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') toggle(e);
    });
    toggle();  // start folded: the rendered score below is the main view
  }
  setInterval(init, 600);
})();"""

HEAD_HTML += "<script>" + ABC_FOLD_JS + "</script>"

# Gradio's frontend re-applies a component's initial value when its tab is re-activated,
# which drops typed text and radio choices. Keep a small client-side store and restore
# the user's values (dispatching events so Gradio's own state follows).
PERSIST_JS = """(function () {
  var store = {};
  var restoreUntil = 0;
  function tabIdx(el) {
    var t = el.closest('.tabitem'); if (!t) return -1;
    return Array.prototype.indexOf.call(document.querySelectorAll('.tabitem'), t);
  }
  function blockIdx(el) {
    var t = el.closest('.tabitem'), b = el.closest('.block');
    if (!t || !b) return -1;
    return Array.prototype.indexOf.call(t.querySelectorAll('.block'), b);
  }
  function labelOf(el) {
    var b = el.closest('.block') || el.parentElement;
    var info = b && b.querySelector('span[data-testid=\"block-info\"]');
    if (!info) return '';
    return (info.textContent || '').replace(/[i+\\-]+$/, '').trim();
  }
  function keyOf(el) { return labelOf(el) + '|' + tabIdx(el) + '|' + blockIdx(el); }
  function optionKey(el) {
    var l = el.closest('label');
    return keyOf(el) + '|' + (l ? (l.innerText || '').trim().slice(0, 40) : el.value);
  }
  function isOutput(el) { return !!el.closest('.bb-output'); }
  function setNative(el, value) {
    var proto = el.tagName === 'TEXTAREA' ? window.HTMLTextAreaElement.prototype
                                          : window.HTMLInputElement.prototype;
    var setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
    setter.call(el, value);
    el.dispatchEvent(new Event('input', {bubbles: true}));
    el.dispatchEvent(new Event('change', {bubbles: true}));
  }
  var EDITABLE = 'textarea, input[type=text], input[type=number], input[type=search], input:not([type])';
  function snapshot(el) {
    if (!el || el.disabled || el.readOnly || isOutput(el)) return;
    if (el.matches('input[type=checkbox], input[type=radio]')) {
      store[optionKey(el)] = el.checked ? 1 : 0;
      return;
    }
    if (el.matches(EDITABLE)) store[keyOf(el)] = el.value;
  }
  function sync() {
    var inWindow = Date.now() < restoreUntil;
    var els = document.querySelectorAll('textarea, input');
    for (var i = 0; i < els.length; i++) {
      var el = els[i];
      if (el.disabled || el.readOnly || el.type === 'file' || isOutput(el)) continue;
      if (el.matches('input[type=checkbox], input[type=radio]')) {
        var ok = optionKey(el);
        if (!(ok in store)) continue;
        if (!!el.checked !== !!store[ok]) {
          if (inWindow && el.offsetParent !== null) el.click();
          else store[ok] = el.checked ? 1 : 0;  // server-driven update wins
        }
        continue;
      }
      if (!el.matches(EDITABLE)) continue;
      var key = keyOf(el);
      if (!(key in store)) continue;
      if (el.value !== store[key]) {
        if (inWindow) setNative(el, store[key]);
        else store[key] = el.value;  // server-driven update wins
      }
    }
  }
  document.addEventListener('input', function (e) {
    if (e.target && e.target.matches && e.target.matches('textarea, input')) snapshot(e.target);
  }, true);
  document.addEventListener('change', function (e) {
    if (e.target && e.target.matches && e.target.matches('textarea, input')) snapshot(e.target);
  }, true);
  document.addEventListener('click', function (e) {
    var el = e.target;
    if (el && el.matches && el.matches('input[type=checkbox], input[type=radio]')) {
      setTimeout(function () { snapshot(el); }, 0);
    }
    var btn = el && el.closest ? el.closest('button') : null;
    if (btn && btn.parentElement && btn.parentElement.className.indexOf('tab-container') >= 0 &&
        btn.parentElement.className.indexOf('visually-hidden') < 0) {
      restoreUntil = Date.now() + 2500;  // a real tab switch: re-apply user values briefly
    }
  }, true);
  setInterval(sync, 400);
})();"""

HEAD_HTML += "<script>" + PERSIST_JS + "</script>"

# Small “fill the repository example” buttons inside the STYLE / LYRICS boxes.
# The example text itself is injected as window.__BB_EXAMPLES__ above.
EXAMPLE_JS = r"""(function () {
  var FIELDS = { STYLE: 'style', LYRICS: 'lyrics' };
  function setNative(area, text) {
    var setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
    setter.call(area, text);
    area.dispatchEvent(new Event('input', { bubbles: true }));
    area.dispatchEvent(new Event('change', { bubbles: true }));
    area.focus();
  }
  function bindFocus(area) {
    if (area.getAttribute('data-bb-ph') === '1') return;
    area.setAttribute('data-bb-ph', '1');
    var hint = area.placeholder || '';
    // clicking in clears the grey example so a paste lands straight away;
    // leaving the field empty restores it
    area.addEventListener('focus', function () { if (!area.value) area.placeholder = ''; });
    area.addEventListener('blur', function () { if (!area.value) area.placeholder = hint; });
  }
  function inject() {
    var examples = window.__BB_EXAMPLES__ || {};
    var tabs = document.querySelectorAll('.tabitem');
    for (var tabIndex = 0; tabIndex < tabs.length; tabIndex++) {
      injectTab(tabs[tabIndex], examples);
    }
  }
  function injectTab(first, examples) {
    var nodes = first.querySelectorAll('span[data-testid="block-info"]');
    for (var i = 0; i < nodes.length; i++) {
      var info = nodes[i];
      var text = (info.textContent || '').trim();
      var key = FIELDS[text.split(/\s+/)[0]];
      if (!key) continue;
      var block = info.closest('.block') || first;
      var host = block.querySelector('.input-container');
      var area = host ? host.querySelector('textarea') : null;
      if (!host || !area || area.readOnly || area.disabled ||
          host.getAttribute('data-bb-eg') === '1') continue;
      host.setAttribute('data-bb-eg', '1');
      host.classList.add('bb-eg-host');
      bindFocus(area);
      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'bb-eg';
      btn.textContent = 'E';
      btn.title = 'Fill the repository example (City Lights)';
      btn.setAttribute('aria-label', btn.title);
      (function (target, value) {
        btn.addEventListener('click', function (event) {
          event.preventDefault();
          event.stopPropagation();
          setNative(target, value);
        });
      })(area, examples[key] || '');
      host.appendChild(btn);
    }
  }
  setInterval(inject, 600);
})();"""
HEAD_HTML += ("<script>window.__BB_EXAMPLES__ = "
              + json.dumps({"style": EXAMPLE_STYLE, "lyrics": EXAMPLE_LYRICS}, ensure_ascii=False)
              + ";</script>")
HEAD_HTML += "<script>" + EXAMPLE_JS + "</script>"
HEAD_HTML += "<script>" + library.LIBRARY_JS + "</script>"

CURRENT_WORK_JS = r"""(function () {
  var KEY = 'bb-current';
  function area() {
    var box = document.getElementById('bb-current-work');
    return box ? box.querySelector('textarea') : null;
  }
  function setNative(el, value) {
    var setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
    setter.call(el, value);
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }
  var restored = false;
  setInterval(function () {
    var el = area();
    if (!el) return;
    var stored = '';
    try { stored = localStorage.getItem(KEY) || ''; } catch (e) {}
    if (!restored) {
      restored = true;
      if (!el.value && stored) setNative(el, stored);   // ask the server to validate it
      return;
    }
    if (el.value && el.value !== stored) {
      try { localStorage.setItem(KEY, el.value); } catch (e) {}   // server-set: remember it
    } else if (!el.value && stored) {
      try { localStorage.removeItem(KEY); } catch (e) {}          // server cleared a stale path
    }
  }, 500);
})();"""
HEAD_HTML += "<script>" + CURRENT_WORK_JS + "</script>"

# ── SONG / STUDIO view switch ──────────────────────────────────────────────
# The two roots are always mounted; the view is a class on <html> so a switch
# never remounts the Studio Blocks (see docs/VIEW_SWITCH_PREFLIGHT.md).  The
# boot script runs in <head> before the body to avoid a flash of both views
# and honours (in order): an explicit --view/env mode, the last stored choice,
# then the SONG default.  `--tab N` is passed through as the studio mode.
VIEW_BOOT_JS = """<script>
(function () {
  var mode = __BB_VIEW_MODE_JSON__;
  window.__BB_VIEW_MODE__ = mode;
  // an explicit --view / --tab must not write through to the remembered choice
  window.__BB_VIEW_FORCED__ = (mode === 'song' || mode === 'studio');
  try {
    var saved = localStorage.getItem('bb-view');
    var view = window.__BB_VIEW_FORCED__ ? mode
             : (saved === 'studio' ? 'studio' : 'song');
    document.documentElement.classList.add('bb-view-' + view);
  } catch (e) {
    document.documentElement.classList.add('bb-view-song');
  }
})();
</script>"""

VIEW_JS = r"""(function () {
  var KEY = 'bb-view';
  function area() {
    var box = document.getElementById('bb-view');
    return box ? box.querySelector('textarea') : null;
  }
  function norm(view) { return view === 'studio' ? 'studio' : 'song'; }
  function setNative(el, value) {
    var setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
    setter.call(el, value);
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }
  function button(id) {
    var wrap = document.getElementById(id);
    return wrap ? (wrap.tagName === 'BUTTON' ? wrap : wrap.querySelector('button')) : null;
  }
  function applyRail(view) {
    // the settings rail sits inside the STUDIO root, so the toggle is a dead
    // control in SONG; disable it there and label it honestly
    var rail = button('bb-rail-btn');
    if (!rail) return;
    var songView = norm(view) !== 'studio';
    var hidden = document.documentElement.classList.contains('bb-rail-hidden');
    rail.disabled = songView;
    rail.classList.toggle('bb-on', !songView && !hidden);
    var label = songView ? 'Settings live in STUDIO'
              : (hidden ? 'Show the settings rail' : 'Hide the settings rail');
    rail.title = label;
    rail.setAttribute('aria-label', label);
  }
  var currentView = 'song';
  window.__bbApplyRail = function () { applyRail(currentView); };
  function apply(view) {
    view = norm(view);
    currentView = view;
    var root = document.documentElement;
    root.classList.toggle('bb-view-song', view !== 'studio');
    root.classList.toggle('bb-view-studio', view === 'studio');
    var song = button('bb-view-song-btn');
    var studio = button('bb-view-studio-btn');
    if (song) song.classList.toggle('bb-active', view !== 'studio');
    if (studio) studio.classList.toggle('bb-active', view === 'studio');
    applyRail(view);
  }
  window.__bbSetView = function (view) {
    view = norm(view);
    try { localStorage.setItem(KEY, view); } catch (e) {}
    apply(view);
    var el = area();
    if (el && el.value !== view) setNative(el, view);
  };
  var booted = false;
  function boot() {
    if (booted) return;
    var el = area();
    if (!el) return;
    booted = true;
    var mode = window.__BB_VIEW_MODE__ || 'auto';
    var view;
    if (mode === 'song' || mode === 'studio') {
      view = mode;
    } else {
      var stored = '';
      try { stored = localStorage.getItem(KEY) || ''; } catch (e) {}
      view = stored === 'studio' ? 'studio' : 'song';
    }
    apply(view);
    if (el.value !== view) setNative(el, view);
  }
  setInterval(function () {
    boot();
    var el = area();
    if (!el || !el.value) return;
    // a forced launch (--view / --tab) must not clobber the last stored choice;
    // only a real click through __bbSetView remembers a new one
    if (!window.__BB_VIEW_FORCED__) {
      try { localStorage.setItem(KEY, el.value); } catch (e) {}
    }
    apply(el.value);
  }, 400);
})();"""
HEAD_HTML += "<script>" + VIEW_JS + "</script>"


def _head_html(view_mode: str = "auto") -> str:
    """The page <head> with the resolved initial view baked into the boot script."""
    if view_mode not in ("song", "studio"):
        view_mode = "auto"
    return VIEW_BOOT_JS.replace("__BB_VIEW_MODE_JSON__", json.dumps(view_mode)) + HEAD_HTML


def VIEW_SET_JS(view: str) -> str:
    """Frontend-only handler for the top-chrome SONG / STUDIO buttons."""
    return f"() => {{ if (window.__bbSetView) window.__bbSetView({json.dumps(view)}); }}"


SONG_LISTEN_JS = """() => {
  var wrap = document.getElementById('bb-song-player');
  if (wrap) wrap.scrollIntoView({ behavior: 'smooth', block: 'center' });
  var audio = wrap ? wrap.querySelector('audio') : null;
  if (audio) { try { audio.play(); } catch (e) {} }
}"""

# Clicking a FAMILY row sets the current work through the same hidden bridge the
# handoffs use, so the band, the stage and the actions follow the clicked work.
SONG_JS = r"""(function () {
  function bridge() {
    var wrap = document.getElementById('bb-current-work');
    return wrap ? wrap.querySelector('textarea') : null;
  }
  function setNative(el, value) {
    var setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
    setter.call(el, value);
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }
  document.addEventListener('click', function (event) {
    var hit = event.target && event.target.closest ? event.target.closest('.bb-family-hit') : null;
    if (!hit) return;
    var run = hit.getAttribute('data-bb-run');
    var el = bridge();
    if (!run || !el) return;
    event.preventDefault();
    var active = document.querySelectorAll('.bb-family-hit.bb-family-active');
    for (var i = 0; i < active.length; i++) active[i].classList.remove('bb-family-active');
    hit.classList.add('bb-family-active');
    setNative(el, run);
  }, true);
})();"""
HEAD_HTML += "<script>" + SONG_JS + "</script>"
HEAD_HTML += (
    "<script>window.__BB_SAMPLING_DEFAULTS__ = "
    + json.dumps({"abc": ABC_DEFAULTS, "sem": SEM_DEFAULTS})
    + ";</script>"
    + f'<script src="/gradio_api/file={SAMPLING_KNOBS_FILE}"></script>'
)


def bb_theme():
    """Bearbone dark scene at startup; bright is switched at runtime via CSS variables."""
    import inspect as _inspect

    theme = gr.themes.Base(font=FONT_STACK, font_mono=FONT_STACK,
                           radius_size=gr.themes.sizes.radius_sm)
    values = _theme_values(DARK)
    valid = set(_inspect.signature(gr.themes.Base.set).parameters)
    both = {}
    for key, val in values.items():
        both[key] = val
        if f"{key}_dark" in valid:
            both[f"{key}_dark"] = val
    return theme.set(**both)


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

    with gr.Blocks(title="YUE2 // GROOVE") as demo:
        gr.HTML(header)
        current_band = gr.HTML("", elem_id="bb-current-band-wrap")
        current_bridge = gr.Textbox(value="", elem_id="bb-current-work",
                                    elem_classes=["bb-output"])
        # the view is applied client-side (html class) so the roots never remount;
        # this hidden box lets a server handler ask for the Studio view.
        view_mode = defaults.get("view_mode", "auto")
        initial_view = view_mode if view_mode in VIEW_CHOICES else "song"
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
                    song_score = gr.HTML(_score_panel("SONG SCORE", "No current work.", abc=""),
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
            with gr.Column(scale=5, min_width=520):
                with gr.Tabs(selected=("gen", "cover", "edit", "library", "tools", "decode", "batch")
                               [int(defaults.get("tab", 0)) % 7]) as tabs:
                    # ───── 01 GENERATE ─────
                    with gr.Tab("01 // GENERATE", id="gen"):
                        style = gr.Textbox(label="STYLE", lines=3, placeholder=EXAMPLE_STYLE)
                        lyrics = gr.Textbox(label="LYRICS", lines=8, placeholder=EXAMPLE_LYRICS)
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
                            with gr.Column(elem_id="bb-sampling-panel",
                                           elem_classes=["bb-view-knobs"]):
                                with gr.Row():
                                    preset = gr.Dropdown(choices=["Protocol defaults (full)",
                                                                  "Preview (~1–1.5 min song)",
                                                                  "Quick test (~20 s)"],
                                                         value="Protocol defaults (full)",
                                                         label="BUDGET PRESET", scale=3)
                                    sampling_view_btn = gr.Button(
                                        "", size="sm", scale=0,
                                        elem_id="bb-sampling-view-btn")
                                    sampling_reset_btn = gr.Button(
                                        "RESET DEFAULTS", size="sm", scale=1,
                                        elem_id="bb-reset-sampling")
                                duration_md = gr.Markdown(
                                    duration_text(SEM_DEFAULTS["max_tokens"]))
                                with gr.Row(elem_classes=["bb-sampling-phases"]):
                                    with gr.Group(elem_classes=["bb-group",
                                                                "bb-sampling-phase"]):
                                        gr.Markdown("**ABC PHASE · score planning**")
                                        abc_temp = gr.Slider(
                                            0, 5, value=ABC_DEFAULTS["temperature"],
                                            step=.05, label="temperature",
                                            elem_id="bb-abc-temp",
                                            elem_classes=["bb-sampling-slider"])
                                        abc_p = gr.Slider(
                                            .05, 1, value=ABC_DEFAULTS["top_p"],
                                            step=.01, label="top_p",
                                            elem_id="bb-abc-p",
                                            elem_classes=["bb-sampling-slider"])
                                        abc_k = gr.Slider(
                                            1, 1000, value=ABC_DEFAULTS["top_k"],
                                            step=1, label="top_k",
                                            elem_id="bb-abc-k",
                                            elem_classes=["bb-sampling-slider"])
                                        abc_rep = gr.Slider(
                                            1, 2, value=ABC_DEFAULTS["repetition_penalty"],
                                            step=.005, label="repetition_penalty",
                                            elem_id="bb-abc-rep",
                                            elem_classes=["bb-sampling-slider"])
                                        abc_win = gr.Slider(
                                            1, 100, value=ABC_DEFAULTS["penalty_window"],
                                            step=1, label="penalty_window",
                                            elem_id="bb-abc-win",
                                            elem_classes=["bb-sampling-slider"])
                                        abc_min = gr.Slider(
                                            0, 4096, value=ABC_DEFAULTS["min_tokens"],
                                            step=8, label="min_tokens",
                                            elem_id="bb-abc-min",
                                            elem_classes=["bb-sampling-slider"])
                                        abc_max = gr.Slider(
                                            64, 4096, value=ABC_DEFAULTS["max_tokens"],
                                            step=32, label="max_tokens",
                                            elem_id="bb-abc-max",
                                            elem_classes=["bb-sampling-slider"])
                                    with gr.Group(elem_classes=["bb-group",
                                                                "bb-sampling-phase"]):
                                        gr.Markdown(
                                            "**SEMANTIC PHASE · 25 tokens ≈ 1 s audio**")
                                        sem_temp = gr.Slider(
                                            0, 5, value=SEM_DEFAULTS["temperature"],
                                            step=.05, label="temperature",
                                            elem_id="bb-sem-temp",
                                            elem_classes=["bb-sampling-slider"])
                                        sem_p = gr.Slider(
                                            .05, 1, value=SEM_DEFAULTS["top_p"],
                                            step=.01, label="top_p",
                                            elem_id="bb-sem-p",
                                            elem_classes=["bb-sampling-slider"])
                                        sem_k = gr.Slider(
                                            1, 1000, value=SEM_DEFAULTS["top_k"],
                                            step=1, label="top_k",
                                            elem_id="bb-sem-k",
                                            elem_classes=["bb-sampling-slider"])
                                        sem_rep = gr.Slider(
                                            1, 2, value=SEM_DEFAULTS["repetition_penalty"],
                                            step=.005, label="repetition_penalty",
                                            elem_id="bb-sem-rep",
                                            elem_classes=["bb-sampling-slider"])
                                        sem_win = gr.Slider(
                                            1, 100, value=SEM_DEFAULTS["penalty_window"],
                                            step=1, label="penalty_window",
                                            elem_id="bb-sem-win",
                                            elem_classes=["bb-sampling-slider"])
                                        sem_min = gr.Slider(
                                            0, 9000, value=SEM_DEFAULTS["min_tokens"],
                                            step=8, label="min_tokens",
                                            elem_id="bb-sem-min",
                                            elem_classes=["bb-sampling-slider"])
                                        sem_max = gr.Slider(
                                            64, 9000, value=SEM_DEFAULTS["max_tokens"],
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
                                + _score_panel("ABC SCORE",
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
                    with gr.Tab("02 // COVER", id="cover") as cover_tab:
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
                                choices=[rel for _label, rel in _library_choices(
                                    _library_mode("time", "desc"))[1]],
                                info="A saved work or transcription with a score.abc",
                                interactive=True)
                        with gr.Row(elem_classes=["bb-tools"]):
                            cover_source_refresh = gr.Button("REFRESH", size="sm")
                            cover_load_btn = gr.Button("LOAD ABC", size="sm")
                            cover_send_edit_btn = gr.Button("SEND TO EDIT", size="sm")
                        cover_abc = gr.Textbox(label="COVER ABC", lines=10, max_lines=24,
                                               elem_id="bb-cover-abc")
                        gr.HTML('<div class="bb-score-title">SCORE VIEW</div>'
                                + _score_panel("COVER ABC",
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
                                                     placeholder=EXAMPLE_STYLE)
                            cover_lyrics = gr.Textbox(label="LYRICS", lines=5,
                                                      placeholder=EXAMPLE_LYRICS)
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
                                    + _score_panel("RESULT ABC", "No cover generated yet."),
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
                    with gr.Tab("03 // EDIT", id="edit") as edit_tab:
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
                                choices=[rel for _label, rel in _library_choices(
                                    _library_mode("time", "desc"))[1]],
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
                                + _score_panel("EDITED ABC",
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
                                + _score_panel("RESULT ABC", "No edit generated yet."),
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
                    with gr.Tab("04 // LIBRARY", id="library") as library_tab:
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
                                                         choices=[c for c, _ in _scan_batches()],
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
                            source_dir = gr.Dropdown(label="RUN DIRECTORY", choices=_scan_runs(),
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
                    dtype = gr.Dropdown(choices=DTYPE_CHOICES, value=defaults["dtype"], label="DTYPE")
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
                    backend = gr.Dropdown(choices=["torch", "torch-eager", "vllm"], value="torch",
                                          label="BACKEND")
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
        run_btn.click(generate, inputs=gen_common,
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
        current_bridge.change(render_song, inputs=[current_bridge], outputs=song_outputs)
        open_library_btn.click(open_last_in_library, inputs=[gen_last_run],
                               outputs=library_outputs_for_flow)
        gen_last_run.change(mirror_current, inputs=[gen_last_run], outputs=[current_bridge])
        edit_last_run.change(mirror_current, inputs=[edit_last_run], outputs=[current_bridge])
        cover_last_run.change(mirror_current, inputs=[cover_last_run], outputs=[current_bridge])
        gen_edit_btn.click(library_open_in_edit, inputs=[gen_last_run],
                           outputs=edit_outputs_for_flow)
        plan_btn.click(plan_only,
                       inputs=[style, lyrics, cot, seed, cfg, out_id,
                               abc_temp, abc_p, abc_k, abc_rep, abc_win, abc_min, abc_max]
                              + model_args,
                       outputs=[score_out, gen_status, files_out, run_btn, plan_btn,
                                current_bridge])
        allmodes_btn.click(generate_all_modes,
                           inputs=[style, lyrics, seed, cfg, abc, out_id,
                                   abc_temp, abc_p, abc_k, abc_rep, abc_win, abc_min, abc_max,
                                   sem_temp, sem_p, sem_k, sem_rep, sem_win, sem_min, sem_max]
                                  + model_args,
                           outputs=[gen_status, files_out, allmodes_link,
                                    run_btn, plan_btn, allmodes_btn, current_bridge])
        # Cooperative cancel only: do NOT use cancels=[...] here, because Gradio would
        # tear down the running generator event and drop its final "re-enable buttons" yield.
        cancel_btn.click(cancel_run, outputs=gen_status)
        preset.change(apply_preset, inputs=preset,
                      outputs=[abc_min, abc_max, sem_min, sem_max])
        sem_max.change(duration_text, inputs=sem_max, outputs=duration_md)
        sampling_reset_btn.click(_reset_sampling_values,
                                 outputs=[abc_temp, abc_p, abc_k, abc_rep, abc_win, abc_min, abc_max,
                                          sem_temp, sem_p, sem_k, sem_rep, sem_win, sem_min, sem_max,
                                          preset])
        # View layer only: knobs ↔ sliders; Gradio slider values / event graph unchanged.
        sampling_view_btn.click(fn=None, js=SAMPLING_VIEW_TOGGLE_JS)
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

        refresh_btn.click(_update_run_choices, outputs=source_dir)
        decode_btn.click(decode_run,
                         inputs=[source_dir, latent_upload, dec_vae_choice, dec_vae_custom,
                                 dec_vae_revision, full_decode] + model_args,
                         outputs=[decode_audio, decode_status, decode_btn, current_bridge])
        decode_reset_btn.click(lambda: ("standard", "", "", False),
                               outputs=[dec_vae_choice, dec_vae_custom, dec_vae_revision, full_decode])

        batch_btn.click(batch_generate,
                        inputs=[batch_text, batch_file, batch_id,
                                abc_temp, abc_p, abc_k, abc_rep, abc_win, abc_min, abc_max,
                                sem_temp, sem_p, sem_k, sem_rep, sem_win, sem_min, sem_max]
                              + model_args,
                        outputs=[batch_table, batch_status, batch_btn, current_bridge])
        cancel_btn.click(cancel_run, outputs=batch_status)

        inspect_btn.click(abc_inspect, inputs=abc_tool_text, outputs=abc_result)
        strip_btn.click(abc_strip_chords, inputs=[abc_tool_text, strip_voice], outputs=abc_result)
        compare_abc_btn.click(abc_compare, inputs=[abc_tool_text, abc_after, compare_voice, allow_tempo],
                              outputs=abc_compare_out)
        compare_btn.click(make_comparison, inputs=compare_paths,
                          outputs=[compare_file, compare_link, compare_status])
        fill_btn.click(_fill_from_batch, inputs=batch_pick, outputs=compare_paths)
        batch_refresh.click(_update_batch_choices, outputs=batch_pick)
        doctor_btn.click(run_doctor,
                         inputs=[model, vae_choice, vae_custom, revision, vae_revision, offline,
                                 verify_hashes],
                         outputs=doctor_out)
        load_btn.click(lambda *a: load_pipeline(*a)[1], inputs=model_args, outputs=env_status)
        unload_btn.click(lambda: (unload_pipeline(), "Model unloaded")[1], outputs=env_status)
        current_bridge.change(publish_current, inputs=[current_bridge],
                              outputs=[current_band, current_bridge])
        # ───── SONG wiring: producer actions reuse the existing handlers ─────
        # (or open Studio on the right tab); no action duplicates editable state.
        view_song_btn.click(fn=None, js=VIEW_SET_JS("song"), outputs=view_song_btn)
        view_studio_btn.click(fn=None, js=VIEW_SET_JS("studio"), outputs=view_studio_btn)
        song_new_btn.click(lambda: (gr.update(selected="gen"), gr.update(value="studio")),
                           outputs=[tabs, view_bridge])
        song_cover_start_btn.click(
            lambda: (gr.update(selected="cover"), gr.update(value="studio")),
            outputs=[tabs, view_bridge])
        song_edit_start_btn.click(
            lambda: (gr.update(selected="library"), gr.update(value="studio")),
            outputs=[tabs, view_bridge])
        song_listen_btn.click(fn=None, js=SONG_LISTEN_JS, outputs=song_listen_btn)
        song_render_btn.click(
            song_render_action, inputs=[current_bridge],
            outputs=[abc, score_input_accordion, style, lyrics,
                     cover_abc, cover_style, cover_lyrics, cover_status, cover_source,
                     current_bridge, tabs, view_bridge])
        song_edit_btn.click(song_open_edit, inputs=[current_bridge],
                            outputs=edit_outputs_for_flow + [view_bridge])
        song_check_btn.click(song_open_edit, inputs=[current_bridge],
                             outputs=edit_outputs_for_flow + [view_bridge])
        song_retry_btn.click(
            song_retry, inputs=[current_bridge],
            outputs=[style, lyrics, cot, seed, current_bridge, tabs, view_bridge])
        song_send_btn.click(
            song_send, inputs=[current_bridge],
            outputs=[abc, cot, style, lyrics, score_input_accordion, tabs, cover_status,
                     view_bridge])
        song_library_btn.click(song_open_library, inputs=[current_bridge],
                               outputs=library_outputs_for_flow + [view_bridge])
        song_studio_btn.click(song_open_studio, inputs=[current_bridge],
                              outputs=[tabs, view_bridge])
        song_compare_btn.click(song_compare, inputs=[current_bridge],
                               outputs=[song_compare_link, song_compare_status])
        cover_detect_btn.click(cover_detect_python, outputs=cover_status)
        busy_timer = gr.Timer(2.0)
        busy_timer.tick(busy_banner, outputs=busy_out)
        theme_btn.click(fn=None, js=THEME_TOGGLE_JS, outputs=theme_btn)
        rail_btn.click(fn=None, js=RAIL_TOGGLE_JS, outputs=rail_btn)

        # ───── cover wiring (SheetSage2 lives in sheetsage_adapter.py) ─────
        cover_controls = [cover_btn, cover_strip_btn, cover_send_btn, cover_generate_btn,
                          cover_env_btn, cover_unload_btn, cover_keep_warm]
        cover_btn.click(cover_transcribe,
                        inputs=[cover_audio, cover_task, cover_max_seconds, cover_model,
                                cover_device, cover_dtype, cover_revision, cover_base_model,
                                cover_keep_warm, cover_offline],
                        outputs=[cover_abc, cover_files, cover_status, *cover_controls,
                                 cover_generate_accordion, cover_source, current_bridge])
        cover_generate_btn.click(
            cover_generate,
            inputs=[cover_style, cover_lyrics, cover_abc, cover_task, cover_keep, cover_seed,
                    cover_cfg,
                    abc_temp, abc_p, abc_k, abc_rep, abc_win, abc_min, abc_max,
                    sem_temp, sem_p, sem_k, sem_rep, sem_win, sem_min, sem_max] + model_args,
            outputs=[cover_status, cover_result_audio, cover_result_abc, cover_gen_files,
                     *cover_controls, cover_last_run])
        cover_open_library_btn.click(open_last_in_library, inputs=[cover_last_run],
                                     outputs=library_outputs_for_flow)
        cover_cancel_btn.click(cancel_run, outputs=cover_status)
        cover_env_btn.click(cover_check_environment, outputs=cover_status)
        cover_unload_btn.click(cover_unload_worker, outputs=cover_status)
        cover_source_refresh.click(cover_choices, outputs=cover_source)
        cover_load_btn.click(cover_load, inputs=[cover_source],
                             outputs=[cover_abc, cover_style, cover_lyrics, cover_status])
        cover_send_edit_btn.click(
            cover_send_to_edit,
            inputs=[cover_abc, cover_source, cover_style, cover_lyrics],
            outputs=[edit_abc, edit_baseline_abc, edit_style, edit_lyrics, edit_source_rel,
                     edit_check_state, edit_baseline_state, edit_baseline_info, edit_status,
                     edit_source, current_bridge, tabs])
        cover_strip_btn.click(cover_strip, inputs=[cover_abc, cover_keep],
                              outputs=[cover_abc, cover_status])
        cover_send_btn.click(cover_send_to_generate,
                             inputs=[cover_abc, cover_task, cover_style, cover_lyrics, cover_keep],
                             outputs=[abc, cot, style, lyrics, score_input_accordion, tabs,
                                      cover_status])

        # ───── edit wiring (pure logic in edit_flow.py) ─────
        edit_tab.select(edit_choices, outputs=edit_source)
        edit_refresh_btn.click(edit_choices, outputs=edit_source)
        edit_tab.select(_sampling_summary,
                        inputs=[abc_temp, abc_p, abc_k, abc_rep, abc_win, abc_min, abc_max,
                                sem_temp, sem_p, sem_k, sem_rep, sem_win, sem_min, sem_max],
                        outputs=edit_sampling_note)
        cover_tab.select(_sampling_summary,
                         inputs=[abc_temp, abc_p, abc_k, abc_rep, abc_win, abc_min, abc_max,
                                 sem_temp, sem_p, sem_k, sem_rep, sem_win, sem_min, sem_max],
                         outputs=cover_sampling_note)
        cover_tab.select(cover_choices, outputs=cover_source)
        # the mirror lives in a closed accordion (not mounted until expanded)
        cover_generate_accordion.expand(
            _sampling_summary,
            inputs=[abc_temp, abc_p, abc_k, abc_rep, abc_win, abc_min, abc_max,
                    sem_temp, sem_p, sem_k, sem_rep, sem_win, sem_min, sem_max],
            outputs=cover_sampling_note)
        edit_load_btn.click(
            edit_load, inputs=[edit_source],
            outputs=[edit_style, edit_lyrics, edit_abc, edit_baseline_abc, edit_source_rel,
                     edit_check_state, edit_baseline_state, edit_baseline_info, edit_status])
        edit_freeze_btn.click(edit_freeze, inputs=[edit_source_rel, edit_source],
                              outputs=[edit_baseline_state, edit_baseline_info, edit_status])
        edit_check_btn.click(
            edit_check,
            inputs=[edit_baseline_abc, edit_abc, edit_voice, edit_allow_tempo,
                    edit_baseline_state, edit_contract, edit_allow_meter],
            outputs=[edit_check_out, edit_check_state])
        edit_run_btn.click(
            edit_generate,
            inputs=[edit_style, edit_lyrics, edit_cot, edit_seed, edit_cfg, edit_abc,
                    edit_baseline_abc, edit_source_rel, edit_check_state, edit_allow_changes,
                    edit_baseline_state,
                    abc_temp, abc_p, abc_k, abc_rep, abc_win, abc_min, abc_max,
                    sem_temp, sem_p, sem_k, sem_rep, sem_win, sem_min, sem_max] + model_args,
            outputs=[edit_audio, edit_result_abc, edit_files, edit_status,
                     edit_run_btn, edit_check_btn, edit_freeze_btn, edit_load_btn,
                     edit_refresh_btn, edit_compare_btn, edit_last_run])
        edit_cancel_btn.click(cancel_run, outputs=edit_status)
        edit_library_btn.click(open_last_in_library, inputs=[edit_last_run],
                               outputs=library_outputs_for_flow)
        edit_compare_btn.click(edit_compare, inputs=[edit_source_rel, edit_last_run],
                               outputs=[edit_compare_file, edit_compare_link, edit_compare_status])

        # ───── library wiring (toolkit in library.py) ─────
        library_outputs = [lib_list, lib_info, lib_style, lib_lyrics, lib_abc,
                           lib_rename_box, lib_rename_btn, lib_status]
        library_tab.select(library_refresh,
                           inputs=[lib_sort_key, lib_sort_dir, lib_list, lib_active],
                           outputs=library_outputs)
        lib_refresh_btn.click(library_refresh,
                              inputs=[lib_sort_key, lib_sort_dir, lib_list, lib_active],
                              outputs=library_outputs)
        lib_sort_key.change(library_refresh,
                            inputs=[lib_sort_key, lib_sort_dir, lib_list, lib_active],
                            outputs=library_outputs)
        lib_sort_dir.change(library_refresh,
                            inputs=[lib_sort_key, lib_sort_dir, lib_list, lib_active],
                            outputs=library_outputs)
        # row click = view/play only (JS sets the hidden bb-lib-active box)
        lib_active.change(library_view, inputs=[lib_active],
                          outputs=[lib_info, lib_style, lib_lyrics, lib_abc,
                                   lib_rename_box, lib_rename_btn])
        lib_rename_btn.click(library_rename,
                             inputs=[lib_active, lib_rename_box, lib_sort_key, lib_sort_dir, lib_list],
                             outputs=[lib_list, lib_info, lib_style, lib_lyrics, lib_abc,
                                      lib_rename_box, lib_rename_btn, lib_status, lib_active])
        lib_edit_btn.click(library_open_in_edit, inputs=[lib_active],
                           outputs=edit_outputs_for_flow)
        lib_cover_btn.click(library_use_in_cover, inputs=[lib_active],
                            outputs=[cover_abc, cover_style, cover_lyrics, cover_status,
                                     cover_source, current_bridge, tabs])
        lib_delete_btn.click(library_delete_prepare,
                             inputs=[lib_list, lib_sort_key, lib_sort_dir],
                             outputs=[lib_confirm, lib_pending, lib_confirm_btn, lib_status])
        # note: delete-confirm returns (…, confirm, pending, button, status, active)
        lib_confirm_btn.click(library_delete_confirm,
                              inputs=[lib_pending, lib_sort_key, lib_sort_dir, lib_active],
                              outputs=[lib_list, lib_info, lib_style, lib_lyrics, lib_abc,
                                       lib_rename_box, lib_rename_btn,
                                       lib_confirm, lib_pending, lib_confirm_btn, lib_status,
                                       lib_active])
        lib_cancel_btn.click(library_delete_cancel,
                             outputs=[lib_confirm, lib_pending, lib_confirm_btn, lib_status])
        demo.load(library_refresh, inputs=[lib_sort_key, lib_sort_dir, lib_list, lib_active],
                  outputs=library_outputs)
        demo.load(edit_choices, outputs=edit_source)
        demo.load(_sampling_summary_pair,
                  inputs=[abc_temp, abc_p, abc_k, abc_rep, abc_win, abc_min, abc_max,
                          sem_temp, sem_p, sem_k, sem_rep, sem_win, sem_min, sem_max],
                  outputs=[edit_sampling_note, cover_sampling_note])

        gr.HTML(footer)

    demo.bb_head = _head_html(view_mode)
    demo.bb_view_mode = view_mode
    return demo


def main():
    # .env must be loaded before the CLI defaults below are resolved and before
    # PyTorch initialises the MPS allocator, so the watermark guard applies here
    # too (serve.sh sources it, a direct `python -m yue2_groove` did not).
    config.load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16"])
    parser.add_argument("--model", default=config.default_model(),
                        help="Hugging Face id or local directory of the 3B model")
    parser.add_argument("--runs", default=None,
                        help="Directory for generated works (default: $YUE2_GROOVE_RUNS or ./runs)")
    parser.add_argument("--vae", default="standard", choices=["standard", "legacy"])
    parser.add_argument("--tab", type=int, default=None,
                        help="Start on Studio tab 0..6 (an explicit --tab forces the Studio view)")
    parser.add_argument("--view", choices=list(VIEW_CHOICES), default=None,
                        help="Start in the SONG or STUDIO view (or set YUE2_GROOVE_VIEW)")
    parser.add_argument("--auth", default=os.environ.get("YUE2_GROOVE_AUTH", ""),
                        help="Login as user:password (or set YUE2_GROOVE_AUTH); recommended on a LAN")
    parser.add_argument("--sheetsage-python", default=None,
                        help="Python of the separate SheetSage2 venv (or set "
                             "YUE2_GROOVE_SHEETSAGE_PYTHON) for the 02 COVER tab")
    parser.add_argument("--no-preload", action="store_true", help="Do not preload the model at startup")
    args = parser.parse_args()
    global RUNS
    if args.runs:
        RUNS = Path(args.runs).expanduser().resolve()
    else:
        RUNS = config.runs_dir()      # honour YUE2_GROOVE_RUNS from .env
    if args.sheetsage_python:
        os.environ["YUE2_GROOVE_SHEETSAGE_PYTHON"] = args.sheetsage_python

    auth = None
    if args.auth:
        if ":" not in args.auth:
            parser.error("--auth must be user:password")
        user, password = args.auth.split(":", 1)
        if not user or not password:
            parser.error("--auth user and password must not be empty")
        auth = (user, password)

    device = _pick_device(args.device)
    dtype = args.dtype
    if dtype == "auto":
        dtype = "bfloat16" if device in ("cuda", "mps") else "float32"
    view_mode, tab = resolve_view(args.view, args.tab, os.environ.get("YUE2_GROOVE_VIEW"))
    RUNS.mkdir(parents=True, exist_ok=True)
    atexit.register(sheetsage_adapter.stop_worker)   # no resident SheetSage2 after exit
    defaults = {"device": device, "dtype": dtype, "model": args.model, "vae": args.vae,
                "tab": tab, "view_mode": view_mode,
                "status": ("Model not loaded yet — it loads automatically on the first generation."
                           if args.no_preload else "Model is preloading in the background…")}
    demo = build_ui(defaults)
    demo.queue(default_concurrency_limit=1)
    if not args.no_preload:
        def preload():
            try:
                load_pipeline(device, dtype, "torch", "none", False, 24, 32, "auto",
                              args.model, args.vae, "", "", "", False)
                print("[yue2_groove] model preload complete", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[yue2_groove] preload failed (will retry on first generation): {exc}", flush=True)
        threading.Thread(target=preload, daemon=True).start()
    demo.launch(server_name=args.host, server_port=args.port, theme=bb_theme(),
                css=BEARBONE_CSS,
                head=getattr(demo, "bb_head", HEAD_HTML),
                allowed_paths=[str(config.STATIC_DIR), str(RUNS)],
                auth=auth,
                share=args.share, inbrowser=not args.share)


if __name__ == "__main__":
    main()
