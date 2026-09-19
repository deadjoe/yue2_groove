"""The kernel the tabs share: the one pipeline, the one job, the run directory.

Module state on purpose — one process serves one model: ``RUNS`` (the works
directory, set once by the CLI), the cached pipeline behind ``load_pipeline`` /
``get_pipe`` / ``unload_pipeline``, the ``RUNNING`` job slot and the ``CANCEL``
flag every generator handler honours.  ``run_generation`` is the single path
from a request to a saved run (with the panic-safe ``pending.json`` protocol),
and ``run_dir`` / ``slug`` / ``artifact_files`` name what it writes.  Handlers
reach all of it as ``runtime.<name>`` so a test can patch one place.
"""
from __future__ import annotations

import contextlib

try:
    import fcntl  # Unix only; optional macOS F_FULLFSYNC in _fsync_fd
except ImportError:  # Windows (and any host without the module)
    fcntl = None
import importlib.util
import json
import logging
import os
import threading
import time
from pathlib import Path

import gradio as gr
import torch

from .. import adapter, config

# Where generated works are stored; main() may override it with --runs.
RUNS = config.runs_dir()
log = logging.getLogger("yue2_groove")

_PIPE = None
_PIPE_KEY = None
_LOCK = threading.Lock()
RUNNING = threading.Lock()
CANCEL = threading.Event()

DTYPE_CHOICES = [
    ("bfloat16 (checkpoint dtype; default on CUDA/MPS)", "bfloat16"),
    ("float32 (cast at load; slower, 2x memory)", "float32"),
]

# vLLM is upstream's optional Linux/CUDA backend (``yue2-infer[fast]``).  Offer it
# only when the package is importable, so the dropdown never promises a backend
# that ends in ImportError at generate time (macOS, Windows, or a plain install).
BACKEND_CHOICES = ["torch", "torch-eager"] + (["vllm"] if importlib.util.find_spec("vllm") else [])

ABC_DEFAULTS = {"temperature": .7, "top_p": .9, "top_k": 30, "repetition_penalty": 1.005,
                "penalty_window": 100, "min_tokens": 32, "max_tokens": 4096}
SEM_DEFAULTS = {"temperature": 1.0, "top_p": .95, "top_k": 100, "repetition_penalty": 1.2,
                "penalty_window": 50, "min_tokens": 200, "max_tokens": 9000}


# ─────────────────────────── helpers ───────────────────────────

def pick_device(device: str) -> str:
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


def sampling(temp, top_p, top_k, rep, window, min_tokens, max_tokens, label):
    try:
        return adapter.sampling(temperature=float(temp), top_p=float(top_p), top_k=int(top_k),
                                repetition_penalty=float(rep), penalty_window=int(window),
                                min_tokens=int(min_tokens), max_tokens=int(max_tokens))
    except (ValueError, TypeError) as exc:
        raise gr.Error(f"{label} sampling parameters invalid: {exc}") from exc


def _pipe_key(device, dtype, backend, quantization, offload_ar, budget, ode_steps,
              vae_core_frames, model, vae_path, revision, vae_revision, offline):
    return (pick_device(device), dtype, backend, quantization, bool(offload_ar),
            float(budget), int(ode_steps), vae_core_frames, model, vae_path,
            revision or "", vae_revision or "", bool(offline))


FLASH_FALLBACK_NOTE = ("this PyTorch build or GPU cannot run FlashAttention; using upstream's "
                       "eager decoder instead of CUDA graphs (slower per token, same model)")


def effective_backend(device: str, backend: str) -> tuple[str, str]:
    """Route a CUDA host that cannot run upstream's FlashAttention decode kernel to torch-eager.

    Only the ``torch`` backend on a CUDA device is ever remapped.  A Linux host with a
    FlashAttention-capable torch build and an Ampere-or-newer GPU keeps the CUDA-graph path
    untouched; MPS and CPU never consult the probe.  Returns ``(backend, reason)``.

    Temporary: works around yue2-v0.1.6 selecting FlashAttention by op presence alone
    (upstream PR #166 fixes it and keeps CUDA graphs via cuDNN/SDPA).  The contract test
    ``test_cuda_graph_still_selects_flash_attention_by_op_presence_only`` fails once the
    installed upstream carries that fix — remove this function and its callers then.
    """
    if device == "cuda" and backend == "torch" and not adapter.cuda_flash_attention_usable():
        return "torch-eager", FLASH_FALLBACK_NOTE
    return backend, ""


def load_pipeline(device, dtype, backend, quantization, offload_ar, budget, ode_steps,
                  vae_core_frames, model, vae_choice, vae_custom, revision, vae_revision,
                  offline, progress=gr.Progress()):
    """(Re)load the pipeline. Reuses the existing one when settings are unchanged."""
    global _PIPE, _PIPE_KEY
    device = pick_device(device)
    backend, fallback = effective_backend(device, backend)
    fallback = f" ({fallback})" if fallback else ""
    vae_path, vae_name = resolve_vae(vae_choice, vae_custom)
    cores = None if vae_core_frames == "auto" else int(vae_core_frames)
    key = _pipe_key(device, dtype, backend, quantization, offload_ar, budget, ode_steps,
                    cores, model, vae_path, revision, vae_revision, offline)
    if _PIPE is not None and key == _PIPE_KEY:
        return _PIPE, (f"Model ready: device={device} dtype={dtype} backend={backend}{fallback} "
                       f"vae={vae_name}")
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
    note = (f"Loaded: device={device} dtype={used_dtype} backend={backend}{fallback} "
            f"vae={vae_name} ode_steps={ode_steps} cores={cores or 'auto'}")
    return _PIPE, note


def unload_pipeline():
    global _PIPE, _PIPE_KEY
    with _LOCK:
        if _PIPE is not None:
            with contextlib.suppress(Exception):   # closing must never raise
                adapter.close_pipeline(_PIPE)
        _PIPE, _PIPE_KEY = None, None
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def get_pipe(device, dtype, backend, quantization, offload_ar, budget, ode_steps,
              vae_core_frames, model, vae_choice, vae_custom, revision, vae_revision,
              offline, progress):
    vae_path, _ = resolve_vae(vae_choice, vae_custom)
    cores = None if vae_core_frames == "auto" else int(vae_core_frames)
    key = _pipe_key(pick_device(device), dtype, backend, quantization, offload_ar, budget,
                    ode_steps, cores, model, vae_path, revision, vae_revision, offline)
    if _PIPE is None or key != _PIPE_KEY:
        return load_pipeline(device, dtype, backend, quantization, offload_ar, budget,
                             ode_steps, vae_core_frames, model, vae_choice, vae_custom,
                             revision, vae_revision, offline, progress)
    return _PIPE, "Model ready"


def write_local_env(directory: Path, pipe, note: str = "") -> None:
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


def run_dir(*parts: str) -> Path:
    """A fresh, time-stamped directory name under RUNS: 20260919-213000-<parts…>."""
    return RUNS / "-".join((time.strftime("%Y%m%d-%H%M%S"), *parts))


def slug(text: str) -> str:
    out = "".join(c if c.isalnum() or c in "-_" else "-" for c in (text or "")[:40].strip())
    return out.strip("-") or "song"


def artifact_files(directory: Path, score: bool):
    names = ["audio.flac", "request.json", "config.json", "result.json",
             "latent.npy", "semantic.npy"] + (["score.abc"] if score else [])
    return [str(directory / n) for n in names if (directory / n).exists()]


def _looks_like_run(path: Path) -> bool:
    return any((path / name).exists()
               for name in ("latent.npy", "result.json", "plan.json", "decode.json"))


def scan_runs():
    if not RUNS.is_dir():
        return []
    return [str(p) for p in sorted(RUNS.iterdir(), reverse=True)
            if p.is_dir() and not p.name.startswith(".") and _looks_like_run(p)]


def cancel_run():
    CANCEL.set()
    return "Cancel requested — will stop after the current token / ODE step"


# ───────────────────── run durability (panic-safe writes) ─────────────────────
# save_artifacts() closes the files but never fsyncs, and the whole point of the
# 2026-09-13 panics was that a run could be listened to and still vanish when the
# kernel died before APFS flushed it.  A run is only "done" once its files and
# directory are flushed; ".pending" is the marker the Library shows when it is not.
PENDING_FILE = "pending.json"


def _fsync_fd(fd: int) -> None:
    with contextlib.suppress(OSError):
        os.fsync(fd)
    # macOS: fsync only reaches the drive cache, F_FULLFSYNC reaches the media.
    # fcntl is absent on Windows; skip the extra flush there.
    if fcntl is not None and hasattr(fcntl, "F_FULLFSYNC"):
        with contextlib.suppress(OSError):
            fcntl.fcntl(fd, fcntl.F_FULLFSYNC)


def _fsync_path(path) -> None:
    try:
        fd = os.open(Path(path), os.O_RDONLY)
    except OSError:
        return
    try:
        _fsync_fd(fd)
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)


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


def run_generation(pipe, request, outdir, *, abc_sampling, semantic_sampling, progress, note,
                    extra_manifest=None):
    """Generate one song into *outdir* and save its artifacts.

    Shared by 01 GENERATE and 03 EDIT so both flows report progress identically.
    ``extra_manifest`` (a dict) is written next to the run as ``edit_manifest.json``.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _write_pending(outdir, "running")
    log.info("run start: %s", outdir)
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
                                semantic_sampling=semantic_sampling, cancelled=CANCEL.is_set,
                                on_token=on_token, on_progress=on_progress)
        progress(1.0, desc="Saving artifacts…")
        result = song.save_artifacts(outdir)
        if extra_manifest is not None:
            (Path(outdir) / "edit_manifest.json").write_text(
                json.dumps(extra_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_local_env(outdir, pipe, note)
    except Exception as exc:
        _write_pending(outdir, "cancelled" if isinstance(exc, InterruptedError) else "failed",
                       error=f"{type(exc).__name__}: {exc}")
        log.warning("run unfinished: %s — %s: %s", outdir, type(exc).__name__, exc)
        raise
    # flush the artifacts first, then remove the marker: the run only stops being
    # "pending" once it is actually on disk
    _fsync_tree(outdir)
    _clear_pending(outdir)
    log.info("run done: %s", outdir)
    return song, result, time.perf_counter() - t0


def generation_status(song, result, outdir, elapsed, request, note):
    return (f"Done: {result['audio_seconds']:.1f}s audio in {elapsed:.0f}s\n"
            f"truncated={result['truncated']}  seed={request.seed}  cfg={request.guidance}\n"
            f"NAR={song.timing['nar_seconds']:.0f}s  VAE={song.timing['vae_seconds']:.0f}s  "
            f"semantic={song.timing['semantic'].get('output_tps', 0):.1f} tok/s  "
            f"ABC={song.timing['abc'].get('output_tokens', 0)} tokens\n"
            f"run directory: {outdir}\n{note}")


def _load_project_example():
    """Example request that feeds the placeholders and the "fill example" buttons."""
    return (("English, warm piano pop, expressive female voice, acoustic piano, "
             "rounded bass and light drums, 88 BPM"),
            ("[Verse]\nNeon fades along the lane\nFootsteps keep the time of rain\n\n"
             "[Chorus]\nLet the day come into view\nEvery road begins with you"))


EXAMPLE_STYLE, EXAMPLE_LYRICS = _load_project_example()


def request_texts(style, lyrics, *, fallback: bool = False):
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
