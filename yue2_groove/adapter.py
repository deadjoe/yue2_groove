"""The only module that imports ``yue2``.

Everything the UI needs from the upstream package goes through here, so an upstream
change shows up in exactly one file (and in ``tests/test_contract.py``, which asserts
the signatures this module relies on).

Fine-grained NAR/VAE progress: upstream YuE2 exposes ``on_token`` for the AR stages but,
as of ``yue2-v0.1.6``, no callback for the ODE steps / VAE chunks.  A pull request
adding an optional ``on_progress`` keyword is open; ``generate()``/``decode()`` use that
keyword when the installed pipeline has it and otherwise wrap the two upstream functions
that report progress internally (``yue2.nar.synthesize`` and
``YuE2VAE.decode_tiled``).  If neither works, generation still runs — the UI just
shows stage-level progress for those phases.
"""
from __future__ import annotations

import contextlib
import inspect
import sys
import threading
from collections.abc import Callable

INSTALL_HINT = (
    "The `yue2` package is not installed in this environment. Install YuE2 first, e.g.\n"
    "    uv pip install /path/to/YuE   (a clone of https://github.com/multimodal-art-projection/YuE)\n"
    "then install yue2-groove into the same virtual environment (see README)."
)

_PATCH_LOCK = threading.Lock()

StageProgress = Callable[[str, int, int], None]   # (stage, completed, total)


def _require():
    try:
        import yue2  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(INSTALL_HINT) from exc


def yue2_version() -> str:
    try:
        from importlib.metadata import version
        return version("yue2-infer")
    except Exception:  # noqa: BLE001
        return "unknown"


# ── protocol objects ─────────────────────────────────────────────────────────

def sampling(**fields):
    _require()
    from yue2.protocol import Sampling
    return Sampling(**fields)


def sampling_fields(value) -> tuple[str, ...]:
    return tuple(value.__dataclass_fields__)


def generation_config(*, ode_steps: int):
    _require()
    from yue2.protocol import GenerationConfig
    return GenerationConfig(ode_steps=int(ode_steps))


def song_request(**fields):
    _require()
    from yue2.protocol import SongRequest
    return SongRequest(**fields)


def resolve_model(model, *, revision=None, local_files_only=False):
    _require()
    from yue2.storage import resolve_model as _resolve
    return _resolve(model, revision=revision, local_files_only=local_files_only)


# ── pipeline lifecycle ───────────────────────────────────────────────────────

def load_pipeline(model, *, vae, device, dtype, backend, quantization, offload_ar,
                  memory_budget_gib, ode_steps, vae_core_frames, revision, vae_revision,
                  local_files_only):
    """Load the pipeline and its weights; returns ``(pipe, dtype_actually_loaded)``.

    ``dtype`` is ``"bfloat16"`` (upstream's checkpoint dtype) or ``"float32"``.  Upstream
    always loads bf16; an explicit float32 is applied by casting the loaded model, which
    is the one place this package touches a private pipeline attribute.
    """
    _require()
    import torch
    from yue2 import YuE2Pipeline

    pipe = YuE2Pipeline.from_pretrained(
        model, vae=vae, revision=revision or None, vae_revision=vae_revision or None,
        device=device, memory_budget_gib=float(memory_budget_gib), backend=backend,
        quantization=quantization, offload_ar=bool(offload_ar), vae_core_frames=vae_core_frames,
        generation_config=generation_config(ode_steps=ode_steps),
        local_files_only=bool(local_files_only), progress=False,
    )
    pipe._load_model()
    if dtype == "float32" and next(pipe._model.parameters()).dtype != torch.float32:
        pipe._model.float()
    return pipe, model_dtype(pipe)


def model_dtype(pipe) -> str:
    model = getattr(pipe, "_model", None)
    if model is None:
        return "unloaded"
    return str(next(model.parameters()).dtype).replace("torch.", "")


def close_pipeline(pipe) -> None:
    pipe.close()


# ── generation ───────────────────────────────────────────────────────────────

def supports_on_progress(pipe) -> bool:
    """True when the installed pipeline accepts ``on_progress`` on ``__call__``."""
    try:
        return "on_progress" in inspect.signature(type(pipe).__call__).parameters
    except (TypeError, ValueError):
        return False


@contextlib.contextmanager
def _acoustic_progress(on_progress: StageProgress | None):
    """Report NAR/VAE progress through the upstream internals when the pipeline
    predates the ``on_progress`` keyword.  Process-wide and therefore serialized."""
    if on_progress is None:
        yield
        return
    report = on_progress          # the inner wrappers shadow the name `on_progress`
    try:
        import yue2.nar as nar
        from yue2.modeling_vae import YuE2VAE
        original_synthesize, original_tiled = nar.synthesize, YuE2VAE.decode_tiled
    except (ImportError, AttributeError):
        yield            # upstream moved things: degrade to stage-level progress
        return

    def synthesize(*args, on_progress=None, **kwargs):
        def chained(done, total):
            if on_progress is not None:
                on_progress(done, total)
            report("nar", done, total)
        return original_synthesize(*args, on_progress=chained, **kwargs)

    def decode_tiled(self, *args, on_progress=None, **kwargs):
        def chained(done, total):
            if on_progress is not None:
                on_progress(done, total)
            report("vae", done, total)
        return original_tiled(self, *args, on_progress=chained, **kwargs)

    with _PATCH_LOCK:
        nar.synthesize, YuE2VAE.decode_tiled = synthesize, decode_tiled
        try:
            yield
        finally:
            nar.synthesize, YuE2VAE.decode_tiled = original_synthesize, original_tiled


def generate(pipe, request, *, abc_sampling, semantic_sampling, cancelled=None,
             on_token=None, on_progress: StageProgress | None = None):
    """Full song generation; ``on_progress(stage, done, total)`` with stage nar/vae."""
    kwargs = {"style": request.style, "lyrics": request.lyrics, "cot": request.cot,
              "seed": request.seed, "abc": request.abc, "cfg_scale": request.cfg_scale,
              "id": request.id, "abc_sampling": abc_sampling,
              "semantic_sampling": semantic_sampling,
              "cancelled": cancelled, "on_token": on_token}
    if on_progress is not None and supports_on_progress(pipe):
        return pipe(on_progress=on_progress, **kwargs)
    with _acoustic_progress(on_progress):
        return pipe(**kwargs)


def plan(pipe, request, *, abc_sampling, cancelled=None):
    return pipe.plan(request=request, abc_sampling=abc_sampling, cancelled=cancelled)


def decode(pipe, latents, *, full=False, vae=None, on_progress=None):
    """Decode ``[T,64]`` latents; ``on_progress(done, total)`` counts VAE chunks."""
    try:
        native = "on_progress" in inspect.signature(type(pipe).decode).parameters
    except (TypeError, ValueError):
        native = False
    if on_progress is not None and native:
        return pipe.decode(latents, full=full, vae=vae, on_progress=on_progress)
    staged = (lambda stage, done, total: on_progress(done, total)) if on_progress else None
    with _acoustic_progress(staged):
        audio = pipe.decode(latents, full=full, vae=vae)
    if on_progress is not None and full:
        on_progress(1, 1)
    return audio


def doctor_command(model, vae, *, revision="", vae_revision="", offline=False,
                   verify=False) -> list[str]:
    """argv for upstream's environment check (``yue2 doctor``)."""
    cmd = [sys.executable, "-m", "yue2.cli", "doctor", "--model", str(model), "--vae", str(vae)]
    if revision:
        cmd += ["--revision", revision]
    if vae_revision:
        cmd += ["--vae-revision", vae_revision]
    if offline:
        cmd += ["--offline"]
    if verify:
        cmd += ["--verify-hashes"]
    return cmd
