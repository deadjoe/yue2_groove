"""Paths and defaults, overridable through environment variables.

* ``YUE2_GROOVE_RUNS``    directory for generated works (default: ``./runs`` in the
                          current working directory — the service script runs from the
                          repository root, so that is ``<repo>/runs``)
* ``YUE2_GROOVE_MODEL``   Hugging Face id or local directory of the 3B model
* ``YUE2_GROOVE_VAE``     Hugging Face id or local directory of the listening decoder
* ``YUE2_GROOVE_VAE_LEGACY``  same for the benchmark decoder
* ``YUE2_GROOVE_MODELS``  optional directory holding ``YuE2-3B`` / ``YuE2-Vae`` /
                          ``YuE2-Vae-legacy`` / ``SheetSage2`` subdirectories (e.g. the
                          ``models/`` folder of a YuE checkout); a subdirectory that exists
                          wins over the id

SheetSage2 (audio → ABC transcription) runs in its own virtual environment because its
pinned torch/transformers conflict with YuE2's.  These settings tell the UI where that
environment is; the environment itself is optional — without it the COVER tab reports a
clear configuration error and the rest of the UI keeps working.

* ``YUE2_GROOVE_SHEETSAGE_PYTHON``  python interpreter of the SheetSage2 venv (required
                                    to transcribe; e.g. ``../YuE/.venv-sheetsage2/bin/python``)
* ``YUE2_GROOVE_SHEETSAGE_MODEL``   Hugging Face id or local snapshot directory
                                    (default: ``m-a-p/SheetSage2``)
* ``YUE2_GROOVE_SHEETSAGE_BASE_MODEL``  optional local path of the parent MERT-v2-FullSong
                                    snapshot for offline adapter loads
* ``YUE2_GROOVE_SHEETSAGE_DEVICE``  device for transcription: ``auto`` / ``cuda`` / ``mps`` /
                                    ``cpu`` (default: ``auto``)
* ``YUE2_GROOVE_TRANSCRIPTIONS``    where transcription outputs are written (default:
                                    ``<runs>/transcriptions``)
"""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"
EXAMPLES_DIR = PACKAGE_DIR / "examples"

HUB_MODEL = "m-a-p/YuE2-3B"
HUB_VAE = "m-a-p/YuE2-Vae"
HUB_VAE_LEGACY = "m-a-p/YuE2-Vae-legacy"
HUB_SHEETSAGE = "m-a-p/SheetSage2"


def runs_dir() -> Path:
    return Path(os.environ.get("YUE2_GROOVE_RUNS") or (Path.cwd() / "runs")).expanduser().resolve()


def _local_or_hub(subdir: str, hub_id: str, env: str) -> str:
    explicit = os.environ.get(env)
    if explicit:
        return explicit
    models = os.environ.get("YUE2_GROOVE_MODELS")
    if models:
        candidate = Path(models).expanduser() / subdir
        if candidate.is_dir():
            return str(candidate)
    return hub_id


def default_model() -> str:
    return _local_or_hub("YuE2-3B", HUB_MODEL, "YUE2_GROOVE_MODEL")


def default_vae() -> str:
    return _local_or_hub("YuE2-Vae", HUB_VAE, "YUE2_GROOVE_VAE")


def default_vae_legacy() -> str:
    return _local_or_hub("YuE2-Vae-legacy", HUB_VAE_LEGACY, "YUE2_GROOVE_VAE_LEGACY")


def sheetsage_python() -> str:
    """Interpreter of the separate SheetSage2 environment (empty when not configured)."""
    return (os.environ.get("YUE2_GROOVE_SHEETSAGE_PYTHON") or "").strip()


def default_sheetsage_model() -> str:
    return _local_or_hub("SheetSage2", HUB_SHEETSAGE, "YUE2_GROOVE_SHEETSAGE_MODEL")


def default_sheetsage_base_model() -> str:
    return (os.environ.get("YUE2_GROOVE_SHEETSAGE_BASE_MODEL") or "").strip()


def default_sheetsage_device() -> str:
    value = (os.environ.get("YUE2_GROOVE_SHEETSAGE_DEVICE") or "auto").strip().lower()
    return value if value in ("auto", "cuda", "mps", "cpu") else "auto"


def transcriptions_dir() -> Path:
    override = os.environ.get("YUE2_GROOVE_TRANSCRIPTIONS")
    return Path(override).expanduser().resolve() if override else runs_dir() / "transcriptions"
