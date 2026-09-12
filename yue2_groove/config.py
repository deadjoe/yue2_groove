"""Paths and defaults, overridable through environment variables.

* ``YUE2_GROOVE_RUNS``    directory for generated works (default: ``./runs`` in the
                          current working directory — the service script runs from the
                          repository root, so that is ``<repo>/runs``)
* ``YUE2_GROOVE_MODEL``   Hugging Face id or local directory of the 3B model
* ``YUE2_GROOVE_VAE``     Hugging Face id or local directory of the listening decoder
* ``YUE2_GROOVE_VAE_LEGACY``  same for the benchmark decoder
* ``YUE2_GROOVE_MODELS``  optional directory holding ``YuE2-3B`` / ``YuE2-Vae`` /
                          ``YuE2-Vae-legacy`` subdirectories (e.g. the ``models/`` folder
                          of a YuE checkout); a subdirectory that exists wins over the id
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
